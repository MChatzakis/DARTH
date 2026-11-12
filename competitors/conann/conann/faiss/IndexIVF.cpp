/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/ConannCache.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/CodePacker.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/utils.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_roots.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <sys/time.h>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace faiss {

using ScopedIds = InvertedLists::ScopedIds;
using ScopedCodes = InvertedLists::ScopedCodes;

/*****************************************
 * Level1Quantizer implementation
 ******************************************/

Level1Quantizer::Level1Quantizer(Index *quantizer, size_t nlist)
    : quantizer(quantizer), nlist(nlist) {
    // here we set a low # iterations because this is typically used
    // for large clusterings (nb this is not used for the MultiIndex,
    // for which quantizer_trains_alone = true)
    cp.niter = 10;
}

Level1Quantizer::Level1Quantizer() = default;

Level1Quantizer::~Level1Quantizer() {
    if (own_fields) {
        delete quantizer;
    }
}

void Level1Quantizer::train_q1(size_t n, const float *x, bool verbose,
                               MetricType metric_type) {
    size_t d = quantizer->d;
    if (quantizer->is_trained && (quantizer->ntotal == nlist)) {
        if (verbose)
            printf("IVF quantizer does not need training.\n");
    } else if (quantizer_trains_alone == 1) {
        if (verbose)
            printf("IVF quantizer trains alone...\n");
        quantizer->verbose = verbose;
        quantizer->train(n, x);
        FAISS_THROW_IF_NOT_MSG(quantizer->ntotal == nlist,
                               "nlist not consistent with quantizer size");
    } else if (quantizer_trains_alone == 0) {
        if (verbose)
            printf("Training level-1 quantizer on %zd vectors in %zdD\n", n, d);

        Clustering clus(d, nlist, cp);
        quantizer->reset();
        if (clustering_index) {
            clus.train(n, x, *clustering_index);
            quantizer->add(nlist, clus.centroids.data());
        } else {
            clus.train(n, x, *quantizer);
        }
        quantizer->is_trained = true;
    } else if (quantizer_trains_alone == 2) {
        if (verbose) {
            printf("Training L2 quantizer on %zd vectors in %zdD%s\n", n, d,
                   clustering_index ? "(user provided index)" : "");
        }
        // also accept spherical centroids because in that case
        // L2 and IP are equivalent
        FAISS_THROW_IF_NOT(
            metric_type == METRIC_L2 ||
            (metric_type == METRIC_INNER_PRODUCT && cp.spherical));

        Clustering clus(d, nlist, cp);
        if (!clustering_index) {
            IndexFlatL2 assigner(d);
            clus.train(n, x, assigner);
        } else {
            clus.train(n, x, *clustering_index);
        }
        if (verbose) {
            printf("Adding centroids to quantizer\n");
        }
        if (!quantizer->is_trained) {
            if (verbose) {
                printf("But training it first on centroids table...\n");
            }
            quantizer->train(nlist, clus.centroids.data());
        }
        quantizer->add(nlist, clus.centroids.data());
    }
}

size_t Level1Quantizer::coarse_code_size() const {
    size_t nl = nlist - 1;
    size_t nbyte = 0;
    while (nl > 0) {
        nbyte++;
        nl >>= 8;
    }
    return nbyte;
}

void Level1Quantizer::encode_listno(idx_t list_no, uint8_t *code) const {
    // little endian
    size_t nl = nlist - 1;
    while (nl > 0) {
        *code++ = list_no & 0xff;
        list_no >>= 8;
        nl >>= 8;
    }
}

idx_t Level1Quantizer::decode_listno(const uint8_t *code) const {
    size_t nl = nlist - 1;
    int64_t list_no = 0;
    int nbit = 0;
    while (nl > 0) {
        list_no |= int64_t(*code++) << nbit;
        nbit += 8;
        nl >>= 8;
    }
    FAISS_THROW_IF_NOT(list_no >= 0 && list_no < nlist);
    return list_no;
}

/*****************************************
 * IndexIVF implementation
 ******************************************/

IndexIVF::IndexIVF(Index *quantizer, size_t d, size_t nlist, size_t code_size,
                   MetricType metric)
    : Index(d, metric), IndexIVFInterface(quantizer, nlist),
      invlists(new ArrayInvertedLists(nlist, code_size)), own_invlists(true),
      code_size(code_size) {
    FAISS_THROW_IF_NOT(d == quantizer->d);
    is_trained = quantizer->is_trained && (quantizer->ntotal == nlist);
    // Spherical by default if the metric is inner_product
    if (metric_type == METRIC_INNER_PRODUCT) {
        cp.spherical = true;
    }

    // ConANN block
    n_list = nlist;
    // ------------------------
}

IndexIVF::IndexIVF() = default;

void IndexIVF::add(idx_t n, const float *x) { add_with_ids(n, x, nullptr); }

void IndexIVF::add_with_ids(idx_t n, const float *x, const idx_t *xids) {
    std::unique_ptr<idx_t[]> coarse_idx(new idx_t[n]);
    quantizer->assign(n, x, coarse_idx.get());
    add_core(n, x, xids, coarse_idx.get());

    // ConANN Block
    // Add data to the exact index
    std::shared_ptr<std::vector<float>> centroids_flat =
        std::make_shared<std::vector<float>>(n_list * quantizer->d);
    quantizer->reconstruct_n(0, n_list, centroids_flat->data());

    for (size_t i = 0; i < n_list; ++i) {
        std::vector<float> centroid(quantizer->d);
        std::copy(centroids_flat->begin() + i * quantizer->d,
                  centroids_flat->begin() + (i + 1) * quantizer->d,
                  centroid.begin());
        centroids.push_back(std::move(centroid));
    }
    // ----------------------------
}

void IndexIVF::add_sa_codes(idx_t n, const uint8_t *codes, const idx_t *xids) {
    size_t coarse_size = coarse_code_size();
    DirectMapAdd dm_adder(direct_map, n, xids);

    for (idx_t i = 0; i < n; i++) {
        const uint8_t *code = codes + (code_size + coarse_size) * i;
        idx_t list_no = decode_listno(code);
        idx_t id = xids ? xids[i] : ntotal + i;
        size_t ofs = invlists->add_entry(list_no, id, code + coarse_size);
        dm_adder.add(i, list_no, ofs);
    }
    ntotal += n;
}

void IndexIVF::add_core(idx_t n, const float *x, const idx_t *xids,
                        const idx_t *coarse_idx, void *inverted_list_context) {
    // do some blocking to avoid excessive allocs
    idx_t bs = 65536;
    if (n > bs) {
        for (idx_t i0 = 0; i0 < n; i0 += bs) {
            idx_t i1 = std::min(n, i0 + bs);
            if (verbose) {
                printf("   IndexIVF::add_with_ids %" PRId64 ":%" PRId64 "\n",
                       i0, i1);
            }
            add_core(i1 - i0, x + i0 * d, xids ? xids + i0 : nullptr,
                     coarse_idx + i0, inverted_list_context);
        }
        return;
    }
    FAISS_THROW_IF_NOT(coarse_idx);
    FAISS_THROW_IF_NOT(is_trained);
    direct_map.check_can_add(xids);

    size_t nadd = 0, nminus1 = 0;

    for (size_t i = 0; i < n; i++) {
        if (coarse_idx[i] < 0)
            nminus1++;
    }

    std::unique_ptr<uint8_t[]> flat_codes(new uint8_t[n * code_size]);
    encode_vectors(n, x, coarse_idx, flat_codes.get());

    DirectMapAdd dm_adder(direct_map, n, xids);

#pragma omp parallel reduction(+ : nadd)
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // each thread takes care of a subset of lists
        for (size_t i = 0; i < n; i++) {
            idx_t list_no = coarse_idx[i];
            if (list_no >= 0 && list_no % nt == rank) {
                idx_t id = xids ? xids[i] : ntotal + i;
                size_t ofs = invlists->add_entry(
                    list_no, id, flat_codes.get() + i * code_size,
                    inverted_list_context);

                dm_adder.add(i, list_no, ofs);

                nadd++;
            } else if (rank == 0 && list_no == -1) {
                dm_adder.add(i, -1, 0);
            }
        }
    }

    if (verbose) {
        printf("    added %zd / %" PRId64 " vectors (%zd -1s)\n", nadd, n,
               nminus1);
    }

    ntotal += n;
}

void IndexIVF::make_direct_map(bool b) {
    if (b) {
        direct_map.set_type(DirectMap::Array, invlists, ntotal);
    } else {
        direct_map.set_type(DirectMap::NoMap, invlists, ntotal);
    }
}

void IndexIVF::set_direct_map_type(DirectMap::Type type) {
    direct_map.set_type(type, invlists, ntotal);
}

/** It is a sad fact of software that a conceptually simple function like this
 * becomes very complex when you factor in several ways of parallelizing +
 * interrupt/error handling + collecting stats + min/max collection. The
 * codepath that is used 95% of time is the one for parallel_mode = 0 */
void IndexIVF::search(idx_t n, const float *x, idx_t k, float *distances,
                      idx_t *labels, const SearchParameters *params_in) const {
    FAISS_THROW_IF_NOT(k > 0);
    const IVFSearchParameters *params = nullptr;
    if (params_in) {
        params = dynamic_cast<const IVFSearchParameters *>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "IndexIVF params have incorrect type");
    }
    const size_t nprobe =
        std::min(nlist, params ? params->nprobe : this->nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    // search function for a subset of queries
    auto sub_search_func = [this, k, nprobe,
                            params](idx_t n, const float *x, float *distances,
                                    idx_t *labels, IndexIVFStats *ivf_stats) {
        std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe]);
        std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

        double t0 = getmillisecs();

        // NOTE(sonia): The quantizer computes distances between the query
        // vector and the centroids, returning the indices (idx) and distances
        // (coarse_dis) of the top nprobe nearest centroids.
        quantizer->search(n, x, nprobe, coarse_dis.get(), idx.get(),
                          params ? params->quantizer_params : nullptr);

        double t1 = getmillisecs();
        invlists->prefetch_lists(idx.get(), n * nprobe);

        // NOTE(sonia): It performs a pairwise distance calculation between the
        // query vector and the vectors in the corresponding clusters (closest
        // nlist).
        search_preassigned(n, x, k, idx.get(), coarse_dis.get(), distances,
                           labels, false, params, ivf_stats);
        double t2 = getmillisecs();
        ivf_stats->quantization_time += t1 - t0;
        ivf_stats->search_time += t2 - t0;
    };

    if ((parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT) == 0) {
        int nt = std::min(omp_get_max_threads(), int(n));
        std::vector<IndexIVFStats> stats(nt);
        std::mutex exception_mutex;
        std::string exception_string;

#pragma omp parallel for if (nt > 1)
        for (idx_t slice = 0; slice < nt; slice++) {
            IndexIVFStats local_stats;
            idx_t i0 = n * slice / nt;
            idx_t i1 = n * (slice + 1) / nt;
            if (i1 > i0) {
                try {
                    sub_search_func(i1 - i0, x + i0 * d, distances + i0 * k,
                                    labels + i0 * k, &stats[slice]);
                } catch (const std::exception &e) {
                    std::lock_guard<std::mutex> lock(exception_mutex);
                    exception_string = e.what();
                }
            }
        }

        if (!exception_string.empty()) {
            FAISS_THROW_MSG(exception_string.c_str());
        }

        // collect stats
        for (idx_t slice = 0; slice < nt; slice++) {
            indexIVF_stats.add(stats[slice]);
        }
    } else {
        // handle parallelization at level below (or don't run in parallel at
        // all)
        sub_search_func(n, x, distances, labels, &indexIVF_stats);
    }
}

void IndexIVF::search_preassigned(idx_t n, const float *x, idx_t k,
                                  const idx_t *keys, const float *coarse_dis,
                                  float *distances, idx_t *labels,
                                  bool store_pairs,
                                  const IVFSearchParameters *params,
                                  IndexIVFStats *ivf_stats) const {
    FAISS_THROW_IF_NOT(k > 0);

    idx_t nprobe = params ? params->nprobe : this->nprobe;
    nprobe = std::min((idx_t)nlist, nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    const idx_t unlimited_list_size = std::numeric_limits<idx_t>::max();
    idx_t max_codes = params ? params->max_codes : this->max_codes;
    IDSelector *sel = params ? params->sel : nullptr;
    const IDSelectorRange *selr = dynamic_cast<const IDSelectorRange *>(sel);
    if (selr) {
        if (selr->assume_sorted) {
            sel = nullptr; // use special IDSelectorRange processing
        } else {
            selr = nullptr; // use generic processing
        }
    }

    FAISS_THROW_IF_NOT_MSG(!(sel && store_pairs),
                           "selector and store_pairs cannot be combined");

    FAISS_THROW_IF_NOT_MSG(
        !invlists->use_iterator || (max_codes == 0 && store_pairs == false),
        "iterable inverted lists don't support max_codes and store_pairs");

    size_t nlistv = 0, ndis = 0, nheap = 0;

    using HeapForIP = CMin<float, idx_t>;
    using HeapForL2 = CMax<float, idx_t>;

    bool interrupt = false;
    std::mutex exception_mutex;
    std::string exception_string;

    int pmode = this->parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT;
    bool do_heap_init = !(this->parallel_mode & PARALLEL_MODE_NO_HEAP_INIT);

    FAISS_THROW_IF_NOT_MSG(
        max_codes == 0 || pmode == 0 || pmode == 3,
        "max_codes supported only for parallel_mode = 0 or 3");

    if (max_codes == 0) {
        max_codes = unlimited_list_size;
    }

    [[maybe_unused]] bool do_parallel =
        omp_get_max_threads() >= 2 && (pmode == 0   ? false
                                       : pmode == 3 ? n > 1
                                       : pmode == 1 ? nprobe > 1
                                                    : nprobe * n > 1);

    void *inverted_list_context =
        params ? params->inverted_list_context : nullptr;

#pragma omp parallel if (do_parallel) reduction(+ : nlistv, ndis, nheap)
    {
        std::unique_ptr<InvertedListScanner> scanner(
            get_InvertedListScanner(store_pairs, sel));

        /*****************************************************
         * Depending on parallel_mode, there are two possible ways
         * to organize the search. Here we define local functions
         * that are in common between the two
         ******************************************************/

        // initialize + reorder a result heap

        // NOTE(sonia): initializes the result heap for each query
        auto init_result = [&](float *simi, idx_t *idxi) {
            if (!do_heap_init)
                return;
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_heapify<HeapForIP>(k, simi, idxi);
            } else {
                heap_heapify<HeapForL2>(k, simi, idxi);
            }
        };

        // NOTE(sonia): updates the heap with new distances and indices from
        // each cluster
        auto add_local_results = [&](const float *local_dis,
                                     const idx_t *local_idx, float *simi,
                                     idx_t *idxi) {
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_addn<HeapForIP>(k, simi, idxi, local_dis, local_idx, k);
            } else {
                heap_addn<HeapForL2>(k, simi, idxi, local_dis, local_idx, k);
            }
        };

        // NOTE(sonia): Once all clusters have been scanned, the heap is
        // reordered to return the results in sorted order
        auto reorder_result = [&](float *simi, idx_t *idxi) {
            if (!do_heap_init)
                return;
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_reorder<HeapForIP>(k, simi, idxi);
            } else {
                heap_reorder<HeapForL2>(k, simi, idxi);
            }
        };

        // single list scan using the current scanner (with query
        // set porperly) and storing results in simi and idxi
        auto scan_one_list = [&](idx_t key, float coarse_dis_i, float *simi,
                                 idx_t *idxi, idx_t list_size_max) {
            if (key < 0) {
                // not enough centroids for multiprobe
                return (size_t)0;
            }
            FAISS_THROW_IF_NOT_FMT(key < (idx_t)nlist,
                                   "Invalid key=%" PRId64 " nlist=%zd\n", key,
                                   nlist);

            // don't waste time on empty lists
            if (invlists->is_empty(key, inverted_list_context)) {
                return (size_t)0;
            }

            scanner->set_list(key, coarse_dis_i);

            nlistv++;

            try {
                if (invlists->use_iterator) {
                    size_t list_size = 0;

                    std::unique_ptr<InvertedListsIterator> it(
                        invlists->get_iterator(key, inverted_list_context));

                    nheap += scanner->iterate_codes(it.get(), simi, idxi, k,
                                                    list_size);

                    return list_size;
                } else {
                    size_t list_size = invlists->list_size(key);
                    if (list_size > list_size_max) {
                        list_size = list_size_max;
                    }

                    InvertedLists::ScopedCodes scodes(invlists, key);
                    const uint8_t *codes = scodes.get();

                    std::unique_ptr<InvertedLists::ScopedIds> sids;
                    const idx_t *ids = nullptr;

                    if (!store_pairs) {
                        sids = std::make_unique<InvertedLists::ScopedIds>(
                            invlists, key);
                        ids = sids->get();
                    }

                    if (selr) { // IDSelectorRange
                        // restrict search to a section of the inverted list
                        size_t jmin, jmax;
                        selr->find_sorted_ids_bounds(list_size, ids, &jmin,
                                                     &jmax);
                        list_size = jmax - jmin;
                        if (list_size == 0) {
                            return (size_t)0;
                        }
                        codes += jmin * code_size;
                        ids += jmin;
                    }

                    nheap += scanner->scan_codes(list_size, codes, ids, simi,
                                                 idxi, k);

                    return list_size;
                }
            } catch (const std::exception &e) {
                std::lock_guard<std::mutex> lock(exception_mutex);
                exception_string =
                    demangle_cpp_symbol(typeid(e).name()) + "  " + e.what();
                interrupt = true;
                return size_t(0);
            }
        };

        /****************************************************
         * Actual loops, depending on parallel_mode
         ****************************************************/

        if (pmode == 0 || pmode == 3) {
#pragma omp for
            for (idx_t i = 0; i < n; i++) {
                if (interrupt) {
                    continue;
                }

                // loop over queries
                scanner->set_query(x + i * d);
                float *simi = distances + i * k;
                idx_t *idxi = labels + i * k;

                init_result(simi, idxi);

                idx_t nscan = 0;

                // NOTE(sonia): adding results by searching each nprobe.
                // Here is where the intermediate results get updated!!

                // loop over probes
                for (size_t ik = 0; ik < nprobe; ik++) {
                    nscan += scan_one_list(keys[i * nprobe + ik],
                                           coarse_dis[i * nprobe + ik], simi,
                                           idxi, max_codes - nscan);
                    if (nscan >= max_codes) {
                        break;
                    }
                }

                ndis += nscan;
                reorder_result(simi, idxi);
                if (InterruptCallback::is_interrupted()) {
                    interrupt = true;
                }

            } // parallel for
        } else if (pmode == 1) {
            std::vector<idx_t> local_idx(k);
            std::vector<float> local_dis(k);

            for (size_t i = 0; i < n; i++) {
                scanner->set_query(x + i * d);
                init_result(local_dis.data(), local_idx.data());

#pragma omp for schedule(dynamic)
                for (idx_t ik = 0; ik < nprobe; ik++) {
                    ndis += scan_one_list(keys[i * nprobe + ik],
                                          coarse_dis[i * nprobe + ik],
                                          local_dis.data(), local_idx.data(),
                                          unlimited_list_size);

                    // can't do the test on max_codes
                }
                // merge thread-local results

                float *simi = distances + i * k;
                idx_t *idxi = labels + i * k;
#pragma omp single
                init_result(simi, idxi);

#pragma omp barrier
#pragma omp critical
                {
                    add_local_results(local_dis.data(), local_idx.data(), simi,
                                      idxi);
                }
#pragma omp barrier
#pragma omp single
                reorder_result(simi, idxi);
            }
        } else if (pmode == 2) {
            std::vector<idx_t> local_idx(k);
            std::vector<float> local_dis(k);

#pragma omp single
            for (int64_t i = 0; i < n; i++) {
                init_result(distances + i * k, labels + i * k);
            }

#pragma omp for schedule(dynamic)
            for (int64_t ij = 0; ij < n * nprobe; ij++) {
                size_t i = ij / nprobe;

                scanner->set_query(x + i * d);
                init_result(local_dis.data(), local_idx.data());
                ndis +=
                    scan_one_list(keys[ij], coarse_dis[ij], local_dis.data(),
                                  local_idx.data(), unlimited_list_size);

#pragma omp critical
                {
                    add_local_results(local_dis.data(), local_idx.data(),
                                      distances + i * k, labels + i * k);
                }
            }
#pragma omp single
            for (int64_t i = 0; i < n; i++) {
                reorder_result(distances + i * k, labels + i * k);
            }
        } else {
            FAISS_THROW_FMT("parallel_mode %d not supported\n", pmode);
        }
    } // parallel section

    if (interrupt) {
        if (!exception_string.empty()) {
            FAISS_THROW_FMT("search interrupted with: %s",
                            exception_string.c_str());
        } else {
            FAISS_THROW_MSG("computation interrupted");
        }
    }

    if (ivf_stats == nullptr) {
        ivf_stats = &indexIVF_stats;
    }
    ivf_stats->nq += n;
    ivf_stats->nlist += nlistv;
    ivf_stats->ndis += ndis;
    ivf_stats->nheap_updates += nheap;
}

void IndexIVF::range_search(idx_t nx, const float *x, float radius,
                            RangeSearchResult *result,
                            const SearchParameters *params_in) const {
    const IVFSearchParameters *params = nullptr;
    const SearchParameters *quantizer_params = nullptr;
    if (params_in) {
        params = dynamic_cast<const IVFSearchParameters *>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "IndexIVF params have incorrect type");
        quantizer_params = params->quantizer_params;
    }
    const size_t nprobe =
        std::min(nlist, params ? params->nprobe : this->nprobe);
    std::unique_ptr<idx_t[]> keys(new idx_t[nx * nprobe]);
    std::unique_ptr<float[]> coarse_dis(new float[nx * nprobe]);

    double t0 = getmillisecs();
    quantizer->search(nx, x, nprobe, coarse_dis.get(), keys.get(),
                      quantizer_params);
    indexIVF_stats.quantization_time += getmillisecs() - t0;

    t0 = getmillisecs();
    invlists->prefetch_lists(keys.get(), nx * nprobe);

    range_search_preassigned(nx, x, radius, keys.get(), coarse_dis.get(),
                             result, false, params, &indexIVF_stats);

    indexIVF_stats.search_time += getmillisecs() - t0;
}

void IndexIVF::range_search_preassigned(
    idx_t nx, const float *x, float radius, const idx_t *keys,
    const float *coarse_dis, RangeSearchResult *result, bool store_pairs,
    const IVFSearchParameters *params, IndexIVFStats *stats) const {
    idx_t nprobe = params ? params->nprobe : this->nprobe;
    nprobe = std::min((idx_t)nlist, nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    idx_t max_codes = params ? params->max_codes : this->max_codes;
    IDSelector *sel = params ? params->sel : nullptr;

    FAISS_THROW_IF_NOT_MSG(
        !invlists->use_iterator || (max_codes == 0 && store_pairs == false),
        "iterable inverted lists don't support max_codes and store_pairs");

    size_t nlistv = 0, ndis = 0;

    bool interrupt = false;
    std::mutex exception_mutex;
    std::string exception_string;

    std::vector<RangeSearchPartialResult *> all_pres(omp_get_max_threads());

    int pmode = this->parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT;
    // don't start parallel section if single query
    [[maybe_unused]] bool do_parallel =
        omp_get_max_threads() >= 2 && (pmode == 3   ? false
                                       : pmode == 0 ? nx > 1
                                       : pmode == 1 ? nprobe > 1
                                                    : nprobe * nx > 1);

    void *inverted_list_context =
        params ? params->inverted_list_context : nullptr;

#pragma omp parallel if (do_parallel) reduction(+ : nlistv, ndis)
    {
        RangeSearchPartialResult pres(result);
        std::unique_ptr<InvertedListScanner> scanner(
            get_InvertedListScanner(store_pairs, sel));
        FAISS_THROW_IF_NOT(scanner.get());
        all_pres[omp_get_thread_num()] = &pres;

        // prepare the list scanning function

        auto scan_list_func = [&](size_t i, size_t ik, RangeQueryResult &qres) {
            idx_t key = keys[i * nprobe + ik]; /* select the list  */
            if (key < 0)
                return;
            FAISS_THROW_IF_NOT_FMT(key < (idx_t)nlist,
                                   "Invalid key=%" PRId64
                                   " at ik=%zd nlist=%zd\n",
                                   key, ik, nlist);

            if (invlists->is_empty(key, inverted_list_context)) {
                return;
            }

            try {
                size_t list_size = 0;
                scanner->set_list(key, coarse_dis[i * nprobe + ik]);
                if (invlists->use_iterator) {
                    std::unique_ptr<InvertedListsIterator> it(
                        invlists->get_iterator(key, inverted_list_context));

                    scanner->iterate_codes_range(it.get(), radius, qres,
                                                 list_size);
                } else {
                    InvertedLists::ScopedCodes scodes(invlists, key);
                    InvertedLists::ScopedIds ids(invlists, key);
                    list_size = invlists->list_size(key);

                    scanner->scan_codes_range(list_size, scodes.get(),
                                              ids.get(), radius, qres);
                }
                nlistv++;
                ndis += list_size;
            } catch (const std::exception &e) {
                std::lock_guard<std::mutex> lock(exception_mutex);
                exception_string =
                    demangle_cpp_symbol(typeid(e).name()) + "  " + e.what();
                interrupt = true;
            }
        };

        if (parallel_mode == 0) {
#pragma omp for
            for (idx_t i = 0; i < nx; i++) {
                scanner->set_query(x + i * d);

                RangeQueryResult &qres = pres.new_result(i);

                for (size_t ik = 0; ik < nprobe; ik++) {
                    scan_list_func(i, ik, qres);
                }
            }

        } else if (parallel_mode == 1) {
            for (size_t i = 0; i < nx; i++) {
                scanner->set_query(x + i * d);

                RangeQueryResult &qres = pres.new_result(i);

#pragma omp for schedule(dynamic)
                for (int64_t ik = 0; ik < nprobe; ik++) {
                    scan_list_func(i, ik, qres);
                }
            }
        } else if (parallel_mode == 2) {
            RangeQueryResult *qres = nullptr;

#pragma omp for schedule(dynamic)
            for (idx_t iik = 0; iik < nx * (idx_t)nprobe; iik++) {
                idx_t i = iik / (idx_t)nprobe;
                idx_t ik = iik % (idx_t)nprobe;
                if (qres == nullptr || qres->qno != i) {
                    qres = &pres.new_result(i);
                    scanner->set_query(x + i * d);
                }
                scan_list_func(i, ik, *qres);
            }
        } else {
            FAISS_THROW_FMT("parallel_mode %d not supported\n", parallel_mode);
        }
        if (parallel_mode == 0) {
            pres.finalize();
        } else {
#pragma omp barrier
#pragma omp single
            RangeSearchPartialResult::merge(all_pres, false);
#pragma omp barrier
        }
    }

    if (interrupt) {
        if (!exception_string.empty()) {
            FAISS_THROW_FMT("search interrupted with: %s",
                            exception_string.c_str());
        } else {
            FAISS_THROW_MSG("computation interrupted");
        }
    }

    if (stats == nullptr) {
        stats = &indexIVF_stats;
    }
    stats->nq += nx;
    stats->nlist += nlistv;
    stats->ndis += ndis;
}

InvertedListScanner *
IndexIVF::get_InvertedListScanner(bool /*store_pairs*/,
                                  const IDSelector * /* sel */) const {
    FAISS_THROW_MSG("get_InvertedListScanner not implemented");
}

void IndexIVF::reconstruct(idx_t key, float *recons) const {
    idx_t lo = direct_map.get(key);
    reconstruct_from_offset(lo_listno(lo), lo_offset(lo), recons);
}

void IndexIVF::reconstruct_n(idx_t i0, idx_t ni, float *recons) const {
    FAISS_THROW_IF_NOT(ni == 0 || (i0 >= 0 && i0 + ni <= ntotal));

    for (idx_t list_no = 0; list_no < nlist; list_no++) {
        size_t list_size = invlists->list_size(list_no);
        ScopedIds idlist(invlists, list_no);

        for (idx_t offset = 0; offset < list_size; offset++) {
            idx_t id = idlist[offset];
            if (!(id >= i0 && id < i0 + ni)) {
                continue;
            }

            float *reconstructed = recons + (id - i0) * d;
            reconstruct_from_offset(list_no, offset, reconstructed);
        }
    }
}

bool IndexIVF::check_ids_sorted() const {
    size_t nflip = 0;

    for (size_t i = 0; i < nlist; i++) {
        size_t list_size = invlists->list_size(i);
        InvertedLists::ScopedIds ids(invlists, i);
        for (size_t j = 0; j + 1 < list_size; j++) {
            if (ids[j + 1] < ids[j]) {
                nflip++;
            }
        }
    }
    return nflip == 0;
}

/* standalone codec interface */
size_t IndexIVF::sa_code_size() const {
    size_t coarse_size = coarse_code_size();
    return code_size + coarse_size;
}

void IndexIVF::sa_encode(idx_t n, const float *x, uint8_t *bytes) const {
    FAISS_THROW_IF_NOT(is_trained);
    std::unique_ptr<int64_t[]> idx(new int64_t[n]);
    quantizer->assign(n, x, idx.get());
    encode_vectors(n, x, idx.get(), bytes, true);
}

void IndexIVF::search_and_reconstruct(idx_t n, const float *x, idx_t k,
                                      float *distances, idx_t *labels,
                                      float *recons,
                                      const SearchParameters *params_in) const {
    const IVFSearchParameters *params = nullptr;
    if (params_in) {
        params = dynamic_cast<const IVFSearchParameters *>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "IndexIVF params have incorrect type");
    }
    const size_t nprobe =
        std::min(nlist, params ? params->nprobe : this->nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe]);
    std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

    quantizer->search(n, x, nprobe, coarse_dis.get(), idx.get());

    invlists->prefetch_lists(idx.get(), n * nprobe);

    // search_preassigned() with `store_pairs` enabled to obtain the list_no
    // and offset into `codes` for reconstruction
    search_preassigned(n, x, k, idx.get(), coarse_dis.get(), distances, labels,
                       true /* store_pairs */, params);
#pragma omp parallel for if (n * k > 1000)
    for (idx_t ij = 0; ij < n * k; ij++) {
        idx_t key = labels[ij];
        float *reconstructed = recons + ij * d;
        if (key < 0) {
            // Fill with NaNs
            memset(reconstructed, -1, sizeof(*reconstructed) * d);
        } else {
            int list_no = lo_listno(key);
            int offset = lo_offset(key);

            // Update label to the actual id
            labels[ij] = invlists->get_single_id(list_no, offset);

            reconstruct_from_offset(list_no, offset, reconstructed);
        }
    }
}

void IndexIVF::search_and_return_codes(
    idx_t n, const float *x, idx_t k, float *distances, idx_t *labels,
    uint8_t *codes, bool include_listno,
    const SearchParameters *params_in) const {
    const IVFSearchParameters *params = nullptr;
    if (params_in) {
        params = dynamic_cast<const IVFSearchParameters *>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "IndexIVF params have incorrect type");
    }
    const size_t nprobe =
        std::min(nlist, params ? params->nprobe : this->nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe]);
    std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

    quantizer->search(n, x, nprobe, coarse_dis.get(), idx.get());

    invlists->prefetch_lists(idx.get(), n * nprobe);

    // search_preassigned() with `store_pairs` enabled to obtain the list_no
    // and offset into `codes` for reconstruction
    search_preassigned(n, x, k, idx.get(), coarse_dis.get(), distances, labels,
                       true /* store_pairs */, params);

    size_t code_size_1 = code_size;
    if (include_listno) {
        code_size_1 += coarse_code_size();
    }

#pragma omp parallel for if (n * k > 1000)
    for (idx_t ij = 0; ij < n * k; ij++) {
        idx_t key = labels[ij];
        uint8_t *code1 = codes + ij * code_size_1;

        if (key < 0) {
            // Fill with 0xff
            memset(code1, -1, code_size_1);
        } else {
            int list_no = lo_listno(key);
            int offset = lo_offset(key);
            const uint8_t *cc = invlists->get_single_code(list_no, offset);

            labels[ij] = invlists->get_single_id(list_no, offset);

            if (include_listno) {
                encode_listno(list_no, code1);
                code1 += code_size_1 - code_size;
            }
            memcpy(code1, cc, code_size);
        }
    }
}

void IndexIVF::reconstruct_from_offset(int64_t /*list_no*/, int64_t /*offset*/,
                                       float * /*recons*/) const {
    FAISS_THROW_MSG("reconstruct_from_offset not implemented");
}

void IndexIVF::reset() {
    direct_map.clear();
    invlists->reset();
    ntotal = 0;
}

size_t IndexIVF::remove_ids(const IDSelector &sel) {
    size_t nremove = direct_map.remove_ids(sel, invlists);
    ntotal -= nremove;
    return nremove;
}

void IndexIVF::update_vectors(int n, const idx_t *new_ids, const float *x) {
    if (direct_map.type == DirectMap::Hashtable) {
        // just remove then add
        IDSelectorArray sel(n, new_ids);
        size_t nremove = remove_ids(sel);
        FAISS_THROW_IF_NOT_MSG(nremove == n,
                               "did not find all entries to remove");
        add_with_ids(n, x, new_ids);
        return;
    }

    FAISS_THROW_IF_NOT(direct_map.type == DirectMap::Array);
    // here it is more tricky because we don't want to introduce holes
    // in continuous range of ids

    FAISS_THROW_IF_NOT(is_trained);
    std::vector<idx_t> assign(n);
    quantizer->assign(n, x, assign.data());

    std::vector<uint8_t> flat_codes(n * code_size);
    encode_vectors(n, x, assign.data(), flat_codes.data());

    direct_map.update_codes(invlists, n, new_ids, assign.data(),
                            flat_codes.data());
}

void IndexIVF::train(idx_t n, const float *x) {
    if (verbose) {
        printf("Training level-1 quantizer\n");
    }

    train_q1(n, x, verbose, metric_type);

    if (verbose) {
        printf("Training IVF residual\n");
    }

    // optional subsampling
    idx_t max_nt = train_encoder_num_vectors();
    if (max_nt <= 0) {
        max_nt = (size_t)1 << 35;
    }

    TransformedVectors tv(
        x, fvecs_maybe_subsample(d, (size_t *)&n, max_nt, x, verbose));

    if (by_residual) {
        std::vector<idx_t> assign(n);
        quantizer->assign(n, tv.x, assign.data());

        std::vector<float> residuals(n * d);
        quantizer->compute_residual_n(n, tv.x, residuals.data(), assign.data());

        train_encoder(n, residuals.data(), assign.data());
    } else {
        train_encoder(n, tv.x, nullptr);
    }

    is_trained = true;
}

void IndexIVF::prep_execution(float alpha, float calib_sz, float tune_sz,
                              const float *queries, size_t nq,
                              const faiss::idx_t *gt) {
    prep_execution_counts(alpha, size_t(calib_sz * nq),
                          size_t(tune_sz * nq), queries, nq, gt);
}

void IndexIVF::prep_execution_counts(float alpha, size_t calib_nq,
                                     size_t tune_nq, const float *queries,
                                     size_t nq, const faiss::idx_t *gt) {

    if (calib_nq + tune_nq >= nq) {
        FAISS_THROW_MSG(
                "ConANN calibration and tuning counts must leave at least one test query");
    }

    std::cout << "Starting to prep execution: " << std::endl;

    const size_t conann_nprobe = std::min(nlist, nprobe);

    // Scores and incremental predictions are limited to the same maximum
    // number of IVF probes configured for the DARTH index.
    std::vector<std::vector<float>> all_nonconf_scores;
    std::vector<std::vector<std::vector<faiss::idx_t>>> all_preds;

    std::string cacheKeyNonConf{dataset_name + "_" + std::to_string(n_list) +
                                "_np" + std::to_string(conann_nprobe) + "_" +
                                std::to_string(K) + "_" +
                                "nonconf_scores"};
    std::string cacheKeyAllPreds{dataset_name + "_" + std::to_string(n_list) +
                                 "_np" + std::to_string(conann_nprobe) + "_" +
                                 std::to_string(K) + "_" + "all_preds"};
    if (enable_cache && conann_cache::check_cached_file(cacheKeyNonConf) &&
        conann_cache::check_cached_file(cacheKeyAllPreds)) {
        all_nonconf_scores =
            conann_cache::read_from_cache<std::vector<std::vector<float>>>(
                cacheKeyNonConf);
        all_preds = conann_cache::read_from_cache<
            std::vector<std::vector<std::vector<faiss::idx_t>>>>(
            cacheKeyAllPreds);
    } else {
        double t1 = elapsed();
        // NOTE: pass lamhat > 1 here to make sure all scores get computed
        // using std::tie in this instance is really important for performance
        // to avoid the extra copy
        std::tie(all_nonconf_scores, all_preds) =
            compute_scores(CalibrationResults{10, 0, 0}, nq, queries);
        time_report.computeScores = elapsed() - t1;
        time_report.computeScoresCalib =
                time_report.computeScores * static_cast<double>(calib_nq) / nq;
        time_report.computeScoresTune =
                time_report.computeScores * static_cast<double>(tune_nq) / nq;
        std::cout << "Time spent computing scores: " << elapsed() - t1
                  << std::endl;

        if (enable_cache) {
            conann_cache::write_to_cache(cacheKeyNonConf, all_nonconf_scores);
            conann_cache::write_to_cache(cacheKeyAllPreds, all_preds);
        }
    }

    double t1 = elapsed();
    // slice computed data and store on index
    // NOTE: This coping can become quite expensive but should still be worth it
    // in combination with caching
    size_t test_nq = nq - calib_nq - tune_nq;
    std::cout << "Calibration query size: " << calib_nq
              << "\nTune query size: " << tune_nq
              << "\nTest query size: " << test_nq << std::endl;

    // Resize containers for all three parts
    calib_cx.resize(calib_nq);
    calib_labels.resize(calib_nq);
    tune_cx.resize(tune_nq);
    tune_labels.resize(tune_nq);
    test_cx.resize(test_nq);
    test_labels.resize(test_nq);

    // Copy calibration data
    for (size_t i = 0; i < calib_nq; ++i) {
        calib_cx[i].resize(d);
        std::memcpy(calib_cx[i].data(), queries + i * d, d * sizeof(float));

        calib_labels[i].resize(K);
        std::memcpy(calib_labels[i].data(), gt + i * K,
                    K * sizeof(faiss::idx_t));
    }
    calib_nonconf = std::vector<std::vector<float>>(
        all_nonconf_scores.begin(), all_nonconf_scores.begin() + calib_nq);

    calib_preds = std::vector<std::vector<std::vector<faiss::idx_t>>>(
        all_preds.begin(), all_preds.begin() + calib_nq);

    // Copy tuning data
    for (size_t i = 0; i < tune_nq; ++i) {
        tune_cx[i].resize(d);
        std::memcpy(tune_cx[i].data(), queries + (i + calib_nq) * d,
                    d * sizeof(float));

        tune_labels[i].resize(K);
        std::memcpy(tune_labels[i].data(), gt + (i + calib_nq) * K,
                    K * sizeof(faiss::idx_t));
    }
    tune_nonconf = std::vector<std::vector<float>>(
        all_nonconf_scores.begin() + calib_nq,
        all_nonconf_scores.begin() + calib_nq + tune_nq);
    tune_preds = std::vector<std::vector<std::vector<faiss::idx_t>>>(
        all_preds.begin() + calib_nq, all_preds.begin() + calib_nq + tune_nq);

    // Copy testing data
    for (size_t i = 0; i < test_nq; ++i) {
        test_cx[i].resize(d);
        std::memcpy(test_cx[i].data(), queries + (i + calib_nq + tune_nq) * d,
                    d * sizeof(float));

        test_labels[i].resize(K);
        std::memcpy(test_labels[i].data(), gt + (i + calib_nq + tune_nq) * K,
                    K * sizeof(faiss::idx_t));
    }
    test_nonconf = std::vector<std::vector<float>>(all_nonconf_scores.begin() +
                                                       calib_nq + tune_nq,
                                                   all_nonconf_scores.end());
    test_preds = std::vector<std::vector<std::vector<faiss::idx_t>>>(
        all_preds.begin() + calib_nq + tune_nq, all_preds.end());

    time_report.memoryCopyPostCompute = elapsed() - t1;
    std::cout << "Time spent doing memcpy: " << elapsed() - t1 << std::endl;
}

std::tuple<std::vector<std::vector<float>>,
           std::vector<std::vector<std::vector<faiss::idx_t>>>>
IndexIVF::compute_scores(CalibrationResults cal_params, faiss::idx_t num_queries,
                         const float *queries) {
    // result vector for nearest neighbor ids
    std::vector<faiss::idx_t> nns(K * num_queries);
    // result vector for nearest neigbor distances
    std::vector<float> dis(K * num_queries);

    const size_t conann_nprobe = std::min(nlist, nprobe);

    // Result vectors for the lists reachable under the configured nprobe.
    std::vector<std::vector<float>> nonconf_list(
        num_queries, std::vector<float>(conann_nprobe, 0.0f));

    // result vector for predicted vector ids of all K neighbors for each query
    // as the number of searched lists increases. shape: nq * nprobe * k
    std::vector<std::vector<std::vector<faiss::idx_t>>> all_preds_list(
        num_queries, std::vector<std::vector<faiss::idx_t>>(
                         conann_nprobe, std::vector<faiss::idx_t>(K)));

    search_with_error_quantification(
        cal_params, num_queries, queries, K, dis.data(), nns.data(),
        nonconf_list.data(), all_preds_list.data());

    return std::make_tuple(nonconf_list, all_preds_list);
}

double IndexIVF::elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

IndexIVF::CalibrationResults
IndexIVF::calibrate(float alpha, int k, float calib_sz, float tune_sz,
                    float *xq, size_t nq, faiss::idx_t *gt, float max_distance,
                    std::string dataset) {
    K = k;
    MAX_DISTANCE = max_distance;
    dataset_name = dataset;

    double t0 = elapsed();
    prep_execution(alpha, calib_sz, tune_sz, xq, nq, gt);

    // NOTE: randomization disabled. Can enable easier by having a class-level
    // boolean. int kreg = pickKreg(tune_nonconf, alpha); // Regularization
    // hyperparameter
    double t1 = elapsed();
    int kreg = 1;
    float lambda_reg = pick_lambda_reg(alpha, kreg);
    time_report.pickRegLambda = elapsed() - t1;
    // std::cout << "Calib hyperparameters: kreg=" << kreg
    //           << " reg-lambda=" << lambda_reg << "\n";

    auto lamhat = optimization(alpha, kreg, lambda_reg, calib_cx, calib_labels,
                               calib_nonconf, calib_preds);
    // std::cout << "Time spent optimizing: " << elapsed() - t1 << std::endl;
    time_report.configureTotal = elapsed() - t0; 
    return CalibrationResults{lamhat, kreg, lambda_reg};
}

IndexIVF::CalibrationResults IndexIVF::calibrate_counts(
        float alpha,
        int k,
        size_t calib_nq,
        size_t tune_nq,
        float *xq,
        size_t nq,
        faiss::idx_t *gt,
        float max_distance,
        std::string dataset) {
    K = k;
    MAX_DISTANCE = max_distance;
    dataset_name = dataset;

    double t0 = elapsed();
    prep_execution_counts(alpha, calib_nq, tune_nq, xq, nq, gt);

    int kreg = 1;
    double t1 = elapsed();
    float lambda_reg = pick_lambda_reg(alpha, kreg);
    time_report.pickRegLambda = elapsed() - t1;

    auto lamhat = optimization(alpha, kreg, lambda_reg, calib_cx, calib_labels,
                               calib_nonconf, calib_preds);
    time_report.configureTotal = elapsed() - t0;
    return CalibrationResults{lamhat, kreg, lambda_reg};
}

float IndexIVF::optimization(
    float alpha, int kreg, float lambda_reg,
    const std::vector<std::vector<float>> &queries,
    const std::vector<std::vector<faiss::idx_t>> &labels,
    const std::vector<std::vector<float>> &nonconf_scores,
    const std::vector<std::vector<std::vector<faiss::idx_t>>> &all_preds) {

    double t1 = elapsed();
    auto sorted_indices_cn = compute_sorted_indices(nonconf_scores);
    auto reg_nonconf_scores =
        regularize_scores(nonconf_scores, sorted_indices_cn, lambda_reg, kreg);
    time_report.regularizeScores = elapsed() - t1;

    t1 = elapsed();
    int n = queries.size();
    float target_fnr =
        (static_cast<float>(n) + 1.0f) / n * alpha - 1.0f / (n + 1.0f);

    // Logger to get loss function information
    // std::cout << "Opening log file" << std::endl;
    // freopen("../lossf.log", "w", stdout);

    // Use GSL's root-finding for the brentq method
    gsl_root_fsolver *solver = gsl_root_fsolver_alloc(gsl_root_fsolver_brent);
    gsl_function F;
    struct LamhatParams {
        IndexIVF *index_ivf;
        float target_fnr;
        const std::vector<std::vector<float>> *queries;
        const std::vector<std::vector<faiss::idx_t>> *labels;
        const std::vector<std::vector<float>> *nonconf_scores;
        const std::vector<std::vector<std::vector<faiss::idx_t>>> *all_preds;
    };

    F.function = [](double lambda, void *params) -> double {
        auto *args = static_cast<LamhatParams *>(params);
        return args->index_ivf->lamhat_threshold(
            static_cast<float>(lambda), args->target_fnr, *(args->queries),
            *(args->labels), *(args->nonconf_scores), *(args->all_preds));
    };
    LamhatParams params = {this,    target_fnr,          &queries,
                           &labels, &reg_nonconf_scores, &all_preds};
    F.params = &params;

    double lamhat = 2.0f;
    int status;

    gsl_set_error_handler_off();

    try {
        double lower_bound = 0.0f;
        double upper_bound = 1.0f;
        gsl_root_fsolver_set(solver, &F, lower_bound, upper_bound);

        int max_iter = 100;
        int iter = 0;

        do {
            iter++;
            gsl_root_fsolver_iterate(solver);
            lamhat = gsl_root_fsolver_root(solver);
            lower_bound = gsl_root_fsolver_x_lower(solver);
            upper_bound = gsl_root_fsolver_x_upper(solver);
            status =
                gsl_root_test_interval(lower_bound, upper_bound, 1e-6, 1e-6);
        } while (status == GSL_CONTINUE && iter < max_iter);
    } catch (const std::exception &e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown error occurred." << std::endl;
    }

    // std::cout << std::endl;
    // fclose(stdout);
    // freopen("/dev/tty", "w", stdout); // Restore stdout to terminal
    // std::cout << "Log file closed." << std::endl;
    // gsl_root_fsolver_free(solver);
    time_report.optimize = elapsed() - t1;

    if (status != GSL_SUCCESS) {
        std::cerr << "Root-finding failed to converge.\n";
    }

    return (float)lamhat;
}

double IndexIVF::lamhat_threshold(
    float lambda, float target_fnr,
    const std::vector<std::vector<float>> &queries,
    const std::vector<std::vector<faiss::idx_t>> &labels,
    const std::vector<std::vector<float>> &nonconf_scores,
    const std::vector<std::vector<std::vector<faiss::idx_t>>> &all_preds) {
    auto [preds, _] =
        compute_predictions(lambda, nonconf_scores, all_preds);
    float fnr = false_negative_rate(preds, labels);
    // std::cout << "Optimization: lambda=" << lambda << " fnr=" << fnr << "\n";
    // std::cout << lambda << " " << fnr << "\n";

    return fnr - target_fnr;
}

std::pair<std::vector<std::vector<faiss::idx_t>>, std::vector<int>>
IndexIVF::compute_predictions(
    float lambda,
    const std::vector<std::vector<float>> &nonconf_scores,
    const std::vector<std::vector<std::vector<faiss::idx_t>>> &all_preds) {

    std::vector<std::vector<faiss::idx_t>> test_preds;
    std::vector<int> cl_searched;

    for (size_t query_idx = 0; query_idx < nonconf_scores.size(); ++query_idx) {
        const auto &sc = nonconf_scores[query_idx];
        const auto &p = all_preds[query_idx];

        std::vector<std::pair<float, size_t>> indexed_sc;
        for (size_t i = 0; i < sc.size(); ++i) {
            indexed_sc.push_back(
                {sc[i], i}); // Store the value and its original index
        }

        std::sort(indexed_sc.begin(), indexed_sc.end(),
                  std::less<>()); // Sort by value in ascending order

        // new implementation
        float optimal_sc = -1;
        int index = indexed_sc.size();
        int num_cls_searched = 0;
        for (size_t i = 0; i < indexed_sc.size(); ++i) {
            if (indexed_sc[i].first <= lambda) {
                optimal_sc = indexed_sc[i].first;
                index = indexed_sc[i].second;
                num_cls_searched = i+1;
            } else {
                break;
            }
        }

        if (index < sc.size() && num_cls_searched > 0) {
            test_preds.push_back({p[index]});
            cl_searched.push_back(num_cls_searched);
        } else {
            test_preds.push_back({});
            cl_searched.push_back(-1);
        }
    }
    return {test_preds, cl_searched};
}

void IndexIVF::search_conann(idx_t n, const float *x, float *distances,
                             idx_t *labels, CalibrationResults calib_params) {
    
    search_with_error_quantification(
        calib_params, n, x, K, distances, labels, nullptr, nullptr);
}

std::vector<float> IndexIVF::recall_per_query(
    const std::vector<std::vector<faiss::idx_t>> &prediction_set,
    const std::vector<std::vector<faiss::idx_t>> &gt_labels) {

    int nq = prediction_set.size();
    int k = gt_labels[0].size();
    int total_false_negatives = 0;
    std::vector<float> fnrs_per_query(nq);

    for (size_t i = 0; i < nq; ++i) {
        const std::set<int> pred_set(prediction_set[i].begin(),
                                     prediction_set[i].end());
        const std::set<int> gt_set(gt_labels[i].begin(), gt_labels[i].end());
        // Calculate intersection size
        int intersection_size = 0;
        for (int pred : pred_set) {
            if (gt_set.count(pred) > 0) {
                ++intersection_size;
            }
        }
        // FNR = 1 - (intersection_size / k)
        fnrs_per_query[i] = 1.0f - (static_cast<float>(intersection_size) /
                                    static_cast<float>(k));
        total_false_negatives += k - intersection_size;
    }

    float overall_fnr = static_cast<float>(total_false_negatives) / (nq * k);
    float check_fnr =
        std::accumulate(fnrs_per_query.begin(), fnrs_per_query.end(), 0.0) /
        fnrs_per_query.size();
    std::cout << "Overall FNR: " << overall_fnr << "; Check FNR: " << check_fnr
              << std::endl;
    return fnrs_per_query;
}

double IndexIVF::false_negative_rate(
    const std::vector<std::vector<faiss::idx_t>> &prediction_set,
    const std::vector<std::vector<faiss::idx_t>> &gt_labels) {
    std::vector<int> overlap;
    for (size_t i = 0; i < prediction_set.size(); ++i) {
        const std::set<int> pred_set(prediction_set[i].begin(),
                                     prediction_set[i].end());
        const std::set<int> gt_set(gt_labels[i].begin(), gt_labels[i].end());
        // Calculate intersection size
        int intersection_size = 0;
        for (int pred : pred_set) {
            if (gt_set.count(pred) > 0) {
                ++intersection_size;
            }
        }
        overlap.push_back(intersection_size);
    }

    int sum_overlap = std::accumulate(overlap.begin(), overlap.end(), 0);
    int sum_gt_sums = 0;
    for (const auto &gt : gt_labels) {
        sum_gt_sums += gt.size();
    }
    // Return the false negative rate (1 - sum(overlap) / sum(gt_sums))
    if (sum_gt_sums > 0) {
        return 1.0f - static_cast<float>(sum_overlap) / sum_gt_sums;
    } else {
        return 0.0f;
    }
}

std::pair<std::vector<float>, std::vector<int>>
IndexIVF::evaluate_test(CalibrationResults params) {
    return evaluate(params, test_cx, test_labels, test_nonconf, test_preds);
}

std::pair<std::vector<float>, std::vector<int>> IndexIVF::evaluate(
    CalibrationResults params, const std::vector<std::vector<float>> &queries,
    const std::vector<std::vector<faiss::idx_t>> &labels,
    const std::vector<std::vector<float>> &nonconf_scores,
    const std::vector<std::vector<std::vector<faiss::idx_t>>> &all_preds) {

    float regLambda = params.regLambda; // Regularization hyperparameter
    int kreg = params.kreg;             // Regularization hyperparameter
    std::cout << "eval hyperparameters: kreg=" << kreg
              << " reg-lambda=" << regLambda << "\n";

    auto t1 = elapsed();
    auto sortedIndices = compute_sorted_indices(nonconf_scores);
    auto reg_nonconf_scores =
        regularize_scores(nonconf_scores, sortedIndices, regLambda, kreg);
    std::cout << "Time spent regularizing scores: " << elapsed() - t1
              << std::endl;
          
    t1 = elapsed();
    auto [test_preds, cl_searched] = compute_predictions(
        params.lamhat, reg_nonconf_scores, all_preds);
    std::cout << "Time spent computing predictions: " << elapsed() - t1
              << std::endl;

    auto fnrs = recall_per_query(test_preds, labels);
    return {fnrs, cl_searched};
}

// -------

std::vector<std::pair<int, float>> IndexIVF::sort_classes_by_probability(
    const std::vector<float> &class_probabilities) const {
    std::vector<std::pair<int, float>> sorted_classes;
    for (int i = 0; i < class_probabilities.size(); ++i) {
        sorted_classes.emplace_back(i, class_probabilities[i]);
    }
    // Sort by probability in descending order
    std::sort(sorted_classes.begin(), sorted_classes.end(),
              [](const auto &a, const auto &b) { return b.second < a.second; });
    return sorted_classes;
}

std::vector<int> IndexIVF::compute_ox(
    const std::vector<std::pair<int, float>> &sorted_classes) const {
    std::vector<int> ox(sorted_classes.size(), 0);
    for (size_t i = 0; i < sorted_classes.size(); ++i) {
        ox[i] = i + 1;
    }
    return ox;
}

float IndexIVF::compute_regularization(int ox_y, float lambda, int kreg) const {
    return lambda * std::max(0, ox_y - kreg);
}

std::vector<std::vector<int>> IndexIVF::compute_sorted_indices(
    const std::vector<std::vector<float>> &class_probabilities) const {
    std::vector<std::vector<int>> sorted_indices;
    for (const auto &probs : class_probabilities) {
        std::vector<std::pair<int, float>> indexed_classes;
        for (int i = 0; i < probs.size(); ++i) {
            indexed_classes.emplace_back(i, probs[i]);
        }
        std::sort(
            indexed_classes.begin(), indexed_classes.end(),
            [](const auto &a, const auto &b) { return b.second < a.second; });
        std::vector<int> sorted_class_indices;
        for (const auto &pair : indexed_classes) {
            sorted_class_indices.push_back(pair.first);
        }
        sorted_indices.push_back(sorted_class_indices);
    }
    return sorted_indices;
}

int IndexIVF::pick_kreg(const std::vector<std::vector<float>> &scores_per_q,
                        float alpha) const {
    size_t n = scores_per_q.size();
    std::vector<int> rank_per_query;
    for (const auto &row : scores_per_q) {
        std::set<float> unique_scores(row.begin(), row.end());
        int highest_rank = unique_scores.size();
        rank_per_query.push_back(1);
    }
    std::vector<int> sorted_ranks = rank_per_query;
    std::sort(sorted_ranks.begin(), sorted_ranks.end());
    int kstar_idx = std::ceil((1.0f - alpha) * (n + 1));
    int kstar = sorted_ranks[kstar_idx];
    return kstar;
}

float IndexIVF::pick_lambda_reg(float alpha, int kreg) const {
    int best_size = tune_nonconf.empty() ? 0 : tune_nonconf.front().size();
    float lambda_star = 0;
    std::vector<float> lambda_values = {0.0, 0.001, 0.01, 0.1};
    for (float temp_lambda : lambda_values) {
        auto lamhat = const_cast<faiss::IndexIVF *>(this)->optimization(
            alpha, kreg, temp_lambda, tune_cx, tune_labels, tune_nonconf,
            tune_preds);
        auto params = CalibrationResults{lamhat, kreg, temp_lambda};
        auto [fnrs, cls] = const_cast<faiss::IndexIVF *>(this)->evaluate(
            params, tune_cx, tune_labels, tune_nonconf, tune_preds);
        float average_fnr = std::accumulate(fnrs.begin(), fnrs.end(), 0.0f) / fnrs.size();
        float avg_cls_searched =
            std::accumulate(cls.begin(), cls.end(), 0.0) / cls.size();
        std::cout << "Avg cls searched=" << avg_cls_searched << "\n";
        if (avg_cls_searched < best_size && average_fnr <= alpha) {
            lambda_star = temp_lambda;
            best_size = avg_cls_searched;
            std::cout << "Found better lambda_reg=" << lambda_star
                      << ". Updating.\n";
        }
    }
    std::cout << "Best lambda_reg found=" << lambda_star << "\n";
    return lambda_star;
}

std::vector<std::vector<float>>
IndexIVF::regularize_scores(const std::vector<std::vector<float>> &s,
                            const std::vector<std::vector<int>> &I,
                            float lambda_reg, int kreg) const {
    size_t n = s.size();
    size_t K = s[0].size();
    std::vector<std::vector<float>> E(n, std::vector<float>(K, 0.0f));
    float max_reg_val = (1 + lambda_reg * (K - kreg)) + 10;
    for (size_t i = 0; i < n; ++i) {
        std::vector<std::pair<int, float>> sorted_classes;
        for (int j = 0; j < K; ++j) {
            sorted_classes.push_back({j, s[i][j]});
        }
        std::sort(
            sorted_classes.rbegin(), sorted_classes.rend(),
            [](const std::pair<int, float> &a, const std::pair<int, float> &b) {
                return a.second < b.second;
            });
        auto ox = compute_ox(sorted_classes);
        for (size_t j = 0; j < K; ++j) {
            int original_class_index = sorted_classes[j].first;
            float Eij = 1.0f - s[i][original_class_index];
            Eij += compute_regularization(ox[j], lambda_reg, kreg);
            E[i][original_class_index] = Eij / max_reg_val;
        }
    }
    return E;
}

// ------------

// parallelized search, executes search_preassigned_with_error_quantification
// internally
void IndexIVF::search_with_error_quantification(
    CalibrationResults cal_params, idx_t n, const float *x, idx_t k, float *distances,
    idx_t *labels, std::vector<float> *nonconf_list,
    std::vector<std::vector<faiss::idx_t>> *all_preds_list,
    const SearchParameters *params_in) const {

    FAISS_THROW_IF_NOT(k > 0);

    const IVFSearchParameters *params = nullptr;
    if (params_in) {
        params = dynamic_cast<const IVFSearchParameters *>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "IndexIVF params have incorrect type");
    }
    const size_t nprobe =
        std::min(nlist, params ? params->nprobe : this->nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    // search function for a subset of queries
    auto sub_search_func =
        [this, k, nprobe,
         params](CalibrationResults cal_params, idx_t n, const float *x, float *distances,
                 idx_t *labels, IndexIVFStats *ivf_stats,
                 std::vector<float> *nonconf_list,
                 std::vector<std::vector<faiss::idx_t>> *all_preds_list) {
            // flattened list of the cluster ids of each cluster to
            // incrementally search for current list of queries
            std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe]);
            // flattened list of the distances to each cluster to incrementally
            // search for current list of queries
            std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

            double t0 = getmillisecs();

            quantizer->search(n, x, nprobe, coarse_dis.get(), idx.get(),
                              params ? params->quantizer_params : nullptr);

            double t1 = getmillisecs();
            invlists->prefetch_lists(idx.get(), n * nprobe);

            search_preassigned_with_error_quantification(
                cal_params, n, x, k, idx.get(), coarse_dis.get(), distances, labels,
                false, nonconf_list, all_preds_list, params, ivf_stats);

            double t2 = getmillisecs();
            ivf_stats->quantization_time += t1 - t0;
            ivf_stats->search_time += t2 - t0;
        };

    if ((parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT) == 0) {
        int nt = std::min(omp_get_max_threads(), int(n));
        std::vector<IndexIVFStats> stats(nt);
        std::mutex exception_mutex;
        std::string exception_string;

#pragma omp parallel for if (nt > 1)
        for (idx_t slice = 0; slice < nt; slice++) {
            IndexIVFStats local_stats;
            idx_t i0 = n * slice / nt;
            idx_t i1 = n * (slice + 1) / nt;
            if (i1 > i0) {
                try {
                    if (all_preds_list == nullptr) {
                    // Note: this bit of ugliness is needed because pointer arithmetic on a nullptr is undefined behaviour
                    sub_search_func(cal_params, i1 - i0, x + i0 * d,
                                    distances + i0 * k, labels + i0 * k,
                                    &stats[slice], nullptr, nullptr);
                    } else {
                    // Note: pointer arithmetic is used to share datastructures
                    // between threads
                    sub_search_func(cal_params, i1 - i0, x + i0 * d,
                                    distances + i0 * k, labels + i0 * k,
                                    &stats[slice], nonconf_list + i0,
                                    all_preds_list + i0);
                    }
                } catch (const std::exception &e) {
                    std::lock_guard<std::mutex> lock(exception_mutex);
                    exception_string = e.what();
                }
            }
        }

        if (!exception_string.empty()) {
            FAISS_THROW_MSG(exception_string.c_str());
        }

        for (idx_t slice = 0; slice < nt; slice++) {
            indexIVF_stats.add(stats[slice]);
        }
    } else {
        std::cout << "WARNING:: hopefully this case is never reached...\n";
        // sub_search_func(n, x, distances, labels, &indexIVF_stats);
    }
}

// faiss search execution and conann non-conformity score calculations
void IndexIVF::search_preassigned_with_error_quantification(
    CalibrationResults cal_params, idx_t n, const float *x, idx_t k, const idx_t *keys,
    const float *coarse_dis, float *distances, idx_t *labels, bool store_pairs,
    std::vector<float> *nonconf_list,
    std::vector<std::vector<faiss::idx_t>> *all_preds_list,
    const IVFSearchParameters *params, IndexIVFStats *ivf_stats) const {
    FAISS_THROW_IF_NOT(k > 0);

    idx_t nprobe = params ? params->nprobe : this->nprobe;
    nprobe = std::min((idx_t)nlist, nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    const idx_t unlimited_list_size = std::numeric_limits<idx_t>::max();
    idx_t max_codes = params ? params->max_codes : this->max_codes;
    IDSelector *sel = params ? params->sel : nullptr;
    const IDSelectorRange *selr = dynamic_cast<const IDSelectorRange *>(sel);
    if (selr) {
        if (selr->assume_sorted) {
            sel = nullptr; // use special IDSelectorRange processing
        } else {
            selr = nullptr; // use generic processing
        }
    }

    FAISS_THROW_IF_NOT_MSG(!(sel && store_pairs),
                           "selector and store_pairs cannot be combined");

    FAISS_THROW_IF_NOT_MSG(
        !invlists->use_iterator || (max_codes == 0 && store_pairs == false),
        "iterable inverted lists don't support max_codes and store_pairs");

    size_t nlistv = 0, ndis = 0, nheap = 0;

    using HeapForIP = CMin<float, idx_t>;
    using HeapForL2 = CMax<float, idx_t>;

    bool interrupt = false;
    std::mutex exception_mutex;
    std::string exception_string;

    int pmode = this->parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT;
    bool do_heap_init = !(this->parallel_mode & PARALLEL_MODE_NO_HEAP_INIT);

    FAISS_THROW_IF_NOT_MSG(
        max_codes == 0 || pmode == 0 || pmode == 3,
        "max_codes supported only for parallel_mode = 0 or 3");

    if (max_codes == 0) {
        max_codes = unlimited_list_size;
    }

    [[maybe_unused]] bool do_parallel =
        omp_get_max_threads() >= 2 && (pmode == 0   ? false
                                       : pmode == 3 ? n > 1
                                       : pmode == 1 ? nprobe > 1
                                                    : nprobe * n > 1);

    void *inverted_list_context =
        params ? params->inverted_list_context : nullptr;

#pragma omp parallel if (do_parallel) reduction(+ : nlistv, ndis, nheap)
    {
        std::unique_ptr<InvertedListScanner> scanner(
            get_InvertedListScanner(store_pairs, sel));

        /*****************************************************
         * Depending on parallel_mode, there are two possible ways
         * to organize the search. Here we define local functions
         * that are in common between the two
         ******************************************************/

        // initialize + reorder a result heap

        // NOTE(sonia): initializes the result heap for each query
        auto init_result = [&](float *simi, idx_t *idxi) {
            if (!do_heap_init)
                return;
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_heapify<HeapForIP>(k, simi, idxi);
            } else {
                heap_heapify<HeapForL2>(k, simi, idxi);
            }
        };

        auto add_local_results = [&](const float *local_dis,
                                     const idx_t *local_idx, float *simi,
                                     idx_t *idxi) {
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_addn<HeapForIP>(k, simi, idxi, local_dis, local_idx, k);
            } else {
                heap_addn<HeapForL2>(k, simi, idxi, local_dis, local_idx, k);
            }
        };

        auto reorder_result = [&](float *simi, idx_t *idxi) {
            if (!do_heap_init)
                return;
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_reorder<HeapForIP>(k, simi, idxi);
            } else {
                heap_reorder<HeapForL2>(k, simi, idxi);
            }
        };

        // single list scan using the current scanner (with query
        // set porperly) and storing results in simi and idxi
        auto scan_one_list = [&](idx_t key, float coarse_dis_i, float *simi,
                                 idx_t *idxi, idx_t list_size_max) {
            if (key < 0) {
                // not enough centroids for multiprobe
                return (size_t)0;
            }
            FAISS_THROW_IF_NOT_FMT(key < (idx_t)nlist,
                                   "Invalid key=%" PRId64 " nlist=%zd\n", key,
                                   nlist);

            // don't waste time on empty lists
            if (invlists->is_empty(key, inverted_list_context)) {
                return (size_t)0;
            }

            scanner->set_list(key, coarse_dis_i);

            nlistv++;

            try {
                if (invlists->use_iterator) {
                    size_t list_size = 0;

                    std::unique_ptr<InvertedListsIterator> it(
                        invlists->get_iterator(key, inverted_list_context));

                    nheap += scanner->iterate_codes(it.get(), simi, idxi, k,
                                                    list_size);

                    return list_size;
                } else {
                    // NOTE: here
                    size_t list_size = invlists->list_size(key);
                    if (list_size > list_size_max) {
                        list_size = list_size_max;
                    }

                    InvertedLists::ScopedCodes scodes(invlists, key);
                    const uint8_t *codes = scodes.get();

                    std::unique_ptr<InvertedLists::ScopedIds> sids;
                    const idx_t *ids = nullptr;

                    if (!store_pairs) {
                        sids = std::make_unique<InvertedLists::ScopedIds>(
                            invlists, key);
                        ids = sids->get();
                    }

                    if (selr) { // IDSelectorRange
                        // restrict search to a section of the inverted list
                        size_t jmin, jmax;
                        selr->find_sorted_ids_bounds(list_size, ids, &jmin,
                                                     &jmax);
                        list_size = jmax - jmin;
                        if (list_size == 0) {
                            return (size_t)0;
                        }
                        codes += jmin * code_size;
                        ids += jmin;
                    }

                    nheap += scanner->scan_codes(list_size, codes, ids, simi,
                                                 idxi, k);

                    return list_size;
                }
            } catch (const std::exception &e) {
                std::lock_guard<std::mutex> lock(exception_mutex);
                exception_string =
                    demangle_cpp_symbol(typeid(e).name()) + "  " + e.what();
                interrupt = true;
                return size_t(0);
            }
        };

        /****************************************************
         * Actual loops, depending on parallel_mode
         ****************************************************/

        if (pmode == 0 || pmode == 3) {
#pragma omp for
            for (idx_t i = 0; i < n; i++) {
                if (interrupt) {
                    continue;
                }

                // loop over queries
                scanner->set_query(x + i * d);
                float *simi = distances + i * k;
                idx_t *idxi = labels + i * k;

                init_result(simi, idxi);

                idx_t nscan = 0;
                

                std::vector<faiss::idx_t> prev_idxi(idxi, idxi + k);
                std::vector<float> prev_simi(simi, simi + k);
                for (size_t ik = 0; ik < nprobe; ik++) {
                    nscan += scan_one_list(keys[i * nprobe + ik],
                                           coarse_dis[i * nprobe + ik], simi,
                                           idxi, max_codes - nscan);
                    if (nscan >= max_codes) {
                        break;
                    }

                    // Find largest score corresponding to the kth closest
                    // vector. Needed because FAISS does not maintain them
                    // ordered.
                    float score_k = *std::max_element(simi, simi + k);

                    if (all_preds_list == nullptr) {
                        // NOTE: early stopping mode, code used for search_conann only
                        // check for early stopping; need to attempt to regularize the score
                        if (cal_params.lamhat <= 1) {
                            float max_reg_val = (1 + cal_params.regLambda * (nprobe - cal_params.kreg)) + 10;
                            float nonconf_score = std::min(score_k / MAX_DISTANCE, 1.0f);
                            float reg_score_k = (1 - nonconf_score) + compute_regularization(ik + 1, cal_params.regLambda, cal_params.kreg);
                            reg_score_k = reg_score_k / max_reg_val;
                            if (reg_score_k > cal_params.lamhat) {
                                // We have searched one cluster more than needed so we can return the results of the previous iteration
                                std::memcpy(idxi, prev_idxi.data(), prev_idxi.size() * sizeof(idx_t));
                                std::memcpy(simi, prev_simi.data(), prev_simi.size() * sizeof(float));
                                break;
                            } 
                        }
                    } else {
                        std::vector<faiss::idx_t> idxi_copy(idxi, idxi + k);

                        // add results for query i
                        (*(all_preds_list + i))[ik] = idxi_copy;
                        if (score_k > MAX_DISTANCE) {
                            (*(nonconf_list + i))[ik] = 1.0;
                        } else {
                            (*(nonconf_list + i))[ik] =
                                score_k / MAX_DISTANCE;
                        }
                    }
                    // keep the results of this iteration in case of early stopping
                    std::memcpy(prev_idxi.data(), idxi, k * sizeof(idx_t));
                    std::memcpy(prev_simi.data(), simi, k * sizeof(float));
                }
                

                ndis += nscan;
                reorder_result(simi, idxi);

                if (InterruptCallback::is_interrupted()) {
                    interrupt = true;
                }

            } // parallel for
        }
        // TODO(sonia): other parallel modes
    } // parallel section

    if (interrupt) {
        if (!exception_string.empty()) {
            FAISS_THROW_FMT("search interrupted with: %s",
                            exception_string.c_str());
        } else {
            FAISS_THROW_MSG("computation interrupted");
        }
    }

    if (ivf_stats == nullptr) {
        ivf_stats = &indexIVF_stats;
    }
    ivf_stats->nq += n;
    ivf_stats->nlist += nlistv;
    ivf_stats->ndis += ndis;
    ivf_stats->nheap_updates += nheap;
}

idx_t IndexIVF::train_encoder_num_vectors() const { return 0; }

void IndexIVF::train_encoder(idx_t /*n*/, const float * /*x*/,
                             const idx_t *assign) {
    // does nothing by default
    if (verbose) {
        printf("IndexIVF: no residual training\n");
    }
}

bool check_compatible_for_merge_expensive_check = true;

void IndexIVF::check_compatible_for_merge(const Index &otherIndex) const {
    // minimal sanity checks
    const IndexIVF *other = dynamic_cast<const IndexIVF *>(&otherIndex);
    FAISS_THROW_IF_NOT(other);
    FAISS_THROW_IF_NOT(other->d == d);
    FAISS_THROW_IF_NOT(other->nlist == nlist);
    FAISS_THROW_IF_NOT(quantizer->ntotal == other->quantizer->ntotal);
    FAISS_THROW_IF_NOT(other->code_size == code_size);
    FAISS_THROW_IF_NOT_MSG(typeid(*this) == typeid(*other),
                           "can only merge indexes of the same type");
    FAISS_THROW_IF_NOT_MSG(this->direct_map.no() && other->direct_map.no(),
                           "merge direct_map not implemented");

    if (check_compatible_for_merge_expensive_check) {
        std::vector<float> v(d), v2(d);
        for (size_t i = 0; i < nlist; i++) {
            quantizer->reconstruct(i, v.data());
            other->quantizer->reconstruct(i, v2.data());
            FAISS_THROW_IF_NOT_MSG(v == v2,
                                   "coarse quantizers should be the same");
        }
    }
}

void IndexIVF::merge_from(Index &otherIndex, idx_t add_id) {
    check_compatible_for_merge(otherIndex);
    IndexIVF *other = static_cast<IndexIVF *>(&otherIndex);
    invlists->merge_from(other->invlists, add_id);

    ntotal += other->ntotal;
    other->ntotal = 0;
}

CodePacker *IndexIVF::get_CodePacker() const {
    return new CodePackerFlat(code_size);
}

void IndexIVF::replace_invlists(InvertedLists *il, bool own) {
    if (own_invlists) {
        delete invlists;
        invlists = nullptr;
    }
    // FAISS_THROW_IF_NOT (ntotal == 0);
    if (il) {
        FAISS_THROW_IF_NOT(il->nlist == nlist);
        FAISS_THROW_IF_NOT(il->code_size == code_size ||
                           il->code_size == InvertedLists::INVALID_CODE_SIZE);
    }
    invlists = il;
    own_invlists = own;
}

void IndexIVF::copy_subset_to(IndexIVF &other,
                              InvertedLists::subset_type_t subset_type,
                              idx_t a1, idx_t a2) const {
    other.ntotal +=
        invlists->copy_subset_to(*other.invlists, subset_type, a1, a2);
}

IndexIVF::~IndexIVF() {
    if (own_invlists) {
        delete invlists;
    }
}

/*************************************************************************
 * IndexIVFStats
 *************************************************************************/

void IndexIVFStats::reset() { memset((void *)this, 0, sizeof(*this)); }

void IndexIVFStats::add(const IndexIVFStats &other) {
    nq += other.nq;
    nlist += other.nlist;
    ndis += other.ndis;
    nheap_updates += other.nheap_updates;
    quantization_time += other.quantization_time;
    search_time += other.search_time;
}

IndexIVFStats indexIVF_stats;

/*************************************************************************
 * InvertedListScanner
 *************************************************************************/

size_t InvertedListScanner::scan_codes(size_t list_size, const uint8_t *codes,
                                       const idx_t *ids, float *simi,
                                       idx_t *idxi, size_t k) const {
    size_t nup = 0;

    if (!keep_max) {
        for (size_t j = 0; j < list_size; j++) {
            float dis = distance_to_code(codes);
            if (dis < simi[0]) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                maxheap_replace_top(k, simi, idxi, dis, id);
                nup++;
            }
            codes += code_size;
        }
    } else {
        for (size_t j = 0; j < list_size; j++) {
            float dis = distance_to_code(codes);
            if (dis > simi[0]) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                minheap_replace_top(k, simi, idxi, dis, id);
                nup++;
            }
            codes += code_size;
        }
    }
    return nup;
}

size_t InvertedListScanner::iterate_codes(InvertedListsIterator *it,
                                          float *simi, idx_t *idxi, size_t k,
                                          size_t &list_size) const {
    size_t nup = 0;
    list_size = 0;

    if (!keep_max) {
        for (; it->is_available(); it->next()) {
            auto id_and_codes = it->get_id_and_codes();
            float dis = distance_to_code(id_and_codes.second);
            if (dis < simi[0]) {
                maxheap_replace_top(k, simi, idxi, dis, id_and_codes.first);
                nup++;
            }
            list_size++;
        }
    } else {
        for (; it->is_available(); it->next()) {
            auto id_and_codes = it->get_id_and_codes();
            float dis = distance_to_code(id_and_codes.second);
            if (dis > simi[0]) {
                minheap_replace_top(k, simi, idxi, dis, id_and_codes.first);
                nup++;
            }
            list_size++;
        }
    }
    return nup;
}

void InvertedListScanner::scan_codes_range(size_t list_size,
                                           const uint8_t *codes,
                                           const idx_t *ids, float radius,
                                           RangeQueryResult &res) const {
    for (size_t j = 0; j < list_size; j++) {
        float dis = distance_to_code(codes);
        bool keep = !keep_max
                        ? dis < radius
                        : dis > radius; // TODO templatize to remove this test
        if (keep) {
            int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
            res.add(dis, id);
        }
        codes += code_size;
    }
}

void InvertedListScanner::iterate_codes_range(InvertedListsIterator *it,
                                              float radius,
                                              RangeQueryResult &res,
                                              size_t &list_size) const {
    list_size = 0;
    for (; it->is_available(); it->next()) {
        auto id_and_codes = it->get_id_and_codes();
        float dis = distance_to_code(id_and_codes.second);
        bool keep = !keep_max
                        ? dis < radius
                        : dis > radius; // TODO templatize to remove this test
        if (keep) {
            res.add(dis, id_and_codes.first);
        }
        list_size++;
    }
}

} // namespace faiss
