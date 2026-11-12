// Included at the original FAISS scope to use its internal search helpers.
void IndexIVF::search_preassigned_LAET(
    idx_t n,
    const float* x,
    idx_t k,
    const idx_t* keys,
    const float* coarse_dis,
    float* distances,
    idx_t* labels,
    bool store_pairs,
    LAETPredictorIVF* laet_predictor,
    const IVFSearchParameters* params,
    IndexIVFStats* ivf_stats) const {
  FAISS_THROW_IF_NOT(k > 0);

  idx_t nprobe = params ? params->nprobe : this->nprobe;
  nprobe = std::min((idx_t)nlist, nprobe);
  FAISS_THROW_IF_NOT(nprobe > 0);

  const idx_t unlimited_list_size = std::numeric_limits<idx_t>::max();
  idx_t max_codes = params ? params->max_codes : this->max_codes;
  IDSelector* sel = params ? params->sel : nullptr;
  const IDSelectorRange* selr = dynamic_cast<const IDSelectorRange*>(sel);
  if (selr) {
    if (selr->assume_sorted) {
      sel = nullptr;  // use special IDSelectorRange processing
    } else {
      selr = nullptr;  // use generic processing
    }
  }

  FAISS_THROW_IF_NOT_MSG(
      !(sel && store_pairs),
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

  [[maybe_unused]] bool do_parallel = omp_get_max_threads() >= 2 &&
                                      (pmode == 0   ? false
                                       : pmode == 3 ? n > 1
                                       : pmode == 1 ? nprobe > 1
                                                    : nprobe * n > 1);

  if (verbose) printf("DO PARALLEL? = %d\n", do_parallel);

  void* inverted_list_context =
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

    auto init_result = [&](float* simi, idx_t* idxi) {
      if (!do_heap_init)
        return;
      if (metric_type == METRIC_INNER_PRODUCT) {
        heap_heapify<HeapForIP>(k, simi, idxi);
      } else {
        heap_heapify<HeapForL2>(k, simi, idxi);
      }
    };

    auto add_local_results = [&](const float* local_dis,
                                 const idx_t* local_idx,
                                 float* simi,
                                 idx_t* idxi) {
      if (metric_type == METRIC_INNER_PRODUCT) {
        heap_addn<HeapForIP>(k, simi, idxi, local_dis, local_idx, k);
      } else {
        heap_addn<HeapForL2>(k, simi, idxi, local_dis, local_idx, k);
      }
    };

    auto reorder_result = [&](float* simi, idx_t* idxi) {
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
    auto scan_one_list = [&](idx_t key,
                             float coarse_dis_i,
                             float* simi,
                             idx_t* idxi,
                             idx_t list_size_max,
                             idx_t qid,
                             size_t nprobe_id,
                             float q_first_centroid_dist,
                             double q_search_start_time,
                             LAETPredictorIVF* laet_predictor,
                             size_t* q_ndis,
                             size_t* q_ninserts,
                             size_t* q_ndis_interval,
                             float* predicted_recall_out,
                             bool* early_termination_triggered,
                             int* prediction_interval,
                             int* query_predictor_calls,
                             double* predictor_time,
                             std::vector<float> centroid_distances,
                             int* distance_calculation_threshold,
                             bool* prediction_made) {
      if (key < 0) {
        // not enough centroids for multiprobe
        return (size_t)0;
      }
      FAISS_THROW_IF_NOT_FMT(
          key < (idx_t)nlist,
          "Invalid key=%" PRId64 " nlist=%zd\n",
          key,
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

          nheap += scanner->iterate_codes(
              it.get(), simi, idxi, k, list_size);

          return list_size;
        } else {
          size_t list_size = invlists->list_size(key);
          if (list_size > list_size_max) {
            list_size = list_size_max;
          }

          InvertedLists::ScopedCodes scodes(invlists, key);
          const uint8_t* codes = scodes.get();

          std::unique_ptr<InvertedLists::ScopedIds> sids;
          const idx_t* ids = nullptr;

          if (!store_pairs) {
            sids = std::make_unique<InvertedLists::ScopedIds>(
                invlists, key);
            ids = sids->get();
          }

          if (selr) {  // IDSelectorRange
            // restrict search to a section of the inverted list
            size_t jmin, jmax;
            selr->find_sorted_ids_bounds(
                list_size, ids, &jmin, &jmax);
            list_size = jmax - jmin;
            if (list_size == 0) {
              return (size_t)0;
            }
            codes += jmin * code_size;
            ids += jmin;
          }

          // it was like this before:
          // nheap += scanner->scan_codes(list_size, codes, ids, simi, idxi, k);

          // ACTUAL DIST CALCS START

          // new code below
          size_t nup = 0;
          if (!scanner->keep_max) {
            for (size_t j = 0; j < list_size; j++) {
              float dis = scanner->distance_to_code(codes);
              (*q_ndis)++;
              (*q_ndis_interval)++;

              if (dis < simi[0]) {
                int64_t id = store_pairs ? lo_build(scanner->list_no, j) : ids[j];
                maxheap_replace_top(k, simi, idxi, dis, id);
                nup++;

                (*q_ninserts)++;
              }
              codes += code_size;

              if (/**q_ndis >= laet_predictor->fixed_amount_of_distance_calcs*/ nprobe_id == 5 && !*prediction_made) {
                *distance_calculation_threshold = laet_predictor->predict_distance_calcs(qid,
                                                                                        nprobe_id,
                                                                                        *q_ndis,
                                                                                        laet_predictor->data_manager->elapsed_secs() - q_search_start_time,
                                                                                        q_first_centroid_dist,
                                                                                        centroid_distances,
                                                                                        predictor_time);
                if (verbose) printf("[Query %ld - Probe %ld] Distance calculation threshold predicted: %d at q_ndis: %ld\n", qid, nprobe_id, *distance_calculation_threshold, *q_ndis);
                *prediction_made = true;
              }

              if (*q_ndis >= *distance_calculation_threshold) {
                *early_termination_triggered = true;
                break;
              }
            }
          } else {
            for (size_t j = 0; j < list_size; j++) {
              float dis = scanner->distance_to_code(codes);
              (*q_ndis)++;
              (*q_ndis_interval)++;

              if (dis > simi[0]) {
                int64_t id = store_pairs ? lo_build(scanner->list_no, j) : ids[j];
                minheap_replace_top(k, simi, idxi, dis, id);
                nup++;

                (*q_ninserts)++;
              }
              codes += code_size;

              if (/**q_ndis >= laet_predictor->fixed_amount_of_distance_calcs*/ nprobe_id == 5 && !*prediction_made) {
                *distance_calculation_threshold = laet_predictor->predict_distance_calcs(qid,
                                                                                        nprobe_id,
                                                                                        *q_ndis,
                                                                                        laet_predictor->data_manager->elapsed_secs() - q_search_start_time,
                                                                                        q_first_centroid_dist,
                                                                                        centroid_distances,
                                                                                        predictor_time);
                if (verbose) printf("[Query %ld - Probe %ld] Distance calculation threshold predicted: %d at q_ndis: %ld\n", qid, nprobe_id, *distance_calculation_threshold, *q_ndis);
                *prediction_made = true;
              }

              if (*q_ndis >= *distance_calculation_threshold) {
                *early_termination_triggered = true;
                break;
              }
            }
          }

          // end new code
          // ACTUAL DIST CALCS END

          nheap += nup;

          return list_size;
        }
      } catch (const std::exception& e) {
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

        scanner->set_query(x + i * d);
        // darthPredictor->prepare_for_next_query();

        std::vector<float> centroid_distances;
        for (size_t ik = 0; ik < nprobe; ik++) {
          centroid_distances.push_back(coarse_dis[i * nprobe + ik]);
        }

        float* simi = distances + i * k;
        idx_t* idxi = labels + i * k;

        idx_t qid = i;
        size_t q_ndis = 0;
        size_t q_ninserts = 0;

        size_t q_ndis_interval = 0;

        double q_search_start_time = laet_predictor->data_manager->elapsed_secs();
        int distance_calculation_threshold = 2147483647;
        float first_centroid_dist = -1;
        int prediction_interval = 0;
        int query_predictor_calls = 0;
        double predictor_time = 0.0;
        bool prediction_made = false;

        init_result(simi, idxi);

        idx_t nscan = 0;

        float predicted_target_recall = 0.0;
        bool early_termination_triggered = false;

        // loop over probes
        size_t q_nprobe_id = 0;
        for (size_t ik = 0; ik < nprobe; ik++) {
          if (ik == 0) {
            first_centroid_dist = centroid_distances[ik];
          }

          q_nprobe_id = ik;
          nscan += scan_one_list(
              keys[i * nprobe + ik],
              coarse_dis[i * nprobe + ik],
              simi,
              idxi,
              max_codes - nscan,
              qid,
              q_nprobe_id,
              first_centroid_dist,
              q_search_start_time,
              laet_predictor,
              &q_ndis,
              &q_ninserts,
              &q_ndis_interval,
              &predicted_target_recall,
              &early_termination_triggered,
              &prediction_interval,
              &query_predictor_calls,
              &predictor_time,
              centroid_distances,
              &distance_calculation_threshold,
              &prediction_made);

          if (nscan >= max_codes) {
            break;
          }

          if (early_termination_triggered) {
            break;
          }
        }

        double q_elapsed_time = laet_predictor->data_manager->elapsed_secs() - q_search_start_time;
        float actual_recall = laet_predictor->data_manager->get_recallk(qid);
        if (verbose) printf("[Query %ld] => R: %f, ndc_p: %d, time: %lf, ndc: %ld, nprobes: %ld, predictor calls: %d\n", qid, actual_recall, distance_calculation_threshold, q_elapsed_time, q_ndis, q_nprobe_id, query_predictor_calls);
        laet_predictor->log_final_result(
            qid,
            q_nprobe_id,
            q_ndis,
            q_elapsed_time,
            first_centroid_dist,
            distance_calculation_threshold,
            predictor_time);

        ndis += nscan;
        reorder_result(simi, idxi);

        if (InterruptCallback::is_interrupted()) {
          interrupt = true;
        }

      }  // parallel for
    } else if (pmode == 1) {
      FAISS_THROW_FMT("parallel_mode %d not supported\n", pmode);
      /*
      std::vector<idx_t> local_idx(k);
      std::vector<float> local_dis(k);

      for (size_t i = 0; i < n; i++) {
          scanner->set_query(x + i * d);
          init_result(local_dis.data(), local_idx.data());

#pragma omp for schedule(dynamic)
          for (idx_t ik = 0; ik < nprobe; ik++) {
              ndis += scan_one_list(
                      keys[i * nprobe + ik],
                      coarse_dis[i * nprobe + ik],
                      local_dis.data(),
                      local_idx.data(),
                      unlimited_list_size);

              // can't do the test on max_codes
          }
          // merge thread-local results

          float* simi = distances + i * k;
          idx_t* idxi = labels + i * k;
#pragma omp single
          init_result(simi, idxi);

#pragma omp barrier
#pragma omp critical
          {
              add_local_results(
                      local_dis.data(), local_idx.data(), simi, idxi);
          }
#pragma omp barrier
#pragma omp single
          reorder_result(simi, idxi);
      }
  */
    } else if (pmode == 2) {
      FAISS_THROW_FMT("parallel_mode %d not supported\n", pmode);

      /*std::vector<idx_t> local_idx(k);
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
          ndis += scan_one_list(
                  keys[ij],
                  coarse_dis[ij],
                  local_dis.data(),
                  local_idx.data(),
                  unlimited_list_size);
#pragma omp critical
          {
              add_local_results(
                      local_dis.data(),
                      local_idx.data(),
                      distances + i * k,
                      labels + i * k);
          }
      }
#pragma omp single
      for (int64_t i = 0; i < n; i++) {
          reorder_result(distances + i * k, labels + i * k);
      }
      */
    } else {
      FAISS_THROW_FMT("parallel_mode %d not supported\n", pmode);
    }
  }  // parallel section

  if (interrupt) {
    if (!exception_string.empty()) {
      FAISS_THROW_FMT(
          "search interrupted with: %s", exception_string.c_str());
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

