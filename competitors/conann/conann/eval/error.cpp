#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <unordered_set>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <omp.h>
#include <sys/time.h>

#include "faiss/AutoTune.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/index_io.h"

#include <fstream>
#include <iostream>

/*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/

#define DC(classname) classname *ix = dynamic_cast<classname *>(index)

float *fvecs_read(const char *fname, size_t *d_out, size_t *n_out) {
    FILE *f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, sizeof(int), 1, f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float *x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int *ivecs_read(const char *fname, size_t *d_out, size_t *n_out) {
    return (int *)fvecs_read(fname, d_out, n_out);
}

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Datasets that don't have a hand-tuned MAX_DISTANCE constant (see below) get
// one estimated from the data: max squared L2 distance from a fixed anchor
// vector to a random sample of the database, times a safety factor. This
// keeps score_k / MAX_DISTANCE (the nonconformity score) from saturating to
// 1.0 immediately while still being on the right order of magnitude.
float estimate_max_distance(const float *xb, size_t nb, size_t d,
                             size_t sample_size = 5000,
                             float safety_factor = 2.0f) {
    std::srand(1234);
    size_t anchor = std::rand() % nb;
    float max_dist = 0.0f;
    for (size_t s = 0; s < sample_size; s++) {
        size_t j = std::rand() % nb;
        double dist = 0.0;
        for (size_t k = 0; k < d; k++) {
            double diff = xb[anchor * d + k] - xb[j * d + k];
            dist += diff * diff;
        }
        if ((float)dist > max_dist) {
            max_dist = (float)dist;
        }
    }
    return max_dist * safety_factor;
}

// Estimate the same normalization scale without loading the complete base
// dataset. This is used when a prebuilt DARTH IVF index is loaded.
float estimate_max_distance_from_file(const char *fname,
                                      size_t expected_d,
                                      size_t sample_size = 5000,
                                      float safety_factor = 2.0f) {
    FILE *f = fopen(fname, "rb");
    if (!f) {
        throw std::runtime_error(std::string("Could not open base vectors: ") + fname);
    }

    int file_d = 0;
    if (fread(&file_d, sizeof(file_d), 1, f) != 1 ||
        file_d <= 0 || static_cast<size_t>(file_d) != expected_d) {
        fclose(f);
        throw std::runtime_error("Invalid base-vector dimension");
    }

    struct stat st;
    if (fstat(fileno(f), &st) != 0) {
        fclose(f);
        throw std::runtime_error("Could not stat base-vector file");
    }
    const size_t row_bytes = (expected_d + 1) * sizeof(float);
    const size_t nb = static_cast<size_t>(st.st_size) / row_bytes;
    if (nb == 0 || static_cast<size_t>(st.st_size) % row_bytes != 0) {
        fclose(f);
        throw std::runtime_error("Invalid base-vector file size");
    }

    auto read_vector = [&](size_t row, std::vector<float> &out) {
        const off_t offset = static_cast<off_t>(row) * row_bytes;
        if (fseeko(f, offset, SEEK_SET) != 0) {
            throw std::runtime_error("Could not seek in base-vector file");
        }
        int row_d = 0;
        if (fread(&row_d, sizeof(row_d), 1, f) != 1 ||
            static_cast<size_t>(row_d) != expected_d ||
            fread(out.data(), sizeof(float), expected_d, f) != expected_d) {
            throw std::runtime_error("Could not read sampled base vector");
        }
    };

    std::mt19937 generator(1234);
    std::uniform_int_distribution<size_t> sample_index(0, nb - 1);
    std::vector<float> anchor(expected_d);
    std::vector<float> sample(expected_d);
    try {
        read_vector(sample_index(generator), anchor);
        float max_dist = 0.0f;
        for (size_t s = 0; s < sample_size; ++s) {
            read_vector(sample_index(generator), sample);
            double dist = 0.0;
            for (size_t j = 0; j < expected_d; ++j) {
                const double diff = anchor[j] - sample[j];
                dist += diff * diff;
            }
            max_dist = std::max(max_dist, static_cast<float>(dist));
        }
        fclose(f);
        return max_dist * safety_factor;
    } catch (...) {
        fclose(f);
        throw;
    }
}

template <typename T> double computeAverage(const std::vector<T> &numbers) {
    if (numbers.empty())
        return 0.0;
    double sum = 0.0;
    int negativeCount = 0;
    for (const auto &num : numbers) {
        if (num > 0) {
            sum += num;
        } else {
            ++negativeCount;
        }
    }
    if (negativeCount > 0) {
        std::cout << "\nWARNING! Number of negative values: " << negativeCount
                  << std::endl;
    }
    return sum / (numbers.size() - negativeCount);
}

template <typename T>
void write_to_file(const std::vector<T> &data, const std::string &filename) {
    std::ofstream file(filename);
    for (const auto &value : data) {
        file << value << '\n';
    }
    file.close();
}

void write_time_report_csv(const std::string& filename, const faiss::IndexIVF::TimeReport time_report) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    // Write header
    // file << "Metric,Time\n";
    
    // Write data with 6 decimal precision
    file << std::fixed << std::setprecision(6)
         << "ComputeScores," << time_report.computeScores << "\n"
         << "ComputeScoresCalib," << time_report.computeScoresCalib << "\n"
         << "ComputeScoresTune," << time_report.computeScoresTune << "\n"
         << "MemoryCopyPostCompute," << time_report.memoryCopyPostCompute << "\n"
         << "PickRegLambda," << time_report.pickRegLambda << "\n"
         << "RegularizeScores," << time_report.regularizeScores << "\n"
         << "Optimize," << time_report.optimize << "\n"
         << "ConfigureTotal," << time_report.configureTotal << "\n";

    file.close();
}

/// Command:
/// ./error DATASET CALIBRATION_FRACTION TUNING_FRACTION ALPHA NLIST K
/// ./error DATASET --counts CALIBRATION_QUERIES TUNING_QUERIES ALPHA
///         NLIST NPROBE K INDEX_FILE [OUTPUT_CSV]
int main(int argc, char **argv) {
    std::cout << argc - 1 << " arguments" << std::endl;
    bool exact_count_mode = argc >= 3 && std::string(argv[2]) == "--counts";
    int minimum_argc = exact_count_mode ? 10 : 7;
    if (argc < minimum_argc) {
        fprintf(stderr,
                "Usage:\n"
                "  %s DATASET CALIBRATION_FRACTION TUNING_FRACTION "
                "ALPHA NLIST K\n"
                "  %s DATASET --counts CALIBRATION_QUERIES "
                "TUNING_QUERIES ALPHA NLIST NPROBE K INDEX_FILE "
                "[OUTPUT_CSV]\n",
                argv[0], argv[0]);
        return 1;
    }

    std::string dataset_name = argv[1];
    size_t calib_nq = 0;
    size_t tune_nq = 0;
    float calib_sz = 0.0f;
    float tune_sz = 0.0f;
    size_t input_nprobe = 0;
    std::string index_filepath;
    std::string selection_k;
    std::string output_csv;
    float alpha;
    int input_nlist;
    if (exact_count_mode) {
        calib_nq = std::stoull(argv[3]);
        tune_nq = std::stoull(argv[4]);
        alpha = std::stof(argv[5]);
        input_nlist = std::stoi(argv[6]);
        input_nprobe = std::stoull(argv[7]);
        selection_k = argv[8];
        index_filepath = argv[9];
        output_csv = argc >= 11 ? argv[10] : "";
    } else {
        calib_sz = std::stof(argv[2]);
        tune_sz = std::stof(argv[3]);
        alpha = std::stof(argv[4]);
        input_nlist = std::stoi(argv[5]);
        selection_k = argv[6];
    }

    // The release comparison uses the five datasets prepared by the launcher.
    if (dataset_name != "DEEP100M" && dataset_name != "T2I100M" &&
        dataset_name != "SIFT100M" && dataset_name != "GIST1M" &&
        dataset_name != "GLOVE100") {
        std::cerr << "Unsupported dataset: " << dataset_name << std::endl;
        return 1;
    }
    const std::string db = "../data/" + dataset_name + "/db.fvecs";
    const std::string query = "../data/" + dataset_name + "/queries.fvecs";
    const std::string gtI = "../data/" + dataset_name + "/indices-" + selection_k + ".fvecs";
    const std::string gtD = "../data/" + dataset_name + "/distances-" + selection_k + ".fvecs";
    float max_distance = 0.0f;
    const bool auto_max_distance = true;

    omp_set_num_threads(60);
    double t0 = elapsed();

    // Milestone timestamps (seconds since t0), collected for the
    // preparation-timings report written near the end of main().
    double t_train_set_loaded = 0, t_training_done = 0, t_db_loaded = 0,
           t_indexing_done = 0, t_queries_loaded = 0, t_gt_loaded = 0,
           t_calibration_done = 0, t_prequery_setup_done = 0;
    double index_load_time = 0;
    double max_distance_estimation_time = 0;
    size_t reported_ntt = 0;

    faiss::IndexIVFFlat *index = nullptr;

    size_t d = 0;

    if (exact_count_mode) {
        printf("[%.3f s] Loading DARTH IVF index from %s\n",
               elapsed() - t0, index_filepath.c_str());
        double index_load_start = elapsed();
        faiss::Index *loaded_index = faiss::read_index(index_filepath.c_str());
        index_load_time = elapsed() - index_load_start;
        index = dynamic_cast<faiss::IndexIVFFlat *>(loaded_index);
        if (!index) {
            delete loaded_index;
            throw std::runtime_error("DARTH index is not an IndexIVFFlat");
        }
        if (index->nlist != static_cast<size_t>(input_nlist)) {
            delete index;
            throw std::runtime_error("Loaded index nlist does not match the requested nlist");
        }
        if (input_nprobe == 0 || input_nprobe > index->nlist) {
            delete index;
            throw std::runtime_error("nprobe must be between 1 and nlist");
        }
        if (index->metric_type != faiss::METRIC_L2) {
            delete index;
            throw std::runtime_error("The DARTH ConANN comparison requires an L2 IVF index");
        }

        d = index->d;
        // nprobe is a query-time parameter; mirror the DARTH runner, which
        // explicitly reapplies it after deserialization.
        index->nprobe = input_nprobe;
        // n_list is ConANN-only state and is therefore not serialized in the
        // standard FAISS index produced by DARTH.
        index->n_list = index->nlist;
        printf("[%.3f s] Loaded index: d=%ld, ntotal=%ld, nlist=%ld, nprobe=%ld\n",
               elapsed() - t0, d, index->ntotal, index->nlist, index->nprobe);

        if (auto_max_distance) {
            double max_distance_start = elapsed();
            max_distance = estimate_max_distance_from_file(db.c_str(), d);
            max_distance_estimation_time = elapsed() - max_distance_start;
            printf("[%.3f s] Auto-estimated MAX_DISTANCE=%.3f for dataset %s\n",
                   elapsed() - t0, max_distance, dataset_name.c_str());
        }
        t_prequery_setup_done = elapsed() - t0;
    } else {
        {
            printf("[%.3f s] Loading train set\n", elapsed() - t0);

            size_t nt;
            float *xt = fvecs_read(db.c_str(), &d, &nt);
            t_train_set_loaded = elapsed() - t0;

            printf("[%.3f s] Preparing index IVFFlat_%i d=%ld\n", elapsed() - t0,
                   input_nlist, d);

            int nlist = input_nlist; // 1024 originally

            faiss::IndexFlatL2 *flat_index = new faiss::IndexFlatL2(d);
            index = new faiss::IndexIVFFlat(flat_index, d, nlist, faiss::METRIC_L2);

            // Make clustering seed explicit
            index->cp.seed = 420;

            index->nprobe = nlist;
            // train on half the dataset
            // TODO: Are we sure the data is shuffled otherwise we train K-means out
            // of distribution.
            auto ntt = size_t(0.5 * nt);
            reported_ntt = ntt;
            printf("[%.3f s] Training on %ld vectors\n", elapsed() - t0, ntt);

            index->train(ntt, xt);
            t_training_done = elapsed() - t0;
            delete[] xt;
        }

        {
            printf("[%.3f s] Loading database\n", elapsed() - t0);

            size_t nb, d2;
            float *xb = fvecs_read(db.c_str(), &d2, &nb);
            t_db_loaded = elapsed() - t0;
            assert(d == d2 || !"dataset does not have same dimension as train set");

            printf("[%.3f s] Indexing database, size %ld*%ld\n", elapsed() - t0, nb,
                   d);

            index->add(nb, xb);
            t_indexing_done = elapsed() - t0;

            if (auto_max_distance) {
                double max_distance_start = elapsed();
                max_distance = estimate_max_distance(xb, nb, d2);
                max_distance_estimation_time = elapsed() - max_distance_start;
                printf("[%.3f s] Auto-estimated MAX_DISTANCE=%.3f for dataset %s\n",
                       elapsed() - t0, max_distance, dataset_name.c_str());
            }

            delete[] xb;
        }
        t_prequery_setup_done = elapsed() - t0;
    }

    size_t nq;
    float *xq;

    {
        printf("[%.3f s] Loading queries\n", elapsed() - t0);

        size_t d2;
        xq = fvecs_read(query.c_str(), &d2, &nq);
        t_queries_loaded = elapsed() - t0;
        assert(d == d2 || !"query does not have same dimension as train set");
    }

    if (!exact_count_mode) {
        calib_nq = size_t(calib_sz * nq);
        tune_nq = size_t(tune_sz * nq);
    }

    if (calib_nq + tune_nq >= nq) {
        std::cerr << "Invalid split: calibration=" << calib_nq
                  << ", tuning=" << tune_nq << ", total=" << nq
                  << ". At least one query must remain for testing.\n";
        delete[] xq;
        delete index;
        return 1;
    }

    size_t k;         // nb of results per query in the GT
    faiss::idx_t *gt; // nq * k matrix of ground-truth nearest-neighbors

    {
        printf("[%.3f s] Loading ground truth for %ld queries\n",
               elapsed() - t0, nq);

        // load ground-truth and convert int to long
        size_t nq2;
        int *gt_int = ivecs_read(gtI.c_str(), &k, &nq2);
        assert(nq2 == nq || !"incorrect nb of ground truth entries");

        gt = new faiss::idx_t[k * nq];
        for (int i = 0; i < k * nq; i++) {
            gt[i] = gt_int[i];
        }
        delete[] gt_int;
    }

    size_t kk;
    float *gt_v;

    {
        printf("[%.3f s] Loading groud truth vector\n", elapsed() - t0);
        size_t nq3;
        gt_v = fvecs_read(gtD.c_str(), &kk, &nq3);
        t_gt_loaded = elapsed() - t0;
        assert(kk == k ||
               !"gt distances does not have same dimension as gt IDs");
        assert(nq3 == nq || !"incorrect nb of ground truth entries");
    }

    printf("[%.3f s] ConANN Calibration\n", elapsed() - t0);
    auto t1 = elapsed();
    auto calib_res = index->calibrate_counts(
            alpha, k, calib_nq, tune_nq, xq, nq, gt, max_distance,
            dataset_name);
    t_calibration_done = elapsed() - t0;
    std::cout << "Calibration-time=" << elapsed() - t1 << "\n";
    std::cout << "Found lamhat=" << calib_res.lamhat << "\n";

    std::ostringstream key;
    key << dataset_name << "-" << std::to_string(input_nlist) << "-"
        << selection_k << "-" << alpha;
    if (exact_count_mode) {
        key << "-np" << input_nprobe << "-c" << calib_nq << "-t" << tune_nq;
    } else {
        key << "-" << calib_sz << "-" << tune_sz;
    }

    if (output_csv.empty()) {
        output_csv = "../ConANN-perquery-" + key.str() + ".csv";
    }

    std::string preparation_filename =
            "../ConANN-prep-" + key.str() + ".txt";
    if (exact_count_mode) {
        preparation_filename = output_csv;
        const std::string csv_suffix = ".csv";
        if (preparation_filename.size() >= csv_suffix.size() &&
            preparation_filename.compare(
                    preparation_filename.size() - csv_suffix.size(),
                    csv_suffix.size(), csv_suffix) == 0) {
            preparation_filename.erase(
                    preparation_filename.size() - csv_suffix.size());
        }
        preparation_filename += "_preparation.txt";
    }

    // Preparation-timings report: dataset/config + how long each setup phase
    // took. DARTH comparison runs report index loading separately and never
    // train or add database vectors here.
    {
        std::ofstream prep_file(preparation_filename);
        if (!prep_file.is_open()) {
            throw std::runtime_error(
                    "Could not open preparation report: " +
                    preparation_filename);
        }
        prep_file << "dataset=" << dataset_name << "\n";
        prep_file << "k=" << k << "\n";
        prep_file << "nlist=" << input_nlist << "\n";
        prep_file << "nprobe=" << index->nprobe << "\n";
        prep_file << "d=" << d << "\n";
        prep_file << "alpha=" << alpha << "\n";
        if (exact_count_mode) {
            prep_file << "calibration_queries=" << calib_nq << "\n";
            prep_file << "tuning_queries=" << tune_nq << "\n";
            prep_file << "index_file=" << index_filepath << "\n";
            prep_file << "index_source=prebuilt_darth\n";
        } else {
            prep_file << "calib_sz=" << calib_sz << "\n";
            prep_file << "tune_sz=" << tune_sz << "\n";
        }
        prep_file << "testing_queries=" << (nq - calib_nq - tune_nq) << "\n";
        prep_file << "max_distance=" << max_distance
                  << (auto_max_distance ? " (auto-estimated)" : " (hand-tuned)")
                  << "\n";
        prep_file << "nq_total=" << nq << "\n";
        prep_file << "\n# wall-clock seconds spent in each preparation phase\n";
        if (exact_count_mode) {
            prep_file << "index_load_time_s=" << index_load_time << "\n";
            prep_file << "index_training_time_s=0\n";
            prep_file << "database_add_time_s=0\n";
        } else {
            prep_file << "train_vectors=" << reported_ntt << "\n";
            prep_file << "train_set_load_time_s=" << t_train_set_loaded << "\n";
            prep_file << "training_time_s=" << (t_training_done - t_train_set_loaded)
                      << "\n";
            prep_file << "db_load_time_s=" << (t_db_loaded - t_training_done) << "\n";
            prep_file << "indexing_time_s=" << (t_indexing_done - t_db_loaded) << "\n";
        }
        prep_file << "max_distance_estimation_time_s="
                  << max_distance_estimation_time << "\n";
        prep_file << "queries_load_time_s="
                  << (t_queries_loaded - t_prequery_setup_done)
                  << "\n";
        prep_file << "gt_load_time_s=" << (t_gt_loaded - t_queries_loaded) << "\n";
        prep_file << "score_precomputation_time_s="
                  << index->time_report.computeScores << "\n";
        prep_file << "post_score_copy_time_s="
                  << index->time_report.memoryCopyPostCompute << "\n";
        prep_file << "lambda_selection_time_s="
                  << index->time_report.pickRegLambda << "\n";
        prep_file << "regularization_time_s="
                  << index->time_report.regularizeScores << "\n";
        prep_file << "optimization_time_s="
                  << index->time_report.optimize << "\n";
        prep_file << "calibration_time_s=" << (t_calibration_done - t_gt_loaded)
                  << "\n";
        prep_file << "total_preparation_time_s=" << t_calibration_done << "\n";
        prep_file << "ready_to_query_time_s=" << t_calibration_done << "\n";
        prep_file.close();
        std::cout << "Wrote " << preparation_filename << "\n";
    }

    // Per-query inference report on DARTH's testing pool (the tail of the
    // concatenated query file -- see prepare_darth_data.py /
    // experiments/run_conann.sh for why this is exactly DARTH's testing
    // queries, untouched by calibration/tuning). This runs ConANN's actual
    // deployment-time early-stopping search (search_conann) one query at a
    // time so each query's wall-clock latency is isolated from the others,
    // and reads the deltas of the global faiss::indexIVF_stats counters
    // (ndis = distance computations, nlist = clusters visited) around each
    // call to get exact per-query values -- these aren't otherwise exposed
    // per-query. test_start/test_count below use the exact same truncation
    // IndexIVF::prep_execution uses internally, so this is guaranteed to be
    // the same query range evaluate_test() reports on in aggregate.
    {
        size_t test_start = calib_nq + tune_nq;
        size_t test_count = (test_start < nq) ? (nq - test_start) : 0;

        bool darth_csv = exact_count_mode;
        std::ofstream pq_file(output_csv);
        if (!pq_file.is_open()) {
            throw std::runtime_error("Could not open result CSV: " + output_csv);
        }
        if (darth_csv) {
            pq_file << "qid,dists,elaps_ms,r,clusters_searched\n";
        } else {
            pq_file << "qid,ndis,clusters_searched,latency_ms,recall\n";
        }
        pq_file << std::setprecision(9);

        std::vector<float> dist_buf(k);
        std::vector<faiss::idx_t> label_buf(k);

        for (size_t i = 0; i < test_count; i++) {
            size_t qi = test_start + i;

            size_t ndis_before = faiss::indexIVF_stats.ndis;
            size_t nlist_before = faiss::indexIVF_stats.nlist;
            double tq0 = elapsed();
            index->search_conann(1, xq + qi * d, dist_buf.data(),
                                 label_buf.data(), calib_res);
            double tq1 = elapsed();
            size_t ndis_used = faiss::indexIVF_stats.ndis - ndis_before;
            size_t clusters_used = faiss::indexIVF_stats.nlist - nlist_before;

            std::unordered_set<faiss::idx_t> gt_set(gt + qi * k, gt + qi * k + k);
            int matches = 0;
            for (size_t j = 0; j < k; j++) {
                if (gt_set.count(label_buf[j])) {
                    matches++;
                }
            }
            float recall = static_cast<float>(matches) / static_cast<float>(k);

            if (darth_csv) {
                pq_file << i << "," << ndis_used << ","
                        << (tq1 - tq0) * 1000.0 << "," << recall << ","
                        << clusters_used << "\n";
            } else {
                pq_file << i << "," << ndis_used << "," << clusters_used
                        << "," << (tq1 - tq0) * 1000.0 << "," << recall
                        << "\n";
            }
        }
        pq_file.close();
        std::cout << "Wrote " << output_csv << " (" << test_count
                  << " queries)\n";
    }

    // Around half of GT was mem_copied into calib_cx and calib_labels so we can
    // free up this memory here
    delete[] xq;
    delete[] gt;
    delete[] gt_v;

    printf("[%.3f s] ConANN Evaluation\n", elapsed() - t0);
    auto [fnr, cls] = index->evaluate_test(calib_res);
    float avg_fnr = std::accumulate(fnr.begin(), fnr.end(), 0.0) / fnr.size();
    printf(
        "[%.3f s] Finished: alpha=%.3f, test fnr=%.3f, avg cls searched=%.3f\n",
        elapsed() - t0, alpha, avg_fnr, computeAverage(cls));
    auto c = computeAverage(cls);
    std::cout << "alpha=" << alpha << ", test fnr=" << avg_fnr
              << ", avg cls searched=" << c << std::endl;

    std::ostringstream fnr_filename;
    fnr_filename << "../ConANN-error-" << key.str() << ".log";
    write_to_file(fnr, fnr_filename.str());

    std::ostringstream cls_filename;
    cls_filename << "../ConANN-efficiency-" << key.str() << ".log";
    write_to_file(cls, cls_filename.str());

    std::ostringstream time_filename;
    time_filename << "../ConANN-timing-" << key.str() << ".csv";
    write_time_report_csv(time_filename.str(), index->time_report);

    delete index;
    return 0;
}
