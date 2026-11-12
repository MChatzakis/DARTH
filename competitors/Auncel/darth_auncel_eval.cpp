// Auncel evaluator for the DARTH+ IVF comparison.
//
// For one dataset this program loads DARTH's prebuilt IndexIVFFlat, trains
// Auncel's error-profile model on the first learn queries, tunes Auncel's two
// hyperparameters (multipler, std_m) on the first validation queries for every
// recall target, and then answers the first test queries one at a time.
//
// Auncel only ever sees ground truth for its training (learn) queries. Rows for
// validation and test queries carry dummy ground truth; the evaluator scores
// returned ids against the real ground truth itself.

#include "AutoTune.h"
#include "IndexFlat.h"
#include "IndexIVF.h"
#include "IndexIVFFlat.h"
#include "IVF_pro.h"
#include "index_io.h"
#include "profile.h"

#include <omp.h>
#include <sys/stat.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using idx_t = faiss::Index::idx_t;

constexpr const char* kVersion = "darth_auncel_eval 1.0";
constexpr const char* kCsvHeader = "qid,dists,elaps_ms,r,clusters_searched,hits,neighbor_ids";

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void log(const std::string& message) { std::cout << message << std::endl; }

std::string fixed(double value, int precision) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

struct VecsInfo {
    size_t rows, dim;
};

VecsInfo vecs_info(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) fail("cannot open " + path);
    int32_t dim = 0;
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    if (!in || dim <= 0) fail("invalid vecs header in " + path);
    in.seekg(0, std::ios::end);
    const size_t bytes = static_cast<size_t>(in.tellg());
    const size_t row_bytes = 4 * (static_cast<size_t>(dim) + 1);
    if (bytes % row_bytes != 0) fail("size of " + path + " is not a multiple of its row size");
    return {bytes / row_bytes, static_cast<size_t>(dim)};
}

// First `rows` rows of a vecs file, keeping their first `cols` 32-bit values.
template <typename T>
std::vector<T> read_vecs(const std::string& path, size_t rows, size_t cols) {
    static_assert(sizeof(T) == 4, "vecs values are 32-bit");
    const VecsInfo info = vecs_info(path);
    if (rows > info.rows) fail(path + " has " + std::to_string(info.rows) + " rows; " + std::to_string(rows) + " needed");
    if (cols > info.dim) fail(path + " has " + std::to_string(info.dim) + " columns; " + std::to_string(cols) + " needed");
    std::vector<T> out(rows * cols);
    std::vector<char> row(4 * (info.dim + 1));
    std::ifstream in(path, std::ios::binary);
    for (size_t i = 0; i < rows; ++i) {
        in.read(row.data(), static_cast<std::streamsize>(row.size()));
        if (!in) fail("truncated vecs file " + path);
        std::memcpy(out.data() + i * cols, row.data() + 4, cols * 4);
    }
    return out;
}

void write_atomically(const std::string& path, const std::string& contents) {
    const std::string temporary = path + ".tmp";
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) fail("cannot write " + temporary);
        out << contents;
        if (!out) fail("cannot write " + temporary);
    }
    if (std::rename(temporary.c_str(), path.c_str()) != 0) fail("cannot rename " + temporary);
}

struct Options {
    std::string dataset, index, learn, learn_gt, learn_gt_dist, validation, validation_gt, query, query_gt, output_dir;
    size_t k = 50, nprobe = 0, train_queries = 10000, tune_queries = 1000, test_queries = 1000, gt_width = 100;
    int threads = 1;
    std::vector<std::string> targets = {"0.80", "0.85", "0.90", "0.95", "0.99"};
    std::vector<float> std_grid{0.0f, 0.2f, 1.0f, 2.0f, 4.0f, 6.0f, 8.0f, 12.0f};
    float multiplier_min = 1.0f, multiplier_max = 30.0f, multiplier_step = 0.1f;
};

void usage(std::ostream& out) {
    out << "Usage: darth_auncel_eval --dataset NAME --index DARTH_IVF.index --nprobe N\n"
           "         --learn L.fvecs --learn-gt L.ivecs --learn-gt-dist L.fvecs\n"
           "         --validation V.fvecs --validation-gt V.ivecs --query Q.fvecs --query-gt Q.ivecs\n"
           "         [--targets 0.80,0.85,0.90,0.95,0.99] --output-dir DIR [--k 50] [--train-queries 10000]\n"
           "         [--tune-queries 1000] [--test-queries 1000] [--gt-width 100] [--threads N]\n"
           "       darth_auncel_eval --version\n";
}

size_t parse_size(const std::string& name, const std::string& value) {
    size_t position = 0;
    unsigned long long parsed = 0;
    try {
        parsed = std::stoull(value, &position);
    } catch (const std::exception&) {
        position = 0;
    }
    if (position != value.size() || parsed == 0) fail(name + " must be a positive integer, got '" + value + "'");
    return static_cast<size_t>(parsed);
}

Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; i += 2) {
        const std::string name = argv[i];
        if (i + 1 >= argc) fail("missing value for " + name);
        const std::string value = argv[i + 1];
        if (name == "--dataset") o.dataset = value;
        else if (name == "--index") o.index = value;
        else if (name == "--learn") o.learn = value;
        else if (name == "--learn-gt") o.learn_gt = value;
        else if (name == "--learn-gt-dist") o.learn_gt_dist = value;
        else if (name == "--validation") o.validation = value;
        else if (name == "--validation-gt") o.validation_gt = value;
        else if (name == "--query") o.query = value;
        else if (name == "--query-gt") o.query_gt = value;
        else if (name == "--output-dir") o.output_dir = value;
        else if (name == "--k") o.k = parse_size(name, value);
        else if (name == "--nprobe") o.nprobe = parse_size(name, value);
        else if (name == "--train-queries") o.train_queries = parse_size(name, value);
        else if (name == "--tune-queries") o.tune_queries = parse_size(name, value);
        else if (name == "--test-queries") o.test_queries = parse_size(name, value);
        else if (name == "--gt-width") o.gt_width = parse_size(name, value);
        else if (name == "--threads") o.threads = static_cast<int>(parse_size(name, value));
        else if (name == "--targets") {
            o.targets.clear();
            std::stringstream stream(value);
            std::string target;
            while (std::getline(stream, target, ',')) o.targets.push_back(target);
        } else fail("unknown option " + name);
    }
    const std::pair<const char*, const std::string*> required[] = {
        {"--dataset", &o.dataset}, {"--index", &o.index}, {"--learn", &o.learn}, {"--learn-gt", &o.learn_gt},
        {"--learn-gt-dist", &o.learn_gt_dist}, {"--validation", &o.validation},
        {"--validation-gt", &o.validation_gt}, {"--query", &o.query}, {"--query-gt", &o.query_gt},
        {"--output-dir", &o.output_dir}};
    for (const auto& item : required) {
        if (item.second->empty()) fail(std::string(item.first) + " is required");
    }
    if (o.nprobe == 0) fail("--nprobe is required");
    if (o.targets.empty()) fail("--targets is required");
    for (const auto& target : o.targets) {
        const float value = std::stof(target);
        if (!(value > 0.0f && value < 1.0f)) fail("recall targets must be in (0, 1), got " + target);
    }
    // Auncel's search returns gt_width results; its upstream driver used 100 and k <= 100.
    if (o.k > o.gt_width || o.gt_width < 4 || o.gt_width > 100) fail("--k must be <= --gt-width, which must be between 4 and 100");
    if (o.train_queries < 10) fail("Auncel requires at least 10 training queries");
    if ((o.train_queries + o.tune_queries + o.test_queries) % 10 != 0) {
        fail("Auncel requires the total number of queries to be a multiple of 10");
    }
    return o;
}

size_t count_hits(const idx_t* returned, const int* truth, size_t k) {
    std::unordered_set<int> truth_set(truth, truth + k);
    size_t hits = 0;
    for (size_t i = 0; i < k; ++i) hits += truth_set.count(static_cast<int>(returned[i]));
    return hits;
}

struct Evaluation {
    double avg_recall = 0.0, avg_dists = 0.0;
};

struct Setting {
    float multiplier = 0.0f, std_m = 0.0f;
    Evaluation tuning;
};

class AuncelRun {
public:
    AuncelRun(faiss::IndexIVFFlat* index, const Options& options, std::vector<float> queries,
              std::vector<idx_t> gt_ids, std::vector<float> gt_dists)
        : index_(index), o_(options), queries_(std::move(queries)),
          total_(o_.train_queries + o_.tune_queries + o_.test_queries),
          system_(index, total_, o_.gt_width), required_(total_, 0.0f) {
        system_.set_gt(gt_dists.data(), gt_ids.data());
    }

    void train() {
        system_.sys_train(o_.train_queries, queries_.data());
        system_.set_topk(o_.k);
        index_->t->max_scan_lists = o_.nprobe;
        index_->t->profile = false;
    }

    // Resets Auncel's per-query state and sets the hyperparameters.
    void configure(float target, const Setting& setting) {
        std::fill(required_.begin(), required_.end(), target);
        system_.set_queries(total_, queries_.data(), required_.data(), total_);
        index_->t->multipler = setting.multiplier;
        index_->t->std_m = setting.std_m;
    }

    // Batched, multi-threaded search of rows [start, start + n).
    void search_batch(size_t start, size_t n, std::vector<float>& distances, std::vector<idx_t>& labels) {
        distances.assign(n * o_.gt_width, 0.0f);
        labels.assign(n * o_.gt_width, -1);
        system_.search(distances.data(), labels.data(), start, n);
    }

    // One query on the calling thread; returns {lists scanned, vectors compared}.
    std::pair<size_t, size_t> search_one(size_t row, float* distances, idx_t* labels) {
        faiss::indexIVF_stats.reset();
        system_.search(distances, labels, row, 1);
        return {faiss::indexIVF_stats.nlist, faiss::indexIVF_stats.ndis};
    }

    size_t total() const { return total_; }

private:
    faiss::IndexIVFFlat* index_;
    const Options& o_;
    std::vector<float> queries_;
    size_t total_;
    faiss::Error_sys system_;
    std::vector<float> required_;
};

void run(const Options& o) {
    const auto program_start = Clock::now();
    omp_set_num_threads(o.threads);
    // Coarse quantization always uses FAISS's multithreaded SIMD path, so
    // batched and single-query searches order centroids the same way.
    faiss::distance_compute_blas_threshold = std::numeric_limits<int>::max();

    auto start = Clock::now();
    std::unique_ptr<faiss::Index> loaded(faiss::read_index(o.index.c_str()));
    auto* index = dynamic_cast<faiss::IndexIVFFlat*>(loaded.get());
    if (index == nullptr) fail(o.index + " is not an IndexIVFFlat");
    if (index->metric_type != faiss::METRIC_L2) fail(o.index + " is not an L2 index");
    const size_t d = index->d, nlist = index->nlist;
    if (nlist < 8) fail("Auncel requires an IVF index with at least 8 lists");
    if (o.nprobe > nlist) fail("--nprobe exceeds nlist");
    const double index_load_s = seconds_since(start);
    log(o.dataset + ": loaded " + o.index + " (d=" + std::to_string(d) + ", ntotal=" +
        std::to_string(index->ntotal) + ", nlist=" + std::to_string(nlist) + ") in " + fixed(index_load_s, 1) + " s");

    // Auncel computes these centroid-to-centroid distances while training its
    // own index (Level1Quantizer::train_q1, L2 branch). DARTH's index is
    // already trained, so the same computation runs here on its centroids.
    start = Clock::now();
    auto* centroids = dynamic_cast<faiss::IndexFlat*>(index->quantizer);
    if (centroids == nullptr || centroids->xb.size() != nlist * d) fail("unexpected coarse quantizer");
    index->interdis_cem.resize(nlist * (nlist - 1) / 2);
    faiss::fvec_inter_vecs(index->interdis_cem.data(), centroids->xb.data(), nlist, d);
    const double geometry_s = seconds_since(start);

    // Query rows: [train (learn) | tune (validation) | test (query)].
    const size_t T = o.train_queries, V = o.tune_queries, Q = o.test_queries, W = o.gt_width, N = T + V + Q;
    std::vector<float> queries(N * d);
    for (const auto& part : {std::make_tuple(&o.learn, size_t{0}, T), std::make_tuple(&o.validation, T, V),
                             std::make_tuple(&o.query, T + V, Q)}) {
        const std::string& path = *std::get<0>(part);
        if (vecs_info(path).dim != d) fail(path + " does not have dimension " + std::to_string(d));
        const auto rows = read_vecs<float>(path, std::get<2>(part), d);
        std::copy(rows.begin(), rows.end(), queries.begin() + std::get<1>(part) * d);
    }
    std::vector<idx_t> gt_ids(N * W, -1);
    std::vector<float> gt_dists(N * W, FLT_MAX);
    {
        const auto ids = read_vecs<int>(o.learn_gt, T, W);
        const auto dists = read_vecs<float>(o.learn_gt_dist, T, W);
        std::copy(ids.begin(), ids.end(), gt_ids.begin());
        std::copy(dists.begin(), dists.end(), gt_dists.begin());
    }
    const auto validation_gt = read_vecs<int>(o.validation_gt, V, o.k);
    const auto query_gt = read_vecs<int>(o.query_gt, Q, o.k);
    for (const auto* truth : {&validation_gt, &query_gt}) {
        for (int id : *truth) {
            if (id < 0 || static_cast<idx_t>(id) >= index->ntotal) fail("ground truth id out of range");
        }
    }

    // Sanity reference: plain IVF search with the fixed nprobe budget.
    std::vector<float> plain_distances(Q * o.k);
    std::vector<idx_t> plain_labels(Q * o.k);

    AuncelRun auncel(index, o, std::move(queries), std::move(gt_ids), std::move(gt_dists));
    start = Clock::now();
    auncel.train();
    const double training_s = seconds_since(start);
    log("Auncel training on " + std::to_string(T) + " learn queries: " + fixed(training_s, 1) + " s");

    {
        index->nprobe = o.nprobe;
        const auto rows = read_vecs<float>(o.query, Q, d);
        index->search(Q, rows.data(), o.k, plain_distances.data(), plain_labels.data());
    }
    double plain_recall = 0.0;
    for (size_t q = 0; q < Q; ++q) plain_recall += count_hits(&plain_labels[q * o.k], &query_gt[q * o.k], o.k);
    plain_recall /= static_cast<double>(Q * o.k);
    log("Plain IVF search at nprobe=" + std::to_string(o.nprobe) + ": test recall " + fixed(plain_recall, 4));

    auto evaluate_tuning = [&](float target, const Setting& setting) {
        auncel.configure(target, setting);
        std::vector<float> distances;
        std::vector<idx_t> labels;
        faiss::indexIVF_stats.reset();
        auncel.search_batch(T, V, distances, labels);
        Evaluation evaluation;
        for (size_t q = 0; q < V; ++q) {
            evaluation.avg_recall += count_hits(&labels[q * W], &validation_gt[q * o.k], o.k);
        }
        evaluation.avg_recall /= static_cast<double>(V * o.k);
        evaluation.avg_dists = static_cast<double>(faiss::indexIVF_stats.ndis) / V;
        return evaluation;
    };

    std::vector<float> multipliers;
    for (int step = 0;; ++step) {
        const float value = o.multiplier_min + step * o.multiplier_step;
        if (value > o.multiplier_max + 1e-4f) break;
        multipliers.push_back(value);
    }

    mkdir(o.output_dir.c_str(), 0775);
    std::ostringstream stem_prefix;
    stem_prefix << o.output_dir << "/nlist" << nlist << "_nprobe" << o.nprobe << "_qs" << Q << "_tr";

    for (const std::string& target_text : o.targets) {
        const float target = std::stof(target_text);
        log("--- " + o.dataset + " target_recall=" + target_text + " ---");

        // For each std_m, binary-search the smallest multiplier whose average
        // validation recall reaches the target (recall grows with the multiplier,
        // which only extends the scan). Keep the cheapest such setting; if none
        // reaches the target, keep the highest-recall one.
        const auto tuning_start = Clock::now();
        size_t evaluations = 0;
        Setting best;
        bool have_best = false, best_feasible = false;
        std::ostringstream tuning_log;
        for (float std_m : o.std_grid) {
            size_t low = 0, high = multipliers.size() - 1;
            Setting chosen{multipliers[high], std_m, evaluate_tuning(target, {multipliers[high], std_m, {}})};
            ++evaluations;
            const bool feasible = chosen.tuning.avg_recall >= target;
            if (feasible) {
                while (low < high) {
                    const size_t middle = (low + high) / 2;
                    const Setting candidate{multipliers[middle], std_m, evaluate_tuning(target, {multipliers[middle], std_m, {}})};
                    ++evaluations;
                    if (candidate.tuning.avg_recall >= target) {
                        high = middle;
                        chosen = candidate;
                    } else {
                        low = middle + 1;
                    }
                }
            }
            tuning_log << "tuning std_m=" << std_m << " multipler=" << chosen.multiplier
                       << " avg_recall=" << chosen.tuning.avg_recall << " avg_dists=" << chosen.tuning.avg_dists
                       << (feasible ? "" : " (target not reached)") << '\n';
            const bool better = !have_best ||
                                (feasible && (!best_feasible || chosen.tuning.avg_dists < best.tuning.avg_dists)) ||
                                (!feasible && !best_feasible && chosen.tuning.avg_recall > best.tuning.avg_recall);
            if (better) {
                best = chosen;
                best_feasible = feasible;
                have_best = true;
            }
        }
        const double tuning_s = seconds_since(tuning_start);
        log(tuning_log.str() + "chosen: multipler=" + fixed(best.multiplier, 1) + " std_m=" + fixed(best.std_m, 1) +
            " (validation recall " + fixed(best.tuning.avg_recall, 4) + ", " + std::to_string(evaluations) +
            " evaluations, " + fixed(tuning_s, 1) + " s)");

        // Test queries, one at a time on one thread.
        auncel.configure(target, best);
        omp_set_num_threads(1);
        std::ostringstream csv;
        csv << kCsvHeader << '\n';
        std::vector<float> distances(W);
        std::vector<idx_t> labels(W);
        double recall_sum = 0.0, dists_sum = 0.0, ms_sum = 0.0, lists_sum = 0.0;
        size_t at_target = 0;
        for (size_t q = 0; q < Q; ++q) {
            const auto query_start = Clock::now();
            const auto counts = auncel.search_one(T + V + q, distances.data(), labels.data());
            const double elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - query_start).count();
            const size_t hits = count_hits(labels.data(), &query_gt[q * o.k], o.k);
            const double recall = static_cast<double>(hits) / o.k;
            csv << q << ',' << counts.second << ',' << fixed(elapsed_ms, 6) << ',' << recall << ',' << counts.first
                << ',' << hits << ',';
            for (size_t i = 0; i < o.k; ++i) csv << (i ? ";" : "") << labels[i];
            csv << '\n';
            recall_sum += recall;
            dists_sum += counts.second;
            ms_sum += elapsed_ms;
            lists_sum += counts.first;
            at_target += recall + 1e-9 >= target;
        }
        omp_set_num_threads(o.threads);
        const double n = static_cast<double>(Q);

        std::ostringstream report;
        report << "dataset=" << o.dataset << "\nmetric=L2\nk=" << o.k << "\nnlist=" << nlist << "\nnprobe=" << o.nprobe
               << "\nd=" << d << "\nntotal=" << index->ntotal << "\ntarget_recall=" << target_text << "\ntrain_queries=" << T
               << "\ntune_queries=" << V << "\ntesting_queries=" << Q << "\ngt_width=" << W
               << "\nindex_file=" << o.index << "\nindex_source=prebuilt_darth"
               << "\nmultipler=" << best.multiplier << "\nstd_m=" << best.std_m
               << "\ntuning_target_reached=" << (best_feasible ? 1 : 0)
               << "\ntuning_avg_recall=" << best.tuning.avg_recall << "\ntuning_avg_dists=" << best.tuning.avg_dists
               << "\ntuning_evaluations=" << evaluations << "\nplain_nprobe_test_recall=" << plain_recall
               << "\n\n# wall-clock seconds"
               << "\nindex_load_time_s=" << index_load_s << "\ngeometry_time_s=" << geometry_s
               << "\ntraining_time_s=" << training_s << "\ntuning_time_s=" << tuning_s
               << "\n\n# tuning grid (best multipler per std_m)\n" << tuning_log.str()
               << "\n# testing summary\navg_recall=" << recall_sum / n << "\nfraction_at_target=" << at_target / n
               << "\navg_dists=" << dists_sum / n << "\navg_clusters_searched=" << lists_sum / n
               << "\navg_elaps_ms=" << ms_sum / n << '\n';
        const std::string stem = stem_prefix.str() + target_text;
        write_atomically(stem + "_preparation.txt", report.str());
        write_atomically(stem + ".csv", csv.str());
        log("target=" + target_text + ": avg_recall=" + fixed(recall_sum / n, 4) + ", at_target=" +
            fixed(at_target / n, 3) + ", avg_dists=" + fixed(dists_sum / n, 1) + ", avg_lists=" +
            fixed(lists_sum / n, 1) + ", avg_ms=" + fixed(ms_sum / n, 3) + " (multipler=" +
            fixed(best.multiplier, 1) + ", std_m=" + fixed(best.std_m, 1) + ")");
    }
    log(o.dataset + " finished in " + fixed(seconds_since(program_start), 1) + " s");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--version") {
        std::cout << kVersion << std::endl;
        return 0;
    }
    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        usage(std::cout);
        return 0;
    }
    try {
        run(parse_options(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << std::endl;
        usage(std::cerr);
        return 1;
    }
    return 0;
}
