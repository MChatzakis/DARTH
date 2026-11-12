// Ada-ef evaluator for the DARTH+ inner-product HNSW comparison.
//
// For one dataset this program builds (or loads) an hnswlib index with the
// DARTH+ M/efConstruction, computes (or loads) Ada-ef's dataset statistics,
// and then, for every recall target, builds Ada-ef's ef-estimation table from
// the first learn queries and answers the first test queries one at a time.
//
// Distances are hnswlib's inner-product distance 1 - <q, x> on the original,
// unnormalized vectors. Ada-ef's distance-side estimator ("cd" upstream) models
// exactly this quantity: mean 1 - <q, mean>, variance q Cov q^T. Its "ipd"
// estimator is not used because it models the similarity <q, x> while the
// search hands it distances.

#include "hnswlib/adaptive_ef.h"

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

constexpr const char* kVersion = "darth_adaef_eval 1.2 (zero-recall learn queries count as 0; neighbor ids)";
constexpr const char* kCsvHeader = "qid,dists,elaps_ms,r,hits,ef,score,neighbor_ids";

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void log(const std::string& message) {
    std::cout << message << std::endl;
}

struct VecsInfo {
    size_t rows;
    size_t dim;
};

// fvecs/ivecs: every row is an int32 dimension followed by dim 32-bit values.
VecsInfo vecs_info(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) fail("cannot open " + path);
    int32_t dim = 0;
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    if (!in || dim <= 0) fail("invalid vecs header in " + path);
    const size_t row_bytes = 4 * (static_cast<size_t>(dim) + 1);
    const auto bytes = static_cast<size_t>(fs::file_size(path));
    if (bytes % row_bytes != 0) fail("size of " + path + " is not a multiple of its row size");
    return {bytes / row_bytes, static_cast<size_t>(dim)};
}

// Reads the first `rows` rows of a vecs file, keeping their first `cols` values.
template <typename Matrix>
Matrix read_vecs(const std::string& path, size_t rows, size_t cols) {
    const VecsInfo info = vecs_info(path);
    if (rows > info.rows) {
        fail(path + " has " + std::to_string(info.rows) + " rows; " + std::to_string(rows) + " needed");
    }
    if (cols > info.dim) {
        fail(path + " has " + std::to_string(info.dim) + " columns; " + std::to_string(cols) + " needed");
    }
    Matrix matrix(rows, cols);
    std::ifstream in(path, std::ios::binary);
    const size_t row_bytes = 4 * (info.dim + 1);
    const size_t chunk_rows = std::max<size_t>(1, (size_t{64} << 20) / row_bytes);
    std::vector<char> buffer(chunk_rows * row_bytes);
    for (size_t first = 0; first < rows; first += chunk_rows) {
        const size_t count = std::min(chunk_rows, rows - first);
        in.read(buffer.data(), static_cast<std::streamsize>(count * row_bytes));
        if (!in) fail("truncated vecs file " + path);
        for (size_t i = 0; i < count; ++i) {
            const char* row = buffer.data() + i * row_bytes;
            int32_t dim = 0;
            std::memcpy(&dim, row, sizeof(dim));
            if (static_cast<size_t>(dim) != info.dim) fail("inconsistent row dimension in " + path);
            std::memcpy(matrix.row(first + i).data(), row + 4, cols * 4);
        }
    }
    return matrix;
}

std::vector<float> read_fvecs_row(const std::string& path, size_t row_id, size_t dim) {
    std::ifstream in(path, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(row_id * 4 * (dim + 1) + 4));
    std::vector<float> row(dim);
    in.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(dim * 4));
    if (!in) fail("cannot read row " + std::to_string(row_id) + " of " + path);
    return row;
}

// Writes through a temporary file so a partial output never looks complete.
void write_atomically(const std::string& path, const std::string& contents) {
    const std::string temporary = path + ".tmp";
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) fail("cannot write " + temporary);
        out << contents;
        if (!out) fail("cannot write " + temporary);
    }
    fs::rename(temporary, path);
}

struct Options {
    std::string dataset, base, learn, learn_gt, query, query_gt, index, statistics, output_dir;
    size_t M = 0, ef_construction = 0, k = 50, learn_queries = 200, test_queries = 1000;
    std::vector<std::string> targets = {"0.80", "0.85", "0.90", "0.95", "0.99"};
    size_t build_threads = std::max(1u, std::thread::hardware_concurrency());
    size_t distance_list_size = 1 + 32 + 31 * 32;  // upstream default (2-hop size for M=16)
    int ef_upper_bound = 5000;
    float quantile_step = 1e-3f;
};

void usage(std::ostream& out) {
    out << "Usage: darth_adaef_eval --dataset NAME --base B.fvecs --learn L.fvecs --learn-gt L.ivecs\n"
           "         --query Q.fvecs --query-gt Q.ivecs --M M --ef-construction EFC\n"
           "         --index FILE --statistics FILE --output-dir DIR [--targets 0.80,0.85,0.90,0.95,0.99]\n"
           "         [--k 50] [--learn-queries 200] [--test-queries 1000] [--ef-upper-bound 5000]\n"
           "         [--distance-list-size 1057] [--quantile-step 0.001] [--build-threads N]\n"
           "       darth_adaef_eval --version\n";
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
    Options options;
    for (int i = 1; i < argc; i += 2) {
        const std::string name = argv[i];
        if (i + 1 >= argc) fail("missing value for " + name);
        const std::string value = argv[i + 1];
        if (name == "--dataset") options.dataset = value;
        else if (name == "--base") options.base = value;
        else if (name == "--learn") options.learn = value;
        else if (name == "--learn-gt") options.learn_gt = value;
        else if (name == "--query") options.query = value;
        else if (name == "--query-gt") options.query_gt = value;
        else if (name == "--index") options.index = value;
        else if (name == "--statistics") options.statistics = value;
        else if (name == "--output-dir") options.output_dir = value;
        else if (name == "--M") options.M = parse_size(name, value);
        else if (name == "--ef-construction") options.ef_construction = parse_size(name, value);
        else if (name == "--k") options.k = parse_size(name, value);
        else if (name == "--learn-queries") options.learn_queries = parse_size(name, value);
        else if (name == "--test-queries") options.test_queries = parse_size(name, value);
        else if (name == "--build-threads") options.build_threads = parse_size(name, value);
        else if (name == "--distance-list-size") options.distance_list_size = parse_size(name, value);
        else if (name == "--ef-upper-bound") options.ef_upper_bound = static_cast<int>(parse_size(name, value));
        else if (name == "--quantile-step") options.quantile_step = std::stof(value);
        else if (name == "--targets") {
            options.targets.clear();
            std::stringstream stream(value);
            std::string target;
            while (std::getline(stream, target, ',')) options.targets.push_back(target);
        } else fail("unknown option " + name);
    }
    for (const auto& [name, value] : {std::pair<std::string, std::string>{"--dataset", options.dataset},
                                      {"--base", options.base}, {"--learn", options.learn},
                                      {"--learn-gt", options.learn_gt}, {"--query", options.query},
                                      {"--query-gt", options.query_gt}, {"--index", options.index},
                                      {"--statistics", options.statistics}, {"--output-dir", options.output_dir}}) {
        if (value.empty()) fail(name + " is required");
    }
    if (options.M == 0 || options.ef_construction == 0) fail("--M and --ef-construction are required");
    if (options.targets.empty()) fail("--targets is required");
    for (const auto& target : options.targets) {
        const float value = std::stof(target);
        if (!(value > 0.0f && value < 1.0f)) fail("recall targets must be in (0, 1), got " + target);
    }
    if (!(options.quantile_step > 0.0f && options.quantile_step < 0.2f)) fail("--quantile-step must be in (0, 0.2)");
    if (options.ef_upper_bound < static_cast<int>(options.k)) fail("--ef-upper-bound must be at least k");
    return options;
}

using Index = hnswlib::HierarchicalNSW<float>;

// Closer-first labels of an hnswlib result queue.
std::vector<size_t> labels_of(std::priority_queue<std::pair<float, hnswlib::labeltype>> queue) {
    std::vector<size_t> labels(queue.size());
    for (size_t position = queue.size(); !queue.empty(); queue.pop()) labels[--position] = queue.top().second;
    return labels;
}

size_t count_hits(const std::vector<size_t>& labels, const hnswdis::MatrixXi& ground_truth, size_t row, size_t k) {
    std::unordered_set<size_t> truth;
    for (size_t j = 0; j < k; ++j) truth.insert(static_cast<size_t>(ground_truth(row, j)));
    size_t hits = 0;
    for (size_t label : labels) hits += truth.count(label);
    return hits;
}

// Counting is done in a separate, untimed pass so the timed search runs the
// native distance function unchanged.
hnswlib::DISTFUNC<float> native_distance = nullptr;
size_t distance_calls = 0;
float counting_distance(const void* a, const void* b, const void* parameters) {
    ++distance_calls;
    return native_distance(a, b, parameters);
}

std::shared_ptr<Index> build_index(hnswlib::SpaceInterface<float>& space, const hnswdis::MatrixXf& base,
                                   const Options& options) {
    auto index = std::make_shared<Index>(&space, base.rows(), options.M, options.ef_construction);
    std::atomic<size_t> inserted{0};
    const size_t report_every = std::max<size_t>(1, base.rows() / 20);
    const auto start = Clock::now();
    hnswdis::ParallelFor(0, base.rows(), options.build_threads, [&](size_t row, size_t) {
        index->addPoint(base.row(row).data(), row);
        const size_t done = ++inserted;
        if (done % report_every == 0) {
            std::ostringstream message;
            message << "  inserted " << done << "/" << base.rows() << " (" << std::fixed << std::setprecision(0)
                    << seconds_since(start) << " s)";
            log(message.str());
        }
    });
    return index;
}

// Spot-checks that the index stores the base vectors under their row ids.
void check_index(const Index& index, const Options& options, size_t base_rows, size_t dim) {
    if (index.cur_element_count != base_rows) {
        fail(options.index + " holds " + std::to_string(index.cur_element_count.load()) + " vectors; base has " +
             std::to_string(base_rows));
    }
    if (index.M_ != options.M || index.ef_construction_ != options.ef_construction) {
        fail(options.index + " was built with M=" + std::to_string(index.M_) + ", efC=" +
             std::to_string(index.ef_construction_));
    }
    for (size_t row : {size_t{0}, base_rows / 2, base_rows - 1}) {
        const auto stored = index.getDataByLabel<float>(row);
        const auto expected = read_fvecs_row(options.base, row, dim);
        if (stored != expected) fail(options.index + " does not match " + options.base + " at row " + std::to_string(row));
    }
}

struct QueryRecord {
    std::vector<size_t> labels;
    double elapsed_ms = 0.0;
    size_t dists = 0;
    size_t hits = 0;
    size_t ef = 0;
    float score = 0.0f;
};

std::string format_double(double value, int precision) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

void run(const Options& options) {
    const auto program_start = Clock::now();
    const VecsInfo base_info = vecs_info(options.base);
    const size_t dim = base_info.dim;
    for (const auto& [path, rows] : {std::pair<std::string, size_t>{options.learn, options.learn_queries},
                                     {options.query, options.test_queries}}) {
        const VecsInfo info = vecs_info(path);
        if (info.dim != dim) fail(path + " has dimension " + std::to_string(info.dim) + "; base has " + std::to_string(dim));
        if (info.rows < rows) fail(path + " has only " + std::to_string(info.rows) + " rows");
    }
    log(options.dataset + ": n=" + std::to_string(base_info.rows) + ", d=" + std::to_string(dim) +
        ", M=" + std::to_string(options.M) + ", efC=" + std::to_string(options.ef_construction) +
        ", k=" + std::to_string(options.k) + ", ef_upper_bound=" + std::to_string(options.ef_upper_bound) +
        ", distance_list_size=" + std::to_string(options.distance_list_size));

    hnswlib::InnerProductSpace space(dim);
    const bool have_index = fs::exists(options.index);
    const bool have_statistics = fs::exists(options.statistics);

    hnswdis::MatrixXf base;
    double base_load_s = 0.0;
    if (!have_index || !have_statistics) {
        const auto start = Clock::now();
        log("Loading base vectors from " + options.base);
        base = read_vecs<hnswdis::MatrixXf>(options.base, base_info.rows, dim);
        base_load_s = seconds_since(start);
    }

    std::shared_ptr<Index> index;
    double index_build_s = 0.0, index_load_s = 0.0;
    if (have_index) {
        const auto start = Clock::now();
        log("Loading hnswlib index " + options.index);
        index = std::make_shared<Index>(&space, options.index);
        index_load_s = seconds_since(start);
    } else {
        log("Building hnswlib index with " + std::to_string(options.build_threads) + " threads");
        const auto start = Clock::now();
        index = build_index(space, base, options);
        index_build_s = seconds_since(start);
        fs::create_directories(fs::path(options.index).parent_path());
        index->saveIndex(options.index + ".tmp");
        fs::rename(options.index + ".tmp", options.index);
        log("Built and saved index in " + format_double(index_build_s, 1) + " s");
    }
    check_index(*index, options, base_info.rows, dim);

    std::shared_ptr<hnswdis::Estimator> estimator;
    double statistics_compute_s = 0.0, statistics_load_s = 0.0;
    if (have_statistics) {
        const auto start = Clock::now();
        estimator = hnswdis::load_estimator_from_file(options.statistics);
        statistics_load_s = seconds_since(start);
    } else {
        const auto start = Clock::now();
        estimator = std::make_shared<hnswdis::CosineDistanceEstimator>(base);
        statistics_compute_s = seconds_since(start);
        fs::create_directories(fs::path(options.statistics).parent_path());
        hnswdis::save_estimator_to_file(*estimator, options.statistics + ".tmp");
        fs::rename(options.statistics + ".tmp", options.statistics);
    }
    if (estimator->get_name() != "CosineDistanceEstimator") {
        fail(options.statistics + " holds a " + estimator->get_name() + ", not the distance-side estimator");
    }
    base.resize(0, 0);  // Neither the table builder nor the search reads the base matrix.
    auto unused_base = std::make_shared<hnswdis::MatrixXf>();

    // Online search and table construction are single-threaded, as upstream.
    omp_set_num_threads(1);
    Eigen::setNbThreads(1);

    auto learn = std::make_shared<hnswdis::MatrixXf>(
        read_vecs<hnswdis::MatrixXf>(options.learn, options.learn_queries, dim));
    auto learn_gt = std::make_shared<hnswdis::MatrixXi>(
        read_vecs<hnswdis::MatrixXi>(options.learn_gt, options.learn_queries, options.k));
    const auto queries = read_vecs<hnswdis::MatrixXf>(options.query, options.test_queries, dim);
    const auto query_gt = read_vecs<hnswdis::MatrixXi>(options.query_gt, options.test_queries, options.k);
    for (const hnswdis::MatrixXi* gt : {static_cast<const hnswdis::MatrixXi*>(learn_gt.get()), &query_gt}) {
        if (gt->minCoeff() < 0 || static_cast<size_t>(gt->maxCoeff()) >= base_info.rows) {
            fail("ground truth contains ids outside [0, " + std::to_string(base_info.rows) + ")");
        }
    }

    fs::create_directories(options.output_dir);
    native_distance = index->fstdistfunc_;
    const std::string stem_prefix = "M" + std::to_string(options.M) + "_efC" + std::to_string(options.ef_construction) +
                                    "_qs" + std::to_string(options.test_queries) + "_tr";

    for (const std::string& target_text : options.targets) {
        const float target = std::stof(target_text);
        const std::string stem = (fs::path(options.output_dir) / (stem_prefix + target_text + "_inner-product")).string();
        log("--- " + options.dataset + " target_recall=" + target_text + " ---");

        const auto table_start = Clock::now();
        hnswdis::EfAdapter adapter(index, unused_base, options.k, "cd", target, options.quantile_step,
                                   options.distance_list_size, learn, learn_gt, estimator, options.ef_upper_bound);
        const double table_s = seconds_since(table_start);
        adapter.serialize(stem + "_ef_table.bin.tmp");
        fs::rename(stem + "_ef_table.bin.tmp", stem + "_ef_table.bin");

        hnswdis::Sketch sketch(adapter.get_ef_recall_estimators(), target);
        const float wae = adapter.get_wae();
        index->setEf(wae);  // upstream online protocol: wae is the per-query ef floor
        const size_t ef_floor = std::max(index->ef_, options.k);
        hnswdis::ApproximatedScoreCalculator score_calculator(estimator, options.quantile_step);

        std::vector<QueryRecord> records(options.test_queries);
        for (size_t q = 0; q < options.test_queries; ++q) {
            const auto start = Clock::now();
            auto result = index->adaptiveSearchKnnTest(queries.row(q).data(), options.k, options.distance_list_size,
                                                       score_calculator, &sketch);
            records[q].elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            records[q].labels = labels_of(std::move(result));
        }

        index->fstdistfunc_ = counting_distance;
        for (size_t q = 0; q < options.test_queries; ++q) {
            distance_calls = 0;
            auto [result, score] = index->adaptiveSearchKnn(queries.row(q).data(), options.k,
                                                            options.distance_list_size, score_calculator, &sketch);
            QueryRecord& record = records[q];
            record.dists = distance_calls;
            if (labels_of(std::move(result)) != record.labels) {
                index->fstdistfunc_ = native_distance;
                fail("query " + std::to_string(q) + " returned different neighbors in the counting pass");
            }
            record.score = score;
            record.ef = std::max(sketch.estimate_ef2(score), ef_floor);
            record.hits = count_hits(record.labels, query_gt, q, options.k);
        }
        index->fstdistfunc_ = native_distance;

        std::ostringstream csv;
        csv << kCsvHeader << '\n';
        double recall_sum = 0.0, dists_sum = 0.0, ms_sum = 0.0, ef_sum = 0.0;
        size_t at_target = 0;
        for (size_t q = 0; q < options.test_queries; ++q) {
            const QueryRecord& record = records[q];
            const double recall = static_cast<double>(record.hits) / options.k;
            csv << q << ',' << record.dists << ',' << format_double(record.elapsed_ms, 6) << ',' << recall << ','
                << record.hits << ',' << record.ef << ',' << record.score << ',';
            for (size_t i = 0; i < record.labels.size(); ++i) csv << (i ? ";" : "") << record.labels[i];
            csv << '\n';
            recall_sum += recall;
            dists_sum += record.dists;
            ms_sum += record.elapsed_ms;
            ef_sum += record.ef;
            at_target += recall + 1e-9 >= target;
        }
        const double n = static_cast<double>(options.test_queries);

        std::ostringstream report;
        report << "dataset=" << options.dataset << "\nmetric=inner-product\nk=" << options.k
               << "\nM=" << options.M << "\nef_construction=" << options.ef_construction
               << "\ntarget_recall=" << target_text << "\nlearn_queries=" << options.learn_queries
               << "\ntesting_queries=" << options.test_queries << "\nef_upper_bound=" << options.ef_upper_bound
               << "\ndistance_list_size=" << options.distance_list_size << "\nquantile_step=" << options.quantile_step
               << "\nestimator=" << estimator->get_name() << "\nindex_file=" << options.index
               << "\nstatistics_file=" << options.statistics << "\nef_table_rows=" << adapter.get_ef_recall_estimators().size()
               << "\nweighted_average_ef=" << wae << "\nef_floor=" << ef_floor
               << "\n\n# wall-clock seconds; zero means the step was not needed in this process"
               << "\nbase_load_time_s=" << base_load_s << "\nindex_build_time_s=" << index_build_s
               << "\nindex_load_time_s=" << index_load_s << "\nstatistics_time_s=" << statistics_compute_s
               << "\nstatistics_load_time_s=" << statistics_load_s << "\nef_table_time_s=" << table_s
               << "\n\n# testing summary\navg_recall=" << recall_sum / n << "\nfraction_at_target=" << at_target / n
               << "\navg_dists=" << dists_sum / n << "\navg_elaps_ms=" << ms_sum / n << "\navg_ef=" << ef_sum / n
               << '\n';

        write_atomically(stem + "_preparation.txt", report.str());
        write_atomically(stem + ".csv", csv.str());
        log("target=" + target_text + ": avg_recall=" + format_double(recall_sum / n, 4) +
            ", at_target=" + format_double(at_target / n, 3) + ", avg_dists=" + format_double(dists_sum / n, 1) +
            ", avg_ms=" + format_double(ms_sum / n, 3) + ", avg_ef=" + format_double(ef_sum / n, 1) +
            ", wae=" + format_double(wae, 1) + ", table_s=" + format_double(table_s, 2));
    }
    log(options.dataset + " finished in " + format_double(seconds_since(program_start), 1) + " s");
}

}  // namespace

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
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
