// Static-index evaluation of the pinned upstream Quake APS implementation.
// No maintenance policy, index updates, background workers, or model tuning.
#include <quake_index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>

#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

static void require(bool condition, const string& message) {
    if (!condition) throw std::runtime_error(message);
}

static double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static string json_string(const string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
        else out << c;
    }
    out << '"';
    return out.str();
}

template <typename T>
static vector<T> read_vecs_prefix(const string& filename, int rows, int wanted_dim, int& stored_dim) {
    static_assert(sizeof(T) == 4, "vecs requires four-byte elements");
    std::ifstream in(filename, std::ios::binary);
    require(bool(in), "Cannot open " + filename);
    int32_t d = 0;
    in.read(reinterpret_cast<char*>(&d), sizeof(d));
    require(bool(in) && d > 0 && d <= 1000000, "Invalid dimension in " + filename);
    stored_dim = d;
    const auto row_bytes = uint64_t(d + 1) * sizeof(T);
    const auto bytes = fs::file_size(filename);
    require(bytes % row_bytes == 0 && bytes / row_bytes >= uint64_t(rows),
            "Invalid size or insufficient query rows in " + filename);
    if (wanted_dim == 0) wanted_dim = d;
    require(wanted_dim <= d, "Insufficient ground-truth width in " + filename);
    in.seekg(0);
    vector<T> values(size_t(rows) * wanted_dim);
    for (int q = 0; q < rows; ++q) {
        int32_t row_dim;
        in.read(reinterpret_cast<char*>(&row_dim), sizeof(row_dim));
        require(bool(in) && row_dim == d, "Inconsistent row dimension in " + filename);
        in.read(reinterpret_cast<char*>(values.data() + size_t(q) * wanted_dim), wanted_dim * sizeof(T));
        in.seekg((d - wanted_dim) * sizeof(T), std::ios::cur);
        require(bool(in), "Truncated row in " + filename);
    }
    return values;
}

struct Configuration {
    string index_path, query_path, gt_path, output_directory, dataset;
    int nlist = 0, nprobe = 0, k = 50, query_count = 1000;
    vector<double> targets;
};

struct QueryResult {
    vector<int64_t> ids, lists;
    int64_t distances = 0;
    double coarse_ms = 0, scan_ms = 0;
};

class StaticQuake {
public:
    shared_ptr<QuakeIndex> index;
    vector<int64_t> list_sizes;
    vector<float> centroids;
    int dimension;
    int64_t ntotal;

    explicit StaticQuake(const faiss::IndexIVFFlat& source) {
        require(source.metric_type == faiss::METRIC_L2 && source.is_trained,
                "Expected a trained L2 IndexIVFFlat");
        require(source.nlist > 1 && source.ntotal > 0 && source.d > 0, "Invalid IVF index");
        auto* lists = dynamic_cast<faiss::ArrayInvertedLists*>(source.invlists);
        require(lists != nullptr, "Expected ArrayInvertedLists in the DARTH IVF index");
        require(lists->code_size == size_t(source.d) * sizeof(float), "Expected uncompressed float32 lists");
        dimension = source.d;
        ntotal = source.ntotal;
        centroids.resize(source.nlist * dimension);
        source.quantizer->reconstruct_n(0, source.nlist, centroids.data());

        // Native Quake flat parent over the ORIGINAL centroids, preserving IDs.
        auto parent_store = std::make_shared<faiss::DynamicInvertedLists>(1, dimension * sizeof(float));
        vector<int64_t> centroid_ids(source.nlist);
        std::iota(centroid_ids.begin(), centroid_ids.end(), 0);
        parent_store->add_entries(0, source.nlist, centroid_ids.data(),
                                  reinterpret_cast<const uint8_t*>(centroids.data()));
        auto parent = make_index(nullptr, parent_store, 1);

        // Upstream conversion copies each list without training or reassigning.
        auto store = std::shared_ptr<faiss::DynamicInvertedLists>(faiss::convert_from_array_invlists(lists));
        index = make_index(parent, store, 0);
        require(index->ntotal() == source.ntotal && index->nlist() == int64_t(source.nlist),
                "Quake conversion changed ntotal/nlist");
        for (size_t p = 0; p < source.nlist; ++p) {
            const auto count = lists->list_size(p);
            require(store->list_size(p) == count, "Quake conversion changed a list size");
            if (count) {
                require(std::memcmp(store->get_ids(p), lists->get_ids(p), count * sizeof(int64_t)) == 0,
                        "Quake conversion changed vector IDs");
                require(std::memcmp(store->get_codes(p), lists->get_codes(p), count * lists->code_size) == 0,
                        "Quake conversion changed vectors");
            }
            list_sizes.push_back(count);
        }
        assert_static();
    }

    void assert_static() const {
        require(!index->maintenance_policy_ && !index->parent_->maintenance_policy_, "Maintenance must be absent");
        require(!index->query_coordinator_->maintenance_policy_ &&
                !index->parent_->query_coordinator_->maintenance_policy_, "Coordinator maintenance must be absent");
        require(index->ntotal() == ntotal && index->nlist() == int64_t(list_sizes.size()),
                "Index size changed during evaluation");
        const auto store = index->partition_manager_->partition_store_;
        for (size_t p = 0; p < list_sizes.size(); ++p)
            require(store->list_size(p) == size_t(list_sizes[p]), "Partition size changed during evaluation");
        const auto parent = index->parent_->partition_manager_->partition_store_;
        require(parent->list_size(0) == list_sizes.size(), "Centroid count changed");
        require(std::memcmp(parent->get_codes(0), centroids.data(), centroids.size() * sizeof(float)) == 0,
                "Centroids changed during evaluation");
    }

    QueryResult search(const Tensor& query, int k, int nprobe, float recall_target) {
        require(nprobe > 0 && nprobe <= int(list_sizes.size()), "Invalid candidate-list budget");
        auto coarse_params = std::make_shared<SearchParams>();
        coarse_params->k = nprobe;
        coarse_params->recall_target = -1;
        coarse_params->num_threads = 1;
        coarse_params->track_hits = false;

        auto params = std::make_shared<SearchParams>();
        params->k = k;
        params->nprobe = nprobe;
        params->recall_target = recall_target;
        params->num_threads = 1;
        params->batched_scan = false;
        params->track_hits = false;
        params->use_auncel = false;
        params->use_spann = false;
        // Preserve upstream APS defaults, including its incomplete-beta lookup.
        // Provide exactly nprobe candidates; upstream search() would instead
        // truncate nlist * initial_search_fraction when recall_target is set.
        const auto start = Clock::now();
        auto candidates = index->parent_->search(query, coarse_params);
        const auto coarse_end = Clock::now();
        auto result = index->query_coordinator_->serial_scan(query, candidates->ids, params);
        const auto scan_end = Clock::now();

        QueryResult out;
        out.coarse_ms = elapsed_ms(start, coarse_end);
        out.scan_ms = elapsed_ms(coarse_end, scan_end);
        auto ids = result->ids.contiguous();
        out.ids.assign(ids.data_ptr<int64_t>(), ids.data_ptr<int64_t>() + k);
        const int scanned = result->timing_info->partitions_scanned;
        require(scanned > 0 && scanned <= nprobe, "Invalid upstream scanned-partition count");
        auto pids = candidates->ids.contiguous();
        const auto* p = pids.data_ptr<int64_t>();
        std::unordered_set<int64_t> seen;
        for (int rank = 0; rank < scanned; ++rank) {
            require(p[rank] >= 0 && p[rank] < int64_t(list_sizes.size()) && seen.insert(p[rank]).second,
                    "Invalid/repeated candidate list");
            out.lists.push_back(p[rank]);
            // serial_scan evaluates every vector in each visited list once.
            out.distances += list_sizes[p[rank]];
        }
        return out;
    }

private:
    static shared_ptr<QuakeIndex> make_index(shared_ptr<QuakeIndex> parent,
                                            shared_ptr<faiss::DynamicInvertedLists> store, int level) {
        auto out = std::make_shared<QuakeIndex>(level);
        out->metric_ = faiss::METRIC_L2;
        out->parent_ = parent;
        out->partition_manager_ = std::make_shared<PartitionManager>();
        out->partition_manager_->parent_ = parent;
        out->partition_manager_->partition_store_ = store;
        out->partition_manager_->curr_partition_id_ = store->nlist;
        out->build_params_ = std::make_shared<IndexBuildParams>();
        out->build_params_->nlist = store->nlist;
        out->build_params_->num_workers = 0;
        out->build_params_->use_gpu = false;
        out->build_params_->use_numa = false;
        // Deliberately never construct/initialize a MaintenancePolicy.
        out->query_coordinator_ = std::make_shared<QueryCoordinator>(
            parent, out->partition_manager_, nullptr, faiss::METRIC_L2, level, 0, false, 1);
        return out;
    }
};

static int hits_at_k(const vector<int64_t>& ids, const int32_t* gt, int k, int64_t ntotal) {
    std::unordered_set<int64_t> truth(gt, gt + k), returned;
    require(truth.size() == size_t(k), "Duplicate ground-truth IDs");
    int hits = 0;
    for (auto id : ids) {
        if (id == -1) continue;
        require(id >= 0 && id < ntotal && returned.insert(id).second, "Invalid/repeated returned ID");
        hits += truth.count(id);
    }
    return hits;
}

static void write_ids(std::ostream& out, const vector<int64_t>& ids) {
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) out << ';';
        out << ids[i];
    }
}

static string result_stem(const Configuration& c, double target) {
    std::ostringstream out;
    out << "nlist" << c.nlist << "_nprobe" << c.nprobe << "_qs" << c.query_count
        << "_tr" << std::fixed << std::setprecision(2) << target;
    return out.str();
}

static void evaluate(const Configuration& c) {
    int query_dim, gt_dim;
    auto queries = read_vecs_prefix<float>(c.query_path, c.query_count, 0, query_dim);
    auto gt = read_vecs_prefix<int32_t>(c.gt_path, c.query_count, c.k, gt_dim);
    require(std::all_of(queries.begin(), queries.end(), [](float v) { return std::isfinite(v); }),
            "Non-finite query vector");

    const auto load_start = Clock::now();
    std::unique_ptr<faiss::Index> faiss_index(faiss::read_index(c.index_path.c_str()));
    auto* ivf = dynamic_cast<faiss::IndexIVFFlat*>(faiss_index.get());
    require(ivf && typeid(*ivf) == typeid(faiss::IndexIVFFlat), "Expected a plain IndexIVFFlat");
    require(ivf->nlist == size_t(c.nlist) && ivf->d == query_dim && ivf->ntotal >= c.k,
            "Index nlist/dimension/size does not match experiment");
    for (auto id : gt) require(id >= 0 && id < ivf->ntotal, "Ground-truth ID outside index");
    const auto loaded = Clock::now();
    StaticQuake quake(*ivf);
    const auto converted = Clock::now();
    faiss_index.reset(); // Release the source copy before running any queries.
    const auto query_tensor = torch::from_blob(queries.data(), {c.query_count, query_dim}, torch::kFloat32);
    fs::create_directories(c.output_directory);

    for (double target : c.targets) {
        quake.assert_static();
        const fs::path file = fs::path(c.output_directory) / (result_stem(c, target) + ".csv");
        const string temporary = file.string() + ".tmp";
        std::ofstream out(temporary);
        require(bool(out), "Cannot create " + temporary);
        out << "qid,dists,elaps_ms,r,clusters_searched,hits,k,target_recall,nlist,nprobe,coarse_ms,scan_ms,neighbor_ids,scanned_list_ids\n";
        out << std::setprecision(17);
        int64_t total_hits = 0;
        int below = 0;
        for (int qid = 0; qid < c.query_count; ++qid) {
            auto result = quake.search(query_tensor.slice(0, qid, qid + 1), c.k, c.nprobe, float(target));
            const int hits = hits_at_k(result.ids, gt.data() + size_t(qid) * c.k, c.k, quake.ntotal);
            const double recall = double(hits) / c.k;
            total_hits += hits;
            below += recall < target;
            out << qid << ',' << result.distances << ',' << result.coarse_ms + result.scan_ms << ','
                << recall << ',' << result.lists.size() << ',' << hits << ',' << c.k << ',' << target << ','
                << c.nlist << ',' << c.nprobe << ',' << result.coarse_ms << ',' << result.scan_ms << ',';
            write_ids(out, result.ids);
            out << ',';
            write_ids(out, result.lists);
            out << '\n';
            if ((qid + 1) % 100 == 0) std::cout << c.dataset << " Rt=" << target << ": " << qid + 1 << '/' << c.query_count << std::endl;
        }
        quake.assert_static();
        out.close();
        require(bool(out), "Failed writing " + temporary);
        fs::rename(temporary, file);
        std::cout << "Saved " << file << "; mean recall=" << double(total_hits) / (c.query_count * c.k)
                  << "; RQUT=" << double(below) / c.query_count << std::endl;
    }

    // Partition sizes permit independent verification of every CSV work count.
    const fs::path metadata = fs::path(c.output_directory) / "index_metadata.json";
    std::ofstream out(metadata.string() + ".tmp");
    require(bool(out), "Cannot create index metadata");
    out << std::setprecision(17)
        << "{\n  \"upstream_commit\": " << json_string(QUAKE_UPSTREAM_COMMIT)
        << ",\n  \"index_source\": " << json_string(fs::absolute(c.index_path).string())
        << ",\n  \"index_conversion\": \"exact_faiss_lists_and_centroids\""
        << ",\n  \"metric\": \"l2\",\n  \"nlist\": " << c.nlist << ",\n  \"nprobe\": " << c.nprobe
        << ",\n  \"dimension\": " << query_dim << ",\n  \"ntotal\": " << quake.ntotal
        << ",\n  \"maintenance\": false,\n  \"track_hits\": false,\n  \"num_workers\": 0,\n  \"search_threads\": 1"
        << ",\n  \"use_numa\": false,\n  \"use_gpu\": false,\n  \"use_auncel\": false,\n  \"use_spann\": false"
        << ",\n  \"candidate_selection\": \"exact_nprobe_nearest_original_centroids\""
        << ",\n  \"use_precomputed\": " << (DEFAULT_PRECOMPUTED ? "true" : "false")
        << ",\n  \"recompute_threshold\": " << DEFAULT_RECOMPUTE_THRESHOLD
        << ",\n  \"static_index_checks_passed\": true"
        << ",\n  \"source_load_time_s\": " << elapsed_ms(load_start, loaded) / 1000
        << ",\n  \"conversion_time_s\": " << elapsed_ms(loaded, converted) / 1000
        << ",\n  \"distance_count_scope\": \"database_vectors_in_visited_lists\""
        << ",\n  \"elapsed_time_scope\": \"coarse_selection_and_native_aps_scan\""
        << ",\n  \"partition_sizes\": [";
    for (size_t i = 0; i < quake.list_sizes.size(); ++i) {
        if (i) out << ',';
        out << quake.list_sizes[i];
    }
    out << "]\n}\n";
    out.close();
    require(bool(out), "Failed writing index metadata");
    fs::rename(metadata.string() + ".tmp", metadata);
}

template <typename T>
static void write_vecs(const fs::path& file, const vector<T>& values, int rows, int dim) {
    std::ofstream out(file, std::ios::binary);
    for (int q = 0; q < rows; ++q) {
        const int32_t d = dim;
        out.write(reinterpret_cast<const char*>(&d), sizeof(d));
        out.write(reinterpret_cast<const char*>(values.data() + size_t(q) * dim), dim * sizeof(T));
    }
    require(bool(out), "Failed writing smoke fixture");
}

// Small deterministic fixture exercises real FAISS serialization and native APS.
// Its GLOVE100 directory name allows testing the same shell/Python orchestration.
static void make_fixture(const fs::path& root) {
    constexpr int n = 4000, d = 100, nlist = 1000, nq = 12, k = 50;
    std::mt19937 rng(42);
    std::normal_distribution<float> normal;
    vector<float> centers(nlist * d), base(n * d), queries(nq * d);
    for (auto& v : centers) v = normal(rng) * 5;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < d; ++j) base[i * d + j] = centers[(i % nlist) * d + j] + 0.1f * normal(rng);
    for (int i = 0; i < nq; ++i)
        for (int j = 0; j < d; ++j) queries[i * d + j] = centers[i * d + j] + normal(rng);
    faiss::IndexFlatL2 quantizer(d);
    quantizer.add(nlist, centers.data());
    faiss::IndexIVFFlat ivf(&quantizer, d, nlist, faiss::METRIC_L2);
    ivf.is_trained = true;
    ivf.add(n, base.data());
    StaticQuake quake(ivf);
    auto tensor = torch::from_blob(queries.data(), {nq, d}, torch::kFloat32);
    ivf.nprobe = 100;
    vector<float> reference_distances(k);
    vector<int64_t> reference_ids(k);
    bool stopped_early = false;
    for (int q = 0; q < nq; ++q) {
        auto query = tensor.slice(0, q, q + 1);
        auto full = quake.search(query, k, 100, -1);
        require(full.lists.size() == 100, "Disabled APS did not scan full budget");
        ivf.search(1, queries.data() + q * d, k, reference_distances.data(), reference_ids.data());
        auto got = full.ids;
        auto expected = reference_ids;
        std::sort(got.begin(), got.end());
        std::sort(expected.begin(), expected.end());
        require(got == expected, "Converted Quake differs from original full-budget IVF");
        auto early = quake.search(query, k, 100, 0.80f);
        require(early.distances <= full.distances, "APS exceeded fixed search budget");
        stopped_early |= early.lists.size() < 100;
        auto exhaustive = quake.search(query, k, nlist, -1);
        faiss::IndexFlatL2 exact(d);
        exact.add(n, base.data());
        exact.search(1, queries.data() + q * d, k, reference_distances.data(), reference_ids.data());
        got = exhaustive.ids;
        expected = reference_ids;
        std::sort(got.begin(), got.end());
        std::sort(expected.begin(), expected.end());
        require(got == expected, "Exhaustive converted Quake differs from exact ground truth");
    }
    require(stopped_early, "Smoke fixture did not exercise early termination");
    quake.assert_static();
    const auto index_dir = root / "indexes/GLOVE100";
    const auto data_dir = root / "data/GLOVE100";
    fs::create_directories(index_dir);
    fs::create_directories(data_dir);
    faiss::write_index(&ivf, (index_dir / "GLOVE100.nlist1000.index").c_str());
    faiss::IndexFlatL2 exact(d);
    exact.add(n, base.data());
    vector<float> distances(nq * k);
    vector<int64_t> ids(nq * k);
    exact.search(nq, queries.data(), k, distances.data(), ids.data());
    vector<int32_t> gt(ids.begin(), ids.end());
    write_vecs(data_dir / "query.10K.fvecs", queries, nq, d);
    write_vecs(data_dir / "query.groundtruth.10K.k1000.ivecs", gt, nq, k);
    std::cout << "Smoke fixture passed: exact conversion, full-budget/exhaustive equivalence, APS early stopping, static index.\n";
}

int main(int argc, char** argv) {
    try {
        torch::set_num_threads(1);
        torch::set_num_interop_threads(1);
        omp_set_num_threads(1);
        torch::NoGradGuard no_grad;
        if (argc == 3 && string(argv[1]) == "--make-fixture") {
            make_fixture(argv[2]);
            return 0;
        }
        if (argc == 2 && string(argv[1]) == "--version") {
            std::cout << "DARTH Quake evaluator; upstream " << QUAKE_UPSTREAM_COMMIT << '\n';
            return 0;
        }
        Configuration c;
        for (int i = 1; i < argc; ++i) {
            const string key = argv[i];
            require(i + 1 < argc, "Missing value for " + key);
            const string value = argv[++i];
            if (key == "--index") c.index_path = value;
            else if (key == "--queries") c.query_path = value;
            else if (key == "--ground-truth") c.gt_path = value;
            else if (key == "--output-directory") c.output_directory = value;
            else if (key == "--dataset") c.dataset = value;
            else if (key == "--nlist") c.nlist = std::stoi(value);
            else if (key == "--nprobe") c.nprobe = std::stoi(value);
            else if (key == "--k") c.k = std::stoi(value);
            else if (key == "--query-count") c.query_count = std::stoi(value);
            else if (key == "--target-recalls") {
                std::istringstream in(value);
                string item;
                while (std::getline(in, item, ',')) c.targets.push_back(std::stod(item));
            } else throw std::runtime_error("Unknown argument " + key);
        }
        require(!c.index_path.empty() && !c.query_path.empty() && !c.gt_path.empty() &&
                !c.output_directory.empty() && !c.dataset.empty(), "Missing experiment paths/dataset");
        require(c.nlist > 1 && c.nprobe > 0 && c.nprobe <= c.nlist && c.k > 0 && c.query_count > 0,
                "Invalid nlist/nprobe/k/query count");
        require(!c.targets.empty(), "Missing recall targets");
        std::set<string> filenames;
        for (auto target : c.targets) {
            require(std::isfinite(target) && target > 0 && target <= 1, "Invalid recall target");
            require(filenames.insert(result_stem(c, target)).second, "Duplicate target output filename");
        }
        evaluate(c);
    } catch (const std::exception& e) {
        std::cerr << "Quake evaluation failed: " << e.what() << '\n';
        return 1;
    }
}
