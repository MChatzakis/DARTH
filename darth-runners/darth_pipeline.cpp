// Native DARTH deployment: generate observations, train, validate and search.
// Intermediate observations and LightGBM boosters never need to leave memory.
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include <faiss/impl/DeclarativeRecall.h>
#include <faiss/utils/distances.h>
#include <omp.h>
#include <common/ComUtil.h>
#include <common/VectorDataLoader.h>
#include <algorithm>
#include <chrono>
#include <clocale>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace
{
  using faiss::idx_t;
  using Model = faiss::LGBMPredictor::Model;
  using Clock = std::chrono::steady_clock;
  double seconds(Clock::time_point start)
  {
    return std::chrono::duration<double>(Clock::now() - start).count();
  }
  void require(bool condition, const std::string &message)
  {
    if (!condition)
      throw std::invalid_argument(message);
  }
  void lgb_check(int status)
  {
    if (status != 0)
      throw std::runtime_error(LGBM_GetLastError());
  }
  struct DatasetDeleter
  {
    void operator()(void *value) const { LGBM_DatasetFree(value); }
  };
  using Dataset = std::unique_ptr<void, DatasetDeleter>;

  struct Options
  {
    std::map<std::string, std::string> values;
    Options(int argc, char **argv)
    {
      const std::set<std::string> allowed = {
          "index", "method", "dataset", "dataset-dir-prefix",
          "index-filepath", "save-index", "save-models", "results", "metric", "k",
          "training-query-num", "validation-query-num", "query-num", "calibration-query-num",
          "M", "efConstruction", "efSearch", "nlist", "nprobe", "target-recall", "target-recalls",
          "threads", "training-threads", "n-estimators", "learning-rate", "num-leaves",
          "min-child-samples", "seed", "logging-interval", "logging-intervals", "cutoff",
          "lower-quantile", "upper-quantile", "coverage", "ipi-divisor", "mpi-divisor",
          "initial-prediction-interval", "min-prediction-interval", "evaluate"};
      for (int i = 1; i < argc; ++i)
      {
        std::string name = argv[i];
        if (name == "--help")
        {
          help();
          std::exit(0);
        }
        require(name.rfind("--", 0) == 0, "Expected a --name value option: " + name);
        name = name.substr(2);
        require(allowed.count(name), "Unknown option: --" + name);
        if (name == "evaluate")
        {
          require(values.emplace(name, "1").second, "Repeated --evaluate");
          continue;
        }
        require(i + 1 < argc, "Missing value for --" + name);
        require(values.emplace(name, argv[++i]).second, "Repeated option: --" + name);
      }
    }
    std::string get(const std::string &name, const std::string &fallback = "") const
    {
      auto value = values.find(name);
      return value == values.end() ? fallback : value->second;
    }
    bool has(const std::string &name) const { return values.count(name); }
    template <class T>
    T number(const std::string &name, T fallback) const
    {
      return has(name) ? parse_numeric_option<T>(get(name).c_str(), ("--" + name).c_str()) : fallback;
    }
    static void help()
    {
      std::cout << R"(Usage: darth_pipeline --dataset NAME --dataset-dir-prefix DIR [options]

  --dataset NAME --dataset-dir-prefix DIR  use VectorDataLoader dataset mappings
  --index hnsw|ivf                default hnsw
  --method darth|darth-plus|darth-plus-lb  default darth-plus
  --index-filepath PATH             load if present; otherwise build and save
  --metric L2|inner-product         default L2
  --k N --target-recall R           default k 10; choose one target
  --target-recalls R,R,...          default 0.8,0.85,0.90,0.95,0.99
  --evaluate                       report search recall using mapped ground truth,
                                   or compute exact truth if its file is absent
  --training-query-num N           default 10000
  --validation-query-num N         default 1000
  --query-num N                    default 1000
  --calibration-query-num N        default all rows; 0 means all for each count
  --threads N --training-threads N  generation/search and LightGBM workers
  --M N --efConstruction N --efSearch N  defaults 32, 500, 500
  --nlist N --nprobe N              IVF defaults 1000, 100
  --n-estimators N --learning-rate R --num-leaves N --min-child-samples N
                                   defaults 100, 0.1, 31, 20; --seed defaults 42
  --logging-interval N             classic trace sampling, default 10
  --logging-intervals LOW,MID,HIGH  plus defaults HNSW 20,10,5; IVF 100,50,20
  --cutoff PERCENT                 plus high-recall cutoff, default 30
  --lower-quantile Q               LB default 0.1; optional --upper-quantile Q
  --coverage C                     optional conformal calibration using the
                                   dataset's separate calibration queries
  --ipi-divisor X --mpi-divisor X  defaults 2 and 10 (classic IVF: 8 and 40)
  --initial-prediction-interval N --min-prediction-interval N  optional overrides
  --save-models PREFIX             optional final model export (.regression.txt etc.)
  --save-index PATH                optional final index export
  --results PREFIX                optional final neighbors (.ivecs and .fvecs)

Training features retain legacy six-decimal rounding. No intermediate CSV or
model files are created. MSE, MAE and R^2 are reported on training and separate
validation observations. IPI/MPI come only from training queries. Validation and
search queries never enter model fitting. Input vectors use FAISS fvecs format.
)";
    }
  };
  struct Vectors
  {
    size_t d = 0, n = 0;
    std::unique_ptr<float[]> values;
    Vectors(VectorDataLoader &loader, bool database, query_type_t type = TRAINING, size_t limit = 0)
    {
      const auto &path = database ? loader.basePath() : loader.queryPath(type);
      values.reset(database ? loader.loadDB(&d, &n, limit) : loader.loadQueries(type, &d, &n, limit));
      require(n && d && (!limit || n == limit), "Vector file has too few rows: " + path);
      require(n <= size_t(std::numeric_limits<idx_t>::max()), "Too many vectors");
    }
  };
  struct Queries
  {
    Vectors vectors;
    std::vector<idx_t> truth;
    std::vector<float> truth_distances;
    Queries(VectorDataLoader &loader, query_type_t type, size_t limit,
            const Vectors &base, int k, faiss::MetricType metric, int threads, bool need_truth = true)
        : vectors(loader, false, type, limit)
    {
      require(vectors.d == base.d, "Query and base dimensions differ: " + loader.queryPath(type));
      if (!need_truth)
        return;
      const auto &gt_path = loader.groundtruthPath(type);
      truth.resize(vectors.n * k);
      truth_distances.resize(vectors.n * k);
      if (!std::filesystem::is_regular_file(gt_path))
      {
        std::cout << "Computing exact ground truth for " << vectors.n << " queries\n";
        // Borrow the database: avoid a second full database copy in IndexFlat.
        if (metric == faiss::METRIC_L2)
          faiss::knn_L2sqr(vectors.values.get(), base.values.get(), base.d, vectors.n,
                           base.n, k, truth_distances.data(), truth.data());
        else
          faiss::knn_inner_product(vectors.values.get(), base.values.get(), base.d,
                                   vectors.n, base.n, k, truth_distances.data(), truth.data());
      }
      else
      {
        size_t width, rows;
        std::unique_ptr<int[]> ids(loader.loadQueriesGroundtruths(type, &width, &rows, limit));
        require(width >= size_t(k) && rows >= vectors.n && (limit || rows == vectors.n),
                "Ground-truth shape does not match queries/k: " + gt_path);
        for (size_t i = 0; i < vectors.n; ++i)
        {
          std::set<int> unique;
          for (int j = 0; j < k; ++j)
          {
            auto id = ids[i * width + j];
            require(id >= 0 && size_t(id) < base.n && unique.insert(id).second,
                    "Ground truth contains an invalid or repeated neighbor ID");
            truth[i * k + j] = id;
          }
        }
#pragma omp parallel for num_threads(threads)
        for (size_t i = 0; i < vectors.n; ++i)
          for (int j = 0; j < k; ++j)
          {
            const auto *q = vectors.values.get() + i * base.d;
            const auto *b = base.values.get() + truth[i * k + j] * base.d;
            truth_distances[i * k + j] = metric == faiss::METRIC_L2
                                             ? faiss::fvec_L2sqr(q, b, base.d)
                                             : faiss::fvec_inner_product(q, b, base.d);
          }
      }
    }
  };
  struct Matrix
  {
    std::vector<std::string> names;
    std::vector<double> x, y;
    std::vector<size_t> qids, distances;
    size_t queries = 0;
    size_t rows() const { return y.size(); }
  };
  struct Configuration
  {
    bool ivf, plus, lb;
    int k, threads, train_threads, logging;
    double target, cutoff;
    std::vector<faiss::AdaptiveLogInterval> intervals;
    Configuration(const Options &options)
        : ivf(options.get("index", "hnsw") == "ivf"),
          plus(options.get("method", "darth-plus") != "darth"),
          lb(options.get("method", "darth-plus") == "darth-plus-lb"),
          k(options.number<int>("k", 10)), threads(options.number<int>("threads", omp_get_max_threads())),
          train_threads(options.number<int>("training-threads", threads)),
          logging(options.number<int>("logging-interval", 10)),
          target(options.number<double>("target-recall", .8)), cutoff(options.number<double>("cutoff", 30))
    {
      require(options.get("index", "hnsw") == "hnsw" || ivf, "index must be hnsw or ivf");
      const auto method = options.get("method", "darth-plus");
      require(method == "darth" || method == "darth-plus" || lb, "Unknown method");
      require(k > 0 && threads > 0 && train_threads > 0 && logging > 0, "k, threads and logging interval must be positive");
      require(target > 0 && target <= 1 && cutoff >= 0, "Invalid target recall or cutoff");
      std::istringstream values(options.get("logging-intervals", ivf ? "100,50,20" : "20,10,5"));
      double lower[] = {0, .5, .7}, upper[] = {.5, .7, 1};
      for (int i = 0; i < 3; ++i)
      {
        std::string value;
        require(bool(std::getline(values, value, ',')), "Expected three logging intervals");
        int interval = parse_numeric_option<int>(value.c_str(), "--logging-intervals");
        require(interval > 0, "Logging intervals must be positive");
        intervals.push_back({lower[i], upper[i], interval});
      }
      require(values.eof(), "Expected exactly three logging intervals");
    }
  };
  std::vector<double> recall_targets(const Options &options, const Configuration &config)
  {
    require(!(options.has("target-recall") && options.has("target-recalls")), "Use either --target-recall or --target-recalls");
    std::vector<double> requested_targets;
    if (options.has("target-recalls"))
    {
      std::istringstream input(options.get("target-recalls"));
      std::string value;
      while (std::getline(input, value, ','))
      {
        double target = parse_numeric_option<double>(value.c_str(), "--target-recalls");
        require(target > 0 && target <= 1, "Invalid target recall");
        require(std::find(requested_targets.begin(), requested_targets.end(), target) == requested_targets.end(), "Duplicate target recall");
        requested_targets.push_back(target);
      }
      require(!requested_targets.empty() && options.get("target-recalls").back() != ',', "Expected comma-separated target recalls");
    }
    else if (options.has("target-recall"))
      requested_targets.push_back(config.target);
    else
      requested_targets = {0.8, 0.85, 0.90, 0.95, 0.99};
    return requested_targets;
  }
  std::unique_ptr<faiss::Index> make_index(const Options &options, const Configuration &config,
                                           const Vectors &base, faiss::MetricType metric)
  {
    auto start = Clock::now();
    std::unique_ptr<faiss::Index> index;
    const bool cached = options.has("index-filepath") && index_file_exists(options.get("index-filepath").c_str());
    if (cached)
    {
      index.reset(faiss::read_index(options.get("index-filepath").c_str()));
      std::cout << "Loaded index: " << options.get("index-filepath") << '\n';
    }
    else if (!config.ivf)
    {
      int m = options.number<int>("M", 32), ef = options.number<int>("efConstruction", 500);
      require(m > 0 && ef > 0, "M and efConstruction must be positive");
      auto hnsw = std::make_unique<faiss::IndexHNSWFlat>(base.d, m, metric);
      hnsw->hnsw.efConstruction = ef;
      hnsw->add(base.n, base.values.get());
      index = std::move(hnsw);
    }
    else
    {
      int nlist = options.number<int>("nlist", 1000);
      require(nlist > 0 && size_t(nlist) <= base.n / 10, "IVF nlist exceeds the 10% index-training sample");
      auto quantizer = std::make_unique<faiss::IndexFlat>(base.d, metric);
      auto ivf = std::make_unique<faiss::IndexIVFFlat>(quantizer.get(), base.d, nlist, metric);
      ivf->own_fields = true;
      quantizer.release();
      std::vector<float> sample((base.n / 10) * base.d);
      for (size_t i = 0; i < base.n / 10; ++i)
        std::copy_n(base.values.get() + 10 * i * base.d, base.d, sample.data() + i * base.d);
      ivf->train(base.n / 10, sample.data());
      ivf->add(base.n, base.values.get());
      index = std::move(ivf);
    }
    require(index->d == base.d && index->ntotal == base.n && index->metric_type == metric && index->is_trained,
            "Index dimensions, size, metric or training state do not match the base vectors");
    if (config.ivf)
    {
      auto *ivf = dynamic_cast<faiss::IndexIVFFlat *>(index.get());
      require(ivf != nullptr, "Expected an IndexIVFFlat");
      require(!ivf->invlists->use_iterator, "The native pipeline requires array-backed IVF lists");
      if (cached && options.has("nlist"))
        require(ivf->nlist == size_t(options.number<int>("nlist", 0)), "Cached index nlist differs from --nlist");
      auto probes = options.number<int>("nprobe", 100);
      require(probes > 0, "nprobe must be positive");
      ivf->nprobe = std::min<size_t>(probes, ivf->nlist);
      ivf->parallel_mode = 0;
    }
    else
    {
      auto *hnsw = dynamic_cast<faiss::IndexHNSWFlat *>(index.get());
      require(hnsw != nullptr, "Expected an IndexHNSWFlat");
      if (cached && options.has("M"))
        require(hnsw->hnsw.nb_neighbors(1) == options.number<int>("M", 0), "Cached index M differs from --M");
      if (cached && options.has("efConstruction"))
        require(hnsw->hnsw.efConstruction == options.number<int>("efConstruction", 0),
                "Cached index efConstruction differs from --efConstruction");
      hnsw->hnsw.efSearch = options.number<int>("efSearch", 500);
      require(hnsw->hnsw.efSearch > 0, "efSearch must be positive");
    }
    if (cached)
      validate_cached_vectors(index.get(), base.values.get(), base.n, base.d);
    if (!cached && options.has("index-filepath"))
    {
      save_new_index(index.get(), options.get("index-filepath").c_str());
      std::cout << "Saved new index: " << options.get("index-filepath") << '\n';
    }
    if (options.has("save-index") && (!options.has("index-filepath") || options.get("save-index") != options.get("index-filepath") || cached))
    {
      prepare_index_parent(options.get("save-index").c_str());
      faiss::write_index(index.get(), options.get("save-index").c_str());
    }
    std::cout << "INDEX seconds=" << seconds(start) << '\n';
    return index;
  }
  // Keep only the rows used by the selected method and release each query's
  // observations as soon as they have been copied into the training matrix.
  Matrix make_training_matrix(faiss::DarthTrainingData &observations, const Configuration &config)
  {
    Matrix out;
    out.names = observations.feature_names();
    out.queries = observations.queries.size();
    size_t rows = 0;
    for (const auto &query : observations.queries)
      for (const auto &row : query)
        if (config.plus || row.distances % config.logging == 0)
          ++rows;
    require(rows && rows <= size_t(std::numeric_limits<int32_t>::max()), "No usable observations or too many rows for LightGBM");
    out.x.reserve(rows * out.names.size());
    out.y.reserve(rows);
    out.qids.reserve(rows);
    out.distances.reserve(rows);
    for (size_t qid = 0; qid < observations.queries.size(); ++qid)
    {
      const auto before = out.rows();
      for (const auto &row : observations.queries[qid])
      {
        // Original classic trainers retained only distances divisible by li.
        if (!config.plus && row.distances % config.logging != 0)
          continue;
        require(row.features.size() == out.names.size(), "Training feature width mismatch");
        out.x.insert(out.x.end(), row.features.begin(), row.features.end());
        out.y.push_back(row.recall);
        out.qids.push_back(qid);
        out.distances.push_back(row.distances);
      }
      require(out.rows() > before, "A query produced no usable training observations; lower logging interval or increase search budget");
      std::vector<faiss::DarthTrainingRow>().swap(observations.queries[qid]);
    }
    return out;
  }
  Matrix generate(faiss::Index *index, const Vectors &base, Queries &queries,
                  const Configuration &config, const char *split)
  {
    auto start = Clock::now();
    omp_set_num_threads(config.threads);
    const auto n = queries.vectors.n;
    std::vector<float> distances(n * config.k);
    std::vector<idx_t> ids(n * config.k);
    faiss::DeclarativeRecallDataManager manager(distances.data(), ids.data(), queries.truth.data(),
                                                queries.truth_distances.data(), n, base.d, config.k, queries.vectors.values.get(), nullptr,
                                                base.values.get(), base.n, queries.truth.data(), config.k);
    if (config.plus)
      manager.precompute_query_dimension_features();
    faiss::DarthTrainingData observations(n, config.ivf, config.plus);
    if (config.ivf)
    {
      auto *ivf = static_cast<faiss::IndexIVFFlat *>(index);
      std::unique_ptr<faiss::DeclarativeRecallDataCollectorIVF> collector;
      if (config.plus)
        collector = std::make_unique<faiss::DeclarativeRecallDataCollectorIVF>(&manager, config.intervals, config.cutoff);
      else
        collector = std::make_unique<faiss::DeclarativeRecallDataCollectorIVF>(&manager, config.logging);
      collector->training_data = &observations;
      if (config.plus)
        ivf->darth_plus_generate_training_data(n, queries.vectors.values.get(), config.k, distances.data(), ids.data(), collector.get());
      else
        ivf->search_generate_training_data(n, queries.vectors.values.get(), config.k, distances.data(), ids.data(), collector.get());
    }
    else
    {
      auto *hnsw = static_cast<faiss::IndexHNSWFlat *>(index);
      std::unique_ptr<faiss::DeclarativeRecallDataCollectorHNSW> collector;
      if (config.plus)
        collector = std::make_unique<faiss::DeclarativeRecallDataCollectorHNSW>(&manager, config.intervals, config.cutoff);
      else
        collector = std::make_unique<faiss::DeclarativeRecallDataCollectorHNSW>(&manager, config.logging);
      collector->training_data = &observations;
      if (config.plus)
        hnsw->search_darth_plus_data_generation(n, queries.vectors.values.get(), config.k, distances.data(), ids.data(), collector.get());
      else
        hnsw->search_declarative_recall_data_generation(n, queries.vectors.values.get(), config.k, distances.data(), ids.data(), collector.get());
    }
    Matrix out = make_training_matrix(observations, config);
    std::cout << "GENERATE split=" << split << " queries=" << n << " rows=" << out.rows()
              << " features=" << out.names.size() << " seconds=" << seconds(start) << '\n';
    return out;
  }
  std::string training_parameters(const Options &options, const Configuration &config, bool quantile, double alpha)
  {
    int leaves = options.number<int>("num-leaves", 31), minimum = options.number<int>("min-child-samples", 20);
    double rate = options.number<double>("learning-rate", .1);
    require(leaves > 1 && minimum > 0 && rate > 0, "Invalid LightGBM training parameters");
    std::ostringstream p;
    p << std::setprecision(17)
      << "objective=" << (quantile ? "quantile" : "regression")
      << " verbosity=-1 seed=" << options.number<int>("seed", 42)
      << " num_threads=" << config.train_threads << " learning_rate=" << rate
      << " num_leaves=" << leaves << " min_data_in_leaf=" << minimum
      << " max_depth=-1 min_sum_hessian_in_leaf=0.001 min_gain_to_split=0"
      << " feature_fraction=1 bagging_fraction=1 bagging_freq=0 lambda_l1=0 lambda_l2=0"
      << " bin_construct_sample_cnt=200000";
    if (quantile)
      p << " alpha=" << alpha;
    return p.str();
  }
  Model train(const Matrix &data, const Options &options, const Configuration &config,
              bool quantile, double alpha, const std::string &label)
  {
    auto start = Clock::now();
    const auto parameters = training_parameters(options, config, quantile, alpha);
    int rounds = options.number<int>("n-estimators", 100);
    require(rounds > 0, "n-estimators must be positive");
    DatasetHandle raw_data = nullptr;
    lgb_check(LGBM_DatasetCreateFromMat(data.x.data(), C_API_DTYPE_FLOAT64, data.rows(), data.names.size(), 1, parameters.c_str(), nullptr, &raw_data));
    Dataset dataset(raw_data);
    std::vector<const char *> names;
    for (const auto &name : data.names)
      names.push_back(name.c_str());
    lgb_check(LGBM_DatasetSetFeatureNames(dataset.get(), names.data(), names.size()));
    std::vector<float> labels(data.y.begin(), data.y.end());
    lgb_check(LGBM_DatasetSetField(dataset.get(), "label", labels.data(), labels.size(), C_API_DTYPE_FLOAT32));
    BoosterHandle raw_model = nullptr;
    lgb_check(LGBM_BoosterCreate(dataset.get(), parameters.c_str(), &raw_model));
    Model model(raw_model);
    int finished = 0;
    for (int i = 0; i < rounds && !finished; ++i)
      lgb_check(LGBM_BoosterUpdateOneIter(model.get(), &finished));
    // Match LightGBM Python's default post-fit conversion: detach inference
    // trees from the training Dataset using RAM only. The original booster is
    // destroyed before its Dataset; deployment keeps neither training object.
    std::vector<char> model_text(65536);
    int64_t required = 0;
    lgb_check(LGBM_BoosterSaveModelToString(model.get(), 0, -1, 0, model_text.size(), &required, model_text.data()));
    if (required > int64_t(model_text.size()))
    {
      model_text.resize(required);
      lgb_check(LGBM_BoosterSaveModelToString(model.get(), 0, -1, 0, model_text.size(), &required, model_text.data()));
    }
    BoosterHandle inference_handle = nullptr;
    int iterations = 0;
    lgb_check(LGBM_BoosterLoadModelFromString(model_text.data(), &iterations, &inference_handle));
    Model inference(inference_handle);
    std::cout << "TRAIN model=" << label << " seconds=" << seconds(start) << '\n';
    if (options.has("save-models"))
    {
      auto path = std::filesystem::path(options.get("save-models") + "." + label + ".txt");
      if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());
      lgb_check(LGBM_BoosterSaveModel(inference.get(), 0, -1, 0, path.string().c_str()));
    }
    return inference;
  }
  std::vector<double> predict(BoosterHandle model, const Matrix &data, int threads)
  {
    std::vector<double> result(data.rows());
    int64_t length = 0;
    lgb_check(LGBM_BoosterPredictForMat(model, data.x.data(), C_API_DTYPE_FLOAT64, data.rows(), data.names.size(), 1,
                                        C_API_PREDICT_NORMAL, 0, -1, ("num_threads=" + std::to_string(threads)).c_str(), &length, result.data()));
    require(length == int64_t(result.size()), "LightGBM returned an unexpected prediction count");
    return result;
  }
  void metrics(const Matrix &data, const std::vector<double> &predicted, const char *split, const std::string &model)
  {
    double mean = std::accumulate(data.y.begin(), data.y.end(), 0.0) / data.rows(), sse = 0, mae = 0, sst = 0;
    for (size_t i = 0; i < data.rows(); ++i)
    {
      double error = predicted[i] - data.y[i];
      sse += error * error;
      mae += std::abs(error);
      double delta = data.y[i] - mean;
      sst += delta * delta;
    }
    double r2;
    if (data.rows() < 2)
      r2 = std::numeric_limits<double>::quiet_NaN();
    else if (sst > 0)
      r2 = 1 - sse / sst;
    else
      r2 = sse == 0 ? 1 : 0;
    std::cout << "METRICS split=" << split << " model=" << model << " rows=" << data.rows()
              << " mse=" << sse / data.rows() << " mae=" << mae / data.rows() << " r2=" << r2 << '\n';
  }
  Model train_and_report(const Matrix &training, const Matrix &validation,
                         const Options &options, const Configuration &config,
                         bool quantile, double alpha, const std::string &label)
  {
    auto model = train(training, options, config, quantile, alpha, label);
    metrics(training, predict(model.get(), training, config.train_threads), "training", label);
    metrics(validation, predict(model.get(), validation, config.train_threads), "validation", label);
    return model;
  }
  struct Intervals
  {
    int initial, minimum;
  };
  Intervals intervals(const Matrix &data, const Options &options, const Configuration &config)
  {
    std::vector<size_t> first(data.queries, std::numeric_limits<size_t>::max());
    for (size_t i = 0; i < data.rows(); ++i)
      if (data.y[i] >= config.target)
        first[data.qids[i]] = std::min(first[data.qids[i]], data.distances[i]);
    double total = 0;
    size_t count = 0;
    for (auto d : first)
      if (d != std::numeric_limits<size_t>::max())
      {
        total += d;
        ++count;
      }
    require(count > 0, "No training query reaches target recall; increase efSearch/nprobe or lower target recall");
    const double average = total / count;
    const double ipi_div = options.number<double>("ipi-divisor", config.ivf && !config.plus ? 8 : 2);
    const double mpi_div = options.number<double>("mpi-divisor", config.ivf && !config.plus ? 40 : 10);
    require(ipi_div > 0 && mpi_div > 0, "Interval divisors must be positive");
    auto rounded = [&](double value)
    {
      // nearbyint uses ties-to-even, matching Python round / {:.0f}.
      value = std::nearbyint(value);
      if (config.ivf && !config.plus)
        value = config.logging * std::nearbyint(value / config.logging);
      require(value <= std::numeric_limits<int>::max(), "Prediction interval exceeds int range");
      return std::max(1, int(value));
    };
    Intervals pi{options.number<int>("initial-prediction-interval", rounded(average / ipi_div)),
                 options.number<int>("min-prediction-interval", rounded(average / mpi_div))};
    faiss::validate_darth_intervals(config.target, pi.initial, pi.minimum);
    std::cout << "INTERVALS target=" << config.target << " queries_reaching_target=" << count
              << " average_dists=" << average << " ipi=" << pi.initial << " mpi=" << pi.minimum << '\n';
    return pi;
  }
  std::pair<double, double> calibrate(const Matrix &data, BoosterHandle regression,
                                      BoosterHandle lower, BoosterHandle upper, double coverage, int threads)
  {
    auto lo = predict(lower ? lower : regression, data, threads), hi = predict(upper ? upper : regression, data, threads);
    for (size_t i = 0; i < data.rows(); ++i)
    {
      if (lower)
        lo[i] = std::clamp(lo[i], 0.0, 1.0);
      if (upper)
        hi[i] = std::clamp(hi[i], 0.0, 1.0);
      lo[i] = std::max(0.0, lo[i] - data.y[i]);
      hi[i] = std::max(0.0, data.y[i] - hi[i]);
    }
    size_t n = data.rows();
    size_t rank = lower ? std::min(n - 1, size_t(std::ceil((n + 1) * coverage) - 1))
                        : size_t(std::ceil(std::min(1.0, std::ceil((n + 1) * coverage) / n) * (n - 1)));
    std::nth_element(lo.begin(), lo.begin() + rank, lo.end());
    std::nth_element(hi.begin(), hi.begin() + rank, hi.end());
    std::cout << "CALIBRATION coverage=" << coverage << " rows=" << n << " lower_hat=" << lo[rank] << " upper_hat=" << hi[rank] << '\n';
    return {lo[rank], hi[rank]};
  }
  void export_results(const std::string &prefix, const std::vector<idx_t> &ids, const std::vector<float> &distances, int k)
  {
    std::ofstream labels(prefix + ".ivecs", std::ios::binary), values(prefix + ".fvecs", std::ios::binary);
    require(bool(labels) && bool(values), "Cannot open result files");
    for (size_t i = 0; i < ids.size(); i += k)
    {
      labels.write(reinterpret_cast<const char *>(&k), 4);
      values.write(reinterpret_cast<const char *>(&k), 4);
      for (int j = 0; j < k; ++j)
      {
        require(ids[i + j] <= std::numeric_limits<int32_t>::max(), "Result ID exceeds ivecs range");
        int32_t id = ids[i + j];
        labels.write(reinterpret_cast<const char *>(&id), 4);
      }
      values.write(reinterpret_cast<const char *>(distances.data() + i), 4 * k);
    }
    require(bool(labels) && bool(values), "Failed writing results");
  }
  struct SearchWorkspace
  {
    std::vector<float> distances;
    std::vector<idx_t> ids;
    faiss::DeclarativeRecallDataManager manager;
    SearchWorkspace(const Vectors &base, Queries &queries, const Configuration &config)
        : distances(queries.vectors.n * config.k), ids(queries.vectors.n * config.k),
          manager(distances.data(), ids.data(), queries.truth.data(), queries.truth_distances.data(),
                  queries.vectors.n, base.d, config.k, queries.vectors.values.get(), nullptr,
                  base.values.get(), base.n, queries.truth.data(), config.k)
    {
      if (config.plus)
        manager.precompute_query_dimension_features();
    }
  };
  void search(faiss::Index *index, Queries &queries, const Configuration &config, SearchWorkspace &workspace,
              const std::string &results, Intervals pi, std::unique_ptr<faiss::LGBMPredictor> &reg,
              std::unique_ptr<faiss::ConformalLGBMPredictor> &conformal)
  {
    omp_set_num_threads(config.threads);
    const auto n = queries.vectors.n;
    auto &D = workspace.distances;
    auto &I = workspace.ids;
    auto &manager = workspace.manager;
    auto start = Clock::now();
    if (config.ivf)
    {
      auto *ivf = static_cast<faiss::IndexIVFFlat *>(index);
      if (config.plus)
      {
        faiss::DARTHPlusPredictorIVF model(&manager, config.target, pi.initial, pi.minimum, false, std::move(conformal));
        ivf->search_DARTH_plus(n, queries.vectors.values.get(), config.k, D.data(), I.data(), &model);
        conformal = std::move(model.conformal_predictor);
      }
      else
      {
        faiss::DARTHPredictorIVF model(&manager, config.target, pi.initial, pi.minimum, false, std::move(reg), config.logging);
        ivf->search_DARTH(n, queries.vectors.values.get(), config.k, D.data(), I.data(), &model);
        reg = std::move(model.predictor);
      }
    }
    else
    {
      auto *hnsw = static_cast<faiss::IndexHNSWFlat *>(index);
      if (config.plus)
      {
        faiss::DARTHPlusPredictorHNSW model(&manager, config.target, pi.initial, pi.minimum, false, std::move(conformal));
        hnsw->search_DARTH_plus(n, queries.vectors.values.get(), config.k, D.data(), I.data(), &model);
        conformal = std::move(model.conformal_predictor);
      }
      else
      {
        faiss::DARTHPredictorHNSW model(&manager, config.target, pi.initial, pi.minimum, false, std::move(reg));
        hnsw->search_DARTH(n, queries.vectors.values.get(), config.k, D.data(), I.data(), &model);
        reg = std::move(model.predictor);
      }
    }
    double elapsed = seconds(start);
    std::cout << "SEARCH target=" << config.target << " queries=" << n;
    if (!queries.truth.empty())
    {
      double recall = 0;
      size_t below_target = 0;
      for (size_t i = 0; i < n; ++i)
      {
        const auto truth_begin = queries.truth.begin() + i * config.k;
        const std::unordered_set<idx_t> truth(truth_begin, truth_begin + config.k);
        int matches = 0;
        for (int j = 0; j < config.k; ++j)
          matches += truth.count(I[i * config.k + j]) != 0;
        const double r = double(matches) / config.k;
        recall += r;
        below_target += r < config.target;
      }
      std::cout << " recall=" << recall / n << " rqut=" << double(below_target) / n
                << " queries_below_target=" << below_target;
    }
    std::cout << " seconds=" << elapsed << " qps=" << n / elapsed << '\n';
    if (!results.empty())
      export_results(results, I, D, config.k);
  }
} // namespace

int main(int argc, char **argv)
try
{
  std::setlocale(LC_NUMERIC, "C");
  std::cout << std::setprecision(12);
  Options options(argc, argv);
  Configuration config(options);
  omp_set_num_threads(config.threads);
  require(options.has("dataset"), "Required option: --dataset");
  const char *env_dir = std::getenv("DATASET_DIRECTORY");
  if (!env_dir || !*env_dir)
    env_dir = std::getenv("DARTH_DATASET_DIR");
  const std::string dataset_dir = options.get("dataset-dir-prefix", env_dir ? env_dir : "");
  require(!dataset_dir.empty(), "Specify --dataset-dir-prefix, DATASET_DIRECTORY or DARTH_DATASET_DIR");
  VectorDataLoader loader(options.get("dataset"), TRAINING, "5", dataset_dir, options.get("metric", "L2"));
  loader.initializeDataMaps();
  loader.basePath(); // Validate the dataset before checking its query mappings.
  loader.validateDistinctQueryFiles({TRAINING, VALIDATION, TESTING});
  std::string metric_name = options.get("metric", "L2");
  require(metric_name == "L2" || metric_name == "inner-product", "Unknown metric");
  auto metric = metric_name == "L2" ? faiss::METRIC_L2 : faiss::METRIC_INNER_PRODUCT;
  double coverage = options.number<double>("coverage", 0);
  double lower_quantile = options.number<double>("lower-quantile", .1);
  double upper_quantile = options.number<double>("upper-quantile", 0);
  require(coverage >= 0 && coverage < 1, "coverage must be in [0,1)");
  require(!coverage || config.plus, "Conformal calibration requires a plus method");
  require(!config.lb || (lower_quantile > 0 && lower_quantile < 1), "Lower quantile must be in (0,1)");
  require(!upper_quantile || (config.lb && upper_quantile > lower_quantile && upper_quantile < 1), "Upper quantile must exceed the lower quantile and be below 1");
  if (coverage)
    loader.validateDistinctQueryFiles({TRAINING, VALIDATION, TESTING, CALIBRATION});
  auto requested_targets = recall_targets(options, config);

  // Load the database and collect training observations once for all targets.
  Vectors base(loader, true);
  require(base.n >= size_t(config.k), "k exceeds database size");
  auto index = make_index(options, config, base, metric);
  auto load = [&](query_type_t type, const std::string &count, size_t default_count, bool need_truth = true)
  {
    return Queries(loader, type, options.number<size_t>(count, default_count), base, config.k, metric, config.threads, need_truth);
  };
  auto training_data = [&]
  {
    auto training = load(TRAINING, "training-query-num", 10000);
    return generate(index.get(), base, training, config, "training");
  }();
  // Prediction intervals depend on the target, but the trained models do not.
  std::vector<std::pair<double, Intervals>> targets;
  bool target_failed = false;
  for (double target : requested_targets)
  {
    config.target = target;
    try
    {
      targets.emplace_back(target, intervals(training_data, options, config));
    }
    catch (const std::exception &error)
    {
      std::cerr << "Target " << target << ": " << error.what() << '\n';
      target_failed = true;
    }
  }
  require(!targets.empty(), "No target could be prepared");

  // Validate and train the regression model and any requested quantile models.
  auto validation_data = [&]
  {
    auto validation = load(VALIDATION, "validation-query-num", 1000);
    return generate(index.get(), base, validation, config, "validation");
  }();
  omp_set_num_threads(config.train_threads);
  auto regression = train_and_report(training_data, validation_data, options, config, false, 0, "regression");
  Model lower, upper;
  if (config.lb)
    lower = train_and_report(training_data, validation_data, options, config, true, lower_quantile, "lower");
  if (upper_quantile)
    upper = train_and_report(training_data, validation_data, options, config, true, upper_quantile, "upper");
  training_data = Matrix{};
  validation_data = Matrix{}; // Release training matrices before deployment.

  // Calibration uses its own query split and is shared by all targets.
  std::pair<double, double> calibration_offsets{0, 0};
  if (coverage)
  {
    auto queries = load(CALIBRATION, "calibration-query-num", 0);
    auto calibration = generate(index.get(), base, queries, config, "calibration");
    calibration_offsets = calibrate(calibration, regression.get(), lower.get(), upper.get(), coverage, config.train_threads);
  }
  omp_set_num_threads(config.threads);
  auto queries = load(TESTING, "query-num", 1000, options.has("evaluate"));
  auto regression_predictor = std::make_unique<faiss::LGBMPredictor>(std::move(regression));
  std::unique_ptr<faiss::ConformalLGBMPredictor> conformal;
  if (config.plus)
  {
    auto lo = lower ? std::make_unique<faiss::LGBMPredictor>(std::move(lower)) : nullptr;
    auto hi = upper ? std::make_unique<faiss::LGBMPredictor>(std::move(upper)) : nullptr;
    conformal = std::make_unique<faiss::ConformalLGBMPredictor>(
        std::move(regression_predictor), std::move(lo), std::move(hi),
        calibration_offsets.first, calibration_offsets.second);
  }
  // Reuse predictors and query features across the target sweep.
  SearchWorkspace workspace(base, queries, config);
  for (const auto &target : targets)
  {
    config.target = target.first;
    std::string result = options.get("results");
    if (!result.empty() && (options.has("target-recalls") || requested_targets.size() > 1))
    {
      std::ostringstream suffix;
      suffix << std::setprecision(12) << config.target;
      result += "-target" + suffix.str();
    }
    search(index.get(), queries, config, workspace, result, target.second, regression_predictor, conformal);
  }
  return target_failed ? 1 : 0;
}
catch (const std::exception &error)
{
  std::cerr << "DARTH pipeline: " << error.what() << '\n';
  return 1;
}
