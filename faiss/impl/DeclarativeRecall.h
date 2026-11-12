#pragma once

#include <LightGBM/boosting.h>
#include <LightGBM/c_api.h>
#include <faiss/Index.h>
#include <faiss/impl/platform_macros.h>
#include <sys/time.h>

#include <algorithm>
#include <functional>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <string>
#include <memory>
#include <mutex>
#include <array>
#include <omp.h>

#define INTERVAL_DISABLED_VALUE -1
#define INTERVAL_WHEN_RECALL_UPDATES_VALUE 0

namespace faiss {

// Validate once at construction, outside the query hot path.
inline void validate_darth_intervals(double target_recall, int initial, int minimum) {
  if (!std::isfinite(target_recall) || target_recall < 0.0 || target_recall > 1.0)
    throw std::invalid_argument("Target recall must be finite and in [0, 1]");
  if (minimum <= 0 || initial < minimum)
    throw std::invalid_argument("Prediction intervals must satisfy initial >= minimum > 0");
}

typedef BoosterHandle LGBMModel;

// Model handles and prediction scratch space have one owner. Fast predictors
// are initialized before entering OpenMP and use separate buffers per worker.
struct LGBMPredictor {
  struct BoosterDeleter {
    void operator()(void* handle) const { LGBM_BoosterFree(handle); }
  };
  struct FastDeleter {
    void operator()(void* handle) const { LGBM_FastConfigFree(handle); }
  };
  using Model = std::unique_ptr<void, BoosterDeleter>;
  Model model;
  std::vector<std::unique_ptr<void, FastDeleter>> workers;
  std::vector<int> feature_order;
  int feature_count = 0;
  int input_feature_count = 0;

  explicit LGBMPredictor(const char* model_path) {
    BoosterHandle handle = nullptr;
    int iterations;
    if (!model_path || LGBM_BoosterCreateFromModelfile(
            model_path, &iterations, &handle) != 0) {
      throw std::runtime_error("Error loading LightGBM model");
    }
    model.reset(handle);
    initialize_workers();
  }

  // Take ownership of a freshly trained native booster, without serializing it.
  explicit LGBMPredictor(Model trained_model) : model(std::move(trained_model)) {
    if (!model) throw std::invalid_argument("A trained model is required");
    initialize_workers();
  }

  void initialize_workers() {
    auto handle = model.get();
    if (LGBM_BoosterGetNumFeature(handle, &feature_count) != 0) {
      throw std::runtime_error(LGBM_GetLastError());
    }
    const int worker_count = omp_get_max_threads();
    workers.reserve(worker_count);
    for (int i = 0; i < worker_count; ++i) {
      FastConfigHandle fast = nullptr;
      if (LGBM_BoosterPredictForMatSingleRowFastInit(
              handle, C_API_PREDICT_NORMAL, 0, -1, C_API_DTYPE_FLOAT64,
              feature_count, "num_threads=1", &fast) != 0) {
        throw std::runtime_error(LGBM_GetLastError());
      }
      workers.emplace_back(fast);
    }
  }

  LGBMPredictor(const LGBMPredictor&) = delete;
  LGBMPredictor& operator=(const LGBMPredictor&) = delete;

  // Bind the inference schema to the feature names stored in each model.
  // This also supports revision models trained with a different column order.
  void set_feature_names(const std::vector<std::string>& names) {
    int count;
    size_t required = 0;
    std::vector<std::vector<char>> buffers(feature_count, std::vector<char>(128));
    std::vector<char*> pointers;
    for (auto& buffer : buffers) pointers.push_back(buffer.data());
    if (LGBM_BoosterGetFeatureNames(model.get(), feature_count, &count,
            128, &required, pointers.data()) != 0 || required > 128) {
      throw std::runtime_error("Cannot read LightGBM feature names");
    }
    std::vector<int> order;
    for (int i = 0; i < count; ++i) {
      auto it = std::find(names.begin(), names.end(), pointers[i]);
      if (it == names.end()) {
        throw std::runtime_error("Unsupported DARTH model feature: " + std::string(pointers[i]));
      }
      order.push_back(static_cast<int>(it - names.begin()));
    }
    input_feature_count = names.size();
    feature_order = std::move(order);
  }

  double predict(const double* features, int num_features);
};

inline std::vector<std::string> darth_feature_names(bool query_dimensions = false) {
  std::vector<std::string> names = {"step", "dists", "inserts", "first_nn_dist",
      "nn_dist", "furthest_dist", "avg_dist", "variance", "percentile_25",
      "percentile_50", "percentile_75"};
  if (query_dimensions) {
    const std::vector<std::string> dims = {"q_dim_avg", "q_dim_std", "q_dim_min",
        "q_dim_max", "q_dim_median", "q_dim_l1", "q_dim_l2", "q_dim_range"};
    names.insert(names.end(), dims.begin(), dims.end());
  }
  return names;
}

typedef struct ConformalLGBMPredictor {
  double low_quantile_hat = 0.0;
  double high_quantile_hat = 0.0;
  std::unique_ptr<LGBMPredictor> low_quantile_model;
  std::unique_ptr<LGBMPredictor> high_quantile_model;
  std::unique_ptr<LGBMPredictor> regression_model;

  ConformalLGBMPredictor(
      std::unique_ptr<LGBMPredictor> regression,
      std::unique_ptr<LGBMPredictor> lower = nullptr,
      std::unique_ptr<LGBMPredictor> upper = nullptr,
      double lower_hat = 0.0, double upper_hat = 0.0)
      : low_quantile_hat(lower_hat), high_quantile_hat(upper_hat),
        low_quantile_model(std::move(lower)), high_quantile_model(std::move(upper)),
        regression_model(std::move(regression)) {
    if (!regression_model) throw std::invalid_argument("A regression model is required");
  }

  ConformalLGBMPredictor(
      char* low_quantile_model_path,
      char* high_quantile_model_path,
      char* regression_model_path,
      double low_quantile_hat,
      double high_quantile_hat) {
    if (!regression_model_path)
      throw std::invalid_argument("Regression model path is required");
    if (!low_quantile_model_path)
      throw std::invalid_argument("Lower quantile model path is required");
    printf("Loading ConformalLGBMPredictor with quantile & regression models\n");

    this->low_quantile_hat = low_quantile_hat;
    this->high_quantile_hat = high_quantile_hat;
    if (low_quantile_model_path)
      low_quantile_model = std::make_unique<LGBMPredictor>(low_quantile_model_path);

    if (high_quantile_model_path)
      high_quantile_model = std::make_unique<LGBMPredictor>(high_quantile_model_path);

    if (regression_model_path)
      regression_model = std::make_unique<LGBMPredictor>(regression_model_path);
  }

  ConformalLGBMPredictor(
      char* regression_model_path,
      double low_quantile_hat,
      double high_quantile_hat) {
    this->low_quantile_hat = low_quantile_hat;
    this->high_quantile_hat = high_quantile_hat;
    if (regression_model_path)
      regression_model = std::make_unique<LGBMPredictor>(regression_model_path);
    else
      throw std::runtime_error("Regression model path is null");

    printf("Loading ConformalLGBMPredictor with regression model\n");
  }

  void set_feature_names(const std::vector<std::string>& names) {
    if (low_quantile_model) low_quantile_model->set_feature_names(names);
    if (high_quantile_model) high_quantile_model->set_feature_names(names);
    if (regression_model) regression_model->set_feature_names(names);
  }

  void predict_range_with_quantile(const double* features, int num_features, double* lower_bound, double* upper_bound) {
    double low_pred = 0.0;
    if (low_quantile_model)
      low_pred = low_quantile_model->predict(features, num_features);

    double high_pred = 1.0;
    if (high_quantile_model)
      high_pred = high_quantile_model->predict(features, num_features);

    if (!high_quantile_model && !low_quantile_model)
      throw std::runtime_error("Quantile models are both null!");

    *lower_bound = low_pred - low_quantile_hat;
    *upper_bound = high_pred + high_quantile_hat;

    if (*lower_bound < 0.0)
      *lower_bound = 0.0;

    if (*upper_bound > 1.0)
      *upper_bound = 1.0;
  }

  void predict_upper_with_quantile(const double* features, int num_features, double* upper_bound) {
    if (!high_quantile_model)
      throw std::runtime_error("High quantile model is null!");

    double high_pred = high_quantile_model->predict(features, num_features);
    *upper_bound = high_pred + high_quantile_hat;
    if (*upper_bound > 1.0)
      *upper_bound = 1.0;
  }

  void predict_lower_with_quantile(const double* features, int num_features, double* lower_bound) {
    if (!low_quantile_model)
      throw std::runtime_error("Low quantile model is null!");

    double low_pred = low_quantile_model->predict(features, num_features);
    *lower_bound = low_pred - low_quantile_hat;
    if (*lower_bound < 0.0)
      *lower_bound = 0.0;
  }

  void predict_avg_with_regression(const double* feature, int num_features, double* avg_out) {
    if (regression_model)
      *avg_out = regression_model->predict(feature, num_features);
    else
      throw std::runtime_error("Regression model not loaded");
  }

  void predict_range_with_regression(const double* features, int num_features, double* lower_bound, double* upper_bound, double* avg) {
    double pred = 0.0;
    if (regression_model)
      pred = regression_model->predict(features, num_features);
    else
      throw std::runtime_error("Regression model not loaded");

    if (pred < 0.0)
      pred = 0.0;

    if (pred > 1.0)
      pred = 1.0;

    *lower_bound = pred - low_quantile_hat;
    *upper_bound = pred + high_quantile_hat;
    *avg = pred;

    if (*lower_bound < 0.0)
      *lower_bound = 0.0;

    if (*upper_bound > 1.0)
      *upper_bound = 1.0;
  }
} ConformalLGBMPredictor;

// One query is generated by one worker. Separate per-query buffers preserve
// CSV query/observation order independently of OpenMP scheduling.
struct DarthTrainingRow {
  std::vector<double> features;
  double recall;
  size_t distances;
};
struct DarthTrainingData {
  std::vector<std::vector<DarthTrainingRow>> queries;
  bool ivf;
  bool query_dimensions;
  DarthTrainingData(size_t query_count, bool ivf, bool query_dimensions)
      : queries(query_count), ivf(ivf), query_dimensions(query_dimensions) {}
  std::vector<std::string> feature_names() const;
};

typedef struct DeclarativeRecallDataManager {
  idx_t nq = 0, ndb = 0, k = 0, d = 0;

  const float* queries = nullptr;
  const float* db = nullptr;

  char* log_filename = nullptr;

  float* distances = nullptr;
  idx_t* labels = nullptr;

  idx_t* gt = nullptr;
  float* gt_dist = nullptr;

  // including all possible k-nn neighbors and their size (needed for normalized rank sum)
  idx_t* gt_for_all_k = nullptr;
  idx_t k_all = 0;

  // for darth-plus: calculate query stats
  bool use_query_dim_features = false;
  char query_dim_features_names[8][25] = {"q_dim_avg", "q_dim_std", "q_dim_min", "q_dim_max", "q_dim_median", "q_dim_l1", "q_dim_l2", "q_dim_range"};
  int total_query_dim_features = 8;      // features: mean, std, min, max, median, l1, l2, ...
  std::vector<std::array<float, 8>> query_dim_storage;
  std::vector<float*> query_dim_rows;
  float** query_dim_features = nullptr;  // this is meant to keep pre-computed query dimension features.

  DeclarativeRecallDataManager();

  DeclarativeRecallDataManager(
      float* distances,
      idx_t* labels,
      idx_t* gt,
      float* gt_dist,
      idx_t nq,
      idx_t d,
      idx_t k,
      const float* queries,
      char* log_filename,
      const float* db,
      idx_t ndb,
      idx_t* gt_for_all_k,
      idx_t k_all);

  void record_training_observation(DarthTrainingData* output, idx_t query_idx,
      int step, int distances, int insertions, float first_distance,
      float current_centroid_distance, float recall);

  float get_ed(const float* x, const float* y, size_t d);

  double elapsed_secs();

  float get_avg_dist_of_query(idx_t q_id);

  float get_nearest_dist_of_query(idx_t q_id);

  float get_furthest_dist_of_query(idx_t q_id);

  float get_dist_of_query_to_medoid(idx_t q_id);

  float get_recallk(idx_t query_idx);

  float get_kth_nearest_dist_of_query(idx_t q_id, int kth);

  float get_variance_of_query(idx_t q_id);

  float get_percentile_of_query(idx_t q_id, float percentile);

  float get_skewness_of_query(idx_t q_id);

  float get_energy_of_query(idx_t q_id);

  float get_kurtosis_of_query(idx_t q_id);

  float get_TDR(idx_t q_id);

  float get_RDE(idx_t q_id);

  float get_NRS(idx_t q_id);

  void precompute_query_dimension_features() {
    // this is meant to be called externally if we need to use the features. It is "plug-in" for the darth-plus data manager functionality.
    double start_time = elapsed_secs();
    use_query_dim_features = true;
    query_dim_storage.resize(nq);
    query_dim_rows.resize(nq);
    query_dim_features = query_dim_rows.data();

    printf("Allocation with nq=%ld queries\n", nq);

    for (idx_t i = 0; i < nq; i++) {
      query_dim_features[i] = query_dim_storage[i].data();
      // compute features denoted in query_dim_features_names:
      const float* query = queries + i * d;

      // Calculate average
      float sum = 0;
      for (idx_t j = 0; j < d; j++) {
        sum += query[j];
      }
      query_dim_features[i][0] = sum / d;  // avg

      // Calculate std dev
      float variance = 0;
      for (idx_t j = 0; j < d; j++) {
        variance += (query[j] - query_dim_features[i][0]) * (query[j] - query_dim_features[i][0]);
      }
      query_dim_features[i][1] = std::sqrt(variance / d);  // std

      // Calculate min/max
      float min_val = query[0];
      float max_val = query[0];
      for (idx_t j = 1; j < d; j++) {
        if (query[j] < min_val) min_val = query[j];
        if (query[j] > max_val) max_val = query[j];
      }
      query_dim_features[i][2] = min_val;  // min
      query_dim_features[i][3] = max_val;  // max

      // Calculate median (approximate using mean of middle values)
      std::vector<float> sorted_dims(query, query + d);
      std::sort(sorted_dims.begin(), sorted_dims.end());
      query_dim_features[i][4] = d % 2 == 0 ? (sorted_dims[d / 2 - 1] + sorted_dims[d / 2]) / 2.0f : sorted_dims[d / 2];  // median

      // Calculate L1 norm
      float l1_norm = 0;
      for (idx_t j = 0; j < d; j++) {
        l1_norm += std::abs(query[j]);
      }
      query_dim_features[i][5] = l1_norm;  // l1

      // Calculate L2 norm
      float l2_norm = 0;
      for (idx_t j = 0; j < d; j++) {
        l2_norm += query[j] * query[j];
      }
      query_dim_features[i][6] = std::sqrt(l2_norm);  // l2

      // Calculate range
      query_dim_features[i][7] = max_val - min_val;  // range
    }

    double end_time = elapsed_secs();
    printf("Precomputing query dimension features took %.3f seconds\n", end_time - start_time);
  }

  DeclarativeRecallDataManager(const DeclarativeRecallDataManager&) = delete;
  DeclarativeRecallDataManager& operator=(const DeclarativeRecallDataManager&) = delete;

} DeclarativeRecallDataManager;

typedef struct AdaptiveLogInterval {
  double min_recall;
  double max_recall;
  int interval;
} AdaptiveLogInterval;

inline float centroid_distance_at_step(
    const std::vector<float>& centroid_distances,
    int nstep) {
  if (centroid_distances.empty()) {
    throw std::runtime_error("Centroid distances are empty");
  }
  const size_t centroid_index = std::min(
      static_cast<size_t>(std::max(nstep, 0)),
      centroid_distances.size() - 1);
  return centroid_distances[centroid_index];
}

typedef struct DeclarativeRecallDataCollectorHNSW {
  std::function<void(idx_t, int, int, int, float, float)> observation_sink;
  DarthTrainingData* training_data = nullptr;  // borrowed, optional in-memory sink

  DeclarativeRecallDataCollectorHNSW(const DeclarativeRecallDataCollectorHNSW&) = delete;
  DeclarativeRecallDataCollectorHNSW& operator=(const DeclarativeRecallDataCollectorHNSW&) = delete;
  ~DeclarativeRecallDataCollectorHNSW() { close_log_file(); }
  std::mutex log_mutex;

  DeclarativeRecallDataManager* data_manager = nullptr;
  FILE* log_file = nullptr;

  int logging_interval = 1;

  const std::vector<AdaptiveLogInterval> adaptive_logging_intervals;  // this is for darth_plus
  double high_recall_cutoff_percentage = 30.0;  // extra search after recall first reaches 1.0

  bool do_naive_early_stop = false;
  int dist_early_stop_threshold = 1000;

  bool include_data_dimensions = true;

  DeclarativeRecallDataCollectorHNSW();

  DeclarativeRecallDataCollectorHNSW(
      DeclarativeRecallDataManager* data_manager,
      int logging_interval = INTERVAL_DISABLED_VALUE,
      int dist_early_stop_threshold = 0);

  DeclarativeRecallDataCollectorHNSW(  // for DARTH plus
      DeclarativeRecallDataManager* data_manager,
      const std::vector<AdaptiveLogInterval>& adaptive_logging_intervals,
      double high_recall_cutoff_percentage)
      : data_manager(data_manager),
        adaptive_logging_intervals(adaptive_logging_intervals),
        high_recall_cutoff_percentage(high_recall_cutoff_percentage) {
  }

  void init_log_file();

  void close_log_file();

  void append_to_log(
      idx_t query_idx,
      int nstep,
      int ndis,
      int total_insertions,
      double elapsed,
      float first_nn_dis,
      float recall_k);

  std::string get_observation_data_str(
      idx_t query_idx,
      int nstep,
      int ndis,
      int total_insertions,
      double elapsed,
      float first_nn_dis,
      float recall_k);

  void flush_observation_to_log(std::string observation_data);

  void flush_all_observations_to_log(std::string* observations, int n);

  int get_adaptive_interval(float recall) {
    for (const auto& ri : adaptive_logging_intervals) {
      if (recall >= ri.min_recall && recall < ri.max_recall)
        return ri.interval;
    }
    return adaptive_logging_intervals.back().interval;  // default fallback
  }

} DeclarativeRecallDataCollectorHNSW;

typedef struct DARTHPredictorHNSW {
  DARTHPredictorHNSW(const DARTHPredictorHNSW&) = delete;
  DARTHPredictorHNSW& operator=(const DARTHPredictorHNSW&) = delete;
  ~DARTHPredictorHNSW() { close_log_file(); }
  std::mutex log_mutex;

  DeclarativeRecallDataManager* data_manager = nullptr;

  FILE* log_file = nullptr;
  bool per_prediction_logging = false;

  double target_recall;

  int initial_prediction_interval = 2000, min_prediction_interval = 100;

  std::unique_ptr<LGBMPredictor> predictor;

  DARTHPredictorHNSW(
      DeclarativeRecallDataManager* manager, double target, int initial, int minimum,
      bool logging, std::unique_ptr<LGBMPredictor> model)
      : data_manager(manager), target_recall(target), initial_prediction_interval(initial),
        min_prediction_interval(minimum), per_prediction_logging(logging),
        predictor(std::move(model)) {
    validate_darth_intervals(target, initial, minimum);
    if (!predictor) throw std::invalid_argument("A regression model is required");
    predictor->set_feature_names(darth_feature_names());
  }

  DARTHPredictorHNSW(
      DeclarativeRecallDataManager* data_manager,
      double target_recall,
      int initial_prediction_interval,
      int min_prediction_interval,
      bool per_prediction_logging,
      char* predictor_model_path);

  void init_log_file();

  void close_log_file();

  float predict_recall(
      idx_t query_idx,
      int nstep,
      int ndis,
      int total_insertions,
      double elapsed,
      float first_nn_dis,
      int query_predictor_calls,
      int* prediction_interval,
      double* predictor_time);

  void log_final_recall_result(
      idx_t query_idx,
      int nstep,
      int ndis,
      int total_insertions,
      double elapsed,
      float first_nn_dis,
      double last_predicted_recall,
      int prediction_interval,
      int query_predictor_calls,
      double predictor_time);

} DARTHPredictorHNSW;



typedef struct DARTHPlusPredictorHNSW {
  DARTHPlusPredictorHNSW(const DARTHPlusPredictorHNSW&) = delete;
  DARTHPlusPredictorHNSW& operator=(const DARTHPlusPredictorHNSW&) = delete;
  ~DARTHPlusPredictorHNSW() { close_log_file(); }
  std::mutex log_mutex;

  DeclarativeRecallDataManager* data_manager = nullptr;

  FILE* log_file = nullptr;
  bool per_prediction_logging = false;

  double target_recall = 0.90;

  int initial_prediction_interval = 2000, min_prediction_interval = 100;

  enum inference_mode {
    QUANTILE_REGRESSION = 0,
    CLASSIC_REGRESSION = 1
  } inference_mode;
  enum inference_mode inf_mode = QUANTILE_REGRESSION;

  std::unique_ptr<ConformalLGBMPredictor> conformal_predictor;

  DARTHPlusPredictorHNSW(
      DeclarativeRecallDataManager* manager, double target, int initial, int minimum,
      bool logging, std::unique_ptr<ConformalLGBMPredictor> model)
      : data_manager(manager), target_recall(target), initial_prediction_interval(initial),
        min_prediction_interval(minimum), per_prediction_logging(logging),
        conformal_predictor(std::move(model)) {
    validate_darth_intervals(target, initial, minimum);
    if (!conformal_predictor) throw std::invalid_argument("A conformal predictor is required");
    inf_mode = conformal_predictor->low_quantile_model ? QUANTILE_REGRESSION : CLASSIC_REGRESSION;
    conformal_predictor->set_feature_names(darth_feature_names(manager->use_query_dim_features));
  }

  DARTHPlusPredictorHNSW(
      DeclarativeRecallDataManager* data_manager,
      double target_recall,
      int initial_prediction_interval,
      int min_prediction_interval,
      bool per_prediction_logging,
      char* low_quantile_model_path,
      char* high_quantile_model_path,
      char* regression_model_path,
      double low_quantile_hat,
      double high_quantile_hat) {
    validate_darth_intervals(target_recall, initial_prediction_interval, min_prediction_interval);
    conformal_predictor = std::make_unique<ConformalLGBMPredictor>(low_quantile_model_path, high_quantile_model_path, regression_model_path, low_quantile_hat, high_quantile_hat);

    conformal_predictor->set_feature_names(darth_feature_names(data_manager->use_query_dim_features));
    this->data_manager = data_manager;
    this->target_recall = target_recall;
    this->initial_prediction_interval = initial_prediction_interval;
    this->min_prediction_interval = min_prediction_interval;
    this->per_prediction_logging = per_prediction_logging;
    this->inf_mode = QUANTILE_REGRESSION;
  }

  DARTHPlusPredictorHNSW(
      DeclarativeRecallDataManager* data_manager,
      double target_recall,
      int initial_prediction_interval,
      int min_prediction_interval,
      bool per_prediction_logging,
      char* predictor_model_path,
      double low_quantile_hat,
      double high_quantile_hat) {
    validate_darth_intervals(target_recall, initial_prediction_interval, min_prediction_interval);
    conformal_predictor = std::make_unique<ConformalLGBMPredictor>(predictor_model_path, low_quantile_hat, high_quantile_hat);

    conformal_predictor->set_feature_names(darth_feature_names(data_manager->use_query_dim_features));
    this->data_manager = data_manager;
    this->target_recall = target_recall;
    this->initial_prediction_interval = initial_prediction_interval;
    this->min_prediction_interval = min_prediction_interval;
    this->per_prediction_logging = per_prediction_logging;
    this->inf_mode = CLASSIC_REGRESSION;
  }

  void init_log_file() {
    if (!data_manager->log_filename) {
      return;
    }

    close_log_file();
    log_file = fopen(data_manager->log_filename, "w");
    if (!log_file) {
      printf("Error opening recall data log file\n");
      throw std::runtime_error("DARTH log or prediction failure");
    }

    fprintf(log_file, "qid,");

    if (data_manager->use_query_dim_features && data_manager->query_dim_features != nullptr) {
      for (int i = 0; i < data_manager->total_query_dim_features; i++) {
        fprintf(log_file, "%s,", data_manager->query_dim_features_names[i]);
      }
    }

    fprintf(log_file, "step,");
    fprintf(log_file, "dists,");
    fprintf(log_file, "elaps_ms,");
    fprintf(log_file, "inserts,");

    fprintf(log_file, "first_nn_dist,");

    fprintf(log_file, "nn_dist,");
    fprintf(log_file, "avg_dist,");
    fprintf(log_file, "furthest_dist,");

    fprintf(log_file, "variance,");
    fprintf(log_file, "median,");

    fprintf(log_file, "percentile_25,");
    fprintf(log_file, "percentile_75,");

    fprintf(log_file, "r_current_interval,");
    fprintf(log_file, "r_predictor_calls,");
    fprintf(log_file, "r_predictor_time_ms,");

    // Quality approximation measures
    fprintf(log_file, "RDE,");  // relative distance error
    fprintf(log_file, "TDR,");  // total distance ratio
    fprintf(log_file, "NRS,");  // normalized rank sum

    fprintf(log_file, "r_actual,");
    // fprintf(log_file, "r_predicted");
    fprintf(log_file, "r_lower,");
    fprintf(log_file, "r_upper");

    fprintf(log_file, "\n");
  }

  void close_log_file() {
    if (log_file) {
      fclose(log_file);
      log_file = nullptr;
    }
  }

  float predict_recall(
      idx_t query_idx,
      int nstep,
      int ndis,
      int total_insertions,
      double elapsed,
      float first_nn_dis,
      int query_predictor_calls,
      int* prediction_interval,
      double* predictor_time,
      float* lower_bound_out,
      float* upper_bound_out,
      char* quantile_regression_mode) {
    if (total_insertions < data_manager->k) {
      return 0.0;
    }

    if (!prediction_interval || !predictor_time || !lower_bound_out || !upper_bound_out || !quantile_regression_mode) {
      throw std::runtime_error("predict_recall: null pointer argument");
    }

    double predictor_start_time = data_manager->elapsed_secs();

    float nn_dist = data_manager->get_nearest_dist_of_query(query_idx);
    float furthest_dist = data_manager->get_furthest_dist_of_query(query_idx);
    float avg_dist = data_manager->get_avg_dist_of_query(query_idx);
    float variance = data_manager->get_variance_of_query(query_idx);
    float median = data_manager->get_percentile_of_query(query_idx, 0.50);
    float percentile_25 = data_manager->get_percentile_of_query(query_idx, 0.25);
    float percentile_75 = data_manager->get_percentile_of_query(query_idx, 0.75);

    const int query_dim_feats = data_manager->use_query_dim_features ? data_manager->total_query_dim_features : 0;

    std::vector<double> all_data_vec;
    all_data_vec.push_back((double)nstep);
    all_data_vec.push_back((double)ndis);
    all_data_vec.push_back((double)total_insertions);
    all_data_vec.push_back((double)first_nn_dis);
    all_data_vec.push_back(nn_dist);
    all_data_vec.push_back(furthest_dist);
    all_data_vec.push_back(avg_dist);
    all_data_vec.push_back(variance);
    all_data_vec.push_back(percentile_25);
    all_data_vec.push_back(median);
    all_data_vec.push_back(percentile_75);

    for (int i = 0; i < query_dim_feats; i++) {
      all_data_vec.push_back((double)data_manager->query_dim_features[query_idx][i]);
    }
    const int num_feats = (int)all_data_vec.size();
    const double* data = all_data_vec.data();

    double upper_bound = 1.0;
    double lower_bound = 0.0;
    double avg_recall = 0.0;
    float predicted_recall = 0.0;
    if (inf_mode == CLASSIC_REGRESSION) {
      conformal_predictor->predict_range_with_regression(data, num_feats, &lower_bound, &upper_bound, &avg_recall);
      predicted_recall = lower_bound;
    } else if (inf_mode == QUANTILE_REGRESSION) {
      if (*quantile_regression_mode == 'u') {  // upper
        // v1 old
        // conformal_predictor->predict_upper_with_quantile(data, num_feats, &upper_bound);
        // predicted_recall = upper_bound;
        // avg_recall = upper_bound;

        // v2 new
        conformal_predictor->predict_avg_with_regression(data, num_feats, &avg_recall);
        upper_bound = avg_recall;
        predicted_recall = avg_recall;
        if (predicted_recall >= target_recall) {
          conformal_predictor->predict_lower_with_quantile(data, num_feats, &lower_bound);
          predicted_recall = lower_bound;
          *quantile_regression_mode = 'l';  // switch the mode if we are above target
        }
      } else if (*quantile_regression_mode == 'l') {  // lower
        conformal_predictor->predict_lower_with_quantile(data, num_feats, &lower_bound);
        avg_recall = lower_bound;
        predicted_recall = lower_bound;
      }
    }

    *lower_bound_out = lower_bound;
    *upper_bound_out = upper_bound;
    predicted_recall = std::min(1.0f, std::max(0.0f, predicted_recall));

    if (data_manager->k == 1) {
      predicted_recall = predicted_recall >= 0.5 ? 1.0f : 0.0f;
    }

    float recall_diff = (float)(target_recall - avg_recall);
    if (recall_diff < 0.0f) recall_diff = 0.0f;
    if (recall_diff > 1.0f) recall_diff = 1.0f;

    int next_interval = (int)std::lround(min_prediction_interval + (initial_prediction_interval - min_prediction_interval) * recall_diff);
    if (next_interval < min_prediction_interval) next_interval = min_prediction_interval;
    if (next_interval > initial_prediction_interval) next_interval = initial_prediction_interval;
    *prediction_interval = next_interval;

    double predictor_end_time = data_manager->elapsed_secs();
    *predictor_time += predictor_end_time - predictor_start_time;

    // Keep each CSV row intact when queries run in parallel.
    if (log_file && per_prediction_logging) {
      std::lock_guard<std::mutex> row_lock(log_mutex);
      double actual_recall_k = data_manager->get_recallk(query_idx);

      fprintf(log_file, "%lld,", (long long)query_idx);

      if (data_manager->use_query_dim_features && data_manager->query_dim_features != nullptr) {
        for (int i = 0; i < data_manager->total_query_dim_features; i++) {
          fprintf(log_file, "%f,", data_manager->query_dim_features[query_idx][i]);
        }
      }

      fprintf(log_file, "%d,", nstep);
      fprintf(log_file, "%d,", ndis);
      fprintf(log_file, "%f,", elapsed * 1000);
      fprintf(log_file, "%d,", total_insertions);

      fprintf(log_file, "%f,", first_nn_dis);
      fprintf(log_file, "%f,", nn_dist);
      fprintf(log_file, "%f,", avg_dist);
      fprintf(log_file, "%f,", furthest_dist);

      fprintf(log_file, "%f,", variance);
      fprintf(log_file, "%f,", median);

      fprintf(log_file, "%f,", percentile_25);
      fprintf(log_file, "%f,", percentile_75);

      fprintf(log_file, "%d,", *prediction_interval);
      fprintf(log_file, "%d,", query_predictor_calls);
      fprintf(log_file, "%f,", *predictor_time * 1000);

      fprintf(log_file, "%f,", data_manager->get_RDE(query_idx));
      fprintf(log_file, "%f,", data_manager->get_TDR(query_idx));
      fprintf(log_file, "%f,", data_manager->get_NRS(query_idx));

      fprintf(log_file, "%f,", actual_recall_k);

      fprintf(log_file, "%f,", *lower_bound_out);
      fprintf(log_file, "%f", *upper_bound_out);
      fprintf(log_file, "\n");
    }

    return predicted_recall;
  }

  void log_final_recall_result(
      idx_t query_idx,
      int nstep,
      int ndis,
      int total_insertions,
      double elapsed,
      float first_nn_dis,
      // double last_predicted_recall,
      float last_lower_bound,
      float last_upper_bound,
      int prediction_interval,
      int query_predictor_calls,
      double predictor_time) {
    if (!log_file || per_prediction_logging) {
      return;
    }

    std::lock_guard<std::mutex> row_lock(log_mutex);
    fprintf(log_file, "%lld,", (long long)query_idx);

    if (data_manager->use_query_dim_features && data_manager->query_dim_features != nullptr) {
      for (int i = 0; i < data_manager->total_query_dim_features; i++) {
        fprintf(log_file, "%f,", data_manager->query_dim_features[query_idx][i]);
      }
    }

    fprintf(log_file, "%d,", nstep);
    fprintf(log_file, "%d,", ndis);
    fprintf(log_file, "%f,", elapsed * 1000);
    fprintf(log_file, "%d,", total_insertions);

    fprintf(log_file, "%f,", first_nn_dis);

    fprintf(log_file, "%f,", data_manager->get_nearest_dist_of_query(query_idx));
    fprintf(log_file, "%f,", data_manager->get_avg_dist_of_query(query_idx));
    fprintf(log_file, "%f,", data_manager->get_furthest_dist_of_query(query_idx));
    fprintf(log_file, "%f,", data_manager->get_variance_of_query(query_idx));
    fprintf(log_file, "%f,", data_manager->get_percentile_of_query(query_idx, 0.50));
    fprintf(log_file, "%f,", data_manager->get_percentile_of_query(query_idx, 0.25));
    fprintf(log_file, "%f,", data_manager->get_percentile_of_query(query_idx, 0.75));

    fprintf(log_file, "%d,", prediction_interval);
    fprintf(log_file, "%d,", query_predictor_calls);
    fprintf(log_file, "%f,", predictor_time * 1000);

    fprintf(log_file, "%f,", data_manager->get_RDE(query_idx));
    fprintf(log_file, "%f,", data_manager->get_TDR(query_idx));
    fprintf(log_file, "%f,", data_manager->get_NRS(query_idx));

    fprintf(log_file, "%f,", data_manager->get_recallk(query_idx));
    // fprintf(log_file, "%f\n", last_predicted_recall);
    fprintf(log_file, "%f,", last_lower_bound);
    fprintf(log_file, "%f", last_upper_bound);

    fprintf(log_file, "\n");
  }

} DARTHPlusPredictorHNSW;

/*
IVF Stuff below!
*/

typedef struct DeclarativeRecallDataCollectorIVF {
  std::function<void(idx_t, int, int, int, float, const std::vector<float>&, float)> observation_sink;
  DarthTrainingData* training_data = nullptr;  // borrowed, optional in-memory sink

  DeclarativeRecallDataCollectorIVF(const DeclarativeRecallDataCollectorIVF&) = delete;
  DeclarativeRecallDataCollectorIVF& operator=(const DeclarativeRecallDataCollectorIVF&) = delete;
  ~DeclarativeRecallDataCollectorIVF() { close_log_file(); }
  std::mutex log_mutex;

  DeclarativeRecallDataManager* dataManager = nullptr;
  FILE* log_file = nullptr;

  int logging_interval = 1;

  bool do_naive_early_stop = false;
  int dist_early_stop_threshold = 1000;

  bool include_data_dimensions = true;

  const std::vector<AdaptiveLogInterval> adaptive_logging_intervals;
  double high_recall_cutoff_percentage = 30.0;

  DeclarativeRecallDataCollectorIVF() {}

  DeclarativeRecallDataCollectorIVF(
      DeclarativeRecallDataManager* dataManager,
      int logging_interval,
      int dist_early_stop_threshold = 0)
      : dataManager(dataManager),
        logging_interval(logging_interval),
        dist_early_stop_threshold(dist_early_stop_threshold) {
    if (dist_early_stop_threshold > 0) {
      do_naive_early_stop = true;
    }
  }

  DeclarativeRecallDataCollectorIVF(  // for DARTH plus
      DeclarativeRecallDataManager* dataManager,
      const std::vector<AdaptiveLogInterval>& adaptive_logging_intervals,
      double high_recall_cutoff_percentage)
      : dataManager(dataManager),
        adaptive_logging_intervals(adaptive_logging_intervals),
        high_recall_cutoff_percentage(high_recall_cutoff_percentage) {
  }

  int get_adaptive_interval(float recall) {
    for (const auto& ri : adaptive_logging_intervals) {
      if (recall >= ri.min_recall && recall < ri.max_recall)
        return ri.interval;
    }
    return adaptive_logging_intervals.back().interval;  // default fallback
  }

  void init_log_file();

  void close_log_file();

  void append_to_log(
      idx_t query_idx,
      int nstep,
      int ndis,
      int total_insertions,
      double elapsed,
      float first_nn_dis,
      std::vector<float> centroid_distances,
      float recall_k);

  std::string get_observation_data_str(
      idx_t query_idx,
      int nstep,
      int ndis,
      int total_insertions,
      double elapsed,
      float first_nn_dis,
      std::vector<float> centroid_distances,
      float recall_k) {
    if (observation_sink && total_insertions >= dataManager->k)
      observation_sink(query_idx, nstep, ndis, total_insertions, first_nn_dis, centroid_distances, recall_k);
    if (training_data && total_insertions >= dataManager->k) {
      dataManager->record_training_observation(training_data, query_idx, nstep,
          ndis, total_insertions, first_nn_dis,
          centroid_distance_at_step(centroid_distances, nstep), recall_k);
    }
    std::string observation_data = "";

    if (!log_file || total_insertions < dataManager->k) {
      return observation_data;
    }

    double feature_collection_time_start = dataManager->elapsed_secs();

    observation_data += std::to_string(query_idx) + ",";

    // add the query dims if they are available and precomputed:
    if (dataManager->use_query_dim_features && dataManager->query_dim_features != nullptr) {
      for (int i = 0; i < dataManager->total_query_dim_features; i++) {
        observation_data += std::to_string(dataManager->query_dim_features[query_idx][i]) + ",";
      }
    }

    observation_data += std::to_string(nstep) + ",";
    observation_data += std::to_string(ndis) + ",";
    observation_data += std::to_string(elapsed * 1000) + ",";
    observation_data += std::to_string(total_insertions) + ",";
    observation_data += std::to_string(first_nn_dis) + ",";

    float curr_centroid_dist =
        centroid_distance_at_step(centroid_distances, nstep);
    observation_data += std::to_string(curr_centroid_dist) + ",";

    float nn_dist = dataManager->get_nearest_dist_of_query(query_idx);
    float avg_dist = dataManager->get_avg_dist_of_query(query_idx);
    float furthest_dist = dataManager->get_furthest_dist_of_query(query_idx);
    observation_data += std::to_string(nn_dist) + ",";
    observation_data += std::to_string(avg_dist) + ",";
    observation_data += std::to_string(furthest_dist) + ",";

    float perc_25 = dataManager->get_percentile_of_query(query_idx, 0.25);
    float perc_50 = dataManager->get_percentile_of_query(query_idx, 0.50);
    float perc_75 = dataManager->get_percentile_of_query(query_idx, 0.75);
    float perc_95 = dataManager->get_percentile_of_query(query_idx, 0.95);
    observation_data += std::to_string(perc_25) + ",";
    observation_data += std::to_string(perc_50) + ",";
    observation_data += std::to_string(perc_75) + ",";
    observation_data += std::to_string(perc_95) + ",";

    float variance = dataManager->get_variance_of_query(query_idx);
    observation_data += std::to_string(variance) + ",";

    // New includes start
    float std = std::sqrt(variance);
    float range = furthest_dist - nn_dist;
    float skewness = dataManager->get_skewness_of_query(query_idx);
    float kurtosis = dataManager->get_kurtosis_of_query(query_idx);
    float energy = dataManager->get_energy_of_query(query_idx);
    observation_data += std::to_string(std) + ",";
    observation_data += std::to_string(range) + ",";
    observation_data += std::to_string(skewness) + ",";
    observation_data += std::to_string(kurtosis) + ",";
    observation_data += std::to_string(energy) + ",";
    // New includes end

    // CMU features

    float dist_10 = -1;
    float dist_1st_to_10 = -1;
    float dist_1st_to_first_centroid = nn_dist / centroid_distances[0];

    if (dataManager->k >= 10) {
      dist_10 = dataManager->get_kth_nearest_dist_of_query(query_idx, 9);
      dist_1st_to_10 = nn_dist / dist_10;
    }

    observation_data += std::to_string(dist_10) + ",";
    observation_data += std::to_string(dist_1st_to_10) + ",";
    observation_data += std::to_string(dist_1st_to_first_centroid) + ",";

    for (size_t i = 9; i <= 100; i += 10) {
      if (centroid_distances.size() > i) {
        float dist_1st_to_i = nn_dist / centroid_distances[i];
        observation_data += std::to_string(dist_1st_to_i) + ",";
      } else {
        observation_data += "-1,";
      }
    }
    // LAET end

    double feature_collection_time_end = dataManager->elapsed_secs();
    double feature_collection_time =
        (feature_collection_time_end - feature_collection_time_start) *
        1000;
    observation_data += std::to_string(feature_collection_time) + ",";

    observation_data += std::to_string(recall_k) + "\n";

    return observation_data;
  }

  void flush_all_observations_to_log(std::string* observations, int n) {
    if (!log_file) {
      return;
    }

    for (int i = 0; i < n; i++) {
      fprintf(log_file, "%s", observations[i].c_str());
    }
  }

} DeclarativeRecallDataCollectorIVF;

typedef struct DARTHPredictorIVF {
  DARTHPredictorIVF(const DARTHPredictorIVF&) = delete;
  DARTHPredictorIVF& operator=(const DARTHPredictorIVF&) = delete;
  ~DARTHPredictorIVF() { close_log_file(); }
  std::mutex log_mutex;

  DeclarativeRecallDataManager* dataManager = nullptr;

  int logging_interval = 1;

  FILE* log_file = nullptr;
  bool per_prediction_logging = false;

  double target_recall;

  int initial_prediction_interval = 2000, min_prediction_interval = 100;

  std::unique_ptr<LGBMPredictor> predictor;

  DARTHPredictorIVF(
      DeclarativeRecallDataManager* manager, double target, int initial, int minimum,
      bool logging, std::unique_ptr<LGBMPredictor> model, int log_interval = 1)
      : dataManager(manager), target_recall(target), initial_prediction_interval(initial),
        min_prediction_interval(minimum), per_prediction_logging(logging),
        predictor(std::move(model)) {
    validate_darth_intervals(target, initial, minimum);
    if (!predictor) throw std::invalid_argument("A regression model is required");
    if (log_interval <= 0) throw std::invalid_argument("Logging interval must be positive");
    logging_interval = log_interval;
    predictor->set_feature_names(darth_feature_names());
  }

  DARTHPredictorIVF(
      DeclarativeRecallDataManager* dataManager,
      double target_recall,
      int initial_prediction_interval,
      int min_prediction_interval,
      bool per_prediction_logging,
      char* predictor_model_path,
      int logging_interval);

  void init_log_file();

  void close_log_file();

  void prepare_for_next_query();

  float predictRecall(
      idx_t query_idx,
      int nstep,
      int ndis,
      int total_insertions,
      double elapsed,
      float first_nn_dis,
      int query_predictor_calls,
      int* prediction_interval,
      double* predictor_time);

  void log_final_recall_result(
      idx_t query_idx,
      int nstep,
      int ndis,
      int total_insertions,
      double elapsed,
      float first_nn_dis,
      double last_predicted_recall,
      int prediction_interval,
      int query_predictor_calls,
      double predictor_time);

} DARTHPredictorIVF;



typedef struct DARTHPlusPredictorIVF {
  DARTHPlusPredictorIVF(const DARTHPlusPredictorIVF&) = delete;
  DARTHPlusPredictorIVF& operator=(const DARTHPlusPredictorIVF&) = delete;
  ~DARTHPlusPredictorIVF() { close_log_file(); }
  std::mutex log_mutex;

  DeclarativeRecallDataManager* data_manager = nullptr;

  FILE* log_file = nullptr;
  bool per_prediction_logging = false;

  double target_recall = 0.90;

  int initial_prediction_interval = 2000, min_prediction_interval = 100;

  enum inference_mode {
    QUANTILE_REGRESSION = 0,
    CLASSIC_REGRESSION = 1
  } inference_mode;
  enum inference_mode inf_mode = QUANTILE_REGRESSION;

  std::unique_ptr<ConformalLGBMPredictor> conformal_predictor;

  DARTHPlusPredictorIVF(
      DeclarativeRecallDataManager* manager, double target, int initial, int minimum,
      bool logging, std::unique_ptr<ConformalLGBMPredictor> model)
      : data_manager(manager), target_recall(target), initial_prediction_interval(initial),
        min_prediction_interval(minimum), per_prediction_logging(logging),
        conformal_predictor(std::move(model)) {
    validate_darth_intervals(target, initial, minimum);
    if (!conformal_predictor) throw std::invalid_argument("A conformal predictor is required");
    inf_mode = conformal_predictor->low_quantile_model ? QUANTILE_REGRESSION : CLASSIC_REGRESSION;
    configure_features(manager);
  }

  DARTHPlusPredictorIVF(
      DeclarativeRecallDataManager* data_manager,
      double target_recall,
      int initial_prediction_interval,
      int min_prediction_interval,
      bool per_prediction_logging,
      char* low_quantile_model_path,
      char* high_quantile_model_path,
      char* regression_model_path,
      double low_quantile_hat,
      double high_quantile_hat) {
    validate_darth_intervals(target_recall, initial_prediction_interval, min_prediction_interval);
    conformal_predictor = std::make_unique<ConformalLGBMPredictor>(low_quantile_model_path, high_quantile_model_path, regression_model_path, low_quantile_hat, high_quantile_hat);

    configure_features(data_manager);
    this->data_manager = data_manager;
    this->target_recall = target_recall;
    this->initial_prediction_interval = initial_prediction_interval;
    this->min_prediction_interval = min_prediction_interval;
    this->per_prediction_logging = per_prediction_logging;
    this->inf_mode = QUANTILE_REGRESSION;
  }

  DARTHPlusPredictorIVF(
      DeclarativeRecallDataManager* data_manager,
      double target_recall,
      int initial_prediction_interval,
      int min_prediction_interval,
      bool per_prediction_logging,
      char* predictor_model_path,
      double low_quantile_hat,
      double high_quantile_hat) {
    validate_darth_intervals(target_recall, initial_prediction_interval, min_prediction_interval);
    conformal_predictor = std::make_unique<ConformalLGBMPredictor>(predictor_model_path, low_quantile_hat, high_quantile_hat);

    configure_features(data_manager);
    this->data_manager = data_manager;
    this->target_recall = target_recall;
    this->initial_prediction_interval = initial_prediction_interval;
    this->min_prediction_interval = min_prediction_interval;
    this->per_prediction_logging = per_prediction_logging;
    this->inf_mode = CLASSIC_REGRESSION;
  }

  void configure_features(const DeclarativeRecallDataManager* data_manager) {
    auto names = std::vector<std::string>{"step", "dists", "inserts", "first_nn_dist",
        "curr_centroid_dist", "nn_dist", "avg_dist", "furthest_dist", "variance",
        "percentile_50", "percentile_25", "percentile_75"};
    if (data_manager->use_query_dim_features) {
      for (const auto& name : data_manager->query_dim_features_names)
        names.emplace_back(name);
    }
    conformal_predictor->set_feature_names(names);
  }

  void init_log_file() {
    if (!data_manager->log_filename) {
      return;
    }

    close_log_file();
    log_file = fopen(data_manager->log_filename, "w");
    if (!log_file) {
      printf("Error opening recall data log file\n");
      throw std::runtime_error("DARTH log or prediction failure");
    }

    fprintf(log_file, "qid,");

    if (data_manager->use_query_dim_features && data_manager->query_dim_features != nullptr) {
      for (int i = 0; i < data_manager->total_query_dim_features; i++) {
        fprintf(log_file, "%s,", data_manager->query_dim_features_names[i]);
      }
    }

    fprintf(log_file, "step,");
    fprintf(log_file, "dists,");
    fprintf(log_file, "elaps_ms,");
    fprintf(log_file, "inserts,");

    fprintf(log_file, "first_nn_dist,");
    fprintf(log_file, "curr_centroid_dist,");

    fprintf(log_file, "nn_dist,");
    fprintf(log_file, "avg_dist,");
    fprintf(log_file, "furthest_dist,");

    fprintf(log_file, "variance,");
    fprintf(log_file, "median,");

    fprintf(log_file, "percentile_25,");
    fprintf(log_file, "percentile_75,");

    fprintf(log_file, "r_current_interval,");
    fprintf(log_file, "r_predictor_calls,");
    fprintf(log_file, "r_predictor_time_ms,");

    // Quality approximation measures
    fprintf(log_file, "RDE,");  // relative distance error
    fprintf(log_file, "TDR,");  // total distance ratio
    fprintf(log_file, "NRS,");  // normalized rank sum

    fprintf(log_file, "r_actual,");
    // fprintf(log_file, "r_predicted");
    fprintf(log_file, "r_lower,");
    fprintf(log_file, "r_upper");

    fprintf(log_file, "\n");
  }

  void close_log_file() {
    if (log_file) {
      fclose(log_file);
      log_file = nullptr;
    }
  }

  float predict_recall(
      idx_t query_idx,
      int nstep,
      int ndis,
      int total_insertions,
      double elapsed,
      float first_nn_dis,
      int query_predictor_calls,
      int* prediction_interval,
      double* predictor_time,
      float* lower_bound_out,
      float* upper_bound_out,
      char* quantile_regression_mode,
      std::vector<float> centroid_distances) {
    if (total_insertions < data_manager->k) {
      return 0.0;
    }

    if (!prediction_interval || !predictor_time || !lower_bound_out || !upper_bound_out || !quantile_regression_mode) {
      throw std::runtime_error("predict_recall: null pointer argument");
    }

    double predictor_start_time = data_manager->elapsed_secs();

    float nn_dist = data_manager->get_nearest_dist_of_query(query_idx);
    float furthest_dist = data_manager->get_furthest_dist_of_query(query_idx);
    float avg_dist = data_manager->get_avg_dist_of_query(query_idx);
    float variance = data_manager->get_variance_of_query(query_idx);
    float median = data_manager->get_percentile_of_query(query_idx, 0.50);
    float percentile_25 = data_manager->get_percentile_of_query(query_idx, 0.25);
    float percentile_75 = data_manager->get_percentile_of_query(query_idx, 0.75);

    const int query_dim_feats = data_manager->use_query_dim_features ? data_manager->total_query_dim_features : 0;

    std::vector<double> all_data_vec;
    all_data_vec.push_back((double)nstep);
    all_data_vec.push_back((double)ndis);
    all_data_vec.push_back((double)total_insertions);
    all_data_vec.push_back((double)first_nn_dis);
    all_data_vec.push_back(
        (double)centroid_distance_at_step(centroid_distances, nstep));
    all_data_vec.push_back(nn_dist);
    all_data_vec.push_back(avg_dist);
    all_data_vec.push_back(furthest_dist);
    all_data_vec.push_back(variance);
    all_data_vec.push_back(median);
    all_data_vec.push_back(percentile_25);
    all_data_vec.push_back(percentile_75);

    for (int i = 0; i < query_dim_feats; i++) {
      all_data_vec.push_back((double)data_manager->query_dim_features[query_idx][i]);
    }
    const int num_feats = (int)all_data_vec.size();
    const double* data = all_data_vec.data();

    double upper_bound = 1.0;
    double lower_bound = 0.0;
    double avg_recall = 0.0;
    float predicted_recall = 0.0;
    if (inf_mode == CLASSIC_REGRESSION) {
      conformal_predictor->predict_range_with_regression(data, num_feats, &lower_bound, &upper_bound, &avg_recall);
      predicted_recall = lower_bound;
    } else if (inf_mode == QUANTILE_REGRESSION) {
      if (*quantile_regression_mode == 'u') {  // upper
        conformal_predictor->predict_avg_with_regression(data, num_feats, &avg_recall);
        upper_bound = avg_recall;
        predicted_recall = avg_recall;
        if (predicted_recall >= target_recall) {
          conformal_predictor->predict_lower_with_quantile(data, num_feats, &lower_bound);
          predicted_recall = lower_bound;
          *quantile_regression_mode = 'l';  // switch the mode if we are above target
        }
      } else if (*quantile_regression_mode == 'l') {  // lower
        conformal_predictor->predict_lower_with_quantile(data, num_feats, &lower_bound);
        avg_recall = lower_bound;
        predicted_recall = lower_bound;
      }
    }

    *lower_bound_out = lower_bound;
    *upper_bound_out = upper_bound;
    predicted_recall = std::min(1.0f, std::max(0.0f, predicted_recall));

    if (data_manager->k == 1) {
      predicted_recall = predicted_recall >= 0.5 ? 1.0f : 0.0f;
    }

    float recall_diff = (float)(target_recall - avg_recall);
    if (recall_diff < 0.0f) recall_diff = 0.0f;
    if (recall_diff > 1.0f) recall_diff = 1.0f;

    int next_interval = (int)std::lround(min_prediction_interval + (initial_prediction_interval - min_prediction_interval) * recall_diff);
    if (next_interval < min_prediction_interval) next_interval = min_prediction_interval;
    if (next_interval > initial_prediction_interval) next_interval = initial_prediction_interval;
    *prediction_interval = next_interval;

    double predictor_end_time = data_manager->elapsed_secs();
    *predictor_time += predictor_end_time - predictor_start_time;

    // Keep each CSV row intact when queries run in parallel.
    if (log_file && per_prediction_logging) {
      std::lock_guard<std::mutex> row_lock(log_mutex);
      double actual_recall_k = data_manager->get_recallk(query_idx);

      fprintf(log_file, "%lld,", (long long)query_idx);

      if (data_manager->use_query_dim_features && data_manager->query_dim_features != nullptr) {
        for (int i = 0; i < data_manager->total_query_dim_features; i++) {
          fprintf(log_file, "%f,", data_manager->query_dim_features[query_idx][i]);
        }
      }

      fprintf(log_file, "%d,", nstep);
      fprintf(log_file, "%d,", ndis);
      fprintf(log_file, "%f,", elapsed * 1000);
      fprintf(log_file, "%d,", total_insertions);

      fprintf(log_file, "%f,", first_nn_dis);
      fprintf(
          log_file,
          "%f,",
          centroid_distance_at_step(centroid_distances, nstep));

      fprintf(log_file, "%f,", nn_dist);
      fprintf(log_file, "%f,", avg_dist);
      fprintf(log_file, "%f,", furthest_dist);

      fprintf(log_file, "%f,", variance);
      fprintf(log_file, "%f,", median);

      fprintf(log_file, "%f,", percentile_25);
      fprintf(log_file, "%f,", percentile_75);

      fprintf(log_file, "%d,", *prediction_interval);
      fprintf(log_file, "%d,", query_predictor_calls);
      fprintf(log_file, "%f,", *predictor_time * 1000);

      fprintf(log_file, "%f,", data_manager->get_RDE(query_idx));
      fprintf(log_file, "%f,", data_manager->get_TDR(query_idx));
      fprintf(log_file, "%f,", data_manager->get_NRS(query_idx));

      fprintf(log_file, "%f,", actual_recall_k);

      fprintf(log_file, "%f,", *lower_bound_out);
      fprintf(log_file, "%f", *upper_bound_out);
      fprintf(log_file, "\n");
    }

    return predicted_recall;
  }

  void log_final_recall_result(
      idx_t query_idx,
      int nstep,
      int ndis,
      int total_insertions,
      double elapsed,
      float first_nn_dis,
      // double last_predicted_recall,
      float last_lower_bound,
      float last_upper_bound,
      int prediction_interval,
      int query_predictor_calls,
      double predictor_time,
      std::vector<float> centroid_distances) {
    if (!log_file || per_prediction_logging) {
      return;
    }

    std::lock_guard<std::mutex> row_lock(log_mutex);
    fprintf(log_file, "%lld,", (long long)query_idx);

    if (data_manager->use_query_dim_features && data_manager->query_dim_features != nullptr) {
      for (int i = 0; i < data_manager->total_query_dim_features; i++) {
        fprintf(log_file, "%f,", data_manager->query_dim_features[query_idx][i]);
      }
    }

    fprintf(log_file, "%d,", nstep);
    fprintf(log_file, "%d,", ndis);
    fprintf(log_file, "%f,", elapsed * 1000);
    fprintf(log_file, "%d,", total_insertions);

    fprintf(log_file, "%f,", first_nn_dis);
    fprintf(
        log_file,
        "%f,",
        centroid_distance_at_step(centroid_distances, nstep));

    fprintf(log_file, "%f,", data_manager->get_nearest_dist_of_query(query_idx));
    fprintf(log_file, "%f,", data_manager->get_avg_dist_of_query(query_idx));
    fprintf(log_file, "%f,", data_manager->get_furthest_dist_of_query(query_idx));
    fprintf(log_file, "%f,", data_manager->get_variance_of_query(query_idx));
    fprintf(log_file, "%f,", data_manager->get_percentile_of_query(query_idx, 0.50));
    fprintf(log_file, "%f,", data_manager->get_percentile_of_query(query_idx, 0.25));
    fprintf(log_file, "%f,", data_manager->get_percentile_of_query(query_idx, 0.75));

    fprintf(log_file, "%d,", prediction_interval);
    fprintf(log_file, "%d,", query_predictor_calls);
    fprintf(log_file, "%f,", predictor_time * 1000);

    fprintf(log_file, "%f,", data_manager->get_RDE(query_idx));
    fprintf(log_file, "%f,", data_manager->get_TDR(query_idx));
    fprintf(log_file, "%f,", data_manager->get_NRS(query_idx));

    fprintf(log_file, "%f,", data_manager->get_recallk(query_idx));
    // fprintf(log_file, "%f\n", last_predicted_recall);
    fprintf(log_file, "%f,", last_lower_bound);
    fprintf(log_file, "%f", last_upper_bound);

    fprintf(log_file, "\n");
  }

} DARTHPlusPredictorIVF;

}  // namespace faiss

// LAET remains available to the FAISS search hooks.
#include <competitors/laet/LAETPredictor.h>
