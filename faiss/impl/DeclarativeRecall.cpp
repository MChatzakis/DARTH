#include "DeclarativeRecall.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace faiss {

double LGBMPredictor::predict(const double* features, int num_features) {
  std::vector<double> ordered;
  if (!feature_order.empty()) {
    if (num_features != input_feature_count)
      throw std::runtime_error("DARTH inference feature count mismatch");
    ordered.reserve(feature_order.size());
    for (int column : feature_order) ordered.push_back(features[column]);
    features = ordered.data();
  } else if (num_features != feature_count) {
    throw std::runtime_error("LightGBM inference feature count mismatch");
  }
  double result;
  int64_t length;
  // FastConfig itself synchronizes callers if a later/larger or nested team
  // shares a slot. Ordinarily each OpenMP worker has its own slot.
  auto* fast = workers[omp_get_thread_num() % workers.size()].get();
  int status;
  // LightGBM indexes prediction scratch space by omp_get_thread_num(),
  // even in the single-row API. A serialized region gives this caller ID 0
  // without starting workers or serializing the surrounding query searches.
#pragma omp parallel if (false)
  {
    status = LGBM_BoosterPredictForMatSingleRowFast(fast, features, &length, &result);
  }
  if (status != 0) throw std::runtime_error(LGBM_GetLastError());
  return result;
}

std::vector<std::string> DarthTrainingData::feature_names() const {
  auto names = darth_feature_names(query_dimensions);
  if (ivf && query_dimensions) {
    names = {"step", "dists", "inserts", "first_nn_dist", "curr_centroid_dist",
        "nn_dist", "avg_dist", "furthest_dist", "variance", "percentile_50",
        "percentile_25", "percentile_75"};
    auto all = darth_feature_names(true);
    names.insert(names.end(), all.begin() + 11, all.end());
  }
  return names;
}

namespace {
// Match the old CSV -> pandas float64 path exactly, including six decimals.
// No CSV text is retained and no intermediate files are created.
double csv_precision(float value) {
  char text[128];
  std::snprintf(text, sizeof(text), "%f", double(value));
  return std::strtod(text, nullptr);
}
}

void DeclarativeRecallDataManager::record_training_observation(
    DarthTrainingData* output, idx_t query_idx, int step, int ndis,
    int insertions, float first_distance, float current_centroid_distance,
    float recall) {
  const float nearest = get_nearest_dist_of_query(query_idx);
  const float furthest = get_furthest_dist_of_query(query_idx);
  const float average = get_avg_dist_of_query(query_idx);
  const float variance = get_variance_of_query(query_idx);
  const float p25 = get_percentile_of_query(query_idx, 0.25);
  const float p50 = get_percentile_of_query(query_idx, 0.50);
  const float p75 = get_percentile_of_query(query_idx, 0.75);
  std::vector<double> values = {double(step), double(ndis), double(insertions),
      csv_precision(first_distance)};
  if (output->ivf && output->query_dimensions) {
    for (float value : {current_centroid_distance, nearest, average, furthest,
                       variance, p50, p25, p75}) values.push_back(csv_precision(value));
  } else {
    for (float value : {nearest, furthest, average, variance, p25, p50, p75})
      values.push_back(csv_precision(value));
  }
  if (output->query_dimensions) {
    for (int i = 0; i < total_query_dim_features; ++i)
      values.push_back(csv_precision(query_dim_features[query_idx][i]));
  }
  output->queries[query_idx].push_back({std::move(values), csv_precision(recall), size_t(ndis)});
}

// DeclarativeRecallDataManager
DeclarativeRecallDataManager::DeclarativeRecallDataManager() {}

DeclarativeRecallDataManager::DeclarativeRecallDataManager(
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
    idx_t k_all)
    : distances(distances),
      labels(labels),
      gt(gt),
      gt_dist(gt_dist),
      nq(nq),
      ndb(ndb),
      k(k),
      d(d),
      queries(queries),
      log_filename(log_filename),
      db(db),
      gt_for_all_k(gt_for_all_k),
      k_all(k_all) {}

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
float DeclarativeRecallDataManager::get_ed(const float* x, const float* y, size_t d) {
  size_t i;
  float res = 0;
  FAISS_PRAGMA_IMPRECISE_LOOP
  for (i = 0; i < d; i++) {
    const float tmp = x[i] - y[i];
    res += tmp * tmp;
  }
  return res;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

double DeclarativeRecallDataManager::elapsed_secs() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

float DeclarativeRecallDataManager::get_avg_dist_of_query(idx_t q_id) {
  float sum = 0;
  for (idx_t i = 0; i < k; i++) {
    sum += distances[q_id * k + i];
  }
  return sum / k;
}

float DeclarativeRecallDataManager::get_nearest_dist_of_query(idx_t q_id) {
  float min_dist = std::numeric_limits<float>::max();
  for (idx_t i = 0; i < k; i++) {
    if (distances[q_id * k + i] < min_dist)
      min_dist = distances[q_id * k + i];
  }
  return min_dist;
}

float DeclarativeRecallDataManager::get_furthest_dist_of_query(idx_t q_id) {
  float max_dist = -std::numeric_limits<float>::infinity();
  for (idx_t i = 0; i < k; i++) {
    if (distances[q_id * k + i] > max_dist)
      max_dist = distances[q_id * k + i];
  }
  return max_dist;
}

float DeclarativeRecallDataManager::get_dist_of_query_to_medoid(idx_t q_id) {
  std::vector<float> medoid(static_cast<size_t>(d), 0.0f);

  for (int i = 0; i < k; i++) {
    for (int j = 0; j < d; j++) {
      medoid[static_cast<size_t>(j)] += db[labels[q_id * k + i] * d + j];
    }
  }

  for (int i = 0; i < d; i++) {
    medoid[static_cast<size_t>(i)] /= k;
  }

  const float* query = queries + q_id * d;

  float ed_dist = get_ed(query, medoid.data(), d);

  return ed_dist;
}

float DeclarativeRecallDataManager::get_recallk(idx_t query_idx) {
  // Production inference does not require ground truth.
  if (!gt) return std::numeric_limits<float>::quiet_NaN();
  std::unordered_set<idx_t> gt_set(
      gt + query_idx * k, gt + (query_idx + 1) * k);
  int matches = 0;

  for (int j = 0; j < k; ++j) {
    if (gt_set.count(labels[query_idx * k + j])) {
      matches++;
    }
  }

  float recall = (float)matches / (float)k;
  return recall;
}

float DeclarativeRecallDataManager::get_kth_nearest_dist_of_query(idx_t q_id, int kth) {
  float* query_distances = distances + q_id * k;

  std::vector<float> query_distances_cp(query_distances, query_distances + k);

  std::nth_element(
      query_distances_cp.begin(),
      query_distances_cp.begin() + kth,
      query_distances_cp.end());

  float kth_nearest_dist = query_distances_cp[static_cast<size_t>(kth)];

  return kth_nearest_dist;
}

float DeclarativeRecallDataManager::get_variance_of_query(idx_t q_id) {
  float sum = 0;
  float sum_of_squares = 0;

  for (idx_t i = 0; i < k; i++) {
    float dist = distances[q_id * k + i];
    sum += dist;
    sum_of_squares += dist * dist;
  }

  float mean = sum / k;

  return (sum_of_squares / k) - (mean * mean);
}

float DeclarativeRecallDataManager::get_percentile_of_query(idx_t q_id, float percentile) {
  std::vector<float> query_distances(static_cast<size_t>(k));

  for (idx_t i = 0; i < k; i++) {
    query_distances[static_cast<size_t>(i)] = distances[q_id * k + i];
  }

  idx_t idx = static_cast<idx_t>(percentile * (k - 1));

  std::nth_element(
      query_distances.begin(),
      query_distances.begin() + idx,
      query_distances.end());

  return query_distances[static_cast<size_t>(idx)];
}

float DeclarativeRecallDataManager::get_skewness_of_query(idx_t q_id) {
  float sum = 0;
  float sum_of_cubes = 0;
  float sum_of_squares = 0;

  for (idx_t i = 0; i < k; i++) {
    float dist = distances[q_id * k + i];
    sum += dist;
    sum_of_squares += dist * dist;
    sum_of_cubes += dist * dist * dist;
  }

  float mean = sum / k;
  float variance = (sum_of_squares / k) - (mean * mean);
  float skewness = (sum_of_cubes / k) - (3 * mean * variance) - (mean * mean * mean);

  return skewness;
}

float DeclarativeRecallDataManager::get_energy_of_query(idx_t q_id) {
  float sum = 0;

  for (idx_t i = 0; i < k; i++) {
    float dist = distances[q_id * k + i];
    sum += dist * dist;
  }

  return sum;
}

float DeclarativeRecallDataManager::get_kurtosis_of_query(idx_t q_id) {
  float mean = 0;
  float m2 = 0;  // Second moment (variance)
  float m4 = 0;  // Fourth moment
  idx_t n = k;

  // Calculate the mean
  for (idx_t i = 0; i < n; i++) {
    mean += distances[q_id * k + i];
  }
  mean /= n;

  // Calculate second and fourth moments
  for (idx_t i = 0; i < n; i++) {
    float dist = distances[q_id * k + i];
    float diff = dist - mean;
    m2 += diff * diff;
    m4 += diff * diff * diff * diff;
  }

  m2 /= n;  // Variance
  m4 /= n;  // Fourth moment

  // Calculate kurtosis
  float kurtosis = (m4 / (m2 * m2)) - 3;

  return kurtosis;
}

float DeclarativeRecallDataManager::get_TDR(idx_t q_id) {
  float found_distances_sum = 0;
  float gt_distances_sum = 0;

  for (idx_t i = 0; i < k; i++) {
    found_distances_sum += distances[q_id * k + i];
    gt_distances_sum += gt_dist[q_id * k + i];
  }

  return gt_distances_sum / found_distances_sum;
}

float DeclarativeRecallDataManager::get_RDE(idx_t q_id) {
  std::vector<float> found_distances(static_cast<size_t>(k));
  for (idx_t i = 0; i < k; i++) {
    found_distances[static_cast<size_t>(i)] = distances[q_id * k + i];
  }

  // sort by increasing distance to the query
  std::sort(found_distances.begin(), found_distances.end());

  float RDE = 0;

  for (idx_t i = 0; i < k; i++) {
    RDE += (found_distances[static_cast<size_t>(i)] / gt_dist[q_id * k + i]) - 1;
  }

  RDE /= k;

  return RDE;
}

float DeclarativeRecallDataManager::get_NRS(idx_t q_id) {
  std::vector<std::pair<float, idx_t>> distance_label_pairs;

  // Collect approximate distances and labels
  for (int i = 0; i < k; ++i) {
    distance_label_pairs.emplace_back(
        distances[q_id * k + i], labels[q_id * k + i]);
  }

  // Sort the pairs based on distances
  std::sort(
      distance_label_pairs.begin(),
      distance_label_pairs.end(),
      [](const std::pair<float, idx_t>& a,
         const std::pair<float, idx_t>& b) {
        return a.first < b.first;
      });

  // Compute the rank sum for the ground-truth neighbors
  float rank_sum = 0;
  for (int i = 0; i < k; i++) {
    idx_t current_nn_id = distance_label_pairs[static_cast<size_t>(i)].second;

    // Find the rank of the current nearest neighbor in the ground truth
    idx_t gt_position = -1;
    for (int j = 0; j < k_all; j++) {
      if (gt_for_all_k[q_id * k_all + j] == current_nn_id) {
        gt_position = j + 1;  // rank is 1-based
        break;
      }
    }

    // If a ground-truth neighbor is missing, return -1 (undefined)
    if (gt_position == -1) {
      return -1.0;
    }

    rank_sum += gt_position;
  }

  // Calculate max and min possible rank sums
  float max_rank_sum = k * (k + 1) / 2.0;

  // Normalize the rank sum
  float NRS = 1 - ((rank_sum - k) / (max_rank_sum - k));

  return NRS;
}

// DeclarativeRecallDataCollectorHNSW
DeclarativeRecallDataCollectorHNSW::DeclarativeRecallDataCollectorHNSW() {}

DeclarativeRecallDataCollectorHNSW::DeclarativeRecallDataCollectorHNSW(
    DeclarativeRecallDataManager* data_manager,
    int logging_interval,
    int dist_early_stop_threshold)
    : data_manager(data_manager),
      logging_interval(logging_interval),
      dist_early_stop_threshold(dist_early_stop_threshold) {
  if (dist_early_stop_threshold > 0) {
    do_naive_early_stop = true;
  }
}

// DeclarativeRecallDataCollectorIVF
void DeclarativeRecallDataCollectorIVF::init_log_file() {
  if (!dataManager->log_filename) {
    return;
  }

  close_log_file();
  log_file = fopen(dataManager->log_filename, "w");
  if (!log_file) {
    perror("Error opening recall data log file");
    throw std::runtime_error("DARTH log or prediction failure");
  }

  fprintf(log_file, "qid,");

  if (dataManager->use_query_dim_features && dataManager->query_dim_features != nullptr) {
    for (int i = 0; i < dataManager->total_query_dim_features; i++) {
      fprintf(log_file, "%s,", dataManager->query_dim_features_names[i]);
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

  fprintf(log_file, "percentile_25,");
  fprintf(log_file, "percentile_50,");
  fprintf(log_file, "percentile_75,");
  fprintf(log_file, "percentile_95,");

  fprintf(log_file, "variance,");

  // New
  fprintf(log_file, "std,");
  fprintf(log_file, "range,");
  fprintf(log_file, "skewness,");
  fprintf(log_file, "kurtosis,");
  fprintf(log_file, "energy,");
  // New end

  // LAET
  fprintf(log_file, "dist_10,");
  fprintf(log_file, "d1st_to_10th,");
  fprintf(log_file, "d1st_to_1stcentroid,");
  for (size_t i = 9; i <= 100; i += 10) {
    fprintf(log_file, "centroid_dist_%d_ratio,", i + 1);
  }
  // LAET END

  fprintf(log_file, "feats_collect_time_ms,");

  // Target
  fprintf(log_file, "r\n");
}

void DeclarativeRecallDataCollectorIVF::close_log_file() {
  if (log_file) {
    fclose(log_file);
    log_file = nullptr;
  }
}

void DeclarativeRecallDataCollectorIVF::append_to_log(
    idx_t query_idx,
    int nstep,
    int ndis,
    int total_insertions,
    double elapsed,
    float first_nn_dis,
    std::vector<float> centroid_distances,
    float recall_k) {

  std::string log_str = this->get_observation_data_str(
      query_idx,
      nstep,
      ndis,
      total_insertions,
      elapsed,
      first_nn_dis,
      centroid_distances,
      recall_k);

  if (log_file && !log_str.empty()) {
    fprintf(log_file, "%s", log_str.c_str());
  }
}

void DeclarativeRecallDataCollectorHNSW::init_log_file() {
  if (!data_manager->log_filename) {
    return;
  }

  close_log_file();
  log_file = fopen(data_manager->log_filename, "w");
  if (!log_file) {
    // printf("Error opening recall data log file %s\n",
    // dataManager->log_filename);
    perror("Error opening recall data log file");
    throw std::runtime_error("DARTH log or prediction failure");
  }

  fprintf(log_file, "qid,");

  //
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

  fprintf(log_file, "percentile_25,");
  fprintf(log_file, "percentile_50,");
  fprintf(log_file, "percentile_75,");
  fprintf(log_file, "percentile_95,");

  fprintf(log_file, "variance,");

  // New
  fprintf(log_file, "std,");
  fprintf(log_file, "range,");
  fprintf(log_file, "skewness,");
  fprintf(log_file, "kurtosis,");
  fprintf(log_file, "energy,");
  // New end

  fprintf(log_file, "nn10_dist,");
  fprintf(log_file, "nn_to_first,");
  fprintf(log_file, "nn10_to_first,");

  // Quality approximation measures
  fprintf(log_file, "RDE,");  // relative distance error
  fprintf(log_file, "TDR,");  // total distance ratio
  fprintf(log_file, "NRS,");  // normalized rank sum

  fprintf(log_file, "feats_collect_time_ms,");

  // Target
  fprintf(log_file, "r\n");
}

void DeclarativeRecallDataCollectorHNSW::close_log_file() {
  if (log_file) {
    fclose(log_file);
    log_file = nullptr;
  }
}

void DeclarativeRecallDataCollectorHNSW::append_to_log(
    idx_t query_idx,
    int nstep,
    int ndis,
    int total_insertions,
    double elapsed,
    float first_nn_dis,
    float recall_k) {
  if (observation_sink && total_insertions >= data_manager->k)
    observation_sink(query_idx, nstep, ndis, total_insertions, first_nn_dis, recall_k);
  if (training_data && total_insertions >= data_manager->k) {
    data_manager->record_training_observation(training_data, query_idx, nstep,
        ndis, total_insertions, first_nn_dis, 0.0f, recall_k);
  }

  if (!log_file || total_insertions < data_manager->k) {
    return;
  }

  // Standard values
  std::lock_guard<std::mutex> row_lock(log_mutex);
  fprintf(log_file, "%ld,", query_idx);

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

  double feature_collection_time_start = data_manager->elapsed_secs();

  // version 1 for distances
  float nn_dist = data_manager->get_nearest_dist_of_query(query_idx);
  float avg_dist = data_manager->get_avg_dist_of_query(query_idx);
  float furthest_dist = data_manager->get_furthest_dist_of_query(query_idx);

  fprintf(log_file, "%f,", nn_dist);
  fprintf(log_file, "%f,", avg_dist);
  fprintf(log_file, "%f,", furthest_dist);

  float percentile_25 =
      data_manager->get_percentile_of_query(query_idx, 0.25);
  float percentile_50 =
      data_manager->get_percentile_of_query(query_idx, 0.50);
  float percentile_75 =
      data_manager->get_percentile_of_query(query_idx, 0.75);
  float percentile_95 =
      data_manager->get_percentile_of_query(query_idx, 0.95);

  fprintf(log_file, "%f,", percentile_25);
  fprintf(log_file, "%f,", percentile_50);
  fprintf(log_file, "%f,", percentile_75);
  fprintf(log_file, "%f,", percentile_95);

  float variance = data_manager->get_variance_of_query(query_idx);
  fprintf(log_file, "%f,", variance);

  /* New includes start */
  float stdv = std::sqrt(variance);
  float range = furthest_dist - nn_dist;
  float skewness = data_manager->get_skewness_of_query(query_idx);
  float kurtosis = data_manager->get_kurtosis_of_query(query_idx);
  float energy = data_manager->get_energy_of_query(query_idx);
  fprintf(log_file, "%f,", stdv);
  fprintf(log_file, "%f,", range);
  fprintf(log_file, "%f,", skewness);
  fprintf(log_file, "%f,", kurtosis);
  fprintf(log_file, "%f,", energy);
  /* New includes end */

  // CMU features
  float dist_10 = -1;
  float dnn_to_dstart = -1;
  float d10_to_dstart = -1;

  if (data_manager->k >= 10) {
    dist_10 = data_manager->get_kth_nearest_dist_of_query(query_idx, 9);
  }

  if (first_nn_dis > 0) {
    dnn_to_dstart = nn_dist / first_nn_dis;
  }

  if (dist_10 != -1 && first_nn_dis > 0) {
    d10_to_dstart = dist_10 / first_nn_dis;
  }

  fprintf(log_file, "%f,", dist_10);
  fprintf(log_file, "%f,", dnn_to_dstart);
  fprintf(log_file, "%f,", d10_to_dstart);
  //

  // version 2 for distances
  // int all_result_set_feats = 11;
  // float dist_feats[all_result_set_feats];
  // get_rset_feats_of_query(query_idx, dist_feats, first_nn_dis);

  // for (int i = 0; i < all_result_set_feats; i++) {
  //     fprintf(log_file, "%f,", dist_feats[i]);
  // }

  // distance from query to medoid
  // float distance_from_medoid =
  //        dataManager->get_dist_of_query_to_medoid(query_idx);
  // fprintf(log_file, "%f,", distance_from_medoid);
  //

  float RDE = data_manager->get_RDE(query_idx);
  float TDR = data_manager->get_TDR(query_idx);
  float NRS = data_manager->get_NRS(query_idx);
  fprintf(log_file, "%f,", RDE);
  fprintf(log_file, "%f,", TDR);
  fprintf(log_file, "%f,", NRS);

  // if (include_data_dimensions){
  //    for (int i = 0; i < dataManager->d; i++) {
  //       fprintf(log_file, "%f,", dataManager->queries[query_idx * dataManager->d + i]);
  //    }
  //}

  double feature_collection_time_end = data_manager->elapsed_secs();
  double feature_collection_time =
      (feature_collection_time_end - feature_collection_time_start) *
      1000;
  fprintf(log_file, "%f,", feature_collection_time);

  // Target
  fprintf(log_file, "%f\n", recall_k);
}

std::string DeclarativeRecallDataCollectorHNSW::get_observation_data_str(
    idx_t query_idx,
    int nstep,
    int ndis,
    int total_insertions,
    double elapsed,
    float first_nn_dis,
    float recall_k) {
  if (observation_sink && total_insertions >= data_manager->k)
    observation_sink(query_idx, nstep, ndis, total_insertions, first_nn_dis, recall_k);
  if (training_data && total_insertions >= data_manager->k) {
    data_manager->record_training_observation(training_data, query_idx, nstep,
        ndis, total_insertions, first_nn_dis, 0.0f, recall_k);
  }

  std::string observation_data = "";

  if (!log_file || total_insertions < data_manager->k) {
    return observation_data;
  }

  double feature_collection_time_start = data_manager->elapsed_secs();

  observation_data += std::to_string(query_idx) + ",";

  // add the query dims if they are available and precomputed:
  if (data_manager->use_query_dim_features && data_manager->query_dim_features != nullptr) {
    for (int i = 0; i < data_manager->total_query_dim_features; i++) {
      observation_data += std::to_string(data_manager->query_dim_features[query_idx][i]) + ",";
    }
  }

  observation_data += std::to_string(nstep) + ",";
  observation_data += std::to_string(ndis) + ",";
  observation_data += std::to_string(elapsed * 1000) + ",";
  observation_data += std::to_string(total_insertions) + ",";
  observation_data += std::to_string(first_nn_dis) + ",";

  float nn_dist = data_manager->get_nearest_dist_of_query(query_idx);
  float avg_dist = data_manager->get_avg_dist_of_query(query_idx);
  float furthest_dist = data_manager->get_furthest_dist_of_query(query_idx);
  observation_data += std::to_string(nn_dist) + ",";
  observation_data += std::to_string(avg_dist) + ",";
  observation_data += std::to_string(furthest_dist) + ",";

  float perc_25 = data_manager->get_percentile_of_query(query_idx, 0.25);
  float perc_50 = data_manager->get_percentile_of_query(query_idx, 0.50);
  float perc_75 = data_manager->get_percentile_of_query(query_idx, 0.75);
  float perc_95 = data_manager->get_percentile_of_query(query_idx, 0.95);
  observation_data += std::to_string(perc_25) + ",";
  observation_data += std::to_string(perc_50) + ",";
  observation_data += std::to_string(perc_75) + ",";
  observation_data += std::to_string(perc_95) + ",";

  float variance = data_manager->get_variance_of_query(query_idx);
  observation_data += std::to_string(variance) + ",";

  /* New includes start */
  float stdv = std::sqrt(variance);
  float range = furthest_dist - nn_dist;
  float skewness = data_manager->get_skewness_of_query(query_idx);
  float kurtosis = data_manager->get_kurtosis_of_query(query_idx);
  float energy = data_manager->get_energy_of_query(query_idx);
  observation_data += std::to_string(stdv) + ",";
  observation_data += std::to_string(range) + ",";
  observation_data += std::to_string(skewness) + ",";
  observation_data += std::to_string(kurtosis) + ",";
  observation_data += std::to_string(energy) + ",";
  /* New includes end */

  // LAET features
  float dist_10 = -1;
  float dnn_to_dstart = -1;
  float d10_to_dstart = -1;

  if (data_manager->k >= 10) {
    dist_10 = data_manager->get_kth_nearest_dist_of_query(query_idx, 9);
  }

  if (first_nn_dis > 0) {
    dnn_to_dstart = nn_dist / first_nn_dis;
  }

  if (dist_10 != -1 && first_nn_dis > 0) {
    d10_to_dstart = dist_10 / first_nn_dis;
  }

  observation_data += std::to_string(dist_10) + ",";
  observation_data += std::to_string(dnn_to_dstart) + ",";
  observation_data += std::to_string(d10_to_dstart) + ",";

  // add the quality approximation measures
  float RDE = data_manager->get_RDE(query_idx);
  float TDR = data_manager->get_TDR(query_idx);
  float NRS = data_manager->get_NRS(query_idx);
  observation_data += std::to_string(RDE) + ",";
  observation_data += std::to_string(TDR) + ",";
  observation_data += std::to_string(NRS) + ",";

  double feature_collection_time_end = data_manager->elapsed_secs();
  double feature_collection_time =
      (feature_collection_time_end - feature_collection_time_start) *
      1000;
  observation_data += std::to_string(feature_collection_time) + ",";

  observation_data += std::to_string(recall_k) + "\n";

  return observation_data;
}

void DeclarativeRecallDataCollectorHNSW::flush_observation_to_log(std::string observation_data) {
  if (!log_file) {
    return;
  }

  fprintf(log_file, "%s", observation_data.c_str());
}

void DeclarativeRecallDataCollectorHNSW::flush_all_observations_to_log(std::string* observations, int n) {
  if (!log_file) {
    return;
  }

  for (int i = 0; i < n; i++) {
    fprintf(log_file, "%s", observations[i].c_str());
  }
}

// DARTHPredictorHNSW
DARTHPredictorHNSW::DARTHPredictorHNSW(
    DeclarativeRecallDataManager* data_manager,
    double target_recall,
    int initial_prediction_interval,
    int min_prediction_interval,
    bool per_prediction_logging,
    char* predictor_model_path)
    : data_manager(data_manager),
      target_recall(target_recall),
      initial_prediction_interval(initial_prediction_interval),
      min_prediction_interval(min_prediction_interval),
      per_prediction_logging(per_prediction_logging) {
  validate_darth_intervals(target_recall, initial_prediction_interval, min_prediction_interval);
  predictor = std::make_unique<LGBMPredictor>(predictor_model_path);
  predictor->set_feature_names(darth_feature_names());
}

void DARTHPredictorHNSW::init_log_file() {
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
  fprintf(log_file, "r_predicted\n");
}

void DARTHPredictorHNSW::close_log_file() {
  if (log_file) {
    fclose(log_file);
    log_file = nullptr;
  }
}

float DARTHPredictorHNSW::predict_recall(
    idx_t query_idx,
    int nstep,
    int ndis,
    int total_insertions,
    double elapsed,
    float first_nn_dis,
    int query_predictor_calls,
    int* prediction_interval,
    double* predictor_time) {
  if (total_insertions < data_manager->k) {
    return 0.0;
  }

  // query_predictor_calls++;

  double predictor_start_time = data_manager->elapsed_secs();

  float nn_dist = data_manager->get_nearest_dist_of_query(query_idx);
  float furthest_dist = data_manager->get_furthest_dist_of_query(query_idx);
  float avg_dist = data_manager->get_avg_dist_of_query(query_idx);
  float variance = data_manager->get_variance_of_query(query_idx);
  float median = data_manager->get_percentile_of_query(query_idx, 0.50);
  float percentile_25 = data_manager->get_percentile_of_query(query_idx, 0.25);
  float percentile_75 = data_manager->get_percentile_of_query(query_idx, 0.75);

  const int num_feats = 11;
  const double data[11] = {
      (double)nstep,
      (double)ndis,
      (double)total_insertions,
      (double)first_nn_dis,
      nn_dist,
      furthest_dist,
      avg_dist,
      variance,
      percentile_25,
      median,
      percentile_75};

  float predicted_recall = static_cast<float>(predictor->predict(data, num_feats));
  predicted_recall = std::min(1.0f, std::max(0.0f, predicted_recall));  // Make sure recall is in [0, 1]

  if (data_manager->k == 1) {  // Special case for k=1 where recall is 0 or 1
    predicted_recall = predicted_recall >= 0.5 ? 1.0f : 0.0f;
  }

  float recall_diff = std::max(0.0f, static_cast<float>(target_recall) - predicted_recall);
  *prediction_interval = min_prediction_interval + (initial_prediction_interval - min_prediction_interval) * recall_diff;

  double predictor_end_time = data_manager->elapsed_secs();
  *predictor_time += predictor_end_time - predictor_start_time;

  // Keep each CSV row intact when queries run in parallel.
  if (log_file && per_prediction_logging) {
    std::lock_guard<std::mutex> row_lock(log_mutex);
    double actual_recall_k = data_manager->get_recallk(query_idx);

    fprintf(log_file, "%ld,", query_idx);
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
    fprintf(log_file, "%f\n", predicted_recall);
  }

  return predicted_recall;
}

void DARTHPredictorHNSW::log_final_recall_result(
    idx_t query_idx,
    int nstep,
    int ndis,
    int total_insertions,
    double elapsed,
    float first_nn_dis,
    double last_predicted_recall,
    int prediction_interval,
    int query_predictor_calls,
    double predictor_time) {
  if (!log_file || per_prediction_logging) {
    return;
  }

  std::lock_guard<std::mutex> row_lock(log_mutex);
  fprintf(log_file, "%ld,", query_idx);
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
  fprintf(log_file, "%f\n", last_predicted_recall);
}

// DARTHPredictorIVF
DARTHPredictorIVF::DARTHPredictorIVF(
    DeclarativeRecallDataManager* dataManager,
    double target_recall,
    int initial_prediction_interval,
    int min_prediction_interval,
    bool per_prediction_logging,
    char* predictor_model_path,
    int logging_interval)
    : dataManager(dataManager),
      target_recall(target_recall),
      initial_prediction_interval(initial_prediction_interval),
      min_prediction_interval(min_prediction_interval),
      per_prediction_logging(per_prediction_logging),
      logging_interval(logging_interval) {
  validate_darth_intervals(target_recall, initial_prediction_interval, min_prediction_interval);
  if (logging_interval <= 0)
    throw std::invalid_argument("Logging interval must be positive");
  predictor = std::make_unique<LGBMPredictor>(predictor_model_path);
  predictor->set_feature_names(darth_feature_names());
}

void DARTHPredictorIVF::init_log_file() {
  if (!dataManager->log_filename) {
    return;
  }

  close_log_file();
  log_file = fopen(dataManager->log_filename, "w");
  if (!log_file) {
    printf("Error opening recall data log file\n");
    throw std::runtime_error("DARTH log or prediction failure");
  }

  fprintf(log_file, "qid,");
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
  fprintf(log_file, "r_predicted\n");
}

void DARTHPredictorIVF::close_log_file() {
  if (log_file) {
    fclose(log_file);
    log_file = nullptr;
  }
}

float DARTHPredictorIVF::predictRecall(
    idx_t query_idx,
    int nstep,
    int ndis,
    int total_insertions,
    double elapsed,
    float first_nn_dis,
    int query_predictor_calls,
    int* prediction_interval,
    double* predictor_time) {
  if (total_insertions < dataManager->k) {
    return 0.0f;
  }

  query_predictor_calls++;

  double predictor_start_time = dataManager->elapsed_secs();

  float nn_dist = dataManager->get_nearest_dist_of_query(query_idx);
  float furthest_dist = dataManager->get_furthest_dist_of_query(query_idx);
  float avg_dist = dataManager->get_avg_dist_of_query(query_idx);
  float variance = dataManager->get_variance_of_query(query_idx);
  float median = dataManager->get_percentile_of_query(query_idx, 0.50);
  float percentile_25 = dataManager->get_percentile_of_query(query_idx, 0.25);
  float percentile_75 = dataManager->get_percentile_of_query(query_idx, 0.75);

  const int num_feats = 11;
  const double data[11] = {
      (double)nstep,
      (double)ndis,
      (double)total_insertions,
      (double)first_nn_dis,
      nn_dist,
      furthest_dist,
      avg_dist,
      variance,
      percentile_25,
      median,
      percentile_75};

  float predicted_recall = static_cast<float>(predictor->predict(data, num_feats));
  predicted_recall = std::min(1.0f, std::max(0.0f, predicted_recall));

  if (dataManager->k == 1) {  // Special case for k=1 where recall is 0 or 1
    predicted_recall = predicted_recall >= 0.5 ? 1.0f : 0.0f;
  }

  float recall_diff = std::max(0.0f, static_cast<float>(target_recall) - predicted_recall);
  *prediction_interval = min_prediction_interval +
                         (initial_prediction_interval - min_prediction_interval) *
                             recall_diff;

  const int64_t rounded_interval =
      ((int64_t(*prediction_interval) + logging_interval - 1) / logging_interval) * logging_interval;
  *prediction_interval = static_cast<int>(std::min<int64_t>(rounded_interval, std::numeric_limits<int>::max()));

  double predictor_end_time = dataManager->elapsed_secs();
  *predictor_time += predictor_end_time - predictor_start_time;

  if (log_file && per_prediction_logging) {
    std::lock_guard<std::mutex> row_lock(log_mutex);
    double actual_recall_k = dataManager->get_recallk(query_idx);

    fprintf(log_file, "%ld,", query_idx);
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

    fprintf(log_file, "%f,", dataManager->get_RDE(query_idx));
    fprintf(log_file, "%f,", dataManager->get_TDR(query_idx));
    fprintf(log_file, "%f,", dataManager->get_NRS(query_idx));

    fprintf(log_file, "%f,", actual_recall_k);
    fprintf(log_file, "%f\n", predicted_recall);
  }

  return predicted_recall;
}

void DARTHPredictorIVF::log_final_recall_result(
    idx_t query_idx,
    int nstep,
    int ndis,
    int total_insertions,
    double elapsed,
    float first_nn_dis,
    double last_predicted_recall,
    int prediction_interval,
    int query_predictor_calls,
    double predictor_time) {
  if (!log_file || per_prediction_logging) {
    return;
  }

  std::lock_guard<std::mutex> row_lock(log_mutex);
  fprintf(log_file, "%ld,", query_idx);
  fprintf(log_file, "%d,", nstep);
  fprintf(log_file, "%d,", ndis);
  fprintf(log_file, "%f,", elapsed * 1000);
  fprintf(log_file, "%d,", total_insertions);

  fprintf(log_file, "%f,", first_nn_dis);

  fprintf(log_file, "%f,", dataManager->get_nearest_dist_of_query(query_idx));
  fprintf(log_file, "%f,", dataManager->get_avg_dist_of_query(query_idx));
  fprintf(log_file, "%f,", dataManager->get_furthest_dist_of_query(query_idx));
  fprintf(log_file, "%f,", dataManager->get_variance_of_query(query_idx));
  fprintf(log_file, "%f,", dataManager->get_percentile_of_query(query_idx, 0.50));
  fprintf(log_file, "%f,", dataManager->get_percentile_of_query(query_idx, 0.25));
  fprintf(log_file, "%f,", dataManager->get_percentile_of_query(query_idx, 0.75));

  fprintf(log_file, "%d,", prediction_interval);
  fprintf(log_file, "%d,", query_predictor_calls);
  fprintf(log_file, "%f,", predictor_time * 1000);

  fprintf(log_file, "%f,", dataManager->get_RDE(query_idx));
  fprintf(log_file, "%f,", dataManager->get_TDR(query_idx));
  fprintf(log_file, "%f,", dataManager->get_NRS(query_idx));

  fprintf(log_file, "%f,", dataManager->get_recallk(query_idx));
  fprintf(log_file, "%f\n", last_predicted_recall);
}

}  // namespace faiss
