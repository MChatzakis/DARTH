#include "LAETPredictor.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace faiss {

// LAETPredictorHNSW
LAETPredictorHNSW::LAETPredictorHNSW(
    DeclarativeRecallDataManager* data_manager,
    int fixed_amount_of_distance_calcs,
    float prediction_multiplier,
    char* predictor_model_path)
    : data_manager(data_manager),
      fixed_amount_of_distance_calcs(fixed_amount_of_distance_calcs),
      prediction_multiplier(prediction_multiplier) {
  predictor = std::make_unique<LGBMPredictor>(predictor_model_path);
}

void LAETPredictorHNSW::init_log_file() {
  if (!data_manager->log_filename) {
    return;
  }

  close_log_file();
  log_file = fopen(data_manager->log_filename, "w");
  if (!log_file) {
    // printf("Error opening recall data log file\n");
    perror("Error opening recall data log file");
    throw std::runtime_error("DARTH log or prediction failure");
  }

  fprintf(log_file, "qid,");
  fprintf(log_file, "dists,");
  fprintf(log_file, "elaps_ms,");

  fprintf(log_file, "predictor_time,");
  fprintf(log_file, "predicted_distance_calcs,");

  // Quality approximation measures
  fprintf(log_file, "RDE,");  // relative distance error
  fprintf(log_file, "TDR,");  // total distance ratio
  fprintf(log_file, "NRS,");  // normalized rank sum

  fprintf(log_file, "r\n");
}

void LAETPredictorHNSW::close_log_file() {
  if (log_file) {
    fclose(log_file);
    log_file = nullptr;
  }
}

int LAETPredictorHNSW::predict_distance_calcs(
    idx_t query_idx,
    int nstep,
    int ndis,
    double elapsed,
    float first_nn_dis,
    double* predictor_time) {
  double predictor_start_time = data_manager->elapsed_secs();

  float d_start = first_nn_dis;
  float d_1st = data_manager->get_nearest_dist_of_query(query_idx);
  float d_10th = -1;
  if (data_manager->k >= 10) {
    d_10th = data_manager->get_kth_nearest_dist_of_query(query_idx, 9);
  }

  float d_1st_to_start = -1;
  if (d_start > 0) {
    d_1st_to_start = d_1st / d_start;
  }

  float d_10th_to_start = -1;
  if (d_10th != -1 && d_start > 0) {
    d_10th_to_start = d_10th / d_start;
  }

  int num_nonquery_feats = 5;
  int query_feats = data_manager->d;
  int total_feats = num_nonquery_feats + query_feats;
  std::vector<double> feats(static_cast<size_t>(total_feats));

  feats[0] = d_start;
  feats[1] = d_1st;
  feats[2] = d_10th;
  feats[3] = d_1st_to_start;
  feats[4] = d_10th_to_start;

  for (int i = 0; i < data_manager->d; i++) {
    feats[static_cast<size_t>(num_nonquery_feats + i)] =
        data_manager->queries[query_idx * data_manager->d + i];
  }

  int num_rows = 1;
  int num_feats = total_feats;

  int predicted_distance_calcs = static_cast<int>(predictor->predict(feats.data(), num_feats));

  predicted_distance_calcs = static_cast<int>(std::round(prediction_multiplier * predicted_distance_calcs));

  double predictor_end_time = data_manager->elapsed_secs();
  *predictor_time += predictor_end_time - predictor_start_time;

  if (per_prediction_logging && log_file) {
    std::lock_guard<std::mutex> row_lock(log_mutex);
    fprintf(log_file, "%ld,", query_idx);
    fprintf(log_file, "%d,", ndis);
    fprintf(log_file, "%f,", elapsed * 1000);
    fprintf(log_file, "%f,", *predictor_time * 1000);
    fprintf(log_file, "%d,", predicted_distance_calcs);
    fprintf(log_file, "%f,", data_manager->get_RDE(query_idx));
    fprintf(log_file, "%f,", data_manager->get_TDR(query_idx));
    fprintf(log_file, "%f,", data_manager->get_NRS(query_idx));
    fprintf(log_file, "%f\n", data_manager->get_recallk(query_idx));
  }

  return predicted_distance_calcs;
}

void LAETPredictorHNSW::log_final_result(
    idx_t query_idx,
    int nstep,
    int ndis,
    double elapsed,
    float first_nn_dis,
    int predicted_distance_calcs,
    double predictor_time) {
  if (!log_file || per_prediction_logging) {
    return;
  }

  std::lock_guard<std::mutex> row_lock(log_mutex);
  fprintf(log_file, "%ld,", query_idx);
  fprintf(log_file, "%d,", ndis);
  fprintf(log_file, "%f,", elapsed * 1000);
  fprintf(log_file, "%f,", predictor_time * 1000);
  fprintf(log_file, "%d,", predicted_distance_calcs);
  fprintf(log_file, "%f,", data_manager->get_RDE(query_idx));
  fprintf(log_file, "%f,", data_manager->get_TDR(query_idx));
  fprintf(log_file, "%f,", data_manager->get_NRS(query_idx));
  fprintf(log_file, "%f\n", data_manager->get_recallk(query_idx));
}


} // namespace faiss
