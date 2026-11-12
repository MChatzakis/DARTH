#pragma once

// LAET's predictor types are included after the shared data manager is defined.
#include <faiss/impl/DeclarativeRecall.h>

namespace faiss {

typedef struct LAETPredictorHNSW {
  LAETPredictorHNSW(const LAETPredictorHNSW&) = delete;
  LAETPredictorHNSW& operator=(const LAETPredictorHNSW&) = delete;
  ~LAETPredictorHNSW() { close_log_file(); }
  std::mutex log_mutex;

  DeclarativeRecallDataManager* data_manager = nullptr;
  int fixed_amount_of_distance_calcs = 100;
  float prediction_multiplier = 1;
  FILE* log_file = nullptr;
  bool per_prediction_logging = false;

  std::unique_ptr<LGBMPredictor> predictor;

  LAETPredictorHNSW(
      DeclarativeRecallDataManager* data_manager,
      int fixed_amount_of_distance_calcs,
      float prediction_multiplier,
      char* predictor_model_path);

  LAETPredictorHNSW(DeclarativeRecallDataManager* manager, int threshold, float multiplier,
                    std::unique_ptr<LGBMPredictor> model)
      : data_manager(manager), fixed_amount_of_distance_calcs(threshold),
        prediction_multiplier(multiplier), predictor(std::move(model)) {
    if (!predictor || threshold <= 0 || multiplier <= 0) throw std::invalid_argument("Invalid LAET predictor");
  }

  void init_log_file();

  void close_log_file();

  int predict_distance_calcs(
      idx_t query_idx,
      int nstep,
      int ndis,
      double elapsed,
      float first_nn_dis,
      double* predictor_time);

  void log_final_result(
      idx_t query_idx,
      int nstep,
      int ndis,
      double elapsed,
      float first_nn_dis,
      int predicted_distance_calcs,
      double predictor_time);

} LAETPredictorHNSW;

typedef struct LAETPredictorIVF {
  LAETPredictorIVF(const LAETPredictorIVF&) = delete;
  LAETPredictorIVF& operator=(const LAETPredictorIVF&) = delete;
  ~LAETPredictorIVF() { close_log_file(); }
  std::mutex log_mutex;

  DeclarativeRecallDataManager* data_manager = nullptr;
  int fixed_amount_of_distance_calcs = 100;
  float prediction_multiplier = 1;
  FILE* log_file = nullptr;
  bool per_prediction_logging = false;

  std::unique_ptr<LGBMPredictor> predictor;

  LAETPredictorIVF(
      DeclarativeRecallDataManager* data_manager,
      int fixed_amount_of_distance_calcs,
      float prediction_multiplier,
      char* predictor_model_path) {
    this->data_manager = data_manager;
    this->fixed_amount_of_distance_calcs = fixed_amount_of_distance_calcs;
    this->prediction_multiplier = prediction_multiplier;

    predictor = std::make_unique<LGBMPredictor>(predictor_model_path);
  }

  LAETPredictorIVF(DeclarativeRecallDataManager* manager, int threshold, float multiplier,
                   std::unique_ptr<LGBMPredictor> model)
      : data_manager(manager), fixed_amount_of_distance_calcs(threshold),
        prediction_multiplier(multiplier), predictor(std::move(model)) {
    if (!predictor || threshold <= 0 || multiplier <= 0) throw std::invalid_argument("Invalid LAET predictor");
  }

  void init_log_file() {
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

    fprintf(log_file, "RDE,");  // relative distance error
    fprintf(log_file, "TDR,");  // total distance ratio
    fprintf(log_file, "NRS,");  // normalized rank sum

    fprintf(log_file, "r\n");
  }

  void close_log_file() {
    if (log_file) {
      fclose(log_file);
      log_file = nullptr;
    }
  }

  int predict_distance_calcs(
      idx_t query_idx,
      int nstep,
      int ndis,
      double elapsed,
      float first_nn_dis,
      std::vector<float> centroid_distances,
      double* predictor_time) {
    double predictor_start_time = data_manager->elapsed_secs();

    std::vector<double> feats = std::vector<double>();

    float nn_dist = data_manager->get_nearest_dist_of_query(query_idx);
    float dist_10 = -1;
    float dist_1st_to_10 = -1;
    float dist_1st_to_first_centroid = nn_dist / centroid_distances[0];

    if (data_manager->k >= 10) {
      dist_10 = data_manager->get_kth_nearest_dist_of_query(query_idx, 9);
      dist_1st_to_10 = nn_dist / dist_10;
    }

    feats.push_back(nn_dist);
    feats.push_back(dist_10);
    feats.push_back(dist_1st_to_10);
    feats.push_back(dist_1st_to_first_centroid);

    for (size_t i = 9; i <= 100; i += 10) {
      float dist_1st_to_i = -1;
      if (centroid_distances.size() > i) {
        dist_1st_to_i = nn_dist / centroid_distances[i];
      }
      feats.push_back(dist_1st_to_i);
    }

    for (int i = 0; i < data_manager->d; i++) {
      feats.push_back((double)data_manager->queries[query_idx * data_manager->d + i]);
    }

    int num_rows = 1;
    int num_feats = feats.size();

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

  void log_final_result(
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

} LAETPredictorIVF;

} // namespace faiss
