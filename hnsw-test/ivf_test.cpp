#include <faiss/AutoTune.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "ComUtil.h"
#include "VectorDataLoader.h"

typedef enum {
  TRAINING_DATA_GENERATION = 0,
  DARTH = 1,
  PLAIN_IVF = 2,
} running_mode_t;

char running_mode_str[6][50] = {
    "Training Data Generation",
    "DARTH",
    "Plain-IVF"};

int main(int argc, char** argv) {
  double t0 = elapsed();

  char *dataset, *output = NULL, *index_filepath = NULL, *predictor_model_path = NULL;

  float target_recall = 0.90;
  int initial_prediction_interval = 1000, min_prediction_interval = 100;

  int logging_interval = 10;

  bool save_index = false;
  running_mode_t mode;
  faiss::idx_t nQ = 100, k = 10;

  // int M = 16, efConstruction = 500, efSearch = 500;
  size_t nprobe = 100;
  int nlist = 1000;

  int fixed_amount_of_search = 200;
  float prediction_multiplier = 1.0;

  bool per_prediction_logging = false;
  bool verbose = false;

  char* dataset_dir_prefix = NULL;

  query_type_t query_type;

  char* noise_perc = "5";

  while (1) {
    static struct option long_options[] = {
        {"dataset", required_argument, 0, '0'},
        {"query-num", required_argument, 0, '1'},
        {"k", required_argument, 0, '2'},
        {"output", required_argument, 0, '3'},
        {"nprobe", required_argument, 0, '4'},
        {"nlist", required_argument, 0, '5'},
        //{"efSearch", required_argument, 0, '6'},
        {"index-filepath", required_argument, 0, '7'},
        {"save-index", no_argument, 0, '8'},
        {"mode", required_argument, 0, '9'},
        {"target-recall", required_argument, 0, 'a'},
        {"initial-prediction-interval", required_argument, 0, 'b'},
        {"predictor-model-path", required_argument, 0, 'c'},
        {"logging-interval", required_argument, 0, 'd'},
        {"fixed-amount-of-search", required_argument, 0, 'e'},
        {"prediction-multiplier", required_argument, 0, 'f'},
        {"min-prediction-interval", required_argument, 0, 'g'},
        {"per-prediction-logging", no_argument, 0, 'h'},
        {"verbose", no_argument, 0, 'i'},
        {"query-type", required_argument, 0, 'j'},
        {"dataset-dir-prefix", required_argument, 0, 'k'},
        {"gnoise-perc", required_argument, 0, 'l'},
        {NULL, 0, NULL, 0}};

    int option_index = 0;
    int c = getopt_long(argc, argv, "", long_options, &option_index);
    if (c == -1)
      break;

    switch (c) {
      case '0':
        dataset = optarg;
        break;
      case '1':
        sscanf(optarg, "%ld", &nQ);
        break;
      case '2':
        sscanf(optarg, "%ld", &k);
        break;
      case '3':
        output = optarg;
        break;
      case '4':
        sscanf(optarg, "%ld", &nprobe);
        break;
      case '5':
        sscanf(optarg, "%d", &nlist);
        break;
      case '6':
        // empty
        break;
      case '7':
        index_filepath = optarg;
        break;
      case '8':
        save_index = true;
        break;
      case '9':
        if (strcmp(optarg, "training-data-generation") == 0) {
          mode = TRAINING_DATA_GENERATION;
        } else if (strcmp(optarg, "darth") == 0) {
          mode = DARTH;
        } else if (strcmp(optarg, "plain-ivf") == 0) {
          mode = PLAIN_IVF;
        } else {
          printf("Unknown running mode: %s\n", optarg);
          exit(EXIT_FAILURE);
        }
        break;
      case 'a':
        sscanf(optarg, "%f", &target_recall);
        break;
      case 'b':
        sscanf(optarg, "%d", &initial_prediction_interval);
        break;
      case 'c':
        predictor_model_path = optarg;
        break;
      case 'd':
        sscanf(optarg, "%d", &logging_interval);
        break;
      case 'e':
        sscanf(optarg, "%d", &fixed_amount_of_search);
        break;
      case 'f':
        sscanf(optarg, "%f", &prediction_multiplier);
        break;
      case 'g':
        sscanf(optarg, "%d", &min_prediction_interval);
        break;
      case 'h':
        per_prediction_logging = true;
        break;
      case 'i':
        verbose = true;
        break;
      case 'j':
        if (strcmp(optarg, "training") == 0) {
          query_type = TRAINING;
        } else if (strcmp(optarg, "validation") == 0) {
          query_type = VALIDATION;
        } else if (strcmp(optarg, "testing") == 0) {
          query_type = TESTING;
        } else if (strcmp(optarg, "noisy-testing") == 0) {
          query_type = NOISY_TESTING;
        } else {
          printf("Unknown query type: %s\n", optarg);
          exit(EXIT_FAILURE);
        }
        break;
      case 'k':
        dataset_dir_prefix = optarg;
        break;
      case 'l':
        // sscanf(optarg, "%f", &gnoise);
        noise_perc = optarg;
        break;
      default:
        printf("Unknown option: %c\n", c);
        exit(EXIT_FAILURE);
    }
  }

  if (initial_prediction_interval < min_prediction_interval) {
    printf(
        "Initial prediction interval should be greater than or equal to "
        "min prediction interval\n");
    exit(EXIT_FAILURE);
  }

  // print a log with the parameters
  printf(">>Parameters:\n");
  printf("   dataset: %s\n", dataset);
  printf("   nQ: %ld\n", nQ);
  printf("   output: %s\n", output);
  printf("   nlist: %d, nprobe: %ld, k: %ld\n", nlist, nprobe, k);
  printf("   index_filepath: %s\n", index_filepath);
  printf("   save_index: %s\n", save_index == true ? "yes" : "no");
  printf("   mode: %s\n", running_mode_str[mode]);
  printf("   target_recall: %f\n", target_recall);
  printf("   logging_interval: %d\n", logging_interval);
  printf("   initial_prediction_interval: %d, min_prediction_interval: %d\n", initial_prediction_interval, min_prediction_interval);
  printf("   predictor_model_path: %s\n", predictor_model_path);
  printf("   per_prediction_logging: %s\n", per_prediction_logging == true ? "yes" : "no");
  printf("   query_type: %s\n", query_type_str[query_type]);
  printf("   dataset_dir_prefix: %s\n", dataset_dir_prefix);
  printf("   verbose: %s\n", verbose == true ? "yes" : "no");
  printf("   gnoise perc: %s\n", noise_perc);
  printf("   [LAET] fixed_amount_of_search: %d, prediction_multiplier: %f\n", fixed_amount_of_search, prediction_multiplier);

  VectorDataLoader vector_dataloader(dataset, query_type, noise_perc, dataset_dir_prefix);
  vector_dataloader.initializeDataMaps();

  // Load database vectors
  size_t d, n;
  float* vecsDB = vector_dataloader.loadDB(&d, &n);
  printf(">>%s DB loaded: d = %ld, n = %ld. Elapsed = %.3fs\n",
         dataset,
         d,
         n,
         (elapsed() - t0));

  // Load query vectors
  size_t dQ, nQ_all;
  float* vecsQ_all = vector_dataloader.loadQueries(&dQ, &nQ_all);
  assert(dQ == d);

  // Load ground-truth results
  size_t nQ2, k_all;
  int* gt_int_all = vector_dataloader.loadQueriesGroundtruths(&k_all, &nQ2);
  assert(nQ_all == nQ2);

  // Load ground-truth distances
  size_t nQ3, k_all2;
  float* gt_dist_all = vector_dataloader.loadQueriesGroundtruthDistances(&k_all2, &nQ3);
  assert(nQ_all == nQ3);

  faiss::idx_t* gt_all = new faiss::idx_t[k_all * nQ_all];
  for (faiss::idx_t i = 0; i < k_all * nQ_all; i++) {
    gt_all[i] = static_cast<faiss::idx_t>(gt_int_all[i]);
  }
  delete[] gt_int_all;

  // Sample some queries
  float* vecsQ;
  faiss::idx_t *gt_k_all, *indicesQ;
  float* gt_k_dist_all;
  get_queries(
      vecsQ_all,
      nQ_all,
      d,
      gt_all,
      gt_dist_all,
      k_all,
      nQ,
      &vecsQ,
      &indicesQ,
      &gt_k_all,
      &gt_k_dist_all,
      false);

  printf(">>k=%ld, k_all=%ld\n", k, k_all);
  assert(k_all >= k);
  faiss::idx_t* gt = new faiss::idx_t[k * nQ];
  float* gt_dist = new float[k * nQ];
  for (faiss::idx_t i = 0; i < nQ; i++) {
    for (faiss::idx_t j = 0; j < k; j++) {
      gt[i * k + j] = gt_k_all[i * k_all + j];
      gt_dist[i * k + j] = gt_k_dist_all[i * k_all + j];
    }
  }

  printf(">>Query Vectors and GT loaded: d = %ld, nQ = %ld, k = %ld, Elapsed = %.3fs\n",
         dQ,
         nQ,
         k,
         (elapsed() - t0));

  if (nQ > 10) {
    printf(">>First 30 query IDs: ");
    for (int i = 0; i < 30; i++) {
      printf("%ld ", indicesQ[i]);
    }
    printf("\n");
  }

  double index_build_start = elapsed();

  faiss::Index* quantizer = NULL;
  if (false && n >= 100000000) {
    int hnsw_m = 32;
    quantizer = new faiss::IndexHNSWFlat(d, hnsw_m);
    printf(">>HNSW quantizer with M=%d quantizer is selected.\n", hnsw_m);
  } else {
    quantizer = new faiss::IndexFlatL2(d);
    printf(">>Coarse quantizer is selected.\n");
  }

  faiss::IndexIVFFlat* index;
  if (save_index) {
    index = new faiss::IndexIVFFlat(quantizer, d, nlist, faiss::METRIC_L2);

    index->train(n, vecsDB);
    printf(">> Index trained. Proceeding to add vectors...\n");

    // todo: implement some batching to monitor the progress.
    size_t batch_size = 10000000;
    size_t num_batches = n / batch_size;
    for (size_t i = 0; i < num_batches; i++) {
      size_t start = i * batch_size;
      size_t end = (i + 1) * batch_size;
      if (end > n) {
        end = n;
      }
      printf(">> Adding vectors from %ld to %ld, %ld in total...\n", start, end, end - start);
      index->add(end - start, vecsDB + start * d);
    }
    index->nprobe = nprobe;
    faiss::write_index(index, index_filepath);
    printf(">>Index saved to %s\n", index_filepath);
  } else {
    if (index_filepath) {
      index = dynamic_cast<faiss::IndexIVFFlat*>(faiss::read_index(index_filepath));
      index->nprobe = nprobe;
      printf(">>Index loaded from %s\n", index_filepath);
    } else {
      exit(EXIT_FAILURE);
      /*index = new faiss::IndexIVFFlat(quantizer, d, nlist, faiss::METRIC_L2);
      index->train(n, vecsDB);
      index->add(n, vecsDB);
      index->nprobe = nprobe;*/
    }
  }

  double index_build_time = elapsed() - index_build_start;
  printf(">>Index build/load time: %.3fs\n", index_build_time);

  faiss::idx_t* I = new faiss::idx_t[nQ * k];
  float* D = new float[nQ * k];

  double search_start_time = elapsed();

  faiss::DeclarativeRecallDataManager data_manager(D, I, gt, gt_dist, nQ, d, k, vecsQ, output, vecsDB, n, gt_k_all, k_all);

  switch (mode) {
    case TRAINING_DATA_GENERATION: {
      faiss::DeclarativeRecallDataCollectorIVF data_collector(data_manager, logging_interval);
      index->search_generate_training_data(nQ, vecsQ, k, D, I, data_collector);
      break;
    }
    case PLAIN_IVF: {
      faiss::DeclarativeRecallDataCollectorIVF data_collector(data_manager, logging_interval);
      index->search_plainIVF(nQ, vecsQ, k, D, I, data_collector);
      break;
    }
    case DARTH: {
      faiss::DARTHPredictorIVF darth_predictor(
          data_manager,
          target_recall,
          initial_prediction_interval,
          min_prediction_interval,
          per_prediction_logging,
          predictor_model_path,
          logging_interval);
      index->search_DARTH(nQ, vecsQ, k, D, I, darth_predictor);
      break;
    }
  }

  double search_time = elapsed() - search_start_time;

  printf("\n\nIndex[nlist=%d, nprobes=%ld]IndexTime: %lfs, SearchTime: %lfs, TotalTime: %lfs, Recall@%ld: %.4f\n",
         nlist,
         index->nprobe,
         index_build_time,
         search_time,
         (elapsed() - t0),
         k,
         recall_at_k_avg(gt, I, D, k, nQ, gt_k_all, k_all, verbose));

  delete[] gt_k_all;
  delete[] gt_all;
  delete[] vecsDB;
  delete[] vecsQ;
  delete[] gt;
  delete[] I;
  delete[] D;
  delete index;
  delete quantizer;

  return 0;
}
