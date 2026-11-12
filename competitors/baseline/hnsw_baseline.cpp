#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <omp.h>
#include <cstdio>
#include <cstdlib>

#include <common/ComUtil.h>
#include <common/VectorDataLoader.h>

enum running_mode_t { NO_EARLY_TERMINATION_TESTING, BASELINE_TESTING, OPTIMAL_RESULTS_GENERATION };
int main(int argc, char** argv) try {
  double t0=elapsed();
  const char *dataset=nullptr, *index_filepath=nullptr, *dataset_dir_prefix=nullptr, *noise_perc="5";
  char* output=nullptr;
  float target_recall=.8; bool target_given=false, save_index=false, verbose=false;
  faiss::idx_t nQ=100, k=10;
  running_mode_t mode=static_cast<running_mode_t>(-1);
  query_type_t query_type=static_cast<query_type_t>(-1);
  int M=16, efConstruction=500, efSearch=500;
  const char* metric_str="L2";
  int initial_prediction_interval=1000;

  static struct option long_options[] = {
      {"dataset", required_argument, nullptr, 'd'},
      {"query-num", required_argument, nullptr, 'q'},
      {"k", required_argument, nullptr, 'k'},
      {"output", required_argument, nullptr, 'o'},
      {"index-filepath", required_argument, nullptr, 'i'},
      {"mode", required_argument, nullptr, 'm'},
      {"target-recall", required_argument, nullptr, 'r'},
      {"query-type", required_argument, nullptr, 'y'},
      {"dataset-dir-prefix", required_argument, nullptr, 'p'},
      {"gnoise-perc", required_argument, nullptr, 'g'},
      {"threads", required_argument, nullptr, 't'},
      {"M", required_argument, nullptr, 'M'},
      {"efConstruction", required_argument, nullptr, 'C'},
      {"efSearch", required_argument, nullptr, 'S'},
      {"metric", required_argument, nullptr, 'x'},
      {"initial-prediction-interval", required_argument, nullptr, 'b'},
      {"save-index", no_argument, nullptr, 's'},
      {"verbose", no_argument, nullptr, 'v'},
      {"help", no_argument, nullptr, 'h'},
      {nullptr, 0, nullptr, 0}};
  int option;
  while((option=getopt_long(argc,argv,"",long_options,nullptr))!=-1) {
    switch(option) {
      case 'd': dataset=optarg; break;
      case 'q': nQ=parse_numeric_option<decltype(nQ)>(optarg, "--query-num"); break;
      case 'k': k=parse_numeric_option<decltype(k)>(optarg, "--k"); break;
      case 'o': output=optarg; break;
      case 'i': index_filepath=optarg; break;
      case 'r': target_recall=parse_numeric_option<decltype(target_recall)>(optarg, "--target-recall"); target_given=true; break;
      case 'p': dataset_dir_prefix=optarg; break;
      case 'g': noise_perc=optarg; break;
      case 'M': M=parse_numeric_option<decltype(M)>(optarg, "--M"); break;
      case 'C': efConstruction=parse_numeric_option<decltype(efConstruction)>(optarg, "--efConstruction"); break;
      case 'S': efSearch=parse_numeric_option<decltype(efSearch)>(optarg, "--efSearch"); break;
      case 'x': metric_str=optarg; break;
      case 'b': initial_prediction_interval=parse_numeric_option<decltype(initial_prediction_interval)>(optarg, "--initial-prediction-interval"); break;
      case 't': { int threads=parse_numeric_option<int>(optarg,"--threads");
        if(threads<=0) throw std::invalid_argument("threads must be positive"); omp_set_num_threads(threads); break; }
      case 'y':
        if(!strcmp(optarg,"training"))query_type=TRAINING;
        else if(!strcmp(optarg,"validation"))query_type=VALIDATION;
        else if(!strcmp(optarg,"testing"))query_type=TESTING;
        else if(!strcmp(optarg,"noisy-testing"))query_type=NOISY_TESTING;
        else if(!strcmp(optarg,"calibration"))query_type=CALIBRATION;
        else throw std::invalid_argument("Unknown query type"); break;
      case 'm':
        if(!strcmp(optarg,"no-early-stop"))mode=NO_EARLY_TERMINATION_TESTING;
        else if(!strcmp(optarg,"naive-early-stop-testing"))mode=BASELINE_TESTING;
        else if(!strcmp(optarg,"optimal-early-stop-testing"))mode=OPTIMAL_RESULTS_GENERATION;
        else throw std::invalid_argument("Unknown HNSW baseline mode"); break;
      case 's': save_index=true; break;
      case 'v': verbose=true; break;
      case 'h': printf("Baseline search: --dataset NAME --dataset-dir-prefix DIR --query-type TYPE --mode MODE [options]\n"); return 0;
      default: throw std::invalid_argument("Unknown option");
    }
  }
  if(!dataset || int(mode)<0 || int(query_type)<0 || optind!=argc)
    throw std::invalid_argument("--dataset, --mode and --query-type are required");
  if(nQ<=0 || k<=0 || target_recall<=0 || target_recall>1)
    throw std::invalid_argument("Invalid query count, k or target recall");
  if(save_index && !index_filepath) throw std::invalid_argument("--save-index requires --index-filepath");
  if(M<=0 || efConstruction<=0 || efSearch<=0 || initial_prediction_interval<=0) throw std::invalid_argument("Invalid HNSW search settings");
  if (!dataset_dir_prefix) dataset_dir_prefix = std::getenv("DATASET_DIRECTORY");
  if (!dataset_dir_prefix || !*dataset_dir_prefix) dataset_dir_prefix = std::getenv("DARTH_DATASET_DIR");
  if (!dataset_dir_prefix || !*dataset_dir_prefix)
    throw std::invalid_argument("Specify --dataset-dir-prefix, DATASET_DIRECTORY or DARTH_DATASET_DIR");
  auto vector_dataloader = std::make_unique<VectorDataLoader>(dataset, query_type, noise_perc, dataset_dir_prefix, metric_str);
  vector_dataloader->initializeDataMaps();

  // Load database vectors
  size_t d, n;
  std::unique_ptr<float[]> vecsDB_owner(vector_dataloader->loadDB(&d, &n));
  auto* vecsDB = vecsDB_owner.get();
  printf(">>%s DB loaded: d = %ld, n = %ld. Elapsed = %.3fs\n", dataset, d, n, (elapsed() - t0));

  // Load query vectors
  size_t dQ, nQ_all;
  std::unique_ptr<float[]> vecsQ_all_owner(vector_dataloader->loadQueries(&dQ, &nQ_all));
  auto* vecsQ_all = vecsQ_all_owner.get();
  if (dQ != d) throw std::invalid_argument("Database and query dimensions differ");
  if (nQ > nQ_all) throw std::invalid_argument("Requested query count exceeds available queries");
  if (k > n) throw std::invalid_argument("k exceeds database size");

  // Load ground-truth results
  size_t nQ2, k_all;
  std::unique_ptr<int[]> gt_int_all_owner(vector_dataloader->loadQueriesGroundtruths(&k_all, &nQ2));
  auto* gt_int_all = gt_int_all_owner.get();
  if (nQ_all != nQ2) throw std::invalid_argument("Query and ground-truth row counts differ");
  if (k_all < k) throw std::invalid_argument("k exceeds ground-truth width");

  // Load ground-truth distances
  size_t nQ3, k_all2;
  std::unique_ptr<float[]> gt_dist_all_owner(vector_dataloader->loadQueriesGroundtruthDistances(&k_all2, &nQ3));
  auto* gt_dist_all = gt_dist_all_owner.get();
  if (nQ_all != nQ3 || k_all != k_all2)
    throw std::invalid_argument("Ground-truth IDs and distances have different shapes");

  std::unique_ptr<faiss::idx_t[]> gt_all_owner(new faiss::idx_t[k_all * nQ_all]);
  auto* gt_all = gt_all_owner.get();
  for (faiss::idx_t i = 0; i < k_all * nQ_all; i++) {
    gt_all[i] = static_cast<faiss::idx_t>(gt_int_all[i]);
  }
  gt_int_all_owner.reset();

  float* vecsQ;
  faiss::idx_t *gt_k_all, *indicesQ;
  float* gt_k_dist_all;
  get_queries(vecsQ_all, nQ_all, d, gt_all, gt_dist_all, k_all, nQ, &vecsQ, &indicesQ, &gt_k_all, &gt_k_dist_all, false);
  std::unique_ptr<float[]> vecsQ_owner(vecsQ);
  std::unique_ptr<faiss::idx_t[]> indicesQ_owner(indicesQ);
  std::unique_ptr<faiss::idx_t[]> gt_k_all_owner(gt_k_all);
  std::unique_ptr<float[]> gt_k_dist_all_owner(gt_k_dist_all);


  printf(">>k=%ld, k_all=%ld\n", k, k_all);

  std::unique_ptr<faiss::idx_t[]> gt_owner(new faiss::idx_t[k * nQ]);
  auto* gt = gt_owner.get();
  std::unique_ptr<float[]> gt_dist_owner(new float[k * nQ]);
  auto* gt_dist = gt_dist_owner.get();
  for (faiss::idx_t i = 0; i < nQ; i++) {
    for (faiss::idx_t j = 0; j < k; j++) {
      gt[i * k + j] = gt_k_all[i * k_all + j];
      gt_dist[i * k + j] = gt_k_dist_all[i * k_all + j];
    }
  }

  printf(">>Query Vectors and GT loaded: d = %ld, nQ = %ld, k = %ld, Elapsed = %.3fs\n", dQ, nQ, k, (elapsed() - t0));

  // Convert metric string to enum
  faiss::MetricType metric_type;
  if (strcmp(metric_str, "L2") == 0) {
    metric_type = faiss::METRIC_L2;
  } else if (strcmp(metric_str, "inner-product") == 0) {
    metric_type = faiss::METRIC_INNER_PRODUCT;
  } else {
    printf("Unknown metric: %s. Supported metrics: L2, inner-product\n", metric_str);
    throw std::runtime_error("Invalid runner configuration");
  }

  double index_build_start = elapsed();
  std::unique_ptr<faiss::Index> index_owner;
  faiss::IndexHNSWFlat* index = nullptr;

  if (save_index || (index_filepath && !index_file_exists(index_filepath))) {
    if (index_filepath == NULL) {
      printf(">>Index file path is required to save the index\n");
      throw std::runtime_error("Invalid runner configuration");
    }

    index = new faiss::IndexHNSWFlat(d, M, metric_type);
    index_owner.reset(index);
    index->hnsw.efConstruction = efConstruction;
    index->hnsw.efSearch = efSearch;
    index->add(n, vecsDB);
    if (save_index) {
      prepare_index_parent(index_filepath);
      faiss::write_index(index, index_filepath);
    } else {
      save_new_index(index, index_filepath);
    }

    printf(">>Index saved to %s\n", index_filepath ? index_filepath : "(none)");
  } else {
    if (index_filepath) {
      index_owner.reset(faiss::read_index(index_filepath));
      faiss::Index* loaded_index = index_owner.get();
      index = dynamic_cast<faiss::IndexHNSWFlat*>(loaded_index);
      if (!index) {
        fprintf(stderr, "Index at %s is not an IndexHNSWFlat\n", index_filepath ? index_filepath : "(none)");
        throw std::runtime_error("Invalid runner configuration");
      }
      index->hnsw.efSearch = efSearch;
      printf(">>Index loaded from %s\n", index_filepath ? index_filepath : "(none)");
    } else {
      index = new faiss::IndexHNSWFlat(d, M, metric_type);
      index_owner.reset(index);
      index->hnsw.efConstruction = efConstruction;
      index->hnsw.efSearch = efSearch;
      index->add(n, vecsDB);
    }
  }

  if (index->d != d || index->ntotal != n || index->metric_type != metric_type)
    throw std::invalid_argument("Loaded index dimensions, size or metric do not match the dataset");

  double index_build_time = elapsed() - index_build_start;
  printf(">>Index build time: %.3fs\n", index_build_time);

  std::unique_ptr<faiss::idx_t[]> I_owner(new faiss::idx_t[nQ * k]);
  auto* I = I_owner.get();
  std::unique_ptr<float[]> D_owner(new float[nQ * k]);
  auto* D = D_owner.get();

  // Perform the search
  double search_start_time = elapsed();

  auto data_manager = std::make_unique<faiss::DeclarativeRecallDataManager>(D, I, gt, gt_dist, nQ, d, k, vecsQ, output, vecsDB, n, gt_k_all, k_all);
  switch (mode) {
    case NO_EARLY_TERMINATION_TESTING: {
      auto searchMonitor = std::make_unique<faiss::DeclarativeRecallDataCollectorHNSW>(data_manager.get(), INTERVAL_DISABLED_VALUE);
      searchMonitor->init_log_file();
      index->search_baseline(nQ, vecsQ, k, D, I, searchMonitor.get());
      searchMonitor->close_log_file();
      break;
    }
    case BASELINE_TESTING: {
      int distances_to_stop_at = initial_prediction_interval;
      auto search_monitor = std::make_unique<faiss::DeclarativeRecallDataCollectorHNSW>(data_manager.get(), INTERVAL_DISABLED_VALUE, distances_to_stop_at);
      search_monitor->init_log_file();
      index->search_baseline(nQ, vecsQ, k, D, I, search_monitor.get());
      search_monitor->close_log_file();
      break;
    }
    case OPTIMAL_RESULTS_GENERATION: {
      auto search_monitor = std::make_unique<faiss::DeclarativeRecallDataCollectorHNSW>(data_manager.get(), INTERVAL_WHEN_RECALL_UPDATES_VALUE);
      search_monitor->init_log_file();
      index->search_baseline(nQ, vecsQ, k, D, I, search_monitor.get());
      search_monitor->close_log_file();
      break;
    }
    default:
      throw std::invalid_argument("Unsupported baseline mode");
  }

  double search_time = elapsed() - search_start_time;

  printf(
      "\n\nIndex[M=%d, efC=%d, efS=%d]IndexTime: %lfs, SearchTime: %lfs, "
      "TotalTime: %lfs, Recall@%ld: %.4f, RQUT@%ld: %.4f\n",
      M, index->hnsw.efConstruction, index->hnsw.efSearch, index_build_time,
      search_time, (elapsed() - t0), k,
      recall_at_k_avg(gt, I, D, k, nQ, gt_k_all, k_all, verbose),
      k,
      get_rqut_at_k(gt, I, D, k, nQ, gt_k_all, k_all, target_recall));

  if (!target_given) {
    for (double target : {0.8, 0.85, 0.90, 0.95, 0.99})
      printf("RQUT target=%.2f value=%.4f\n", target,
             get_rqut_at_k(gt, I, D, k, nQ, gt_k_all, k_all, target));
  }
  return 0;
} catch (const std::exception& error) {
  fprintf(stderr, "Baseline runner: %s\n", error.what());
  return EXIT_FAILURE;
}
