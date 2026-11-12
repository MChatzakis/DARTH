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

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <omp.h>
#include <cstdio>
#include <cstdlib>

#include <common/ComUtil.h>
#include <common/VectorDataLoader.h>

enum running_mode_t { PLAIN_IVF };
int main(int argc, char** argv) try {
  double t0=elapsed();
  const char *dataset=nullptr, *index_filepath=nullptr, *dataset_dir_prefix=nullptr, *noise_perc="5";
  char* output=nullptr;
  float target_recall=.8; bool target_given=false, save_index=false, verbose=false;
  faiss::idx_t nQ=100, k=10;
  running_mode_t mode=static_cast<running_mode_t>(-1);
  query_type_t query_type=static_cast<query_type_t>(-1);
  size_t nprobe=100; int nlist=1000, logging_interval=10;

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
      {"nlist", required_argument, nullptr, 'L'},
      {"nprobe", required_argument, nullptr, 'P'},
      {"logging-interval", required_argument, nullptr, 'l'},
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
      case 'L': nlist=parse_numeric_option<decltype(nlist)>(optarg, "--nlist"); break;
      case 'P': nprobe=parse_numeric_option<decltype(nprobe)>(optarg, "--nprobe"); break;
      case 'l': logging_interval=parse_numeric_option<decltype(logging_interval)>(optarg, "--logging-interval"); break;
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
        if(!strcmp(optarg,"plain-ivf"))mode=PLAIN_IVF;
        else throw std::invalid_argument("Unknown IVF baseline mode"); break;
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
  if(nlist<=0 || nprobe==0 || logging_interval < INTERVAL_DISABLED_VALUE) throw std::invalid_argument("Invalid IVF search settings");
  if (!dataset_dir_prefix) dataset_dir_prefix = std::getenv("DATASET_DIRECTORY");
  if (!dataset_dir_prefix || !*dataset_dir_prefix) dataset_dir_prefix = std::getenv("DARTH_DATASET_DIR");
  if (!dataset_dir_prefix || !*dataset_dir_prefix)
    throw std::invalid_argument("Specify --dataset-dir-prefix, DATASET_DIRECTORY or DARTH_DATASET_DIR");
  auto vector_dataloader = std::make_unique<VectorDataLoader>(dataset, query_type, noise_perc, dataset_dir_prefix);
  vector_dataloader->initializeDataMaps();

  // Load database vectors
  size_t d, n;
  std::unique_ptr<float[]> vecsDB_owner(vector_dataloader->loadDB(&d, &n));
  auto* vecsDB = vecsDB_owner.get();
  printf(">>%s DB loaded: d = %ld, n = %ld. Elapsed = %.3fs\n",
         dataset,
         d,
         n,
         (elapsed() - t0));

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

  printf(">>Query Vectors and GT loaded: d = %ld, nQ = %ld, k = %ld, Elapsed = %.3fs\n",
         dQ,
         nQ,
         k,
         (elapsed() - t0));

  if (nQ > 10) {
    const auto display_count = std::min<faiss::idx_t>(30, nQ);
    printf(">>First %ld query IDs: ", display_count);
    for (faiss::idx_t i = 0; i < display_count; i++) {
      printf("%ld ", indicesQ[i]);
    }
    printf("\n");
  }

  double index_build_start = elapsed();

  std::unique_ptr<faiss::Index> quantizer_owner;
  faiss::Index* quantizer = nullptr;
  if (false && n >= 100000000) {
    int hnsw_m = 32;
    quantizer = new faiss::IndexHNSWFlat(d, hnsw_m);
    quantizer_owner.reset(quantizer);
    printf(">>HNSW quantizer with M=%d quantizer is selected.\n", hnsw_m);
  } else {
    quantizer = new faiss::IndexFlatL2(d);
    quantizer_owner.reset(quantizer);
    printf(">>Coarse quantizer is selected.\n");
  }

  std::unique_ptr<faiss::Index> index_owner;
  faiss::IndexIVFFlat* index;
  if (save_index || !index_filepath || !index_file_exists(index_filepath)) {
    index = new faiss::IndexIVFFlat(quantizer, d, nlist, faiss::METRIC_L2);
    index_owner.reset(index);

    // Sample 10% of the dataset vectors for training
    size_t n_train = n / 10;
    if (n_train < size_t(nlist))
      throw std::invalid_argument("IVF nlist exceeds the 10% index-training sample");
    auto vecsDB_sample = std::make_unique<float[]>(n_train * d);
    for (size_t i = 0; i < n_train; i++) {
      memcpy(vecsDB_sample.get() + i * d, vecsDB + (i * 10) * d, d * sizeof(float));
    }

    index->train(n_train, vecsDB_sample.get());
    printf(">> Index trained with %ld vectors (10%% of %ld total). Proceeding to add vectors...\n", n_train, n);
    vecsDB_sample.reset();

    // todo: implement some batching to monitor the progress.
    size_t batch_size = 10000000;
    for (size_t start = 0; start < n; start += batch_size) {
      size_t end = std::min(start + batch_size, n);
      printf(">> Adding vectors from %ld to %ld, %ld in total...\n", start, end, end - start);
      index->add(end - start, vecsDB + start * d);
    }
    index->nprobe = nprobe;
    if (index_filepath) {
      if (save_index) {
        prepare_index_parent(index_filepath);
        faiss::write_index(index, index_filepath);
      } else {
        save_new_index(index, index_filepath);
      }
      printf(">>Index saved to %s\n", index_filepath);
    }
  } else {
    if (index_filepath) {
      index_owner.reset(faiss::read_index(index_filepath));
      index = dynamic_cast<faiss::IndexIVFFlat*>(index_owner.get());
      if (!index) throw std::runtime_error("Loaded index is not an IndexIVFFlat");
      index->nprobe = nprobe;
      printf(">>Index loaded from %s\n", index_filepath ? index_filepath : "(none)");
    } else {
      throw std::runtime_error("Invalid runner configuration");
      /*index = new faiss::IndexIVFFlat(quantizer, d, nlist, faiss::METRIC_L2);
      index->train(n, vecsDB);
      index->add(n, vecsDB);
      index->nprobe = nprobe;*/
    }
  }

  if (index->d != d || index->ntotal != n || index->metric_type != faiss::METRIC_L2)
    throw std::invalid_argument("Loaded index dimensions, size or metric do not match the dataset");

  double index_build_time = elapsed() - index_build_start;
  printf(">>Index build/load time: %.3fs\n", index_build_time);

  std::unique_ptr<faiss::idx_t[]> I_owner(new faiss::idx_t[nQ * k]);
  auto* I = I_owner.get();
  std::unique_ptr<float[]> D_owner(new float[nQ * k]);
  auto* D = D_owner.get();

  double search_start_time = elapsed();

  auto data_manager = std::make_unique<faiss::DeclarativeRecallDataManager>(D, I, gt, gt_dist, nQ, d, k, vecsQ, output, vecsDB, n, gt_k_all, k_all);
  switch (mode) {
    case PLAIN_IVF: {
      auto data_collector = std::make_unique<faiss::DeclarativeRecallDataCollectorIVF>(data_manager.get(), logging_interval);
      index->search_plainIVF(nQ, vecsQ, k, D, I, data_collector.get());
      break;
    }
    default:
      throw std::invalid_argument("Unsupported baseline mode");
  }

  double search_time = elapsed() - search_start_time;

  printf("\n\nIndex[nlist=%d, nprobes=%ld]IndexTime: %lfs, SearchTime: %lfs, TotalTime: %lfs, Recall@%ld: %.4f, RQUT@%ld: %.4f\n",
         nlist,
         index->nprobe,
         index_build_time,
         search_time,
         (elapsed() - t0),
         k,
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
