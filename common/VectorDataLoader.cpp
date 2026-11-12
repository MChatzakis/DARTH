#include "VectorDataLoader.h"

#include <sys/stat.h>

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <cstdlib>
#include <filesystem>

const char query_type_str[5][20] = {
    "Training",
    "Validation",
    "Testing",
    "Noisy Testing",
    "Calibration"};

template <typename T>
T* vecs_read(
    const char* fname,
    size_t* d_out,
    size_t* n_out,
    size_t limit) {
  static_assert(sizeof(T) == sizeof(int), "vecs elements must be 32-bit");

  std::unique_ptr<FILE, decltype(&fclose)> file(fopen(fname, "rb"), &fclose);
  FILE* f = file.get();
  if (!f) {
    fprintf(stderr, "could not open %s\n", fname);
    throw std::runtime_error(std::string("Cannot read vectors: ") + fname);
  }

  int d;
  if (fread(&d, sizeof(d), 1, f) != 1) {
    fprintf(stderr, "could not read vector dimension from %s\n", fname);
    throw std::runtime_error(std::string("Cannot read vectors: ") + fname);
  }
  if (d <= 0 || d >= 1000000) {
    fprintf(stderr, "unreasonable vector dimension %d in %s\n", d, fname);
    throw std::runtime_error(std::string("Cannot read vectors: ") + fname);
  }

  struct stat st;
  if (fstat(fileno(f), &st) != 0) {
    fprintf(stderr, "could not determine file size for %s\n", fname);
    throw std::runtime_error(std::string("Cannot read vectors: ") + fname);
  }
  size_t sz = st.st_size;
  const size_t row_size = sizeof(int) + static_cast<size_t>(d) * sizeof(T);
  if (sz % row_size != 0) {
    fprintf(stderr, "invalid vecs file size for %s\n", fname);
    throw std::runtime_error(std::string("Cannot read vectors: ") + fname);
  }
  size_t n = sz / row_size;

  if (limit > 0 && n > limit) {
    n = limit;
    printf("Limiting dataset %s to %ld vectors\n", fname, n);
  }

  *d_out = d;
  *n_out = n;
  auto values = std::make_unique<T[]>(n * static_cast<size_t>(d));
  T* x = values.get();
  rewind(f);
  for (size_t i = 0; i < n; i++) {
    int row_dimension;
    if (fread(&row_dimension, sizeof(row_dimension), 1, f) != 1 ||
        row_dimension != d ||
        fread(x + i * d, sizeof(T), d, f) != static_cast<size_t>(d)) {
      fprintf(stderr, "invalid or truncated vector row %zu in %s\n", i, fname);
      throw std::runtime_error(std::string("Cannot read vectors: ") + fname);
    }
  }

  return values.release();
}

float* fvecs_read(
    const char* fname,
    size_t* d_out,
    size_t* n_out,
    size_t limit) {
  return vecs_read<float>(fname, d_out, n_out, limit);
}

int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out, size_t limit) {
  return vecs_read<int>(fname, d_out, n_out, limit);
}

VectorDataLoader::VectorDataLoader(std::string dataset_name, query_type_t query_type)
    : query_type(query_type), noise_perc(""), dataset_name(dataset_name), metric("L2") {
}

VectorDataLoader::VectorDataLoader(std::string dataset_name, query_type_t query_type, std::string noise_perc, std::string directory_path)
    : VectorDataLoader(dataset_name, query_type) {
  this->directory_path = directory_path;
  this->noise_perc = noise_perc;
}

VectorDataLoader::VectorDataLoader(std::string dataset_name, query_type_t query_type, std::string noise_perc, std::string directory_path, std::string metric)
    : VectorDataLoader(dataset_name, query_type, noise_perc, directory_path) {
  this->metric = metric;
}

void VectorDataLoader::initializeDataMaps() {
  if (directory_path.empty()) {
    const char* configured = std::getenv("DATASET_DIRECTORY");
    if (!configured || !*configured) configured = std::getenv("DARTH_DATASET_DIR");
    if (!configured || !*configured)
      throw std::invalid_argument("Specify --dataset-dir-prefix, DATASET_DIRECTORY or DARTH_DATASET_DIR");
    directory_path = configured;
  }
  if (directory_path.back() != '/') directory_path += '/';
  // Initialize metric prefix map
  metricPrefixMap["L2"] = "";
  metricPrefixMap["inner-product"] = "innprod.";

  std::string gt_prefix = metricPrefixMap[metric];

  // SIFT10M
  baseVectorsMap["SIFT10M"] = directory_path + "SIFT10M/base.10M.fvecs";

  queryTypeToVectorsMap[TRAINING]["SIFT10M"] = directory_path + "SIFT10M/learn.1M.fvecs";
  queryTypeToVectorsMap[VALIDATION]["SIFT10M"] = directory_path + "SIFT10M/validation.10K.fvecs";
  queryTypeToVectorsMap[TESTING]["SIFT10M"] = directory_path + "SIFT10M/query.10K.fvecs";
  queryTypeToVectorsMap[NOISY_TESTING]["SIFT10M"] = directory_path + "SIFT10M/gauss_noisy_queries/query.10K.noise" + noise_perc + ".fvecs";
  queryTypeToVectorsMap[CALIBRATION]["SIFT10M"] = directory_path + "SIFT10M/calibration.10K.fvecs";

  queryTypeToGroundtruthsMap[TRAINING]["SIFT10M"] = directory_path + "SIFT10M/" + gt_prefix + "learn.groundtruth.1M.k1000.ivecs";
  queryTypeToGroundtruthsMap[VALIDATION]["SIFT10M"] = directory_path + "SIFT10M/" + gt_prefix + "validation.groundtruth.10K.k1000.ivecs";
  queryTypeToGroundtruthsMap[TESTING]["SIFT10M"] = directory_path + "SIFT10M/" + gt_prefix + "query.groundtruth.10K.k1000.ivecs";
  queryTypeToGroundtruthsMap[NOISY_TESTING]["SIFT10M"] = directory_path + "SIFT10M/gauss_noisy_queries/" + gt_prefix + "query.10K.groundtruth.noise" + noise_perc + ".ivecs";
  queryTypeToGroundtruthsMap[CALIBRATION]["SIFT10M"] = directory_path + "SIFT10M/" + gt_prefix + "calibration.groundtruth.10K.k1000.ivecs";

  queryTypeToGroundtruthDistancesMap[TRAINING]["SIFT10M"] = directory_path + "SIFT10M/" + gt_prefix + "learn.groundtruth.1M.k1000.fvecs";
  queryTypeToGroundtruthDistancesMap[VALIDATION]["SIFT10M"] = directory_path + "SIFT10M/" + gt_prefix + "validation.groundtruth.10K.k1000.fvecs";
  queryTypeToGroundtruthDistancesMap[TESTING]["SIFT10M"] = directory_path + "SIFT10M/" + gt_prefix + "query.groundtruth.10K.k1000.fvecs";
  queryTypeToGroundtruthDistancesMap[NOISY_TESTING]["SIFT10M"] = directory_path + "SIFT10M/gauss_noisy_queries/" + gt_prefix + "query.10K.groundtruth.noise" + noise_perc + ".fvecs";
  queryTypeToGroundtruthDistancesMap[CALIBRATION]["SIFT10M"] = directory_path + "SIFT10M/" + gt_prefix + "calibration.groundtruth.10K.k1000.fvecs";

  // SIFT50M
  baseVectorsMap["SIFT50M"] = directory_path + "SIFT50M/base.50M.fvecs";

  queryTypeToVectorsMap[TRAINING]["SIFT50M"] = directory_path + "SIFT50M/learn.1M.fvecs";
  queryTypeToVectorsMap[VALIDATION]["SIFT50M"] = directory_path + "SIFT50M/validation.10K.fvecs";
  queryTypeToVectorsMap[TESTING]["SIFT50M"] = directory_path + "SIFT50M/query.10K.fvecs";
  queryTypeToVectorsMap[NOISY_TESTING]["SIFT50M"] = directory_path + "SIFT50M/gauss_noisy_queries/query.10K.noise" + noise_perc + ".fvecs";
  queryTypeToVectorsMap[CALIBRATION]["SIFT50M"] = directory_path + "SIFT50M/calibration.10K.fvecs";

  queryTypeToGroundtruthsMap[TRAINING]["SIFT50M"] = directory_path + "SIFT50M/" + gt_prefix + "learn.groundtruth.1M.k1000.ivecs";
  queryTypeToGroundtruthsMap[VALIDATION]["SIFT50M"] = directory_path + "SIFT50M/" + gt_prefix + "validation.groundtruth.10K.k1000.ivecs";
  queryTypeToGroundtruthsMap[TESTING]["SIFT50M"] = directory_path + "SIFT50M/" + gt_prefix + "query.groundtruth.10K.k1000.ivecs";
  queryTypeToGroundtruthsMap[NOISY_TESTING]["SIFT50M"] = directory_path + "SIFT50M/gauss_noisy_queries/" + gt_prefix + "query.10K.groundtruth.noise" + noise_perc + ".ivecs";
  queryTypeToGroundtruthsMap[CALIBRATION]["SIFT50M"] = directory_path + "SIFT50M/" + gt_prefix + "calibration.groundtruth.10K.k1000.ivecs";

  queryTypeToGroundtruthDistancesMap[TRAINING]["SIFT50M"] = directory_path + "SIFT50M/" + gt_prefix + "learn.groundtruth.1M.k1000.fvecs";
  queryTypeToGroundtruthDistancesMap[VALIDATION]["SIFT50M"] = directory_path + "SIFT50M/" + gt_prefix + "validation.groundtruth.10K.k1000.fvecs";
  queryTypeToGroundtruthDistancesMap[TESTING]["SIFT50M"] = directory_path + "SIFT50M/" + gt_prefix + "query.groundtruth.10K.k1000.fvecs";
  queryTypeToGroundtruthDistancesMap[NOISY_TESTING]["SIFT50M"] = directory_path + "SIFT50M/gauss_noisy_queries/" + gt_prefix + "query.10K.groundtruth.noise" + noise_perc + ".fvecs";
  queryTypeToGroundtruthDistancesMap[CALIBRATION]["SIFT50M"] = directory_path + "SIFT50M/" + gt_prefix + "calibration.groundtruth.10K.k1000.fvecs";

  // SIFT100M
  baseVectorsMap["SIFT100M"] = directory_path + "SIFT100M/base.100M.fvecs";

  queryTypeToVectorsMap[TRAINING]["SIFT100M"] = directory_path + "SIFT100M/learn.1M.fvecs";
  queryTypeToVectorsMap[VALIDATION]["SIFT100M"] = directory_path + "SIFT100M/validation.10K.fvecs";
  queryTypeToVectorsMap[TESTING]["SIFT100M"] = directory_path + "SIFT100M/query.10K.fvecs";
  queryTypeToVectorsMap[NOISY_TESTING]["SIFT100M"] = directory_path + "SIFT100M/gauss_noisy_queries/query.10K.noise" + noise_perc + ".fvecs";
  queryTypeToVectorsMap[CALIBRATION]["SIFT100M"] = directory_path + "SIFT100M/calibration.10K.fvecs";

  queryTypeToGroundtruthsMap[TRAINING]["SIFT100M"] = directory_path + "SIFT100M/" + gt_prefix + "learn.groundtruth.1M.k1000.ivecs";
  queryTypeToGroundtruthsMap[VALIDATION]["SIFT100M"] = directory_path + "SIFT100M/" + gt_prefix + "validation.groundtruth.10K.k1000.ivecs";
  queryTypeToGroundtruthsMap[TESTING]["SIFT100M"] = directory_path + "SIFT100M/" + gt_prefix + "query.groundtruth.10K.k1000.ivecs";
  queryTypeToGroundtruthsMap[NOISY_TESTING]["SIFT100M"] = directory_path + "SIFT100M/gauss_noisy_queries/" + gt_prefix + "query.10K.groundtruth.noise" + noise_perc + ".ivecs";
  queryTypeToGroundtruthsMap[CALIBRATION]["SIFT100M"] = directory_path + "SIFT100M/" + gt_prefix + "calibration.groundtruth.10K.k1000.ivecs";

  queryTypeToGroundtruthDistancesMap[TRAINING]["SIFT100M"] = directory_path + "SIFT100M/" + gt_prefix + "learn.groundtruth.1M.k1000.fvecs";
  queryTypeToGroundtruthDistancesMap[VALIDATION]["SIFT100M"] = directory_path + "SIFT100M/" + gt_prefix + "validation.groundtruth.10K.k1000.fvecs";
  queryTypeToGroundtruthDistancesMap[TESTING]["SIFT100M"] = directory_path + "SIFT100M/" + gt_prefix + "query.groundtruth.10K.k1000.fvecs";
  queryTypeToGroundtruthDistancesMap[NOISY_TESTING]["SIFT100M"] = directory_path + "SIFT100M/gauss_noisy_queries/" + gt_prefix + "query.10K.groundtruth.noise" + noise_perc + ".fvecs";
  queryTypeToGroundtruthDistancesMap[CALIBRATION]["SIFT100M"] = directory_path + "SIFT100M/" + gt_prefix + "calibration.groundtruth.10K.k1000.fvecs";

  // DEEP10M
  baseVectorsMap["DEEP10M"] = directory_path + "DEEP10M/base.10M.fvecs";

  queryTypeToVectorsMap[TRAINING]["DEEP10M"] = directory_path + "DEEP10M/learn.1M.fvecs";
  queryTypeToVectorsMap[VALIDATION]["DEEP10M"] = directory_path + "DEEP10M/validation.10K.fvecs";
  queryTypeToVectorsMap[TESTING]["DEEP10M"] = directory_path + "DEEP10M/query.10K.fvecs";
  queryTypeToVectorsMap[NOISY_TESTING]["DEEP10M"] = directory_path + "DEEP10M/gauss_noisy_queries/query.10K.noise" + noise_perc + ".fvecs";
  queryTypeToVectorsMap[CALIBRATION]["DEEP10M"] = directory_path + "DEEP10M/calibration.10K.fvecs";

  queryTypeToGroundtruthsMap[TRAINING]["DEEP10M"] = directory_path + "DEEP10M/" + gt_prefix + "learn.groundtruth.1M.k1000.ivecs";
  queryTypeToGroundtruthsMap[VALIDATION]["DEEP10M"] = directory_path + "DEEP10M/" + gt_prefix + "validation.groundtruth.10K.k1000.ivecs";
  queryTypeToGroundtruthsMap[TESTING]["DEEP10M"] = directory_path + "DEEP10M/" + gt_prefix + "query.groundtruth.10K.k1000.ivecs";
  queryTypeToGroundtruthsMap[NOISY_TESTING]["DEEP10M"] = directory_path + "DEEP10M/gauss_noisy_queries/" + gt_prefix + "query.10K.groundtruth.noise" + noise_perc + ".ivecs";
  queryTypeToGroundtruthsMap[CALIBRATION]["DEEP10M"] = directory_path + "DEEP10M/" + gt_prefix + "calibration.groundtruth.10K.k1000.ivecs";

  queryTypeToGroundtruthDistancesMap[TRAINING]["DEEP10M"] = directory_path + "DEEP10M/" + gt_prefix + "learn.groundtruth.1M.k1000.fvecs";
  queryTypeToGroundtruthDistancesMap[VALIDATION]["DEEP10M"] = directory_path + "DEEP10M/" + gt_prefix + "validation.groundtruth.10K.k1000.fvecs";
  queryTypeToGroundtruthDistancesMap[TESTING]["DEEP10M"] = directory_path + "DEEP10M/" + gt_prefix + "query.groundtruth.10K.k1000.fvecs";
  queryTypeToGroundtruthDistancesMap[NOISY_TESTING]["DEEP10M"] = directory_path + "DEEP10M/gauss_noisy_queries/" + gt_prefix + "query.10K.groundtruth.noise" + noise_perc + ".fvecs";
  queryTypeToGroundtruthDistancesMap[CALIBRATION]["DEEP10M"] = directory_path + "DEEP10M/" + gt_prefix + "calibration.groundtruth.10K.k1000.fvecs";

  // DEEP50M
  baseVectorsMap["DEEP50M"] = directory_path + "DEEP50M/base.50M.fvecs";

  queryTypeToVectorsMap[TRAINING]["DEEP50M"] = directory_path + "DEEP50M/learn.1M.fvecs";
  queryTypeToVectorsMap[VALIDATION]["DEEP50M"] = directory_path + "DEEP50M/validation.10K.fvecs";
  queryTypeToVectorsMap[TESTING]["DEEP50M"] = directory_path + "DEEP50M/query.10K.fvecs";
  queryTypeToVectorsMap[NOISY_TESTING]["DEEP50M"] = directory_path + "DEEP50M/gauss_noisy_queries/query.10K.noise" + noise_perc + ".fvecs";
  queryTypeToVectorsMap[CALIBRATION]["DEEP50M"] = directory_path + "DEEP50M/calibration.10K.fvecs";

  queryTypeToGroundtruthsMap[TRAINING]["DEEP50M"] = directory_path + "DEEP50M/" + gt_prefix + "learn.groundtruth.1M.k1000.ivecs";
  queryTypeToGroundtruthsMap[VALIDATION]["DEEP50M"] = directory_path + "DEEP50M/" + gt_prefix + "validation.groundtruth.10K.k1000.ivecs";
  queryTypeToGroundtruthsMap[TESTING]["DEEP50M"] = directory_path + "DEEP50M/" + gt_prefix + "query.groundtruth.10K.k1000.ivecs";
  queryTypeToGroundtruthsMap[NOISY_TESTING]["DEEP50M"] = directory_path + "DEEP50M/gauss_noisy_queries/" + gt_prefix + "query.10K.groundtruth.noise" + noise_perc + ".ivecs";
  queryTypeToGroundtruthsMap[CALIBRATION]["DEEP50M"] = directory_path + "DEEP50M/" + gt_prefix + "calibration.groundtruth.10K.k1000.ivecs";

  queryTypeToGroundtruthDistancesMap[TRAINING]["DEEP50M"] = directory_path + "DEEP50M/" + gt_prefix + "learn.groundtruth.1M.k1000.fvecs";
  queryTypeToGroundtruthDistancesMap[VALIDATION]["DEEP50M"] = directory_path + "DEEP50M/" + gt_prefix + "validation.groundtruth.10K.k1000.fvecs";
  queryTypeToGroundtruthDistancesMap[TESTING]["DEEP50M"] = directory_path + "DEEP50M/" + gt_prefix + "query.groundtruth.10K.k1000.fvecs";
  queryTypeToGroundtruthDistancesMap[NOISY_TESTING]["DEEP50M"] = directory_path + "DEEP50M/gauss_noisy_queries/" + gt_prefix + "query.10K.groundtruth.noise" + noise_perc + ".fvecs";
  queryTypeToGroundtruthDistancesMap[CALIBRATION]["DEEP50M"] = directory_path + "DEEP50M/" + gt_prefix + "calibration.groundtruth.10K.k1000.fvecs";

  // DEEP100M
  baseVectorsMap["DEEP100M"] = directory_path + "DEEP100M/base.100M.fvecs";

  queryTypeToVectorsMap[TRAINING]["DEEP100M"] = directory_path + "DEEP100M/learn.1M.fvecs";
  queryTypeToVectorsMap[VALIDATION]["DEEP100M"] = directory_path + "DEEP100M/validation.10K.fvecs";
  queryTypeToVectorsMap[TESTING]["DEEP100M"] = directory_path + "DEEP100M/query.10K.fvecs";
  queryTypeToVectorsMap[NOISY_TESTING]["DEEP100M"] = directory_path + "DEEP100M/gauss_noisy_queries/query.10K.noise" + noise_perc + ".fvecs";
  queryTypeToVectorsMap[CALIBRATION]["DEEP100M"] = directory_path + "DEEP100M/calibration.10K.fvecs";

  queryTypeToGroundtruthsMap[TRAINING]["DEEP100M"] = directory_path + "DEEP100M/" + gt_prefix + "learn.groundtruth.1M.k1000.ivecs";
  queryTypeToGroundtruthsMap[VALIDATION]["DEEP100M"] = directory_path + "DEEP100M/" + gt_prefix + "validation.groundtruth.10K.k1000.ivecs";
  queryTypeToGroundtruthsMap[TESTING]["DEEP100M"] = directory_path + "DEEP100M/" + gt_prefix + "query.groundtruth.10K.k1000.ivecs";
  queryTypeToGroundtruthsMap[NOISY_TESTING]["DEEP100M"] = directory_path + "DEEP100M/gauss_noisy_queries/" + gt_prefix + "query.10K.groundtruth.noise" + noise_perc + ".ivecs";
  queryTypeToGroundtruthsMap[CALIBRATION]["DEEP100M"] = directory_path + "DEEP100M/" + gt_prefix + "calibration.groundtruth.10K.k1000.ivecs";

  queryTypeToGroundtruthDistancesMap[TRAINING]["DEEP100M"] = directory_path + "DEEP100M/" + gt_prefix + "learn.groundtruth.1M.k1000.fvecs";
  queryTypeToGroundtruthDistancesMap[VALIDATION]["DEEP100M"] = directory_path + "DEEP100M/" + gt_prefix + "validation.groundtruth.10K.k1000.fvecs";
  queryTypeToGroundtruthDistancesMap[TESTING]["DEEP100M"] = directory_path + "DEEP100M/" + gt_prefix + "query.groundtruth.10K.k1000.fvecs";
  queryTypeToGroundtruthDistancesMap[NOISY_TESTING]["DEEP100M"] = directory_path + "DEEP100M/gauss_noisy_queries/" + gt_prefix + "query.10K.groundtruth.noise" + noise_perc + ".fvecs";
  queryTypeToGroundtruthDistancesMap[CALIBRATION]["DEEP100M"] = directory_path + "DEEP100M/" + gt_prefix + "calibration.groundtruth.10K.k1000.fvecs";

  // GLOVE100
  baseVectorsMap["GLOVE100"] = directory_path + "GLOVE100/base.1183514.fvecs";

  queryTypeToVectorsMap[TRAINING]["GLOVE100"] = directory_path + "GLOVE100/learn.100K.fvecs";
  queryTypeToVectorsMap[VALIDATION]["GLOVE100"] = directory_path + "GLOVE100/validation.10K.fvecs";
  queryTypeToVectorsMap[TESTING]["GLOVE100"] = directory_path + "GLOVE100/query.10K.fvecs";
  queryTypeToVectorsMap[NOISY_TESTING]["GLOVE100"] = directory_path + "GLOVE100/gauss_noisy_queries/query.10K.noise" + noise_perc + ".fvecs";
  queryTypeToVectorsMap[CALIBRATION]["GLOVE100"] = directory_path + "GLOVE100/calibration.10K.fvecs";

  queryTypeToGroundtruthsMap[TRAINING]["GLOVE100"] = directory_path + "GLOVE100/" + gt_prefix + "learn.groundtruth.100K.k1000.ivecs";
  queryTypeToGroundtruthsMap[VALIDATION]["GLOVE100"] = directory_path + "GLOVE100/" + gt_prefix + "validation.groundtruth.10K.k1000.ivecs";
  queryTypeToGroundtruthsMap[TESTING]["GLOVE100"] = directory_path + "GLOVE100/" + gt_prefix + "query.groundtruth.10K.k1000.ivecs";
  queryTypeToGroundtruthsMap[NOISY_TESTING]["GLOVE100"] = directory_path + "GLOVE100/gauss_noisy_queries/" + gt_prefix + "query.10K.groundtruth.noise" + noise_perc + ".ivecs";
  queryTypeToGroundtruthsMap[CALIBRATION]["GLOVE100"] = directory_path + "GLOVE100/" + gt_prefix + "calibration.groundtruth.10K.k1000.ivecs";

  queryTypeToGroundtruthDistancesMap[TRAINING]["GLOVE100"] = directory_path + "GLOVE100/" + gt_prefix + "learn.groundtruth.100K.k1000.fvecs";
  queryTypeToGroundtruthDistancesMap[VALIDATION]["GLOVE100"] = directory_path + "GLOVE100/" + gt_prefix + "validation.groundtruth.10K.k1000.fvecs";
  queryTypeToGroundtruthDistancesMap[TESTING]["GLOVE100"] = directory_path + "GLOVE100/" + gt_prefix + "query.groundtruth.10K.k1000.fvecs";
  queryTypeToGroundtruthDistancesMap[NOISY_TESTING]["GLOVE100"] = directory_path + "GLOVE100/gauss_noisy_queries/" + gt_prefix + "query.10K.groundtruth.noise" + noise_perc + ".fvecs";
  queryTypeToGroundtruthDistancesMap[CALIBRATION]["GLOVE100"] = directory_path + "GLOVE100/" + gt_prefix + "calibration.groundtruth.10K.k1000.fvecs";

  // GIST1M
  baseVectorsMap["GIST1M"] = directory_path + "GIST1M/base.1M.fvecs";

  queryTypeToVectorsMap[TRAINING]["GIST1M"] = directory_path + "GIST1M/learn.100K.fvecs";
  queryTypeToVectorsMap[VALIDATION]["GIST1M"] = directory_path + "GIST1M/validation.1K.fvecs";
  queryTypeToVectorsMap[TESTING]["GIST1M"] = directory_path + "GIST1M/query.1K.fvecs";
  queryTypeToVectorsMap[NOISY_TESTING]["GIST1M"] = directory_path + "GIST1M/gauss_noisy_queries/query.1K.noise" + noise_perc + ".fvecs";
  queryTypeToVectorsMap[CALIBRATION]["GIST1M"] = directory_path + "GIST1M/calibration.10K.fvecs";

  queryTypeToGroundtruthsMap[TRAINING]["GIST1M"] = directory_path + "GIST1M/" + gt_prefix + "learn.groundtruth.100K.k1000.ivecs";
  queryTypeToGroundtruthsMap[VALIDATION]["GIST1M"] = directory_path + "GIST1M/" + gt_prefix + "validation.groundtruth.1K.k1000.ivecs";
  queryTypeToGroundtruthsMap[TESTING]["GIST1M"] = directory_path + "GIST1M/" + gt_prefix + "query.groundtruth.1K.k1000.ivecs";
  queryTypeToGroundtruthsMap[NOISY_TESTING]["GIST1M"] = directory_path + "GIST1M/gauss_noisy_queries/" + gt_prefix + "query.1K.groundtruth.noise" + noise_perc + ".ivecs";
  queryTypeToGroundtruthsMap[CALIBRATION]["GIST1M"] = directory_path + "GIST1M/" + gt_prefix + "calibration.groundtruth.10K.k1000.ivecs";

  queryTypeToGroundtruthDistancesMap[TRAINING]["GIST1M"] = directory_path + "GIST1M/" + gt_prefix + "learn.groundtruth.100K.k1000.fvecs";
  queryTypeToGroundtruthDistancesMap[VALIDATION]["GIST1M"] = directory_path + "GIST1M/" + gt_prefix + "validation.groundtruth.1K.k1000.fvecs";
  queryTypeToGroundtruthDistancesMap[TESTING]["GIST1M"] = directory_path + "GIST1M/" + gt_prefix + "query.groundtruth.1K.k1000.fvecs";
  queryTypeToGroundtruthDistancesMap[NOISY_TESTING]["GIST1M"] = directory_path + "GIST1M/gauss_noisy_queries/" + gt_prefix + "query.1K.groundtruth.noise" + noise_perc + ".fvecs";
  queryTypeToGroundtruthDistancesMap[CALIBRATION]["GIST1M"] = directory_path + "GIST1M/" + gt_prefix + "calibration.groundtruth.10K.k1000.fvecs";

  // T2I100M
  baseVectorsMap["T2I100M"] = directory_path + "T2I100M/base.100M.fvecs";
  queryTypeToVectorsMap[TRAINING]["T2I100M"] = directory_path + "T2I100M/learn.1M.fvecs";
  queryTypeToVectorsMap[VALIDATION]["T2I100M"] = directory_path + "T2I100M/validation.10K.fvecs";
  queryTypeToVectorsMap[TESTING]["T2I100M"] = directory_path + "T2I100M/query.10K.fvecs";
  queryTypeToVectorsMap[NOISY_TESTING]["T2I100M"] = directory_path + "T2I100M/gauss_noisy_queries/query.10K.noise" + noise_perc + ".fvecs";
  queryTypeToVectorsMap[CALIBRATION]["T2I100M"] = directory_path + "T2I100M/calibration.10K.fvecs";

  queryTypeToGroundtruthsMap[TRAINING]["T2I100M"] = directory_path + "T2I100M/" + gt_prefix + "learn.groundtruth.1M.k1000.ivecs";
  queryTypeToGroundtruthsMap[VALIDATION]["T2I100M"] = directory_path + "T2I100M/" + gt_prefix + "validation.groundtruth.10K.k1000.ivecs";
  queryTypeToGroundtruthsMap[TESTING]["T2I100M"] = directory_path + "T2I100M/" + gt_prefix + "query.groundtruth.10K.k1000.ivecs";
  queryTypeToGroundtruthsMap[NOISY_TESTING]["T2I100M"] = directory_path + "T2I100M/gauss_noisy_queries/" + gt_prefix + "query.10K.groundtruth.noise" + noise_perc + ".ivecs";
  queryTypeToGroundtruthsMap[CALIBRATION]["T2I100M"] = directory_path + "T2I100M/" + gt_prefix + "calibration.groundtruth.10K.k1000.ivecs";

  queryTypeToGroundtruthDistancesMap[TRAINING]["T2I100M"] = directory_path + "T2I100M/" + gt_prefix + "learn.groundtruth.1M.k1000.fvecs";
  queryTypeToGroundtruthDistancesMap[VALIDATION]["T2I100M"] = directory_path + "T2I100M/" + gt_prefix + "validation.groundtruth.10K.k1000.fvecs";
  queryTypeToGroundtruthDistancesMap[TESTING]["T2I100M"] = directory_path + "T2I100M/" + gt_prefix + "query.groundtruth.10K.k1000.fvecs";
  queryTypeToGroundtruthDistancesMap[NOISY_TESTING]["T2I100M"] = directory_path + "T2I100M/gauss_noisy_queries/" + gt_prefix + "query.10K.groundtruth.noise" + noise_perc + ".fvecs";
  queryTypeToGroundtruthDistancesMap[CALIBRATION]["T2I100M"] = directory_path + "T2I100M/" + gt_prefix + "calibration.groundtruth.10K.k1000.fvecs";

  // T2I10M:
  baseVectorsMap["T2I10M"] = directory_path + "T2I10M/base.10M.fvecs";
  queryTypeToVectorsMap[TRAINING]["T2I10M"] = directory_path + "T2I10M/learn.10K.fvecs";
  queryTypeToVectorsMap[VALIDATION]["T2I10M"] = directory_path + "T2I10M/validation.10K.fvecs";
  queryTypeToVectorsMap[TESTING]["T2I10M"] = directory_path + "T2I10M/query.10K.fvecs";
  queryTypeToVectorsMap[NOISY_TESTING]["T2I10M"] = directory_path + "T2I10M/gauss_noisy_queries/query.10K.noise" + noise_perc + ".fvecs";
  queryTypeToVectorsMap[CALIBRATION]["T2I10M"] = directory_path + "T2I10M/calibration.10K.fvecs"; 

  queryTypeToGroundtruthsMap[TRAINING]["T2I10M"] = directory_path + "T2I10M/" + gt_prefix + "learn.groundtruth.10K.k1000.ivecs";
  queryTypeToGroundtruthsMap[VALIDATION]["T2I10M"] = directory_path + "T2I10M/" + gt_prefix + "validation.groundtruth.10K.k1000.ivecs";
  queryTypeToGroundtruthsMap[TESTING]["T2I10M"] = directory_path + "T2I10M/" + gt_prefix + "query.groundtruth.10K.k1000.ivecs";
  queryTypeToGroundtruthsMap[NOISY_TESTING]["T2I10M"] = directory_path + "T2I10M/gauss_noisy_queries/" + gt_prefix + "query.10K.groundtruth.noise" + noise_perc + ".ivecs";
  queryTypeToGroundtruthsMap[CALIBRATION]["T2I10M"] = directory_path + "T2I10M/" + gt_prefix + "calibration.groundtruth.10K.k1000.ivecs";
  
  queryTypeToGroundtruthDistancesMap[TRAINING]["T2I10M"] = directory_path + "T2I10M/" + gt_prefix + "learn.groundtruth.10K.k1000.fvecs";
  queryTypeToGroundtruthDistancesMap[VALIDATION]["T2I10M"] = directory_path + "T2I10M/" + gt_prefix + "validation.groundtruth.10K.k1000.fvecs";
  queryTypeToGroundtruthDistancesMap[TESTING]["T2I10M"] = directory_path + "T2I10M/" + gt_prefix + "query.groundtruth.10K.k1000.fvecs";
  queryTypeToGroundtruthDistancesMap[NOISY_TESTING]["T2I10M"] = directory_path + "T2I10M/gauss_noisy_queries/" + gt_prefix + "query.10K.groundtruth.noise" + noise_perc + ".fvecs";
  queryTypeToGroundtruthDistancesMap[CALIBRATION]["T2I10M"] = directory_path + "T2I10M/" + gt_prefix + "calibration.groundtruth.10K.k1000.fvecs";

}

const std::string& VectorDataLoader::basePath() const {
  auto path = baseVectorsMap.find(dataset_name);
  if (path == baseVectorsMap.end()) throw std::invalid_argument("Unknown dataset: " + dataset_name);
  return path->second;
}

const std::string& VectorDataLoader::queryPath(query_type_t type) const {
  auto paths = queryTypeToVectorsMap.find(type);
  if (paths == queryTypeToVectorsMap.end() || !paths->second.count(dataset_name))
    throw std::invalid_argument("Dataset has no query mapping: " + dataset_name);
  return paths->second.at(dataset_name);
}

const std::string& VectorDataLoader::groundtruthPath(query_type_t type) const {
  auto paths = queryTypeToGroundtruthsMap.find(type);
  if (paths == queryTypeToGroundtruthsMap.end() || !paths->second.count(dataset_name))
    throw std::invalid_argument("Dataset has no ground-truth mapping: " + dataset_name);
  return paths->second.at(dataset_name);
}

void VectorDataLoader::validateDistinctQueryFiles(std::initializer_list<query_type_t> types) const {
  for (auto first = types.begin(); first != types.end(); ++first) {
    for (auto second = first + 1; second != types.end(); ++second) {
      const auto& a = queryPath(*first);
      const auto& b = queryPath(*second);
      std::error_code error;
      const bool equivalent = std::filesystem::equivalent(a, b, error);
      if (a == b || (!error && equivalent))
        throw std::invalid_argument(std::string(query_type_str[*first]) + " and " +
                                    query_type_str[*second] + " queries must use separate files");
    }
  }
}

float* VectorDataLoader::loadDB(size_t* d_out, size_t* n_out, size_t limit) {
  const std::string& baseVectorsPath = basePath();

  printf(">> Loading base vectors from: %s\n", baseVectorsPath.c_str());
  float* db = fvecs_read(baseVectorsPath.c_str(), d_out, n_out, limit);

  return db;
}

float* VectorDataLoader::loadQueries(size_t* d_out, size_t* n_out) {
  return loadQueries(query_type, d_out, n_out);
}

float* VectorDataLoader::loadQueries(query_type_t type, size_t* d_out, size_t* n_out, size_t limit) {
  const std::string& queryVectorsPath = queryPath(type);

  printf(">> Loading queries from: %s\n", queryVectorsPath.c_str());
  float* queries = fvecs_read(queryVectorsPath.c_str(), d_out, n_out, limit);

  // skip first 5000 queries:
  //*n_out -= 5000;
  // queries += 5000 * *d_out;

  return queries;
}

int* VectorDataLoader::loadQueriesGroundtruths(size_t* k_out, size_t* n_out) {
  return loadQueriesGroundtruths(query_type, k_out, n_out);
}

int* VectorDataLoader::loadQueriesGroundtruths(query_type_t type, size_t* k_out, size_t* n_out, size_t limit) {
  const std::string& queryGroundtruthsPath = groundtruthPath(type);

  printf(">> Loading queries groundtruths from: %s\n", queryGroundtruthsPath.c_str());
  int* gt = ivecs_read(queryGroundtruthsPath.c_str(), k_out, n_out, limit);

  return gt;
}

float* VectorDataLoader::loadQueriesGroundtruthDistances(size_t* k_out, size_t* n_out) {
  std::string queryGroundtruthDistancesPath = queryTypeToGroundtruthDistancesMap[query_type][dataset_name];

  printf(">> Loading queries groundtruth distances from: %s\n", queryGroundtruthDistancesPath.c_str());
  float* gt = fvecs_read(queryGroundtruthDistancesPath.c_str(), k_out, n_out);

  return gt;
}
