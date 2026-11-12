#pragma once

#include <cstddef>
#include <initializer_list>
#include <map>
#include <string>

typedef enum {
  TRAINING = 0,
  VALIDATION = 1,
  TESTING = 2,
  NOISY_TESTING = 3,
  CALIBRATION = 4
} query_type_t;

extern const char query_type_str[5][20];

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out, size_t limit = 0);
int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out, size_t limit = 0);

typedef struct VectorDataLoader {
  query_type_t query_type;

  std::string noise_perc;
  std::string metric;

  std::string dataset_name;
  std::string directory_path;
  std::map<std::string, std::string> baseVectorsMap;
  std::map<query_type_t, std::map<std::string, std::string>> queryTypeToVectorsMap;
  std::map<query_type_t, std::map<std::string, std::string>> queryTypeToGroundtruthsMap;
  std::map<query_type_t, std::map<std::string, std::string>> queryTypeToGroundtruthDistancesMap;
  std::map<std::string, std::string> metricPrefixMap;

  VectorDataLoader(std::string dataset_name, query_type_t query_type);
  VectorDataLoader(std::string dataset_name, query_type_t query_type, std::string noise_perc, std::string directory_path);
  VectorDataLoader(std::string dataset_name, query_type_t query_type, std::string noise_perc, std::string directory_path, std::string metric);

  void initializeDataMaps();

  const std::string& basePath() const;
  const std::string& queryPath(query_type_t type) const;
  const std::string& groundtruthPath(query_type_t type) const;
  void validateDistinctQueryFiles(std::initializer_list<query_type_t> types) const;

  float* loadDB(size_t* d_out, size_t* n_out, size_t limit = 0);
  float* loadQueries(size_t* d_out, size_t* n_out);
  float* loadQueries(query_type_t type, size_t* d_out, size_t* n_out, size_t limit = 0);
  int* loadQueriesGroundtruths(size_t* k_out, size_t* n_out);
  int* loadQueriesGroundtruths(query_type_t type, size_t* k_out, size_t* n_out, size_t limit = 0);
  float* loadQueriesGroundtruthDistances(size_t* k_out, size_t* n_out);

} VectorDataLoader;
