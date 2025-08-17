#pragma once

#include <map>
#include <string>
#include <cstddef>

typedef enum {
    TRAINING = 0,
    VALIDATION = 1,
    TESTING = 2,
    NOISY_TESTING = 3
} query_type_t;

extern const char query_type_str[4][20];

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out, size_t limit = 0);
int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out);

typedef struct VectorDataLoader {
    query_type_t query_type;

    std::string noise_perc;

    std::string dataset_name;
    std::string directory_path = "/data/mchatzakis/datasets/processed/";    
    std::map<std::string, std::string> baseVectorsMap;
    std::map<query_type_t, std::map<std::string, std::string>> queryTypeToVectorsMap;
    std::map<query_type_t, std::map<std::string, std::string>> queryTypeToGroundtruthsMap;
    std::map<query_type_t, std::map<std::string, std::string>> queryTypeToGroundtruthDistancesMap;

    VectorDataLoader(std::string dataset_name, query_type_t query_type);
    VectorDataLoader(std::string dataset_name, query_type_t query_type, std::string noise_perc, std::string directory_path);

    void initializeDataMaps();

    float* loadDB(size_t* d_out, size_t* n_out);
    float* loadQueries(size_t* d_out, size_t* n_out);
    int* loadQueriesGroundtruths(size_t* k_out, size_t* n_out);
    float* loadQueriesGroundtruthDistances(size_t* k_out, size_t* n_out);

} VectorDataLoader;
