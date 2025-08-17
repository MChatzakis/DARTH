#include "VectorDataLoader.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>

const char query_type_str[4][20] = {
    "Training",
    "Validation",
    "Testing",
    "Noisy Testing"
};

float* fvecs_read(
        const char* fname,
        size_t* d_out,
        size_t* n_out,
        size_t limit) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        exit(1);
    }

    int d;
    fread(&d, 1, sizeof(int), f);

    assert((d > 0 && d < 1000000) && "unreasonable dimension");

    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 && "weird file size");
    size_t n = sz / ((d + 1) * 4);

    if (limit > 0 && n > limit) {
        n = limit;
        printf("Limiting dataset %s to %ld vectors\n", fname, n);
    }

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) && "could not read whole file");

    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

VectorDataLoader::VectorDataLoader(std::string dataset_name, query_type_t query_type)
    : query_type(query_type), noise_perc(""), dataset_name(dataset_name) {
}

VectorDataLoader::VectorDataLoader(std::string dataset_name, query_type_t query_type, std::string noise_perc, std::string directory_path)
    : VectorDataLoader(dataset_name, query_type) {
    this->directory_path = directory_path;
    this->noise_perc = noise_perc;
}

void VectorDataLoader::initializeDataMaps(){
    // SIFT10M
    baseVectorsMap["SIFT10M"] = directory_path + "SIFT10M/base.10M.fvecs";
    
    queryTypeToVectorsMap[TRAINING]["SIFT10M"] = directory_path + "SIFT10M/learn.1M.fvecs";
    queryTypeToVectorsMap[VALIDATION]["SIFT10M"] = directory_path + "SIFT10M/validation.10K.fvecs";
    queryTypeToVectorsMap[TESTING]["SIFT10M"] = directory_path + "SIFT10M/query.10K.fvecs";
    queryTypeToVectorsMap[NOISY_TESTING]["SIFT10M"] = directory_path + "SIFT10M/gauss_noisy_queries/query.10K.noise" + noise_perc + ".fvecs";
    
    queryTypeToGroundtruthsMap[TRAINING]["SIFT10M"] = directory_path + "SIFT10M/learn.groundtruth.1M.k1000.ivecs";
    queryTypeToGroundtruthsMap[VALIDATION]["SIFT10M"] = directory_path + "SIFT10M/validation.groundtruth.10K.k1000.ivecs";
    queryTypeToGroundtruthsMap[TESTING]["SIFT10M"] = directory_path + "SIFT10M/query.groundtruth.10K.k1000.ivecs";
    queryTypeToGroundtruthsMap[NOISY_TESTING]["SIFT10M"] = directory_path + "SIFT10M/gauss_noisy_queries/query.10K.groundtruth.noise" + noise_perc + ".ivecs";

    queryTypeToGroundtruthDistancesMap[TRAINING]["SIFT10M"] = directory_path + "SIFT10M/learn.groundtruth.1M.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[VALIDATION]["SIFT10M"] = directory_path + "SIFT10M/validation.groundtruth.10K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[TESTING]["SIFT10M"] = directory_path + "SIFT10M/query.groundtruth.10K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[NOISY_TESTING]["SIFT10M"] = directory_path + "SIFT10M/gauss_noisy_queries/query.10K.groundtruth.noise" + noise_perc + ".fvecs";

    //SIFT50M
    baseVectorsMap["SIFT50M"] = directory_path + "SIFT50M/base.50M.fvecs";

    queryTypeToVectorsMap[TRAINING]["SIFT50M"] = directory_path + "SIFT50M/learn.1M.fvecs";
    queryTypeToVectorsMap[VALIDATION]["SIFT50M"] = directory_path + "SIFT50M/validation.10K.fvecs";
    queryTypeToVectorsMap[TESTING]["SIFT50M"] = directory_path + "SIFT50M/query.10K.fvecs";
    queryTypeToVectorsMap[NOISY_TESTING]["SIFT50M"] = directory_path + "SIFT50M/gauss_noisy_queries/query.10K.noise" + noise_perc + ".fvecs";

    queryTypeToGroundtruthsMap[TRAINING]["SIFT50M"] = directory_path + "SIFT50M/learn.groundtruth.1M.k1000.ivecs";
    queryTypeToGroundtruthsMap[VALIDATION]["SIFT50M"] = directory_path + "SIFT50M/validation.groundtruth.10K.k1000.ivecs";
    queryTypeToGroundtruthsMap[TESTING]["SIFT50M"] = directory_path + "SIFT50M/query.groundtruth.10K.k1000.ivecs";
    queryTypeToGroundtruthsMap[NOISY_TESTING]["SIFT50M"] = directory_path + "SIFT50M/gauss_noisy_queries/query.10K.groundtruth.noise" + noise_perc + ".ivecs";

    queryTypeToGroundtruthDistancesMap[TRAINING]["SIFT50M"] = directory_path + "SIFT50M/learn.groundtruth.1M.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[VALIDATION]["SIFT50M"] = directory_path + "SIFT50M/validation.groundtruth.10K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[TESTING]["SIFT50M"] = directory_path + "SIFT50M/query.groundtruth.10K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[NOISY_TESTING]["SIFT50M"] = directory_path + "SIFT50M/gauss_noisy_queries/query.10K.groundtruth.noise" + noise_perc + ".fvecs";

    //SIFT100M
    baseVectorsMap["SIFT100M"] = directory_path + "SIFT100M/base.100M.fvecs";

    queryTypeToVectorsMap[TRAINING]["SIFT100M"] = directory_path + "SIFT100M/learn.1M.fvecs";
    queryTypeToVectorsMap[VALIDATION]["SIFT100M"] = directory_path + "SIFT100M/validation.10K.fvecs";
    queryTypeToVectorsMap[TESTING]["SIFT100M"] = directory_path + "SIFT100M/query.10K.fvecs";
    queryTypeToVectorsMap[NOISY_TESTING]["SIFT100M"] = directory_path + "SIFT100M/gauss_noisy_queries/query.10K.noise" + noise_perc + ".fvecs";

    queryTypeToGroundtruthsMap[TRAINING]["SIFT100M"] = directory_path + "SIFT100M/learn.groundtruth.1M.k1000.ivecs";
    queryTypeToGroundtruthsMap[VALIDATION]["SIFT100M"] = directory_path + "SIFT100M/validation.groundtruth.10K.k1000.ivecs";
    queryTypeToGroundtruthsMap[TESTING]["SIFT100M"] = directory_path + "SIFT100M/query.groundtruth.10K.k1000.ivecs";
    queryTypeToGroundtruthsMap[NOISY_TESTING]["SIFT100M"] = directory_path + "SIFT100M/gauss_noisy_queries/query.10K.groundtruth.noise" + noise_perc + ".ivecs";

    queryTypeToGroundtruthDistancesMap[TRAINING]["SIFT100M"] = directory_path + "SIFT100M/learn.groundtruth.1M.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[VALIDATION]["SIFT100M"] = directory_path + "SIFT100M/validation.groundtruth.10K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[TESTING]["SIFT100M"] = directory_path + "SIFT100M/query.groundtruth.10K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[NOISY_TESTING]["SIFT100M"] = directory_path + "SIFT100M/gauss_noisy_queries/query.10K.groundtruth.noise" + noise_perc + ".fvecs";

    // DEEP10M
    baseVectorsMap["DEEP10M"] = directory_path + "DEEP10M/base.10M.fvecs";
    
    queryTypeToVectorsMap[TRAINING]["DEEP10M"] = directory_path + "DEEP10M/learn.1M.fvecs";
    queryTypeToVectorsMap[VALIDATION]["DEEP10M"] = directory_path + "DEEP10M/validation.10K.fvecs";
    queryTypeToVectorsMap[TESTING]["DEEP10M"] = directory_path + "DEEP10M/query.10K.fvecs";
    queryTypeToVectorsMap[NOISY_TESTING]["DEEP10M"] = directory_path + "DEEP10M/gauss_noisy_queries/query.10K.noise" + noise_perc + ".fvecs";
    
    queryTypeToGroundtruthsMap[TRAINING]["DEEP10M"] = directory_path + "DEEP10M/learn.groundtruth.1M.k1000.ivecs";
    queryTypeToGroundtruthsMap[VALIDATION]["DEEP10M"] = directory_path + "DEEP10M/validation.groundtruth.10K.k1000.ivecs";
    queryTypeToGroundtruthsMap[TESTING]["DEEP10M"] = directory_path + "DEEP10M/query.groundtruth.10K.k1000.ivecs";
    queryTypeToGroundtruthsMap[NOISY_TESTING]["DEEP10M"] = directory_path + "DEEP10M/gauss_noisy_queries/query.10K.groundtruth.noise" + noise_perc + ".ivecs";

    queryTypeToGroundtruthDistancesMap[TRAINING]["DEEP10M"] = directory_path + "DEEP10M/learn.groundtruth.1M.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[VALIDATION]["DEEP10M"] = directory_path + "DEEP10M/validation.groundtruth.10K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[TESTING]["DEEP10M"] = directory_path + "DEEP10M/query.groundtruth.10K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[NOISY_TESTING]["DEEP10M"] = directory_path + "DEEP10M/gauss_noisy_queries/query.10K.groundtruth.noise" + noise_perc + ".fvecs";

    //DEEP50M
    baseVectorsMap["DEEP50M"] = directory_path + "DEEP50M/base.50M.fvecs";
    
    queryTypeToVectorsMap[TRAINING]["DEEP50M"] = directory_path + "DEEP50M/learn.1M.fvecs";
    queryTypeToVectorsMap[VALIDATION]["DEEP50M"] = directory_path + "DEEP50M/validation.10K.fvecs";
    queryTypeToVectorsMap[TESTING]["DEEP50M"] = directory_path + "DEEP50M/query.10K.fvecs";
    queryTypeToVectorsMap[NOISY_TESTING]["DEEP50M"] = directory_path + "DEEP50M/gauss_noisy_queries/query.10K.noise" + noise_perc + ".fvecs";

    queryTypeToGroundtruthsMap[TRAINING]["DEEP50M"] = directory_path + "DEEP50M/learn.groundtruth.1M.k1000.ivecs";
    queryTypeToGroundtruthsMap[VALIDATION]["DEEP50M"] = directory_path + "DEEP50M/validation.groundtruth.10K.k1000.ivecs";
    queryTypeToGroundtruthsMap[TESTING]["DEEP50M"] = directory_path + "DEEP50M/query.groundtruth.10K.k1000.ivecs";
    queryTypeToGroundtruthsMap[NOISY_TESTING]["DEEP50M"] = directory_path + "DEEP50M/gauss_noisy_queries/query.10K.groundtruth.noise" + noise_perc + ".ivecs";

    queryTypeToGroundtruthDistancesMap[TRAINING]["DEEP50M"] = directory_path + "DEEP50M/learn.groundtruth.1M.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[VALIDATION]["DEEP50M"] = directory_path + "DEEP50M/validation.groundtruth.10K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[TESTING]["DEEP50M"] = directory_path + "DEEP50M/query.groundtruth.10K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[NOISY_TESTING]["DEEP50M"] = directory_path + "DEEP50M/gauss_noisy_queries/query.10K.groundtruth.noise" + noise_perc + ".fvecs";

    //DEEP100M
    baseVectorsMap["DEEP100M"] = directory_path + "DEEP100M/base.100M.fvecs";

    queryTypeToVectorsMap[TRAINING]["DEEP100M"] = directory_path + "DEEP100M/learn.1M.fvecs";
    queryTypeToVectorsMap[VALIDATION]["DEEP100M"] = directory_path + "DEEP100M/validation.10K.fvecs";
    queryTypeToVectorsMap[TESTING]["DEEP100M"] = directory_path + "DEEP100M/query.10K.fvecs";
    queryTypeToVectorsMap[NOISY_TESTING]["DEEP100M"] = directory_path + "DEEP100M/gauss_noisy_queries/query.10K.noise" + noise_perc + ".fvecs";

    queryTypeToGroundtruthsMap[TRAINING]["DEEP100M"] = directory_path + "DEEP100M/learn.groundtruth.1M.k1000.ivecs";
    queryTypeToGroundtruthsMap[VALIDATION]["DEEP100M"] = directory_path + "DEEP100M/validation.groundtruth.10K.k1000.ivecs";
    queryTypeToGroundtruthsMap[TESTING]["DEEP100M"] = directory_path + "DEEP100M/query.groundtruth.10K.k1000.ivecs";
    queryTypeToGroundtruthsMap[NOISY_TESTING]["DEEP100M"] = directory_path + "DEEP100M/gauss_noisy_queries/query.10K.groundtruth.noise" + noise_perc + ".ivecs";

    queryTypeToGroundtruthDistancesMap[TRAINING]["DEEP100M"] = directory_path + "DEEP100M/learn.groundtruth.1M.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[VALIDATION]["DEEP100M"] = directory_path + "DEEP100M/validation.groundtruth.10K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[TESTING]["DEEP100M"] = directory_path + "DEEP100M/query.groundtruth.10K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[NOISY_TESTING]["DEEP100M"] = directory_path + "DEEP100M/gauss_noisy_queries/query.10K.groundtruth.noise" + noise_perc + ".fvecs";

    // GLOVE100
    baseVectorsMap["GLOVE100"] = directory_path + "GLOVE100/base.1183514.fvecs";

    queryTypeToVectorsMap[TRAINING]["GLOVE100"] = directory_path + "GLOVE100/learn.100K.fvecs";
    queryTypeToVectorsMap[VALIDATION]["GLOVE100"] = directory_path + "GLOVE100/validation.10K.fvecs";
    queryTypeToVectorsMap[TESTING]["GLOVE100"] = directory_path + "GLOVE100/query.10K.fvecs";
    queryTypeToVectorsMap[NOISY_TESTING]["GLOVE100"] = directory_path + "GLOVE100/gauss_noisy_queries/query.10K.noise" + noise_perc + ".fvecs";

    queryTypeToGroundtruthsMap[TRAINING]["GLOVE100"] = directory_path + "GLOVE100/learn.groundtruth.100K.k1000.ivecs";
    queryTypeToGroundtruthsMap[VALIDATION]["GLOVE100"] = directory_path + "GLOVE100/validation.groundtruth.10K.k1000.ivecs";
    queryTypeToGroundtruthsMap[TESTING]["GLOVE100"] = directory_path + "GLOVE100/query.groundtruth.10K.k1000.ivecs";
    queryTypeToGroundtruthsMap[NOISY_TESTING]["GLOVE100"] = directory_path + "GLOVE100/gauss_noisy_queries/query.10K.groundtruth.noise" + noise_perc + ".ivecs";

    queryTypeToGroundtruthDistancesMap[TRAINING]["GLOVE100"] = directory_path + "GLOVE100/learn.groundtruth.100K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[VALIDATION]["GLOVE100"] = directory_path + "GLOVE100/validation.groundtruth.10K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[TESTING]["GLOVE100"] = directory_path + "GLOVE100/query.groundtruth.10K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[NOISY_TESTING]["GLOVE100"] = directory_path + "GLOVE100/gauss_noisy_queries/query.10K.groundtruth.noise" + noise_perc + ".fvecs";
    
    // GIST1M
    baseVectorsMap["GIST1M"] = directory_path + "GIST1M/base.1M.fvecs";

    queryTypeToVectorsMap[TRAINING]["GIST1M"] = directory_path + "GIST1M/learn.100K.fvecs";
    queryTypeToVectorsMap[VALIDATION]["GIST1M"] = directory_path + "GIST1M/validation.1K.fvecs";
    queryTypeToVectorsMap[TESTING]["GIST1M"] = directory_path + "GIST1M/query.1K.fvecs";
    queryTypeToVectorsMap[NOISY_TESTING]["GIST1M"] = directory_path + "GIST1M/gauss_noisy_queries/query.1K.noise" + noise_perc + ".fvecs";

    queryTypeToGroundtruthsMap[TRAINING]["GIST1M"] = directory_path + "GIST1M/learn.groundtruth.100K.k1000.ivecs";
    queryTypeToGroundtruthsMap[VALIDATION]["GIST1M"] = directory_path + "GIST1M/validation.groundtruth.1K.k1000.ivecs";
    queryTypeToGroundtruthsMap[TESTING]["GIST1M"] = directory_path + "GIST1M/query.groundtruth.1K.k1000.ivecs";
    queryTypeToGroundtruthsMap[NOISY_TESTING]["GIST1M"] = directory_path + "GIST1M/gauss_noisy_queries/query.1K.groundtruth.noise" + noise_perc + ".ivecs";

    queryTypeToGroundtruthDistancesMap[TRAINING]["GIST1M"] = directory_path + "GIST1M/learn.groundtruth.100K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[VALIDATION]["GIST1M"] = directory_path + "GIST1M/validation.groundtruth.1K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[TESTING]["GIST1M"] = directory_path + "GIST1M/query.groundtruth.1K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[NOISY_TESTING]["GIST1M"] = directory_path + "GIST1M/gauss_noisy_queries/query.1K.groundtruth.noise" + noise_perc + ".fvecs";

    // T2I100M
    baseVectorsMap["T2I100M"] = directory_path + "T2I100M/base.100M.fvecs";        
    queryTypeToVectorsMap[TRAINING]["T2I100M"] = directory_path + "T2I100M/learn.1M.fvecs";
    queryTypeToVectorsMap[VALIDATION]["T2I100M"] = directory_path + "T2I100M/validation.10K.fvecs";
    queryTypeToVectorsMap[TESTING]["T2I100M"] = directory_path + "T2I100M/query.10K.fvecs";
    queryTypeToVectorsMap[NOISY_TESTING]["T2I100M"] = directory_path + "T2I100M/gauss_noisy_queries/query.10K.noise" + noise_perc + ".fvecs";

    queryTypeToGroundtruthsMap[TRAINING]["T2I100M"] = directory_path + "T2I100M/learn.groundtruth.1M.k1000.ivecs";
    queryTypeToGroundtruthsMap[VALIDATION]["T2I100M"] = directory_path + "T2I100M/validation.groundtruth.10K.k1000.ivecs";
    queryTypeToGroundtruthsMap[TESTING]["T2I100M"] = directory_path + "T2I100M/query.groundtruth.10K.k1000.ivecs";
    queryTypeToGroundtruthsMap[NOISY_TESTING]["T2I100M"] = directory_path + "T2I100M/gauss_noisy_queries/query.10K.groundtruth.noise" + noise_perc + ".ivecs";   

    queryTypeToGroundtruthDistancesMap[TRAINING]["T2I100M"] = directory_path + "T2I100M/learn.groundtruth.1M.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[VALIDATION]["T2I100M"] = directory_path + "T2I100M/validation.groundtruth.10K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[TESTING]["T2I100M"] = directory_path + "T2I100M/query.groundtruth.10K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[NOISY_TESTING]["T2I100M"] = directory_path + "T2I100M/gauss_noisy_queries/query.10K.groundtruth.noise" + noise_perc + ".fvecs";
}

float* VectorDataLoader::loadDB(size_t* d_out, size_t* n_out) {
    std::string baseVectorsPath = baseVectorsMap[dataset_name];
    
    printf(">> Loading base vectors from: %s\n", baseVectorsPath.c_str());
    float *db = fvecs_read(baseVectorsPath.c_str(), d_out, n_out);

    return db;
}

float* VectorDataLoader::loadQueries(size_t* d_out, size_t* n_out) {
    std::string queryVectorsPath = queryTypeToVectorsMap[query_type][dataset_name];
    
    printf(">> Loading queries from: %s\n", queryVectorsPath.c_str());
    float *queries = fvecs_read(queryVectorsPath.c_str(), d_out, n_out);

    // skip first 5000 queries:
    //*n_out -= 5000;
    //queries += 5000 * *d_out;

    return queries;
}

int* VectorDataLoader::loadQueriesGroundtruths(size_t* k_out, size_t* n_out) {
    std::string queryGroundtruthsPath = queryTypeToGroundtruthsMap[query_type][dataset_name];
    
    printf(">> Loading queries groundtruths from: %s\n", queryGroundtruthsPath.c_str());
    int *gt = ivecs_read(queryGroundtruthsPath.c_str(), k_out, n_out);

    return gt;
}

float* VectorDataLoader::loadQueriesGroundtruthDistances(size_t* k_out, size_t* n_out) {
    std::string queryGroundtruthDistancesPath = queryTypeToGroundtruthDistancesMap[query_type][dataset_name];
    
    printf(">> Loading queries groundtruth distances from: %s\n", queryGroundtruthDistancesPath.c_str());
    float *gt = fvecs_read(queryGroundtruthDistancesPath.c_str(), k_out, n_out);

    return gt;
}


