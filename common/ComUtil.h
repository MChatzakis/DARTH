#pragma once

#include <faiss/Index.h>
#include <string>
#include <vector>

// Parse a comma-separated recall sweep; empty input is invalid.
std::vector<double> parse_recall_targets(const std::string& text);

// Verify that a cached flat index contains these exact vectors and IDs.
void validate_cached_vectors(const faiss::Index* index, const float* vectors, size_t n, size_t d);

// Helpers for optional on-disk index reuse.
bool index_file_exists(const char* path);
void prepare_index_parent(const char* path);
void save_new_index(const faiss::Index* index, const char* path);
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <type_traits>

// Reject partial parses and out-of-range values instead of retaining defaults.
template <typename T>
T parse_numeric_option(const char* text, const char* name) {
  std::istringstream input(text);
  T value;
  if constexpr (std::is_unsigned<T>::value) {
    if (std::string(text).find('-') != std::string::npos)
      throw std::invalid_argument(std::string(name) + " must not be negative");
  }
  if (!(input >> value) || !(input >> std::ws).eof() || !std::isfinite(double(value)))
    throw std::invalid_argument(std::string("Invalid value for ") + name + ": " + text);
  return value;
}

double recall_at_k_avg(
    const faiss::idx_t* gt,
    const faiss::idx_t* retrieved,
    const float* distances,
    int k,
    int nq,
    const faiss::idx_t* gt_all,
    int k_all,
    bool verbose = false);

void get_queries(
    float* vecsQ,
    faiss::idx_t nQ,
    int dQ,
    faiss::idx_t* gt,
    float* gt_dist,
    faiss::idx_t k,
    faiss::idx_t sample_size,
    float** vecsQ_out,
    faiss::idx_t** indicesQ_out,
    faiss::idx_t** gtQ_out,
    float** gt_dist_out,
    bool shuffle = false);

double get_rqut_at_k(const faiss::idx_t* gt,
                     const faiss::idx_t* retrieved,
                     const float* distances,
                     int k,
                     int nq,
                     const faiss::idx_t* gt_all,
                     int k_all,
                     double recall_target);

double elapsed();
