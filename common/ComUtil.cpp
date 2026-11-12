#include "ComUtil.h"

#include <sys/time.h>

#include <cstdio>
#include <cerrno>
#include <vector>
#include <memory>
#include <filesystem>
#include <system_error>
#include <unistd.h>
#include <faiss/index_io.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/invlists/InvertedLists.h>
#include <cstring>
#include <algorithm>
#include <unordered_set>

std::vector<double> parse_recall_targets(const std::string& text) {
  std::vector<double> targets;
  std::istringstream input(text);
  std::string value;
  while (std::getline(input, value, ',')) {
    const double target = parse_numeric_option<double>(value.c_str(), "--target-recalls");
    if (target <= 0 || target > 1 || std::find(targets.begin(), targets.end(), target) != targets.end())
      throw std::invalid_argument("Invalid or duplicate recall target: " + value);
    targets.push_back(target);
  }
  if (targets.empty() || text.back() == ',') throw std::invalid_argument("Expected comma-separated recall targets");
  return targets;
}

void validate_cached_vectors(const faiss::Index* index, const float* vectors, size_t n, size_t d) {
  auto mismatch = [] { throw std::runtime_error("Cached index vectors or IDs do not match the base dataset"); };
  if (auto h = dynamic_cast<const faiss::IndexHNSWFlat*>(index)) {
    auto flat = dynamic_cast<const faiss::IndexFlat*>(h->storage);
    if (!flat || flat->ntotal != n || std::memcmp(flat->get_xb(), vectors, n * d * sizeof(float)) != 0) mismatch();
    return;
  }
  auto ivf = dynamic_cast<const faiss::IndexIVFFlat*>(index);
  if (!ivf || ivf->invlists->use_iterator || ivf->code_size != d * sizeof(float)) mismatch();
  std::vector<bool> seen(n, false);
  for (size_t list = 0; list < ivf->nlist; ++list) {
    const auto count = ivf->invlists->list_size(list);
    faiss::InvertedLists::ScopedIds ids(ivf->invlists, list);
    faiss::InvertedLists::ScopedCodes codes(ivf->invlists, list);
    for (size_t j = 0; j < count; ++j) {
      auto id = ids.get()[j];
      if (id < 0 || size_t(id) >= n || seen[id] ||
          std::memcmp(codes.get() + j * ivf->code_size, vectors + size_t(id) * d, ivf->code_size) != 0) mismatch();
      seen[id] = true;
    }
  }
}

bool index_file_exists(const char* path) {
  if (!path) return false;
  namespace fs = std::filesystem;
  std::error_code ec;
  const auto status = fs::symlink_status(path, ec);
  if (ec == std::errc::no_such_file_or_directory) return false;
  if (ec) throw std::system_error(ec, std::string("Cannot inspect index ") + path);
  if (!fs::exists(status)) return false;
  if (!fs::is_regular_file(fs::status(path)))
    throw std::runtime_error(std::string("Index path is not a regular file: ") + path);
  return true;
}

void prepare_index_parent(const char* path) {
  const auto parent = std::filesystem::path(path).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
}

// Write to a unique file in the destination directory, then publish it without
// replacing an index another process may have created while we were building.
void save_new_index(const faiss::Index* index, const char* path) {
  namespace fs = std::filesystem;
  prepare_index_parent(path);
  std::string pattern = std::string(path) + ".tmp.XXXXXX";
  std::vector<char> name(pattern.begin(), pattern.end());
  name.push_back('\0');
  int fd = mkstemp(name.data());
  if (fd < 0) throw std::system_error(errno, std::generic_category(), "Cannot create temporary index");
  close(fd);
  struct Cleanup {
    const char* path;
    ~Cleanup() { std::error_code ignored; fs::remove(path, ignored); }
  } cleanup{name.data()};
  faiss::write_index(index, name.data());
  std::error_code ec;
  fs::create_hard_link(name.data(), path, ec);
  if (ec) throw std::system_error(ec, std::string("Cannot publish new index ") + path);
}

double recall_at_k_avg(
    const faiss::idx_t* gt,
    const faiss::idx_t* retrieved,
    const float* distances,
    int k,
    int nq,
    const faiss::idx_t* gt_all,
    int k_all,
    bool verbose) {
  double total = 0.0;
  for (int i = 0; i < nq; ++i) {
    std::unordered_set<faiss::idx_t> gt_set(gt + i * k, gt + (i + 1) * k);
    int matches = 0;

    for (int j = 0; j < k; ++j) {
      if (gt_set.count(retrieved[i * k + j])) {
        matches++;
      }
    }

    double recall = (double)matches / (double)k;

    if (verbose) {
      printf("Query[%d]: Recall@%d: %.4f. NN: %f NN_id: %ld\n",
             i,
             k,
             recall,
             distances[i * k],
             retrieved[i * k]);

      if (gt_all[i * k_all] != retrieved[i * k]) {
        for (int j = 0; j < k_all; j++) {
          if (gt_all[i * k_all + j] == retrieved[i * k]) {
            printf("    NN_id found at %d-th position.\n", j + 1);
            break;
          }
        }
      }
    }

    total += recall;
  }

  return total / (double)(nq);
}

double get_rqut_at_k(const faiss::idx_t* gt,
                     const faiss::idx_t* retrieved,
                     const float* distances,
                     int k,
                     int nq,
                     const faiss::idx_t* gt_all,
                     int k_all,
                     double recall_target) {
  int queries_under_target = 0;
  for (int i = 0; i < nq; ++i) {
    std::unordered_set<faiss::idx_t> gt_set(gt + i * k, gt + (i + 1) * k);
    int matches = 0;

    for (int j = 0; j < k; ++j) {
      if (gt_set.count(retrieved[i * k + j])) {
        matches++;
      }
    }

    double recall = (double)matches / (double)k;

    if (recall < recall_target) {
      queries_under_target++;
    }
  }

  return (double)queries_under_target / (double)(nq);
}

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
    bool shuffle) {
  if (sample_size <= 0 || sample_size > nQ || dQ <= 0 || k <= 0)
    throw std::invalid_argument("Query sample requires 0 < sample_size <= available queries, d > 0 and k > 0");
  if (!vecsQ || !gt || !gt_dist || !vecsQ_out || !indicesQ_out || !gtQ_out || !gt_dist_out)
    throw std::invalid_argument("Query sampling buffers must not be null");
  auto vecsQ_sample = std::make_unique<float[]>(sample_size * dQ);

  auto gtQ_sample = std::make_unique<faiss::idx_t[]>(sample_size * k);
  auto gt_dist_sample = std::make_unique<float[]>(sample_size * k);

  auto indicesQ_sample = std::make_unique<faiss::idx_t[]>(sample_size);

  auto all_indices = std::make_unique<faiss::idx_t[]>(nQ);
  for (faiss::idx_t i = 0; i < nQ; i++) {
    all_indices[i] = i;
  }

  // if (shuffle) {
  //     std::shuffle(
  //             all_indices,
  //             all_indices + nQ,
  //             std::default_random_engine(SEED));
  // }

  for (faiss::idx_t i = 0; i < sample_size; i++) {
    faiss::idx_t idx = all_indices[i];

    for (faiss::idx_t j = 0; j < dQ; j++) {
      vecsQ_sample[i * dQ + j] = vecsQ[idx * dQ + j];
    }

    for (faiss::idx_t j = 0; j < k; j++) {
      gtQ_sample[i * k + j] = gt[idx * k + j];
      gt_dist_sample[i * k + j] = gt_dist[idx * k + j];
    }

    indicesQ_sample[i] = idx;
  }

  *vecsQ_out = vecsQ_sample.release();
  *indicesQ_out = indicesQ_sample.release();
  *gtQ_out = gtQ_sample.release();
  *gt_dist_out = gt_dist_sample.release();

  return;
}

double elapsed() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}
