// Create the shared L2 IVF index used by the competitor experiments.
#include <common/ComUtil.h>
#include <common/VectorDataLoader.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <omp.h>
#include <algorithm>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) try {
  if (argc != 6) throw std::invalid_argument("Usage: ivf_index_builder DATASET DATA_DIR NLIST INDEX_PATH THREADS");
  const int nlist = parse_numeric_option<int>(argv[3], "NLIST");
  const int threads = parse_numeric_option<int>(argv[5], "THREADS");
  if (nlist <= 0 || threads <= 0) throw std::invalid_argument("NLIST and THREADS must be positive");
  if (index_file_exists(argv[4])) return 0;
  omp_set_num_threads(threads);
  VectorDataLoader loader(argv[1], TESTING, "5", argv[2], "L2");
  loader.initializeDataMaps();
  if (!loader.baseVectorsMap.count(argv[1])) throw std::invalid_argument("Unknown dataset");
  size_t d = 0, n = 0;
  std::unique_ptr<float[]> base(loader.loadDB(&d, &n));
  if (!base || !d || n / 10 < size_t(nlist))
    throw std::invalid_argument("IVF nlist exceeds the 10% index-training sample");
  const size_t training_count = n / 10;
  std::vector<float> training(training_count * d);
  for (size_t i = 0; i < training_count; ++i)
    std::memcpy(training.data() + i * d, base.get() + i * 10 * d, d * sizeof(float));
  faiss::IndexFlatL2 quantizer(d);
  faiss::IndexIVFFlat index(&quantizer, d, nlist, faiss::METRIC_L2);
  index.train(training_count, training.data());
  training.clear();
  training.shrink_to_fit();
  for (size_t start = 0; start < n; start += 10000000) {
    const size_t count = std::min(size_t(10000000), n - start);
    index.add(count, base.get() + start * d);
  }
  save_new_index(&index, argv[4]);
  std::cout << "Saved shared IVF index: " << argv[4] << '\n';
  return 0;
} catch (const std::exception& error) {
  std::cerr << "IVF index builder: " << error.what() << '\n';
  return 1;
}
