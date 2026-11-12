// Standalone native competitor pipeline.
// Intermediate observations and LightGBM boosters never need to leave memory.
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include <faiss/impl/DeclarativeRecall.h>
#include <faiss/utils/distances.h>
#include <omp.h>
#include <common/ComUtil.h>
#include <common/VectorDataLoader.h>
#include <algorithm>
#include <chrono>
#include <clocale>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using faiss::idx_t;
using Model = faiss::LGBMPredictor::Model;
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
void require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument(message);
}
void lgb_check(int status) {
  if (status != 0) throw std::runtime_error(LGBM_GetLastError());
}
struct DatasetDeleter { void operator()(void* value) const { LGBM_DatasetFree(value); } };
using Dataset = std::unique_ptr<void, DatasetDeleter>;

struct Options {
  std::map<std::string, std::string> values;
  Options(int argc, char** argv) {
    const std::set<std::string> allowed = {"index", "dataset", "dataset-dir-prefix", "index-filepath", "save-index", "results", "metric", "k", "validation-query-num", "query-num", "M", "efConstruction", "efSearch", "nlist", "nprobe", "target-recall", "target-recalls", "threads", "evaluate", "tuning-metric", "rqut-target", "max-budget"};
    for (int i = 1; i < argc; ++i) {
      std::string name = argv[i];
      if (name == "--help") { help(); std::exit(0); }
      require(name.rfind("--", 0) == 0, "Expected a --name value option: " + name);
      name = name.substr(2);
      require(allowed.count(name), "Unknown option: --" + name);
      if (name == "evaluate") { require(values.emplace(name,"1").second,"Repeated --evaluate"); continue; }
      require(i + 1 < argc, "Missing value for --" + name);
      require(values.emplace(name, argv[++i]).second, "Repeated option: --" + name);
    }
  }
  std::string get(const std::string& name, const std::string& fallback = "") const {
    auto value = values.find(name); return value == values.end() ? fallback : value->second;
  }
  bool has(const std::string& name) const { return values.count(name); }
  template<class T> T number(const std::string& name, T fallback) const {
    return has(name) ? parse_numeric_option<T>(get(name).c_str(), ("--" + name).c_str()) : fallback;
  }
  static void help() {
    std::cout << "Native REM pipeline: --dataset NAME --dataset-dir-prefix DIR "
                 "--index hnsw|ivf [--target-recall R | --target-recalls R,R,...] --max-budget N "
                 "[--tuning-metric recall|rqut --rqut-target R].\n"
                 "Default targets: 0.80,0.85,0.90,0.95,0.99.\n";
  }

};
  struct Vectors
  {
    size_t d = 0, n = 0;
    std::unique_ptr<float[]> values;
    Vectors(VectorDataLoader &loader, bool database, query_type_t type = TRAINING, size_t limit = 0)
    {
      const auto &path = database ? loader.basePath() : loader.queryPath(type);
      values.reset(database ? loader.loadDB(&d, &n, limit) : loader.loadQueries(type, &d, &n, limit));
      require(n && d && (!limit || n == limit), "Vector file has too few rows: " + path);
      require(n <= size_t(std::numeric_limits<idx_t>::max()), "Too many vectors");
    }
  };
  struct Queries
  {
    Vectors vectors;
    std::vector<idx_t> truth;
    std::vector<float> truth_distances;
    Queries(VectorDataLoader &loader, query_type_t type, size_t limit,
            const Vectors &base, int k, faiss::MetricType metric, int threads, bool need_truth = true)
        : vectors(loader, false, type, limit)
    {
      require(vectors.d == base.d, "Query and base dimensions differ: " + loader.queryPath(type));
      if (!need_truth)
        return;
      const auto &gt_path = loader.groundtruthPath(type);
      truth.resize(vectors.n * k);
      truth_distances.resize(vectors.n * k);
      if (!std::filesystem::is_regular_file(gt_path))
      {
        std::cout << "Computing exact ground truth for " << vectors.n << " queries\n";
        // Borrow the database: avoid a second full database copy in IndexFlat.
        if (metric == faiss::METRIC_L2)
          faiss::knn_L2sqr(vectors.values.get(), base.values.get(), base.d, vectors.n,
                           base.n, k, truth_distances.data(), truth.data());
        else
          faiss::knn_inner_product(vectors.values.get(), base.values.get(), base.d,
                                   vectors.n, base.n, k, truth_distances.data(), truth.data());
      }
      else
      {
        size_t width, rows;
        std::unique_ptr<int[]> ids(loader.loadQueriesGroundtruths(type, &width, &rows, limit));
        require(width >= size_t(k) && rows >= vectors.n && (limit || rows == vectors.n),
                "Ground-truth shape does not match queries/k: " + gt_path);
        for (size_t i = 0; i < vectors.n; ++i)
        {
          std::set<int> unique;
          for (int j = 0; j < k; ++j)
          {
            auto id = ids[i * width + j];
            require(id >= 0 && size_t(id) < base.n && unique.insert(id).second,
                    "Ground truth contains an invalid or repeated neighbor ID");
            truth[i * k + j] = id;
          }
        }
#pragma omp parallel for num_threads(threads)
        for (size_t i = 0; i < vectors.n; ++i)
          for (int j = 0; j < k; ++j)
          {
            const auto *q = vectors.values.get() + i * base.d;
            const auto *b = base.values.get() + truth[i * k + j] * base.d;
            truth_distances[i * k + j] = metric == faiss::METRIC_L2
                                             ? faiss::fvec_L2sqr(q, b, base.d)
                                             : faiss::fvec_inner_product(q, b, base.d);
          }
      }
    }
  };
struct Matrix {
  std::vector<std::string> names;
  std::vector<double> x, y;
  std::vector<size_t> qids, distances;
  size_t queries = 0;
  size_t rows() const { return y.size(); }
};
struct Configuration {
  bool ivf, plus, lb;
  int k, threads, train_threads, logging;
  double target, cutoff;
  std::vector<faiss::AdaptiveLogInterval> intervals;
  Configuration(const Options& o)
      : ivf(o.get("index", "hnsw") == "ivf"),
        plus(o.get("method", "rem") != "darth"),
        lb(o.get("method", "rem") == "darth-plus-lb"),
        k(o.number<int>("k", 10)), threads(o.number<int>("threads", omp_get_max_threads())),
        train_threads(o.number<int>("training-threads", threads)),
        logging(o.number<int>("logging-interval", 10)),
        target(o.number<double>("target-recall", .8)), cutoff(o.number<double>("cutoff", 30)) {
    require(o.get("index", "hnsw") == "hnsw" || ivf, "index must be hnsw or ivf");
    const auto method = o.get("method", "rem");
    require(method == "rem", "Unknown method");
    require(k > 0 && threads > 0 && train_threads > 0 && logging > 0, "k, threads and logging interval must be positive");
    require(target > 0 && target <= 1 && cutoff >= 0, "Invalid target recall or cutoff");
    std::istringstream values(o.get("logging-intervals", ivf ? "100,50,20" : "20,10,5"));
    double lower[] = {0,.5,.7}, upper[] = {.5,.7,1};
    for (int i = 0; i < 3; ++i) {
      std::string value;
      require(bool(std::getline(values, value, ',')), "Expected three logging intervals");
      int interval = parse_numeric_option<int>(value.c_str(), "--logging-intervals");
      require(interval > 0, "Logging intervals must be positive");
      intervals.push_back({lower[i],upper[i],interval});
    }
    require(values.eof(), "Expected exactly three logging intervals");
  }
};
std::unique_ptr<faiss::Index> make_index(const Options& o, const Configuration& c,
                                      const Vectors& base, faiss::MetricType metric) {
  auto start = Clock::now();
  std::unique_ptr<faiss::Index> index;
  const bool cached = o.has("index-filepath") && index_file_exists(o.get("index-filepath").c_str());
  if (cached) {
    index.reset(faiss::read_index(o.get("index-filepath").c_str()));
    std::cout << "Loaded index: " << o.get("index-filepath") << '\n';
  } else if (!c.ivf) {
    int m = o.number<int>("M", 32), ef = o.number<int>("efConstruction", 500);
    require(m > 0 && ef > 0, "M and efConstruction must be positive");
    auto h = std::make_unique<faiss::IndexHNSWFlat>(base.d, m, metric);
    h->hnsw.efConstruction = ef; h->add(base.n, base.values.get()); index = std::move(h);
  } else {
    int nlist = o.number<int>("nlist", 1000);
    require(nlist > 0 && size_t(nlist) <= base.n / 10, "IVF nlist exceeds the 10% index-training sample");
    auto quantizer = std::make_unique<faiss::IndexFlat>(base.d, metric);
    auto v = std::make_unique<faiss::IndexIVFFlat>(quantizer.get(), base.d, nlist, metric);
    v->own_fields = true; quantizer.release();
    std::vector<float> sample((base.n / 10) * base.d);
    for (size_t i = 0; i < base.n / 10; ++i)
      std::copy_n(base.values.get() + 10 * i * base.d, base.d, sample.data() + i * base.d);
    v->train(base.n / 10, sample.data()); v->add(base.n, base.values.get()); index = std::move(v);
  }
  require(index->d == base.d && index->ntotal == base.n && index->metric_type == metric && index->is_trained,
          "Index dimensions, size, metric or training state do not match the base vectors");
  if (c.ivf) {
    auto* v = dynamic_cast<faiss::IndexIVFFlat*>(index.get());
    require(v != nullptr, "Expected an IndexIVFFlat");
    require(!v->invlists->use_iterator, "The native pipeline requires array-backed IVF lists");
    if (cached && o.has("nlist")) require(v->nlist == size_t(o.number<int>("nlist", 0)), "Cached index nlist differs from --nlist");
    auto probes = o.number<int>("nprobe", 100); require(probes > 0, "nprobe must be positive");
    v->nprobe = std::min<size_t>(probes, v->nlist);
    v->parallel_mode = 0;
  } else {
    auto* h = dynamic_cast<faiss::IndexHNSWFlat*>(index.get());
    require(h != nullptr, "Expected an IndexHNSWFlat");
    if (cached && o.has("M")) require(h->hnsw.nb_neighbors(1) == o.number<int>("M", 0), "Cached index M differs from --M");
    if (cached && o.has("efConstruction")) require(h->hnsw.efConstruction == o.number<int>("efConstruction", 0), "Cached index efConstruction differs from --efConstruction");
    h->hnsw.efSearch = o.number<int>("efSearch", 500);
    require(h->hnsw.efSearch > 0, "efSearch must be positive");
  }
  if (cached) validate_cached_vectors(index.get(), base.values.get(), base.n, base.d);
  if (!cached && o.has("index-filepath")) {
    save_new_index(index.get(), o.get("index-filepath").c_str());
    std::cout << "Saved new index: " << o.get("index-filepath") << '\n';
  }
  if (o.has("save-index") && (!o.has("index-filepath") || o.get("save-index") != o.get("index-filepath") || cached)) {
    prepare_index_parent(o.get("save-index").c_str());
    faiss::write_index(index.get(), o.get("save-index").c_str());
  }
  std::cout << "INDEX seconds=" << seconds(start) << '\n';
  return index;
}

struct Evaluation { double recall; double rqut; double elapsed; };
Evaluation evaluate(faiss::Index* index, Queries& q, int k, double target, int budget, bool ivf, const std::string& output="") {
  std::vector<float> distances(q.vectors.n*k);
  std::vector<idx_t> labels(q.vectors.n*k);
  if (ivf) static_cast<faiss::IndexIVFFlat*>(index)->nprobe = budget;
  else static_cast<faiss::IndexHNSWFlat*>(index)->hnsw.efSearch = budget;
  auto started=Clock::now();
  index->search(q.vectors.n,q.vectors.values.get(),k,distances.data(),labels.data());
  const double elapsed=seconds(started);
  if(!output.empty()){
    std::ofstream ids(output+".ivecs",std::ios::binary),values(output+".fvecs",std::ios::binary);require(bool(ids)&&bool(values),"Cannot open result files");
    for(size_t i=0;i<q.vectors.n;++i){ids.write(reinterpret_cast<const char*>(&k),4);values.write(reinterpret_cast<const char*>(&k),4);for(int j=0;j<k;++j){require(labels[i*k+j]<=std::numeric_limits<int32_t>::max(),"Result ID exceeds ivecs range");int32_t id=labels[i*k+j];ids.write(reinterpret_cast<const char*>(&id),4);}values.write(reinterpret_cast<const char*>(distances.data()+i*k),4*k);}
    require(bool(ids)&&bool(values),"Failed writing results");
  }
  if(q.truth.empty())return {NAN,NAN,elapsed};
  double sum=0; size_t below=0;
  for(size_t i=0;i<q.vectors.n;++i) {
    size_t hits=0;
    for(int j=0;j<k;++j)
      if(std::find(q.truth.begin()+i*k,q.truth.begin()+(i+1)*k,labels[i*k+j]) != q.truth.begin()+(i+1)*k) ++hits;
    double recall=std::round((double(hits)/k)*1000000.0)/1000000.0;
    sum+=recall; below += double(hits)/k < target;
  }
  return {sum/q.vectors.n,double(below)/q.vectors.n,elapsed};
}
} // namespace
int main(int argc,char** argv) try {
  std::setlocale(LC_NUMERIC,"C"); std::cout << std::setprecision(12);
  Options o(argc,argv); Configuration c(o); omp_set_num_threads(c.threads);
  require(o.has("dataset"), "Required option: --dataset");
  const char* env=std::getenv("DATASET_DIRECTORY"); if(!env||!*env)env=std::getenv("DARTH_DATASET_DIR");
  const auto dir=o.get("dataset-dir-prefix",env?env:""); require(!dir.empty(),"Specify dataset base directory");
  VectorDataLoader loader(o.get("dataset"), TRAINING, "5", dir, o.get("metric","L2"));
  loader.initializeDataMaps(); loader.basePath();
  loader.validateDistinctQueryFiles({VALIDATION, TESTING});
  auto metric_name=o.get("metric","L2");require(metric_name=="L2"||metric_name=="inner-product","Unknown metric");
  auto metric=metric_name=="L2"?faiss::METRIC_L2:faiss::METRIC_INNER_PRODUCT;
  require(!(o.has("target-recall") && o.has("target-recalls")), "Use either --target-recall or --target-recalls");
  const auto targets = o.has("target-recall") ? std::vector<double>{c.target} :
      parse_recall_targets(o.get("target-recalls", "0.8,0.85,0.90,0.95,0.99"));
  Vectors base(loader,true);require(base.n>=size_t(c.k),"k exceeds database size");
  auto index=make_index(o,c,base,metric);
  Queries valid(loader,VALIDATION,o.number<size_t>("validation-query-num",0),base,c.k,metric,c.threads);
  const int max_budget=o.number<int>("max-budget",c.ivf?int(static_cast<faiss::IndexIVFFlat*>(index.get())->nlist):o.number<int>("efSearch",500));
  require(max_budget>0 && (!c.ivf || size_t(max_budget)<=static_cast<faiss::IndexIVFFlat*>(index.get())->nlist),"Invalid max-budget");
  const auto criterion=o.get("tuning-metric","recall");const double rqut=o.number<double>("rqut-target",0.1);
  require(criterion=="recall"||criterion=="rqut","tuning-metric must be recall or rqut");require(rqut>=0&&rqut<=1,"rqut-target must be in [0,1]");
  Queries test(loader,TESTING,o.number<size_t>("query-num",0),base,c.k,metric,c.threads,o.has("evaluate"));
  bool failed=false;
  for (double target : targets) {
    c.target=target;
    std::cout << "TARGET recall=" << target << '\n';
    int lo=1,hi=max_budget,best=0; Evaluation best_eval{};
    while(lo<=hi){int mid=lo+(hi-lo)/2;auto eval=evaluate(index.get(),valid,c.k,c.target,mid,c.ivf);
      std::cout << "TUNE budget=" << mid << " recall=" << eval.recall << " rqut=" << eval.rqut << " seconds=" << eval.elapsed << '\n';
      bool pass=criterion=="recall"?eval.recall>=c.target:eval.rqut<=rqut;
      if(pass){best=mid;best_eval=eval;hi=mid-1;}else lo=mid+1;
    }
    if(best==0){std::cerr<<"Target "<<target<<": no budget reached the validation criterion\n";failed=true;continue;}
    std::cout << "SELECTED budget=" << best << " recall=" << best_eval.recall << " rqut=" << best_eval.rqut << '\n';
    auto output=o.get("results");
    if(!output.empty() && (targets.size()>1 || o.has("target-recalls"))){std::ostringstream suffix;suffix<<std::setprecision(12)<<target;output+="-target"+suffix.str();}
    auto eval=evaluate(index.get(),test,c.k,c.target,best,c.ivf,output);
    std::cout << "SEARCH target=" << target; if(!test.truth.empty())std::cout << " recall=" << eval.recall << " rqut=" << eval.rqut; std::cout << " seconds=" << eval.elapsed << '\n';
  }
  return failed ? 1 : 0;
} catch(const std::exception& e){std::cerr<<"REM pipeline: "<<e.what()<<'\n';return 1;}
