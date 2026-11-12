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
    const std::set<std::string> allowed = {"index", "dataset", "dataset-dir-prefix", "index-filepath", "save-index", "results", "metric", "k", "validation-query-num", "query-num", "M", "efConstruction", "efSearch", "nlist", "nprobe", "target-recall", "target-recalls", "threads", "evaluate", "tuning-metric", "rqut-target", "training-query-num", "training-threads", "n-estimators", "learning-rate", "num-leaves", "min-child-samples", "seed", "logging-interval", "logging-intervals", "cutoff", "fixed-amount-of-search", "save-models"};
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
    std::cout << "Native LAET pipeline: --dataset NAME --dataset-dir-prefix DIR "
                 "--index hnsw|ivf [--target-recall R | --target-recalls R,R,...] [--fixed-amount-of-search N] "
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
        plus(o.get("method", "laet") != "darth"),
        lb(o.get("method", "laet") == "darth-plus-lb"),
        k(o.number<int>("k", 10)), threads(o.number<int>("threads", omp_get_max_threads())),
        train_threads(o.number<int>("training-threads", threads)),
        logging(o.number<int>("logging-interval", 10)),
        target(o.number<double>("target-recall", .8)), cutoff(o.number<double>("cutoff", 30)) {
    require(o.get("index", "hnsw") == "hnsw" || ivf, "index must be hnsw or ivf");
    const auto method = o.get("method", "laet");
    require(method == "laet", "Unknown method");
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

std::string training_parameters(const Options& o, const Configuration& c, bool quantile, double alpha) {
  int leaves = o.number<int>("num-leaves",31), minimum = o.number<int>("min-child-samples",20);
  double rate = o.number<double>("learning-rate",.1);
  require(leaves > 1 && minimum > 0 && rate > 0, "Invalid LightGBM training parameters");
  std::ostringstream p; p << std::setprecision(17)
    << "objective=" << (quantile ? "quantile" : "regression")
    << " verbosity=-1 seed=" << o.number<int>("seed",42)
    << " num_threads=" << c.train_threads << " learning_rate=" << rate
    << " num_leaves=" << leaves << " min_data_in_leaf=" << minimum
    << " max_depth=-1 min_sum_hessian_in_leaf=0.001 min_gain_to_split=0"
    << " feature_fraction=1 bagging_fraction=1 bagging_freq=0 lambda_l1=0 lambda_l2=0"
    << " bin_construct_sample_cnt=200000";
  if (quantile) p << " alpha=" << alpha;
  return p.str();
}
Model train(const Matrix& data, const Options& o, const Configuration& c,
            bool quantile, double alpha, const std::string& label) {
  auto start = Clock::now();
  const auto parameters = training_parameters(o,c,quantile,alpha);
  int rounds = o.number<int>("n-estimators",100); require(rounds > 0,"n-estimators must be positive");
  DatasetHandle raw_data = nullptr;
  lgb_check(LGBM_DatasetCreateFromMat(data.x.data(),C_API_DTYPE_FLOAT64,data.rows(),data.names.size(),1,parameters.c_str(),nullptr,&raw_data));
  Dataset dataset(raw_data);
  std::vector<const char*> names; for (const auto& name : data.names) names.push_back(name.c_str());
  lgb_check(LGBM_DatasetSetFeatureNames(dataset.get(),names.data(),names.size()));
  std::vector<float> labels(data.y.begin(),data.y.end());
  lgb_check(LGBM_DatasetSetField(dataset.get(),"label",labels.data(),labels.size(),C_API_DTYPE_FLOAT32));
  BoosterHandle raw_model = nullptr; lgb_check(LGBM_BoosterCreate(dataset.get(),parameters.c_str(),&raw_model));
  Model model(raw_model);
  int finished = 0;
  for (int i = 0; i < rounds && !finished; ++i) lgb_check(LGBM_BoosterUpdateOneIter(model.get(),&finished));
  // Match LightGBM Python's default post-fit conversion: detach inference
  // trees from the training Dataset using RAM only. The original booster is
  // destroyed before its Dataset; deployment keeps neither training object.
  std::vector<char> model_text(65536);
  int64_t required = 0;
  lgb_check(LGBM_BoosterSaveModelToString(model.get(),0,-1,0,model_text.size(),&required,model_text.data()));
  if (required > int64_t(model_text.size())) {
    model_text.resize(required);
    lgb_check(LGBM_BoosterSaveModelToString(model.get(),0,-1,0,model_text.size(),&required,model_text.data()));
  }
  BoosterHandle inference_handle = nullptr;
  int iterations = 0;
  lgb_check(LGBM_BoosterLoadModelFromString(model_text.data(),&iterations,&inference_handle));
  Model inference(inference_handle);
  std::cout << "TRAIN model=" << label << " seconds=" << seconds(start) << '\n';
  if (o.has("save-models")) {
    auto path=std::filesystem::path(o.get("save-models")+"."+label+".txt");
    if(path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    lgb_check(LGBM_BoosterSaveModel(inference.get(),0,-1,0,path.string().c_str()));
  }
  return inference;
}
std::vector<double> predict(BoosterHandle model, const Matrix& data, int threads) {
  std::vector<double> result(data.rows()); int64_t length = 0;
  lgb_check(LGBM_BoosterPredictForMat(model,data.x.data(),C_API_DTYPE_FLOAT64,data.rows(),data.names.size(),1,
      C_API_PREDICT_NORMAL,0,-1,("num_threads="+std::to_string(threads)).c_str(),&length,result.data()));
  require(length == int64_t(result.size()),"LightGBM returned an unexpected prediction count");
  return result;
}
struct Snapshot { int step, distances; double recall; std::vector<double> features; };
double rounded(float value) { char buf[128]; std::snprintf(buf,sizeof(buf),"%f",double(value)); return std::strtod(buf,nullptr); }
Matrix observations(faiss::Index* index,const Vectors& base,Queries& queries,const Configuration& c,int& threshold) {
  size_t n=queries.vectors.n;std::vector<float>D(n*c.k);std::vector<idx_t>I(n*c.k);
  faiss::DeclarativeRecallDataManager manager(D.data(),I.data(),queries.truth.data(),queries.truth_distances.data(),n,base.d,c.k,queries.vectors.values.get(),nullptr,base.values.get(),base.n,queries.truth.data(),c.k);
  std::vector<std::vector<Snapshot>> rows(n);
  auto capture=[&](idx_t id,int step,int nd,int inserted,float first,const std::vector<float>& centroids,float recall){
    float nearest=manager.get_nearest_dist_of_query(id),tenth=c.k>=10?manager.get_kth_nearest_dist_of_query(id,9):-1;
    std::vector<double> f;
    if(c.ivf){
      f={rounded(nearest),rounded(tenth),rounded(tenth!=-1?nearest/tenth:-1),rounded(nearest/centroids.at(0))};
      for(size_t z=9;z<=100;z+=10) f.push_back(rounded(centroids.size()>z?nearest/centroids[z]:-1));
    }else{
      f={rounded(first),rounded(nearest),rounded(tenth),rounded(first>0?nearest/first:-1),rounded(tenth!=-1&&first>0?tenth/first:-1)};
    }
    rows[id].push_back({step,nd,rounded(recall),std::move(f)});
  };
  if(c.ivf){
    faiss::DeclarativeRecallDataCollectorIVF collector(&manager,c.intervals,c.cutoff);
    collector.observation_sink=capture;
    static_cast<faiss::IndexIVFFlat*>(index)->darth_plus_generate_training_data(n,queries.vectors.values.get(),c.k,D.data(),I.data(),&collector);
    threshold=std::numeric_limits<int>::max();for(auto& q:rows)for(auto& row:q)if(row.step==5)threshold=std::min(threshold,row.distances);
    require(threshold!=std::numeric_limits<int>::max(),"No step-5 observations; increase nprobe");
  }else{
    faiss::DeclarativeRecallDataCollectorHNSW collector(&manager,c.logging);
    collector.observation_sink=[&](idx_t id,int step,int nd,int inserted,float first,float recall){capture(id,step,nd,inserted,first,{},recall);};
    static_cast<faiss::IndexHNSWFlat*>(index)->search_declarative_recall_data_generation(n,queries.vectors.values.get(),c.k,D.data(),I.data(),&collector);
  }
  Matrix out;out.queries=n; const size_t width=base.d+(c.ivf?14:5);
  for(size_t j=0;j<width;++j)out.names.push_back("feature_"+std::to_string(j));
  for(size_t qid=0;qid<n;++qid){
    if(rows[qid].empty())continue;
    double best=-1;int bestdist=std::numeric_limits<int>::max();
    for(const auto& row:rows[qid])if(row.recall>best||(row.recall==best&&row.distances<bestdist)){best=row.recall;bestdist=row.distances;}
    for(const auto& row:rows[qid])if(row.distances==threshold){out.x.insert(out.x.end(),row.features.begin(),row.features.end());for(size_t z=0;z<base.d;++z)out.x.push_back(queries.vectors.values[qid*base.d+z]);out.y.push_back(bestdist);out.qids.push_back(qid);out.distances.push_back(row.distances);}
  }
  require(out.rows()>0 && out.rows()<=size_t(std::numeric_limits<int32_t>::max()),"No observations at the LAET prediction threshold");
  std::cout << "LAET_TRAINING threshold=" << threshold << " rows=" << out.rows() << " queries=" << n << "\n";
  return out;
}
struct Evaluation {double recall; double rqut; double elapsed;};
Evaluation evaluate_laet(faiss::Index* index,const Vectors& base,Queries& q,const Configuration& c,int threshold,float multiplier,std::unique_ptr<faiss::LGBMPredictor>& model, const std::string& output=""){
  size_t n=q.vectors.n;std::vector<float>D(n*c.k);std::vector<idx_t>I(n*c.k);
  faiss::DeclarativeRecallDataManager manager(D.data(),I.data(),q.truth.data(),q.truth_distances.data(),n,base.d,c.k,q.vectors.values.get(),nullptr,base.values.get(),base.n,q.truth.data(),c.k);
  auto start=Clock::now();
  if(c.ivf){faiss::LAETPredictorIVF p(&manager,threshold,multiplier,std::move(model));try {static_cast<faiss::IndexIVFFlat*>(index)->search_LAET(n,q.vectors.values.get(),c.k,D.data(),I.data(),&p);} catch (...) {model=std::move(p.predictor);throw;} model=std::move(p.predictor);}
  else{faiss::LAETPredictorHNSW p(&manager,threshold,multiplier,std::move(model));try {static_cast<faiss::IndexHNSWFlat*>(index)->search_LAET(n,q.vectors.values.get(),c.k,D.data(),I.data(),&p);} catch (...) {model=std::move(p.predictor);throw;} model=std::move(p.predictor);}
  double elapsed=seconds(start),sum=0;size_t below=0;
  if (!output.empty()) {
    std::ofstream labels(output+".ivecs",std::ios::binary), distances(output+".fvecs",std::ios::binary);
    require(bool(labels)&&bool(distances),"Cannot open result files");
    for(size_t i=0;i<n;++i){labels.write(reinterpret_cast<const char*>(&c.k),4);distances.write(reinterpret_cast<const char*>(&c.k),4);for(int j=0;j<c.k;++j){require(I[i*c.k+j]<=std::numeric_limits<int32_t>::max(),"Result ID exceeds ivecs range");int32_t id=I[i*c.k+j];labels.write(reinterpret_cast<const char*>(&id),4);}distances.write(reinterpret_cast<const char*>(D.data()+i*c.k),4*c.k);}
    require(bool(labels)&&bool(distances),"Failed writing results");
  }
  if(q.truth.empty()) return {NAN,NAN,elapsed};
  for(size_t i=0;i<n;++i){
    size_t hits=0;
    for(int j=0;j<c.k;++j)
      if(std::find(q.truth.begin()+i*c.k,q.truth.begin()+(i+1)*c.k,I[i*c.k+j]) != q.truth.begin()+(i+1)*c.k) ++hits;
    double recall=double(hits)/c.k;
    sum+=rounded(recall); below+=recall<c.target;
  }
  return {sum/n,double(below)/n,elapsed};
}
} // namespace
int main(int argc,char**argv) try {
 std::setlocale(LC_NUMERIC,"C");std::cout<<std::setprecision(12);Options o(argc,argv);Configuration c(o);omp_set_num_threads(c.threads);
  require(o.has("dataset"), "Required option: --dataset");
  const char* env=std::getenv("DATASET_DIRECTORY"); if(!env||!*env)env=std::getenv("DARTH_DATASET_DIR");
  const auto dir=o.get("dataset-dir-prefix",env?env:""); require(!dir.empty(),"Specify dataset base directory");
  VectorDataLoader loader(o.get("dataset"), TRAINING, "5", dir, o.get("metric","L2"));
  loader.initializeDataMaps(); loader.basePath();
  loader.validateDistinctQueryFiles({TRAINING, VALIDATION, TESTING});
 auto metric_name=o.get("metric","L2");require(metric_name=="L2"||metric_name=="inner-product","Unknown metric");auto metric=metric_name=="L2"?faiss::METRIC_L2:faiss::METRIC_INNER_PRODUCT;
 require(!(o.has("target-recall") && o.has("target-recalls")), "Use either --target-recall or --target-recalls");
 const auto targets = o.has("target-recall") ? std::vector<double>{c.target} :
     parse_recall_targets(o.get("target-recalls", "0.8,0.85,0.90,0.95,0.99"));
 Vectors base(loader,true);require(base.n>=size_t(c.k),"k exceeds database size");auto index=make_index(o,c,base,metric);
 auto load=[&](query_type_t type,const char* count,bool need_truth=true){return Queries(loader,type,o.number<size_t>(count,0),base,c.k,metric,c.threads,need_truth);};
 auto trainq=load(TRAINING,"training-query-num");int threshold=o.number<int>("fixed-amount-of-search",100);require(threshold>0,"fixed-amount-of-search must be positive");
 auto data=observations(index.get(),base,trainq,c,threshold);omp_set_num_threads(c.train_threads);auto booster=train(data,o,c,false,0,"laet");
 auto fitted=predict(booster.get(),data,c.train_threads);double mse=0,mae=0;for(size_t i=0;i<data.rows();++i){double delta=fitted[i]-data.y[i];mse+=delta*delta;mae+=std::abs(delta);}std::cout<<"METRICS split=training mse="<<mse/data.rows()<<" mae="<<mae/data.rows()<<"\n";
 data=Matrix{};omp_set_num_threads(c.threads);auto model=std::make_unique<faiss::LGBMPredictor>(std::move(booster));auto valid=load(VALIDATION,"validation-query-num");
 std::vector<float> candidates;if(c.ivf)candidates={.001f,.005f,.01f,.02f,.03f,.04f};for(int i=c.ivf?10:2;i<=(c.ivf?120:100);++i)candidates.push_back(static_cast<float>(i*.05));
 auto criterion=o.get("tuning-metric","recall");double rqut=o.number<double>("rqut-target",.1);require(criterion=="recall"||criterion=="rqut","Unknown tuning metric");require(rqut>=0&&rqut<=1,"Invalid rqut-target");
 auto query=load(TESTING,"query-num",o.has("evaluate"));
 bool failed=false;
 for (double target : targets) {
   c.target=target;
   std::cout << "TARGET recall=" << target << '\n';
   int lo=0,hi=int(candidates.size())-1,best=-1;Evaluation found{};
   while(lo<=hi){int m=lo+(hi-lo)/2;auto e=evaluate_laet(index.get(),base,valid,c,threshold,candidates[m],model);std::cout<<"TUNE multiplier="<<candidates[m]<<" recall="<<e.recall<<" rqut="<<e.rqut<<"\n";bool pass=criterion=="recall"?e.recall>=c.target:e.rqut<=rqut;if(pass){best=m;found=e;hi=m-1;}else lo=m+1;}
   if(best<0){std::cerr<<"Target "<<target<<": no multiplier reached the validation criterion\n";failed=true;continue;}
   std::cout<<"SELECTED multiplier="<<candidates[best]<<" recall="<<found.recall<<" rqut="<<found.rqut<<"\n";
   auto output=o.get("results");
   if(!output.empty() && (targets.size()>1 || o.has("target-recalls"))){std::ostringstream suffix;suffix<<std::setprecision(12)<<target;output+="-target"+suffix.str();}
   auto final=evaluate_laet(index.get(),base,query,c,threshold,candidates[best],model,output);
   std::cout<<"SEARCH target="<<target;if(!query.truth.empty())std::cout<<" recall="<<final.recall<<" rqut="<<final.rqut;std::cout<<" seconds="<<final.elapsed<<"\n";
 }
 return failed ? 1 : 0;
}catch(const std::exception&e){std::cerr<<"LAET pipeline: "<<e.what()<<"\n";return 1;}
