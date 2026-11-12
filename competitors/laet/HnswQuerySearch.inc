// Included at the original FAISS scope to use its internal search helpers.
template <class BlockResultHandler>
void hnsw_search_LAET(
    const IndexHNSW* index,
    idx_t n,
    const float* x,
    BlockResultHandler& bres,
    LAETPredictorHNSW* laet_predictor,
    const SearchParameters* params_in) {
  FAISS_THROW_IF_NOT_MSG(
      index->storage,
      "No storage index, please use IndexHNSWFlat (or variants) "
      "instead of IndexHNSW directly");
  const SearchParametersHNSW* params = nullptr;
  const HNSW& hnsw = index->hnsw;

  int efSearch = hnsw.efSearch;
  if (params_in) {
    params = dynamic_cast<const SearchParametersHNSW*>(params_in);
    FAISS_THROW_IF_NOT_MSG(params, "params type invalid");
    efSearch = params->efSearch;
  }
  size_t n1 = 0, n2 = 0, ndis = 0;

  idx_t check_period = InterruptCallback::get_period_hint(
      hnsw.max_level * index->d * efSearch);

  for (idx_t i0 = 0; i0 < n; i0 += check_period) {
    idx_t i1 = std::min(i0 + check_period, n);

    // #pragma omp parallel
    {
      VisitedTable vt(index->ntotal);
      typename BlockResultHandler::SingleResultHandler res(bres);

      std::unique_ptr<DistanceComputer> dis(
          storage_distance_computer(index->storage));

      // #pragma omp for reduction(+ : n1, n2, ndis) schedule(guided)
      for (idx_t i = i0; i < i1; i++) {
        // printf(">> Answering query %ld\n", i);
        res.begin(i);
        dis->set_query(x + i * index->d);

        HNSWStats stats = hnsw.search_LAET(
            *dis, res, vt, i, laet_predictor, params);
        n1 += stats.n1;
        n2 += stats.n2;
        ndis += stats.ndis;
        res.end();
      }
    }
    InterruptCallback::check();
  }

  hnsw_stats.combine({n1, n2, ndis});
}

