// Included at the original FAISS scope to use its internal search helpers.
HNSWStats HNSW::search_LAET(
    DistanceComputer& qdis,
    ResultHandler<C>& res,
    VisitedTable& vt,
    idx_t query_idx,
    LAETPredictorHNSW* laet_predictor,
    const SearchParametersHNSW* params) const {
  HNSWStats stats;
  if (entry_point == -1) {
    return stats;
  }
  int k = extract_k_from_ResultHandler(res);

  if (upper_beam == 1) {
    //  greedy search on upper levels
    storage_idx_t nearest = entry_point;
    float d_nearest = qdis(nearest);

    for (int level = max_level; level >= 1; level--) {
      greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
    }

    int ef = std::max(params ? params->efSearch : efSearch, k);
    if (search_bounded_queue) {  // this is the most common branch
      MinimaxHeap candidates(ef);

      candidates.push(nearest, d_nearest);

      search_from_candidates_LAET(
          *this,
          qdis,
          res,
          candidates,
          vt,
          stats,
          0,
          query_idx,
          laet_predictor,
          0,
          params);
    } else {
      assert(false || !"Declarative recall is implemented with default HNSW.");
    }

    vt.advance();

  } else {
    assert(false || !"Declarative recall is implemented with default HNSW.");
  }

  return stats;
}

