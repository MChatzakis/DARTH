// Included at the original FAISS scope to use its internal search helpers.
void IndexHNSW::search_LAET(
    idx_t n,
    const float* x,
    idx_t k,
    float* distances,
    idx_t* labels,
    LAETPredictorHNSW* laet_predictor,
    const SearchParameters* params_in) const {
  FAISS_THROW_IF_NOT(k > 0);

  using RH = HeapBlockResultHandler<HNSW::C>;
  RH bres(n, distances, labels, k);

  hnsw_search_LAET(this, n, x, bres, laet_predictor, params_in);

  if (is_similarity_metric(this->metric_type)) {
    // we need to revert the negated distances
    for (size_t i = 0; i < k * n; i++) {
      distances[i] = -distances[i];
    }
  }
}

