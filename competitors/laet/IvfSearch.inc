// Included at the original FAISS scope to use its internal search helpers.
void IndexIVF::search_LAET(
    idx_t n,
    const float* x,
    idx_t k,
    float* distances,
    idx_t* labels,
    LAETPredictorIVF* laet_predictor,
    const SearchParameters* params_in) const {
  FAISS_THROW_IF_NOT(k > 0);
  const IVFSearchParameters* params = nullptr;
  if (params_in) {
    params = dynamic_cast<const IVFSearchParameters*>(params_in);
    FAISS_THROW_IF_NOT_MSG(params, "IndexIVF params have incorrect type");
  }
  const size_t nprobe = std::min(nlist, params ? params->nprobe : this->nprobe);
  FAISS_THROW_IF_NOT(nprobe > 0);

  laet_predictor->init_log_file();

  // search function for a subset of queries
  auto sub_search_func = [this, k, nprobe, params](
                             idx_t n,
                             const float* x,
                             float* distances,
                             idx_t* labels,
                             IndexIVFStats* ivf_stats,
                             LAETPredictorIVF* laet_predictor) {
    std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe]);
    std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

    double t0 = getmillisecs();
    quantizer->search(
        n,
        x,
        nprobe,
        coarse_dis.get(),
        idx.get(),
        params ? params->quantizer_params : nullptr);

    double t1 = getmillisecs();
    invlists->prefetch_lists(idx.get(), n * nprobe);

    search_preassigned_LAET(
        n,
        x,
        k,
        idx.get(),
        coarse_dis.get(),
        distances,
        labels,
        false,
        laet_predictor,
        params,
        ivf_stats);

    double t2 = getmillisecs();
    ivf_stats->quantization_time += t1 - t0;
    ivf_stats->search_time += t2 - t0;
  };

  if (verbose) printf("Parallel mode: %d\n", parallel_mode);
  sub_search_func(n, x, distances, labels, &indexIVF_stats, laet_predictor);  // only for the testing

  laet_predictor->close_log_file();
  return;
}

