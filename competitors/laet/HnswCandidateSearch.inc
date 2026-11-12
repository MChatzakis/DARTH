// Included at the original FAISS scope to use its internal search helpers.
int search_from_candidates_LAET(
    const HNSW& hnsw,
    DistanceComputer& qdis,
    ResultHandler<C>& res,
    MinimaxHeap& candidates,
    VisitedTable& vt,
    HNSWStats& stats,
    int level,
    idx_t query_idx,
    LAETPredictorHNSW* laet_predictor,
    int nres_in = 0,
    const SearchParametersHNSW* params = nullptr) {
  int nres = nres_in;
  int ndis = 0;
  int ndis_interval = 0;

  bool do_dis_check = params ? params->check_relative_distance : hnsw.check_relative_distance;
  int efSearch = params ? params->efSearch : hnsw.efSearch;
  const IDSelector* sel = params ? params->sel : nullptr;

  float prev_recall = 0;
  int total_insertions = 0;

  float first_nn_dist = -1;

  double search_time_start = laet_predictor->data_manager->elapsed_secs();

  double predictor_time = 0;

  C::T threshold = res.threshold;
  for (int i = 0; i < candidates.size(); i++) {
    idx_t v1 = candidates.ids[i];
    float d = candidates.dis[i];
    FAISS_ASSERT(v1 >= 0);
    if (!sel || sel->is_member(v1)) {
      if (d < threshold) {
        total_insertions++;

        if (res.add_result(d, v1)) {
          threshold = res.threshold;
        }

        if (first_nn_dist == -1) {
          first_nn_dist = d;
        }
      }
    }
    vt.set(v1);
  }

  int nstep = 0;
  bool early_terminate = false;
  int ndis_target = std::numeric_limits<int>::max();
  while (candidates.size() > 0) {
    float d0 = 0;
    int v0 = candidates.pop_min(&d0);

    size_t begin, end;
    hnsw.neighbor_range(v0, level, &begin, &end);

    size_t jmax = begin;
    for (size_t j = begin; j < end; j++) {
      int v1 = hnsw.neighbors[j];
      if (v1 < 0)
        break;

      prefetch_L2(vt.visited.data() + v1);
      jmax += 1;
    }

    int counter = 0;
    size_t saved_j[4];

    threshold = res.threshold;

    auto add_to_heap = [&](const size_t idx, const float dis) {
      if (!sel || sel->is_member(idx)) {
        if (dis < threshold) {
          total_insertions++;
          if (res.add_result(dis, idx)) {
            threshold = res.threshold;
          }
        }
      }

      candidates.push(idx, dis);

      // LAET CMU term: This should be called **only** once per query
      if (ndis == laet_predictor->fixed_amount_of_distance_calcs) {
        ndis_target = laet_predictor->predict_distance_calcs(
            query_idx,
            nstep,
            ndis,
            laet_predictor->data_manager->elapsed_secs() - search_time_start,
            first_nn_dist,
            &predictor_time);
      }

      if (ndis >= ndis_target) {
        early_terminate = true;
      }
    };

    for (size_t j = begin; j < jmax; j++) {
      int v1 = hnsw.neighbors[j];

      bool vget = vt.get(v1);
      vt.set(v1);
      saved_j[counter] = v1;
      counter += vget ? 0 : 1;

      if (counter == 4) {
        float dis[4];
        qdis.distances_batch_4(
            saved_j[0],
            saved_j[1],
            saved_j[2],
            saved_j[3],
            dis[0],
            dis[1],
            dis[2],
            dis[3]);

        for (size_t id4 = 0; id4 < 4; id4++) {
          ndis++;
          add_to_heap(saved_j[id4], dis[id4]);

          if (early_terminate) {
            break;
          }
        }

        counter = 0;
      }

      if (early_terminate) {
        break;
      }
    }

    if (early_terminate) {
      break;
    }

    for (size_t icnt = 0; icnt < counter; icnt++) {
      float dis = qdis(saved_j[icnt]);
      ndis++;
      add_to_heap(saved_j[icnt], dis);

      if (early_terminate) {
        break;
      }
    }

    if (early_terminate) {
      break;
    }

    nstep++;
  }

  if (level == 0) {
    stats.n1++;
    if (candidates.size() == 0) {
      stats.n2++;
    }
    stats.ndis += ndis;
  }

  laet_predictor->log_final_result(
      query_idx,
      nstep,
      ndis,
      laet_predictor->data_manager->elapsed_secs() - search_time_start,
      first_nn_dist,
      ndis_target,
      predictor_time);

  return nres;
}

