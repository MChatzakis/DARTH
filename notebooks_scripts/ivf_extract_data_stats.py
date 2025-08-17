import pandas as pd
import numpy as np
import json

import dask.dataframe as dd

def get_dataset_name(nlist, nprobe, s, ds_name, k, logint):
    return f"/data/mchatzakis/et_training_data/ivf/training-data-generation/training/{ds_name}/{k}/nlist{nlist}_nprobe{nprobe}_qs{s}_li{logint}.txt"

def extract_data_stats(data):    
    results = {}
    
    target_recalls = [0.8, 0.85, 0.9, 0.95, 0.99, 1.00]
    grouped_by_query = data.groupby("qid")
    
    results["correlations_to_target"] = {}
    correlations_to_target = {}
    for col in data.columns:
        if col != "r":
            correlations_to_target[col] = data["r"].corr(data[col])
    
    correlations_to_target = {k: v for k, v in sorted(correlations_to_target.items(), key=lambda item: abs(item[1]), reverse=True)}
    for col, corr in correlations_to_target.items():
        #print(f"        {col}: {corr:.4f}")
        results["correlations_to_target"][col] = corr
    
    results["distance_calcs_to_reach_target_recall"] = {}
    for target_recall in target_recalls:
        distance_calcs_to_reach_target_recall = []
        for idx, group in grouped_by_query:
            group = group[group["r"] >= target_recall]
            if group.shape[0] > 0:
                row_of_first_reached_recall = group.iloc[0]
                distance_calcs_to_reach_target_recall.append(row_of_first_reached_recall["dists"])
        
        mean_dists = np.mean(distance_calcs_to_reach_target_recall)
        max_dists = np.max(distance_calcs_to_reach_target_recall)
        min_dists = np.min(distance_calcs_to_reach_target_recall)
        results["distance_calcs_to_reach_target_recall"][target_recall] = {
            "avg": mean_dists,
            "max": max_dists,
            "min": min_dists
        }
    
    return results

dataset_params = {
    "GLOVE100": {
        "nlist": 1000,
        "nprobe": 100,
        "d": 100,
        "logging_interval": 20
    },
    "GIST1M":{
        "nlist": 1000,
        "nprobe": 200,
        "d": 960,
        "logging_interval": 20
    },
    "SIFT100M":{
            "nlist": 10000,
            "nprobe": 150,
            "d": 128,
            "logging_interval": 50
        },
    "DEEP100M":{
            "nlist": 10000,
            "nprobe": 150,
            "d": 96,
            "logging_interval": 50
        },
}

index_metric_feats = ["step", "dists", "inserts"]
neighbor_distances_feats = ["first_nn_dist", "nn_dist", "furthest_dist"]
neighbor_stats_feats = ["avg_dist", "variance", "percentile_25", "percentile_50", "percentile_75"]
all_feats = index_metric_feats + neighbor_distances_feats + neighbor_stats_feats
columns_to_load = ["qid", "elaps_ms"] + all_feats + ["r", "feats_collect_time_ms"]


all_datasets = ["SIFT100M", "DEEP100M"]
all_k_values = [50]

for ds_name in all_datasets:
    dataset_stats = {}
    
    nlist = dataset_params[ds_name]["nlist"]
    nprobe = dataset_params[ds_name]["nprobe"]
    for k in all_k_values:
        print(f"{ds_name} | k={k}")
        li = 50
        dask_df = dd.read_csv(get_dataset_name(nlist, nprobe, 10000, ds_name, k, li), usecols=columns_to_load)
        data_all = dask_df.compute()
        
        dataset_stats[k] = extract_data_stats(data_all)
    
    with open(f"../experiments/generated_json/ivf/ivf_data_stats_{ds_name}.json", "w") as f:
        json.dump(dataset_stats, f, indent=4)          
    