import os
import json

from tqdm import tqdm

import lightgbm as lgb
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import dask.dataframe as dd

def compute_P99(y_true, y_pred):
    y_diff = np.abs(y_true - y_pred)
    return np.percentile(y_diff, 99)

def compute_P1(y_true, y_pred):
    y_diff = np.abs(y_true - y_pred)
    return np.percentile(y_diff, 1)

dataset_params = {
        "SIFT100M": {
            "M": 32,
            "efC": 500,
            "efS": 500,
        },
        "GIST1M": {
            "M": 32,
            "efC": 500,
            "efS": 1000,
        },
        "GLOVE100": {
            "M": 16,
            "efC": 500,
            "efS": 500,
        },
        "DEEP100M": {
            "M": 32,
            "efC": 500,
            "efS": 750,
        },
        "T2I100M": {
            "M": 80,
            "efC": 1000,
            "efS": 2500,
        },
    }

SEED = 42

n_estimators = 100

index_metric_feats = ["step", "dists", "inserts"]
neighbor_distances_feats = ["first_nn_dist", "nn_dist", "furthest_dist"]
neighbor_stats_feats = ["avg_dist", "variance", "percentile_25", "percentile_50", "percentile_75"]
all_feats = index_metric_feats + neighbor_distances_feats + neighbor_stats_feats

columns_to_load = ["qid", "elaps_ms"] + all_feats + ["r", "feats_collect_time_ms"]
    
model_conf = {
    "lightgbm": lgb.LGBMRegressor(objective='regression', random_state=SEED, n_estimators=n_estimators, verbose = -1),
}

feature_classes = {
    #"index_metric_feats": index_metric_feats,
    #"neighbor_distances_feats": neighbor_distances_feats,
    #"neighbor_stats_feats": neighbor_stats_feats,
    #"index_metrics_and_neighbor_distances": index_metric_feats + neighbor_distances_feats,
    #"index_metrics_and_neighbor_stats": index_metric_feats + neighbor_stats_feats,
    #"neighbor_distances_and_neighbor_stats": neighbor_distances_feats + neighbor_stats_feats,
    "all_feats": all_feats,
}

def get_validation_dataset_name(M, efC, efS, query_num, ds_name, k, logint): 
    return f"../results/validation_logging/{ds_name}/k{k}/M{M}_efC{efC}_efS{efS}_qs{query_num}_li{logint}.txt"

results = {}
file_path = "predictor_validation_results.json"

if os.path.exists(file_path):
    with open(file_path, "r") as f:
        results = json.load(f)
else:
    print(f"File '{file_path}' does not exist.")
    
    
training_queries_num = ["100", "500", "1000", "5000", "10000"]
all_k_values = ["10", "25", "50", "75", "100"]
all_datasets = ["T2I100M"]
all_lis = ["2"]

for ds_name in all_datasets:
    if ds_name not in results:
        results[ds_name] = {}
    
    for k in tqdm(all_k_values, desc=f"k values for {ds_name}"):
        if k not in results[ds_name]:
            results[ds_name][k] = {}
        M = dataset_params[ds_name]["M"]
        efC = dataset_params[ds_name]["efC"]
        efS = dataset_params[ds_name]["efS"]
        
        validation_data_dd = dd.read_csv(get_validation_dataset_name(M, efC, efS, 1000, ds_name, k, 1), usecols=columns_to_load)
        validation_data_df = validation_data_dd.compute()
        validation_y_true = validation_data_df["r"]
        
        for s in training_queries_num:
            if s not in results[ds_name][k]:
                results[ds_name][k][s] = {}
            
            for li in all_lis:
                if li not in results[ds_name][k][s]:
                    results[ds_name][k][s][li] = {}
                
                for selected_features in feature_classes.keys():
                    
                    feats = feature_classes[selected_features]
                    
                    model_file = f"../predictor_models/lightgbm/{ds_name}_M{M}_efC{efC}_efS{efS}_s{s}_k{k}_nestim{n_estimators}_li{li}_{selected_features}.txt"
                    model = lgb.Booster(model_file=model_file)
                    
                    validation_X = validation_data_df[feats]
                    validation_y_pred = model.predict(validation_X)
                    mse = mean_squared_error(validation_y_true, validation_y_pred)
                    mae = mean_absolute_error(validation_y_true, validation_y_pred)
                    r2 = r2_score(validation_y_true, validation_y_pred)
                    p99 = compute_P99(validation_y_true, validation_y_pred)
                    p1 = compute_P1(validation_y_true, validation_y_pred)
                    
                    #average_relative_error = np.mean(np.abs(validation_y_true - validation_y_pred) / validation_y_true)
                    #maximum_relative_error = np.max(np.abs(validation_y_true - validation_y_pred) / validation_y_true)
                    
                    results[ds_name][k][s][li][selected_features] = {
                        "mse": mse,
                        "mae": mae,
                        "r2": r2,
                        "p99": p99,
                        "p1": p1,
                        #"average_relative_error": average_relative_error,
                        #"maximum_relative_error": maximum_relative_error
                    }
                

with open(file_path, "w") as f:
    json.dump(results, f, indent=4)