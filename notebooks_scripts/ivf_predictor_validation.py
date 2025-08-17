import os
import json

from tqdm import tqdm

import lightgbm as lgb
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import dask.dataframe as dd

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

SEED = 42

n_estimators = 100

index_metric_feats = ["step", "dists", "inserts"]
neighbor_distances_feats = ["first_nn_dist", "nn_dist", "furthest_dist"]
neighbor_stats_feats = ["avg_dist", "variance", "percentile_25", "percentile_50", "percentile_75"]
all_feats = index_metric_feats + neighbor_distances_feats + neighbor_stats_feats
    
model_conf = {
    "lightgbm": lgb.LGBMRegressor(objective='regression', random_state=SEED, n_estimators=n_estimators, verbose = -1),
}

def get_validation_dataset_name(nlist, nprobe, query_num, ds_name, k, logint): 
    return f"/data/mchatzakis/et_training_data/ivf/training-data-generation/validation/{ds_name}/{k}/nlist{nlist}_nprobe{nprobe}_qs{query_num}_li{logint}.txt"

results = {}
file_path = "../experiments/generated_json/ivf/ivf_predictor_validation_results.json"

if os.path.exists(file_path):
    with open(file_path, "r") as f:
        results = json.load(f)
else:
    print(f"File '{file_path}' does not exist.")
    
    
training_queries_num = ["10000"]#["100", "500", "1000", "5000", "10000"]
all_k_values = ["50"]
all_datasets = ["DEEP100M", "SIFT100M"]#["GLOVE100", "GIST1M", "SIFT100M", "DEEP100M"]
all_lis = ["50"]
estimators = ["100"]
learning_rates = ["0.1"]
reg_lambda_values = ["0"]


for ds_name in all_datasets:
    if ds_name not in results:
        results[ds_name] = {}
    
    dims = dataset_params[ds_name]["d"]
    dim_feats = [f"d{d}" for d in range(0, dims)] 
    neighbor_stats_feats_additional = ["std", "range", "energy", "skewness", "kurtosis", "percentile_95"]
    dim_stats_feats = ["dim_l2_norm", "dim_l1_norm", "dim_mean", "dim_median", "dim_std", "dim_var", "dim_min", "dim_max", "dim_range", "dim_energy", "dim_skewness", "dim_kurtosis", "dim_perc_25", "dim_perc_75", "dim_perc_95"]
    
    feature_classes = {
        #"index_metric_feats": index_metric_feats,
        #"neighbor_distances_feats": neighbor_distances_feats,
        #"neighbor_stats_feats": neighbor_stats_feats,
        #"index_metrics_and_neighbor_distances": index_metric_feats + neighbor_distances_feats,
        #"index_metrics_and_neighbor_stats": index_metric_feats + neighbor_stats_feats,
        #"neighbor_distances_and_neighbor_stats": neighbor_distances_feats + neighbor_stats_feats,
        "all_feats": all_feats,
        #"all_feats_and_all_data_dims": all_feats + dim_feats,
        #"all_feats_and_new_nn_stats": all_feats + neighbor_stats_feats_additional,
        #"all_feats_and_new_dim_stats": all_feats + dim_stats_feats,
        #"all_feats_new_nn_stats_and_dim_stats": all_feats + neighbor_stats_feats_additional + dim_stats_feats,
    }
    
    columns_to_load = ["qid", "elaps_ms"] + all_feats + ["r", "feats_collect_time_ms"]# + neighbor_stats_feats_additional + dim_stats_feats

    for k in tqdm(all_k_values, desc=f"k values for {ds_name}"):
        if k not in results[ds_name]:
            results[ds_name][k] = {}
        #M = dataset_params[ds_name]["M"]
        #efC = dataset_params[ds_name]["efC"]
        #efS = dataset_params[ds_name]["efS"]
        nlist = dataset_params[ds_name]["nlist"]
        nprobe = dataset_params[ds_name]["nprobe"]
        local_li = 50
        validation_data_dd = dd.read_csv(get_validation_dataset_name(nlist, nprobe, 1000, ds_name, k, local_li), usecols=columns_to_load)
        validation_data_df = validation_data_dd.compute()
        
        validation_y_true = validation_data_df["r"]
        
        print(f"Validation data shape: {validation_data_df.shape}")
        
        for s in training_queries_num:
            if s not in results[ds_name][k]:
                results[ds_name][k][s] = {}
            
            for li in all_lis:
                if li not in results[ds_name][k][s]:
                    results[ds_name][k][s][li] = {}
                
                for selected_features in feature_classes.keys():
                    feats = feature_classes[selected_features]
                    
                    results[ds_name][k][s][li][selected_features] = {}
                    for n_estimators in estimators:
                        
                        results[ds_name][k][s][li][selected_features][n_estimators] = {}
                        for lr in learning_rates:
                            
                            results[ds_name][k][s][li][selected_features][n_estimators][lr] = {}
                            for reg_lambda in reg_lambda_values:
                                #model_file = f"../predictor_models/darth/{ds_name}_M{M}_efC{efC}_efS{efS}_s{s}_k{k}_nestim{n_estimators}_lr{lr}_li{li}_rl{reg_lambda}_{selected_features}.txt"
                                model_file = f"../predictor_models/ivf/darth/{ds_name}_nlist{nlist}_nprobe{nprobe}_s{s}_k{k}_nestim{n_estimators}_lr{lr}_li{li}_rl{reg_lambda}_{selected_features}.txt"

                                model = lgb.Booster(model_file=model_file)
                                
                                validation_X = validation_data_df[feats]
                                validation_y_pred = model.predict(validation_X)
                                mse = mean_squared_error(validation_y_true, validation_y_pred)
                                mae = mean_absolute_error(validation_y_true, validation_y_pred)
                                r2 = r2_score(validation_y_true, validation_y_pred)
                                
                                results[ds_name][k][s][li][selected_features][n_estimators][lr][reg_lambda] = {
                                    "mse": mse,
                                    "mae": mae,
                                    "r2": r2,
                                }
                                
                                print(f"Validation | {ds_name} | k={k} | s={s} | li={li} | Features: {selected_features} | nest= {n_estimators} | lr={lr} | reg_lambda={reg_lambda} | MSE: {mse:.6f} | MAE: {mae:.6f} | R2: {r2:.4f}")

with open(file_path, "w") as f:
    json.dump(results, f, indent=4)