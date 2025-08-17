import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import time
import json
import os

from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

import dask.dataframe as dd

SEED = 42

def get_dataset_name(nlist, nprobe, num_queries, ds_name, k, logint):
    return f"/data/mchatzakis/et_training_data/ivf/training-data-generation/training/{ds_name}/{k}/nlist{nlist}_nprobe{nprobe}_qs{num_queries}_li{logint}.txt"

def main():
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

    training_queries_num = [10000]
    all_k_values = [50]
    all_datasets = ["SIFT100M", "DEEP100M"]
    all_li = [50]
    n_estimators_list = [100]
    learning_rates = [0.1]
    reg_lambda_values = [0]
    
    for ds_name in all_datasets:
        nprobe = dataset_params[ds_name]["nprobe"]
        nlist = dataset_params[ds_name]["nlist"]
        d = dataset_params[ds_name]["d"]
                
        results = train_predictors(
            ds_name, training_queries_num, all_k_values, all_li,
            nlist, nprobe, d, n_estimators_list, learning_rates, reg_lambda_values)
        
        filename = f"../experiments/generated_json/ivf/ivf_training_results_{ds_name}.json"
        with open(filename, "w") as f:
            json.dump(results, f, indent=4)
    
def train_predictors(ds_name, training_queries_num, all_k_values, all_li, nlist, nprobe, d, n_estimators_list, learning_rates, reg_lambda_values):
    index_metric_feats = ["step", "dists", "inserts"]
    neighbor_distances_feats = ["first_nn_dist", "nn_dist", "furthest_dist"]
    neighbor_stats_feats = ["avg_dist", "variance", "percentile_25", "percentile_50", "percentile_75"]
    
    all_feats = index_metric_feats + neighbor_distances_feats + neighbor_stats_feats
    
    #dim_feats = [f"d{d}" for d in range(0, d)]
    #neighbor_stats_feats_additional = ["std", "range", "energy", "skewness", "kurtosis", "percentile_95"]
    #dim_stats_feats = ["dim_l2_norm", "dim_l1_norm", "dim_mean", "dim_median", "dim_std", "dim_var", "dim_min", "dim_max", "dim_range", "dim_energy", "dim_skewness", "dim_kurtosis", "dim_perc_25", "dim_perc_75", "dim_perc_95"]
    
    columns_to_load = ["qid", "elaps_ms"] + all_feats + ["r", "feats_collect_time_ms"] #+ neighbor_stats_feats_additional + dim_stats_feats
    
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
    
    results = {}
    
    max_query_size = max(training_queries_num)
    
    print(f"Starting training for {ds_name}")
    
    for k in all_k_values:
        results[k] = {}
        min_li = min(all_li)
                
        data_path = get_dataset_name(nlist, nprobe, max_query_size, ds_name, k, min_li)
        
        print(f"{ds_name} | k={k} | Path: {data_path}")
        
        all_queries_dask = dd.read_csv(data_path, usecols=columns_to_load)
        all_queries_data = all_queries_dask.compute()
        
        # Here, first of all assert that the number of queries is correct:
        assert all_queries_data["qid"].nunique() == max_query_size, f"Expected {max_query_size} queries, got {all_queries_data['qid'].nunique()}"
        
        for s in training_queries_num:
            results[k][s] = {}
            query_data_all = all_queries_data[all_queries_data["qid"] < s]
                
            for li in all_li:
                results[k][s][li] = {}
                    
                data_all = query_data_all[query_data_all["dists"] % li == 0]
                    
                print(f"    {s} Queries Data Shape: {data_all.shape} |  Unique queries: {data_all['qid'].nunique()} | Li: {li}")
                
                print(data_all.head())
                
                assert data_all["qid"].nunique() == s, f"Expected {s} queries, got {data_all['qid'].nunique()}"
                assert data_all.shape[1] == len(columns_to_load), f"Expected {len(columns_to_load) + d} columns, got {data_all.shape[1]}"
                            
                y_all = data_all["r"]
                    
                for selected_features in feature_classes.keys():
                    feats = feature_classes[selected_features]
                                                
                    X_all = data_all[feats]
                    X_train, y_train = X_all, y_all
                    
                    results[k][s][li][selected_features] = {}
                    for n_estimators in n_estimators_list:
                        
                        results[k][s][li][selected_features][str(n_estimators)] = {}
                        for learning_rate in learning_rates:
                            
                            results[k][s][li][selected_features][str(n_estimators)][str(learning_rate)] = {}
                            for reg_lambda in reg_lambda_values:
                                model = lgb.LGBMRegressor(objective='regression', random_state=SEED, n_estimators=n_estimators, verbose = -1, learning_rate=learning_rate, reg_lambda=reg_lambda)
                                model_train_time_start = time.time()
                                model.fit(X_train, y_train)
                                model_train_time = time.time() - model_train_time_start  
                                    
                                model_params = model.get_params()
                                learning_rate = model_params["learning_rate"]
                                    
                                feature_importances = pd.DataFrame({'Feature': feats, 'Importance': model.feature_importances_})
                                feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
                                    
                                model_file = f"../predictor_models/ivf/darth/{ds_name}_nlist{nlist}_nprobe{nprobe}_s{s}_k{k}_nestim{n_estimators}_lr{learning_rate}_li{li}_rl{reg_lambda}_{selected_features}.txt"
                                model.booster_.save_model(model_file)
                                print(f"        Model Train Time: {model_train_time:.2f} | learning rate: {learning_rate} | Saved to: {model_file}")

                                results[k][s][li][selected_features][str(n_estimators)][str(learning_rate)][str(reg_lambda)] = {
                                    "training_time": model_train_time,
                                    "training_data_size": X_train.shape[0],
                                    "feature_importances": feature_importances.to_dict(orient="records"),
                                }
                                
                                print(f"        Results[{k}][{s}][{li}][{selected_features}][{str(n_estimators)}][{str(learning_rate)}]:\n{results[k][s][li][selected_features][str(n_estimators)][str(learning_rate)][str(reg_lambda)]}")
            print()
        print("\n\n")    
    print(f"Finished training for {ds_name}")
    
    return results
    

if __name__ == "__main__":
    main()