import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import time
import json
import os

from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

import dask.dataframe as dd

SEED = 42

dataset_params = {
        "SIFT100M": {
            "M": 32,
            "efC": 500,
            "efS": 500,
            "d": 128,
        },
        "GIST1M": {
            "M": 32,
            "efC": 500,
            "efS": 1000,
            "d": 960,
        },
        "GLOVE100": {
            "M": 16,
            "efC": 500,
            "efS": 500,
            "d": 100,
        },
        "DEEP100M": {
            "M": 32,
            "efC": 500,
            "efS": 750,
            "d": 96,
        },
}
    
def get_dataset_name(M, efC, efS, num_queries, ds_name, k, logint):
    return f"/data/mchatzakis/et_training_data/early-stop-training/{ds_name}/k{k}/M{M}_efC{efC}_efS{efS}_qs{num_queries}_li{logint}.txt"

def get_validation_dataset_name(M, efC, efS, query_num, ds_name, k, logint): 
    return f"../experiments/results/validation_logging/{ds_name}/k{k}/M{M}_efC{efC}_efS{efS}_qs{query_num}_li{logint}.txt"

index_metric_feats = ["step", "dists", "inserts"]
neighbor_distances_feats = ["first_nn_dist", "nn_dist", "furthest_dist"]
neighbor_stats_feats = ["avg_dist", "variance", "percentile_25", "percentile_50", "percentile_75"]
    
all_feats = index_metric_feats + neighbor_distances_feats + neighbor_stats_feats

columns_to_load = ["qid", "elaps_ms"] + all_feats + ["r", "feats_collect_time_ms"]
    
feature_classes = {
    "all_feats": all_feats,  
}

def compare_predictors(ds_name, k):  
    li = 1
    s = 10000
    M = dataset_params[ds_name]["M"]
    efC = dataset_params[ds_name]["efC"]
    efS = dataset_params[ds_name]["efS"]
    d = dataset_params[ds_name]["d"]
    
    data_path = get_dataset_name(M, efC, efS, s, ds_name, k, li)
    print(f"{ds_name} | k={k} | Path: {data_path}")
            
    all_queries_dask = dd.read_csv(data_path, usecols=columns_to_load)
    all_queries_data = all_queries_dask.compute()
            
    assert all_queries_data["qid"].nunique() == s, f"Expected {s} queries, got {all_queries_data['qid'].nunique()}"
            
    query_data_all = all_queries_data[all_queries_data["qid"] < s]               
                        
    data_all = query_data_all[query_data_all["dists"] % li == 0]
                        
    print(f"    {s} Queries Data Shape: {data_all.shape} |  Unique queries: {data_all['qid'].nunique()} | Li: {li}")                
                    
    assert data_all["qid"].nunique() == s, f"Expected {s} queries, got {data_all['qid'].nunique()}"
    assert data_all.shape[1] == len(columns_to_load), f"Expected {len(columns_to_load) + d} columns, got {data_all.shape[1]}"
                                
    y_all = data_all["r"]
                        
    feats = feature_classes["all_feats"]
                                                    
    X_all = data_all[feats]
    X_train, y_train = X_all, y_all
    
    # Take a subset, Random Forest is slow:
    subset = 0.4 # take half of the data
    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=1-subset, random_state=SEED)
    
    print(f"    X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")

    trained_models = {}

    n_estimators = 10
    learning_rate = 0.1
    reg_lambda = 0.0
    ml_models = {
        "RandomForest": RandomForestRegressor(n_estimators=n_estimators, random_state=SEED, n_jobs=-1),
        #"DecisionTree": DecisionTreeRegressor(random_state=SEED),
        #"LGBM": lgb.LGBMRegressor(objective='regression', random_state=SEED, n_estimators=n_estimators, verbose = -1, learning_rate=learning_rate, reg_lambda=reg_lambda),
        #"LinearRegression": LinearRegression(),
    }

    for model_name, model in ml_models.items():
        print(f"        Training {model_name}...")
        model_train_time_start = time.time()
        model.fit(X_train, y_train)
        model_train_time = time.time() - model_train_time_start
        trained_models[model_name] = model
        print(f"        {model_name} | Training Time: {model_train_time}")     

    validation_data_dd = dd.read_csv(get_validation_dataset_name(M, efC, efS, 1000, ds_name, k, 1), usecols=columns_to_load)
    validation_data_df = validation_data_dd.compute()
            
    validation_y_true = validation_data_df["r"]
            
    print(f"    Validation data shape: {validation_data_df.shape}")

    feats = feature_classes["all_feats"]
    validation_X = validation_data_df[feats]

    model_results = {}
    
    for model_name, model in trained_models.items():
        validation_y_pred = model.predict(validation_X)
        mse = mean_squared_error(validation_y_true, validation_y_pred)
        mae = mean_absolute_error(validation_y_true, validation_y_pred)
        r2 = r2_score(validation_y_true, validation_y_pred)
        
        model_results[model_name] = {
            "mse": mse,
            "mae": mae,
            "r2": r2,
        }
        
        print(f"        {model_name} | MSE: {mse:.4f} | MAE: {mae:.4f} | R2: {r2:.2f}")
        
    return model_results

# Main
all_ds_names = ["GLOVE100", "GIST1M", "DEEP100M", "SIFT100M"]
all_k = [10, 25, 50, 75, 100]

final_ds_results = {}
for ds_name in all_ds_names:
    ds_results = {}
    for k in all_k:
        ds_results[k] = compare_predictors(ds_name, k)
    
    for model_name in ds_results[all_k[0]]:
        model_results = ds_results[all_k[0]][model_name]
        for k in all_k[1:]:
            model_results["mse"] += ds_results[k][model_name]["mse"]
            model_results["mae"] += ds_results[k][model_name]["mae"]
            model_results["r2"] += ds_results[k][model_name]["r2"]
        
        model_results["mse"] /= len(all_k)
        model_results["mae"] /= len(all_k)
        model_results["r2"] /= len(all_k)
        
        print(f"{ds_name} | {model_name} | MSE: {model_results['mse']:.4f} | MAE: {model_results['mae']:.4f} | R2: {model_results['r2']:.2f}")
        final_ds_results[f"{ds_name}_{model_name}"] = model_results
        
with open("../experiments/generated_json/predictor_model_comparison.json", "w") as f:
    json.dump(final_ds_results, f)