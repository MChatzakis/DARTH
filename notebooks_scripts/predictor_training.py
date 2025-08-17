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

from concurrent.futures import ThreadPoolExecutor, as_completed

import dask.dataframe as dd


def get_dataset_name(M, efC, efS, num_queries, ds_name, k, logint):
    return f"/data/mchatzakis/et_training_data/early-stop-training/{ds_name}/k{k}/M{M}_efC{efC}_efS{efS}_qs{num_queries}_li{logint}.txt"

def main():
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
        }
    }

    training_queries_num = [15000, 20000]
    all_k_values = [10, 25, 50, 75, 100]
    all_datasets = ["T2I100M"]
    all_li = [2]
        
    for ds_name in all_datasets:
        M = dataset_params[ds_name]["M"]
        efC = dataset_params[ds_name]["efC"]
        efS = dataset_params[ds_name]["efS"]
        
        results = train_predictors(
            ds_name, training_queries_num, all_k_values, all_li,
            M, efC, efS, n_estimators, columns_to_load, feature_classes, model_conf)

        with open(f"./training_results_{ds_name}_u10K.json", "w") as f:
            json.dump(results, f, indent=4)
    
def train_predictors(ds_name, training_queries_num, all_k_values, all_li, M, efC, efS, n_estimators, columns_to_load, feature_classes, model_conf):
    results = {}
    
    max_query_size = max(training_queries_num)
    
    print(f"Starting training for {ds_name}")
        
    for k in all_k_values:
        results[k] = {}
        li = 2
        
        data_path = get_dataset_name(M, efC, efS, max_query_size, ds_name, k, 4)
        
        print(f"{ds_name} | k={k} | Path: {data_path}")
        
        all_queries_dask = dd.read_csv(data_path, usecols=columns_to_load)
        all_queries_data = all_queries_dask.compute()
        
        for s in training_queries_num:
            results[k][s] = {}
            query_data_all = all_queries_data[all_queries_data["qid"] < s]
                
            for li in all_li:
                results[k][s][li] = {}
                    
                data_all = query_data_all[query_data_all["dists"] % li == 0]
                    
                print(f"    {s} Queries Data Shape: {data_all.shape} |  Unique queries: {data_all['qid'].nunique()} | Li: {li}")
                        
                y_all = data_all["r"]
                    
                for selected_features in feature_classes.keys():
                    feats = feature_classes[selected_features]
                                                
                    X_all = data_all[feats]
                    X_train, y_train = X_all, y_all
                        
                    model = model_conf["lightgbm"]
                    model_train_time_start = time.time()
                    model.fit(X_train, y_train)
                    model_train_time = time.time() - model_train_time_start  
                        
                    model_params = model.get_params()
                    learning_rate = model_params["learning_rate"]
                        
                    feature_importances = pd.DataFrame({'Feature': feats, 'Importance': model.feature_importances_})
                    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
                        
                    model_file = f"../predictor_models/lightgbm/{ds_name}_M{M}_efC{efC}_efS{efS}_s{s}_k{k}_nestim{n_estimators}_li{li}_{selected_features}.txt"
                    model.booster_.save_model(model_file)
                    print(f"        Model Train Time: {model_train_time:.2f} | learning rate: {learning_rate} | Saved to: {model_file}")

                    results[k][s][li][selected_features] = {
                        "training_time": model_train_time,
                        "training_data_size": X_train.shape[0],
                        "learning_rate": learning_rate,
                        "feature_importances": feature_importances.to_dict(orient="records"),
                        "n_estimators": n_estimators,
                    }
                    
                    print(f"results[{k}][{s}][{li}][{selected_features}]:\n{results[k][s][li][selected_features]}")
                    
            print()
        print("\n\n")    
    print(f"Finished training for {ds_name}")
    
    return results
    

if __name__ == "__main__":
    main()