import os
import time
import subprocess
import json
import pandas as pd
import numpy as np

INDEX_DIRECTORY = "/home/mchatzakis/hnsw-index"
DATASET_DIRECTORY = "/data/mchatzakis/datasets/processed/"

dataset_info = {
        "SIFT100M":{
            "M": 32,
            "efC": 500,
            "efS": 500,
            "d": 128,
            "F": 241
        },
        "GIST1M": {
            "M": 32,
            "efC": 500,
            "efS": 1000,
            "d": 960,
            "F": 1260,
        },
        "GLOVE100": {
            "M": 16,
            "efC": 500,
            "efS": 500,
            "d": 100,
            "F": 200,
        },
        "DEEP100M": {
            "M": 32,
            "efC": 500,
            "efS": 750,
            "d": 96,
            "F": 368,
        },
        "T2I100M":{
            "M": 80,
            "efC": 1000,
            "efS": 2500,
            "d": 96,
            "F": 400,
        },
}

all_datasets = ["T2I100M"]
all_k_values = [10, 25, 50, 75, 100]
all_recall_targets = [0.80, 0.85, 0.90, 0.95,]
    
def get_average_recall(validation_df):
    return validation_df["r"].mean()

def get_total_query_search_time(validation_df):
    return validation_df["elaps_ms"].sum()


# LAET Automated Tuning
def run_laet(ds_name, M, efC, efS, s, F, k, m, target_recall, result_directory, output_log_file):
    mode = "laet-early-stop-testing"
    os.makedirs(f"{result_directory}/{ds_name}/k{k}", exist_ok=True)
    
    command = [
        "./build/hnsw-test/hnsw_test",
        "--dataset", ds_name,
        "--M", str(M),
        "--efConstruction", str(efC),
        "--efSearch", str(efS),
        "--query-num", str(s),
        "--k", str(k),
        "--output", f"{result_directory}/{ds_name}/k{k}/M{M}_efC{efC}_efS{efS}_qs{s}_tr{target_recall:.2f}_multiplier{m}.txt",
        "--mode", mode,
        "--index-filepath", f"{INDEX_DIRECTORY}/{ds_name}/{ds_name}.M{M}.efC{efC}.index",
        "--dataset-dir-prefix", DATASET_DIRECTORY,
        "--target-recall", f"{target_recall:.2f}",
        "--fixed-amount-of-search", str(F),
        "--prediction-multiplier", str(m),
        "--query-type", "validation",
        "--predictor-model-path", f"predictor_models/laet/{ds_name}_M{M}_efC{efC}_efS{efS}_s{10000}_k{k}.txt"
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output_log_file.write(f"{result.stdout}\n\n")
        if result.stderr != "":
            output_log_file.write(f"{result.stderr}\n\n")
            print("Command Errors:", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Command failed with return code:", e.returncode)
        print("Error Output:", e.stderr)
    
def get_laet_validation_result_filename(ds_name, M, efC, efS, s, k, m, target_recall, result_directory):
    return f"{result_directory}/{ds_name}/k{k}/M{M}_efC{efC}_efS{efS}_qs{s}_tr{target_recall:.2f}_multiplier{m}.txt"

def find_min_multiplier(ds_name, M, efC, efS, s, F, k, recall_target, multipliers, min_m_idx, result_directory, output_log_file):
    """
    Find the minimum multiplier that achieves the target recall using binary search,
    starting from a given minimum index.
    """
    low, high = min_m_idx, len(multipliers) - 1
    best_m = None  # To store the minimum multiplier that satisfies the condition
    
    total_experiments = 0
    total_tuning_time = 0
    
    while low <= high:
        mid = (low + high) // 2
        m = multipliers[mid]
        
        # Run the algorithm and retrieve the results
        run_laet(ds_name, M, efC, efS, s, F, k, m, recall_target, result_directory, output_log_file)
        validation_df = pd.read_csv(get_laet_validation_result_filename(ds_name, M, efC, efS, s, k, m, recall_target, result_directory))
        avg_recall = get_average_recall(validation_df)
        
        if avg_recall >= recall_target:
            best_m = m
            high = mid - 1 # Search for smaller m
        else:
            low = mid + 1  # Search for larger m
        
        total_experiments += 1
        total_tuning_time += get_total_query_search_time(validation_df)
    
    if best_m is None:
        print(f"[WARNING] -- Dataset: {ds_name}, k: {k}, R_t: {recall_target}, No suitable multiplier found. Using max multiplier.")
        best_m = multipliers[-1]

    return best_m, low, total_experiments, total_tuning_time

def laet_tuning(s, keep_m_idx_memory = False,):
    result_directory = "./results/laet-automated-tuning"    
    output_log_filename = "./laet_tuning.log"
    output_log_file = open(output_log_filename, "w") 
        
    step = 0.05
    multiplier_range_upper_bound = 5.0
    multiplier_range_lower_bound = 0.1
    multipliers = np.arange(multiplier_range_lower_bound, multiplier_range_upper_bound + step, step)
    multipliers = [f"{m:.2f}" for m in multipliers]
    print(f"LAET Tuning. Memory: {keep_m_idx_memory}, Validation Size: {s}. Multiplier range: {multipliers}")

    tuning_results = {}
    for di, ds_name in enumerate(all_datasets):
        M = dataset_info[ds_name]["M"]
        F = dataset_info[ds_name]["F"]
        efS = dataset_info[ds_name]["efS"]
        efC = dataset_info[ds_name]["efC"]
        
        tuning_results[ds_name] = {}
        
        for k in all_k_values:
            tuning_results[ds_name][k] = {}
            min_m_idx = 0
            for recall_target in all_recall_targets:
                min_m, min_m_idx, total_experiments, total_tuning_time = find_min_multiplier(ds_name, M, efC, efS, s, F, k, recall_target, multipliers, min_m_idx, result_directory, output_log_file)
                
                validation_df = pd.read_csv(get_laet_validation_result_filename(ds_name, M, efC, efS, s, k, min_m, recall_target, result_directory))
                avg_recall = get_average_recall(validation_df)
                    
                print(f"Dataset: {ds_name}, k: {k}, R_t: {recall_target}, Minimum Multiplier: {min_m}, Average Recall: {avg_recall}, Tuning Time: {total_tuning_time} ms, Total Experiments: {total_experiments}")
                    
                tuning_results[ds_name][k][recall_target] = {
                    "min_m": min_m,
                    "avg_recall": avg_recall,
                    "total_experiments": total_experiments,
                    "total_tuning_time": total_tuning_time
                } 

                if not keep_m_idx_memory:
                    min_m_idx = 0
    
    filename = f"./T2I_laet_tuning_results_memory{keep_m_idx_memory}_validationSize{s}.json"
    with open(filename, "w") as f:
        json.dump(tuning_results, f, indent=4)


# Classic HNSW without Early Termination efSearch Tuning
def run_classic_hnsw(ds_name, M, efC, efS, s, k, target_recall, result_directory, output_log_file):
    mode = "no-early-stop"
    os.makedirs(f"{result_directory}/{ds_name}/k{k}", exist_ok=True)
    
    command = [
        "./build/hnsw-test/hnsw_test",
        "--dataset", ds_name,
        "--M", str(M),
        "--efConstruction", str(efC),
        "--efSearch", str(efS),
        "--query-num", str(s),
        "--k", str(k),
        "--output", f"{result_directory}/{ds_name}/k{k}/M{M}_efC{efC}_efS{efS}_qs{s}_tr{target_recall:.2f}.txt",
        "--mode", mode,
        "--index-filepath", f"{INDEX_DIRECTORY}/{ds_name}/{ds_name}.M{M}.efC{efC}.index",
        "--dataset-dir-prefix", DATASET_DIRECTORY,
        "--target-recall", f"{target_recall:.2f}",
        "--query-type", "validation",
    ]
    
    try:
        #print(f"Running HNSW with efSearch: {efS}")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output_log_file.write(f"{result.stdout}\n\n")
        if result.stderr != "":
            print("Command Errors:", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Command failed with return code:", e.returncode)
        print("Error Output:", e.stderr)

def get_classic_hnsw_validation_result_filename(ds_name, M, efC, efS, s, k, target_recall, result_directory):
    return f"{result_directory}/{ds_name}/k{k}/M{M}_efC{efC}_efS{efS}_qs{s}_tr{target_recall:.2f}.txt"

def find_best_efSearch(ds_name, M, efC, s, k, recall_target, efSearch_values, min_efS_idx, result_directory, output_log_file):
    low, high = min_efS_idx, len(efSearch_values) - 1
    best_efS = None
    
    total_experiments = 0
    total_tuning_time = 0
    
    while low <= high:
        mid = (low + high) // 2
        efS = efSearch_values[mid]
        
        run_classic_hnsw(ds_name, M, efC, efS, s, k, recall_target, result_directory, output_log_file)
        validation_df = pd.read_csv(get_classic_hnsw_validation_result_filename(ds_name, M, efC, efS, s, k, recall_target, result_directory))
        avg_recall = get_average_recall(validation_df)
        
        if avg_recall >= recall_target:
            best_efS = efS
            high = mid - 1
        else:
            low = mid + 1
        
        total_experiments += 1
        total_tuning_time += get_total_query_search_time(validation_df)
        
    if best_efS is None:
        print(f"[WARNING] -- Dataset: {ds_name}, k: {k}, R_t: {recall_target}, No suitable efSearch found. Using max efSearch.")
        best_efS = efSearch_values[-1]
        
    return best_efS, low, total_experiments, total_tuning_time

def classic_hnsw_tuning(s, keep_efS_idx_memory = False):
    result_directory = "./results/classic-hnsw-automated-tuning"
    output_log_file = open("classic_hnsw_tuning.log", "w")
    
    step = 10
    efSearch_upper_bound = 3000
    efSearch_lower_bound = 20
    efSearch_values = np.arange(efSearch_lower_bound, efSearch_upper_bound + step, step)
    # make efSearch values ints
    efSearch_values = [int(efS) for efS in efSearch_values]
    print(f"efSearch range: {efSearch_values}")
    
    tuning_results = {}
    for di, ds_name in enumerate(all_datasets):
        M = dataset_info[ds_name]["M"]
        F = dataset_info[ds_name]["F"]
        efC = dataset_info[ds_name]["efC"]
        
        tuning_results[ds_name] = {}
        for k in all_k_values:
            tuning_results[ds_name][k] = {}
            min_efS_idx = 0
            for recall_target in all_recall_targets:
                
                min_efS, min_efS_idx, total_experiments, total_tuning_time = find_best_efSearch(ds_name, M, efC, s, k, recall_target, efSearch_values, min_efS_idx, result_directory, output_log_file)
                
                validation_df = pd.read_csv(get_classic_hnsw_validation_result_filename(ds_name, M, efC, min_efS, s, k, recall_target, result_directory))
                avg_recall = get_average_recall(validation_df)
                    
                print(f"Dataset: {ds_name}, k: {k}, R_t: {recall_target}, Minimum efSearch: {min_efS}, Average Recall: {avg_recall}, Tuning Time: {total_tuning_time} ms, Total Experiments: {total_experiments}")
                    
                tuning_results[ds_name][k][recall_target] = {
                    "min_efS": min_efS,
                    "avg_recall": avg_recall,
                    "total_experiments": total_experiments,
                    "total_tuning_time": total_tuning_time
                } 
               
                if not keep_efS_idx_memory:
                    min_efS_idx = 0
    
    filename = f"./T2I_classic_hnsw_tuning_results_memory{keep_efS_idx_memory}_validationSize{s}.json"
    with open(filename, "w") as f:
        json.dump(tuning_results, f, indent=4)
       

if __name__ == "__main__":
    for s in [1000]:
        laet_tuning(s, keep_m_idx_memory=False)
        #classic_hnsw_tuning(s, keep_efS_idx_memory=False)