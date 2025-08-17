import pandas as pd
import numpy as np
import json

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

all_datasets = ["SIFT100M", "DEEP100M", "GIST1M", "GLOVE100"]
all_k_values = ["50"]

MPI_DIV = 40
IPI_DIV = 8

json_results = {}

for ds_name in all_datasets:
    json_results[ds_name] = {}
    logging_interval = dataset_params[ds_name]["logging_interval"]
    
    for k in all_k_values:
        json_results[ds_name][k] = {}
        
        stats = json.load(open(f"../experiments/generated_json/ivf/ivf_data_stats_{ds_name}.json", "r"))
        stats_per_k = stats[f"{k}"]["distance_calcs_to_reach_target_recall"]
        str_data = f"{ds_name}_tuples_k{k}=(\n"
        for rt_str in stats_per_k.keys():
            f_rt = float(rt_str)
            mpi = round(stats_per_k[rt_str]["avg"] / MPI_DIV)
            ipi = round(stats_per_k[rt_str]["avg"] / IPI_DIV)
            
            mpi = int(logging_interval * round(mpi / logging_interval))
            ipi = int(logging_interval * round(ipi / logging_interval))
            
            json_results[ds_name][k][f_rt] = {
                "mpi": mpi,
                "ipi": ipi
            }
            
            str_data += f"  \"{f_rt:.2f} {ipi} {mpi}\"\n"
        str_data += f")\n"
    
        print(str_data)
        
filename = f"../experiments/generated_json/ivf/ivf_darth_params_ipi{IPI_DIV}_mpi{MPI_DIV}.json"
with open(filename, "w") as f:
    json.dump(json_results, f, indent=4)
print(f"Saved to {filename}")