import numpy as np
import faiss
import sys
import os

import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

DATA_DIR = "/data/mchatzakis/datasets/processed"

def read_fvecs(file_path, limit=None):
    vectors = None
    dim = None
    with open(file_path, 'rb') as f:
        first_entry = np.fromfile(f, dtype=np.int32, count=1)
        if len(first_entry) == 0:
            raise ValueError("The file is empty or not in the expected fvecs format.")
        dim = first_entry[0]

        f.seek(0)

        vector_size = (dim + 1) * 4  # 4 bytes per float

        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0)

        num_vectors = file_size // vector_size

        #print(f">>Dimension: {dim}, Total Num vectors: {num_vectors}")

        if limit is not None and limit < num_vectors:
            num_vectors = limit
            #print(f">>Limiting to {num_vectors} vectors")

        data = np.fromfile(f, dtype=np.float32, count=num_vectors * (dim + 1))
        data = data.reshape(-1, dim + 1)

        vectors = data[:, 1:]

        #print(f">>Vectors shape: {vectors.shape}")
        assert vectors.shape == (num_vectors, dim)

    return vectors, int(dim)

def write_ivecs(file_name, indices):
    with open(file_name, 'wb') as f:
        for vec in indices:
            dim = len(vec)
            f.write(np.int32(dim).tobytes())
            f.write(np.array(vec, dtype=np.int32).tobytes())
            
def write_fvecs(file_name, vectors):
    with open(file_name, 'wb') as f:
        for vec in vectors:
            dim = len(vec)
            f.write(np.int32(dim).tobytes())
            f.write(np.array(vec, dtype=np.float32).tobytes())
            
def add_gaussian_noise_to_query_slow(query, base_vectors, noise_perc):
    dataset_std_dev = np.std(base_vectors, axis=0)  # Compute per-dimension standard deviation
    average_std_dev = np.mean(dataset_std_dev)
    
    noise_scale = (noise_perc / 100) * average_std_dev
    noise = np.random.normal(0, noise_scale, query.shape)
    
    noisy_query = query + noise    
    return noisy_query

def add_gaussian_noise_to_queryA(query, average_std_dev, noise_perc):
    noise_scale = (noise_perc / 100) * average_std_dev
    noise = np.random.normal(0, noise_scale, query.shape)
    
    noisy_query = query + noise    
    return noisy_query

def add_gaussian_noise_to_query(query, noise_perc):
    # Compute the L2 norm of the query vector
    query_norm = np.linalg.norm(query)
    #query_mean = np.abs(np.mean(query))
    #if query_mean == 0:
    #    query_mean = 1e-6
    
    # Compute the standard deviation of the noise
    noise_std = (noise_perc / 100) * query_norm
    
    # Generate Gaussian noise
    noise = np.random.normal(0, noise_std, size=query.shape)
    
    # Add noise to the query
    noisy_query = query + noise
    
    return noisy_query

dataconf = {
    "SIFT10M": {
        "base_vectors_path":    f"{DATA_DIR}/SIFT10M/base.10M.fvecs",
        "query_vectors_path":   f"{DATA_DIR}/SIFT10M/query.10K.fvecs",        
        
        "output_dir":                               f"{DATA_DIR}/SIFT10M/gauss_noisy_queries",
        "query_vectors_path_output_prefix":         "query.10K",
        
        "d": 128,
        "k": 100
    },
    "SIFT50M": {
        "base_vectors_path":    f"{DATA_DIR}/SIFT50M/base.50M.fvecs",
        "query_vectors_path":   f"{DATA_DIR}/SIFT50M/query.10K.fvecs",        
        
        "output_dir":                               f"{DATA_DIR}/SIFT50M/gauss_noisy_queries",
        "query_vectors_path_output_prefix":         "query.10K",
        
        "d": 128,
        "k": 100
    },
    "SIFT100M":{
        "base_vectors_path":    f"{DATA_DIR}/SIFT100M/base.100M.fvecs",
        "query_vectors_path":   f"{DATA_DIR}/SIFT100M/query.10K.fvecs",        
        
        "output_dir":                               f"{DATA_DIR}/SIFT100M/gauss_noisy_queries",
        "query_vectors_path_output_prefix":         "query.10K",
        
        "d": 128,
        "k": 100
    },
    "DEEP10M": {
        "base_vectors_path":    f"{DATA_DIR}/DEEP10M/base.10M.fvecs",
        "query_vectors_path":   f"{DATA_DIR}/DEEP10M/query.10K.fvecs",        
        
        "output_dir":                               f"{DATA_DIR}/DEEP10M/gauss_noisy_queries",
        "query_vectors_path_output_prefix":         "query.10K",
        
        "d": 96,
        "k": 100
    },
    "DEEP50M": {
        "base_vectors_path":    f"{DATA_DIR}/DEEP50M/base.50M.fvecs",
        "query_vectors_path":   f"{DATA_DIR}/DEEP50M/query.10K.fvecs",        
        
        "output_dir":                               f"{DATA_DIR}/DEEP50M/gauss_noisy_queries",
        "query_vectors_path_output_prefix":         "query.10K",
        
        "d": 96,
        "k": 100
    },
    "DEEP100M": {
        "base_vectors_path":    f"{DATA_DIR}/DEEP100M/base.100M.fvecs",
        "query_vectors_path":   f"{DATA_DIR}/DEEP100M/query.10K.fvecs",        
        
        "output_dir":                               f"{DATA_DIR}/DEEP100M/gauss_noisy_queries",
        "query_vectors_path_output_prefix":         "query.10K",
        
        "d": 96,
        "k": 100
    },
    "GIST1M": {
        "base_vectors_path":    f"{DATA_DIR}/GIST1M/base.1M.fvecs",
        "query_vectors_path":   f"{DATA_DIR}/GIST1M/query.1K.fvecs",        
        
        "output_dir":                               f"{DATA_DIR}/GIST1M/gauss_noisy_queries",
        "query_vectors_path_output_prefix":         "query.1K",
        
        "d": 960,
        "k": 100
    },
    "GLOVE100": {
        "base_vectors_path":    f"{DATA_DIR}/GLOVE100/base.1183514.fvecs",
        "query_vectors_path":   f"{DATA_DIR}/GLOVE100/query.10K.fvecs",        
        
        "output_dir":                               f"{DATA_DIR}/GLOVE100/gauss_noisy_queries",
        "query_vectors_path_output_prefix":         "query.10K",
        
        "d": 100,
        "k": 100
    },
    "T2I100M":{
        "base_vectors_path":    f"{DATA_DIR}/T2I100M/base.100M.fvecs",
        "query_vectors_path":   f"{DATA_DIR}/T2I100M/query.10K.fvecs",        
        
        "output_dir":                               f"{DATA_DIR}/T2I100M/gauss_noisy_queries",
        "query_vectors_path_output_prefix":         "query.10K",
        
        "d": 200,
        "k": 100
    }
}

noises_perc = [14, 16, 18, 20, 22, 24, 26, 28, 30]

def generateNoisyQueriesForDS(ds_name):
    print(">> Generating Noisy Queries for", ds_name)
    
    d = dataconf[ds_name]["d"]
    k = dataconf[ds_name]["k"]
    
    base_vectors, _ = read_fvecs(dataconf[ds_name]["base_vectors_path"])
    assert base_vectors.shape[1] == d
    
    index = faiss.IndexFlatL2(d)
    index.add(base_vectors)
    
    query_vectors, _ = read_fvecs(dataconf[ds_name]["query_vectors_path"])
    assert query_vectors.shape[1] == d
    
    out_dir = dataconf[ds_name]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    
    #dataset_std_dev = np.std(base_vectors, axis=0)
    #average_std_dev = np.mean(dataset_std_dev)
    
    for noise_perc in noises_perc:
        noisy_output_vectors = []
        for q in tqdm(query_vectors, desc=f"Noise {noise_perc}"):            
            #noisy_vec = add_gaussian_noise_to_query(q, average_std_dev, noise_perc)
            noisy_vec = add_gaussian_noise_to_query(q, noise_perc)
            assert len(noisy_vec) == d
            noisy_output_vectors.append(noisy_vec)
        
        assert len(noisy_output_vectors) == query_vectors.shape[0]
        
        noisy_output_vectors = np.array(noisy_output_vectors)
        assert noisy_output_vectors.shape == query_vectors.shape
        
        noise_output_fvecs_path = f"{out_dir}/{dataconf[ds_name]['query_vectors_path_output_prefix']}.noise{noise_perc}.fvecs"
        write_fvecs(noise_output_fvecs_path, noisy_output_vectors)
        
        print(">>   Computing groundtruth for noise", noise_perc)
        
        distances, indices = index.search(noisy_output_vectors, k)
        
        noise_output_gt_path =      f"{out_dir}/{dataconf[ds_name]['query_vectors_path_output_prefix']}.groundtruth.noise{noise_perc}.ivecs"
        write_ivecs(noise_output_gt_path, indices)
        
        noise_output_gt_dist_path = f"{out_dir}/{dataconf[ds_name]['query_vectors_path_output_prefix']}.groundtruth.noise{noise_perc}.fvecs"
        write_fvecs(noise_output_gt_dist_path, distances)
        
        print(">>   Done for noise", noise_perc)
    
    print(">> Done for", ds_name)

generateNoisyQueriesForDS("DEEP100M")
generateNoisyQueriesForDS("GIST1M")
generateNoisyQueriesForDS("GLOVE100")
generateNoisyQueriesForDS("SIFT100M")
#generateNoisyQueriesForDS("T2I100M")

