import numpy as np
import faiss
import sys
import os

from concurrent.futures import ThreadPoolExecutor

DIRECTORY_TO_LOAD = "/mnt/hddhelp/mchatzakis/datasets"
DIRECTORY_TO_SAVE = "/mnt/hddhelp/mchatzakis/datasets/processed"

os.makedirs(f"{DIRECTORY_TO_SAVE}", exist_ok=True)

# File writers
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

# File readers
def read_bvecs(file_path, limit=None):
    vectors = None
    dim = None
    with open(file_path, 'rb') as f:
        first_entry = np.fromfile(f, dtype=np.int32, count=1)
        if len(first_entry) == 0:
            raise ValueError("The file is empty or not in the expected bvecs format.")
        dim = first_entry[0]

        f.seek(0)

        vector_size = np.int64(4 + dim)  # 4 bytes for int32 + dim bytes for vector data
        file_size = np.int64(f.seek(0, 2))  # Ensure file_size is an int
        file_size = f.tell()
        f.seek(0)

        num_vectors = np.int64(np.int64(file_size) // np.int64(vector_size))

        print(f">>Dimension: {dim}, Total Num vectors: {num_vectors}")
        
        if limit is not None:
            num_vectors = min(limit, num_vectors)
            
        print(f">>Limiting to {num_vectors} vectors")

        print(type(num_vectors))
        print(type(vector_size))
        print(type(num_vectors * vector_size))
        
        data = np.fromfile(f, dtype=np.uint8, count=num_vectors * vector_size)
        
        print(f">>Data shape: {data.shape}")
        
        data = data.reshape(-1, vector_size)
        
        vectors = data[:, 4:]
        assert vectors.shape[1] == dim  # Ensure shape consistency

    return vectors, int(dim)

def read_fvecs(file_path, limit=None):
    vectors = None
    dim = None
    with open(file_path, 'rb') as f:
        first_entry = np.fromfile(f, dtype=np.int32, count=1)
        if len(first_entry) == 0:
            raise ValueError("The file is empty or not in the expected fvecs format.")
        dim = first_entry[0]

        f.seek(0)

        vector_size = np.int64((dim + 1) * 4)  # 4 bytes per float

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

def read_fbin_vecs(file_path, limit=None):
    vectors = None
    dim = None
    
    with open(file_path, 'rb') as f:
        n = np.fromfile(f, count=1, dtype=np.uint32)[0]
        dim = np.fromfile(f, count=1, dtype=np.uint32)[0]
        
        if limit is not None and n > limit:
            n = limit
            #print(f">> Limiting dataset {file_path} to {n} vectors")
        
        
        n = np.int64(n)
        dim = np.int64(dim)
        vectors = np.fromfile(f, count=np.int64(n * dim), dtype=np.float32)
        vectors = vectors.reshape(n, dim)
    
    return vectors, int(dim)
    
dataset_info = {
    "SIFT10M": {
        "provided_base_filepath":   f"{DIRECTORY_TO_LOAD}/SIFT1B/bigann_base.bvecs",
        "provided_learn_filepath":  f"{DIRECTORY_TO_LOAD}/SIFT1B/bigann_learn.bvecs",
        "provided_query_filepath":  f"{DIRECTORY_TO_LOAD}/SIFT1B/bigann_query.bvecs",
        
        "base_vectors_to_load": 10000000,
        "learn_vectors_to_load": 1000000,
        "query_vectors_to_load": 10000,
        
        "validation_vectors_to_generate": 10000,
        
        "processed_base_filepath":          f"{DIRECTORY_TO_SAVE}/SIFT10M/base.10M.fvecs",
        "processed_learn_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT10M/learn.1M.fvecs",
        "processed_validation_filepath":    f"{DIRECTORY_TO_SAVE}/SIFT10M/validation.10K.fvecs",
        "processed_query_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT10M/query.10K.fvecs",
        
        "processed_learn_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT10M/learn.groundtruth.1M.k1000.ivecs",
        "processed_validation_groundtruth_filepath":    f"{DIRECTORY_TO_SAVE}/SIFT10M/validation.groundtruth.10K.k1000.ivecs",
        "processed_query_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT10M/query.groundtruth.10K.k1000.ivecs",
        
        "processed_learn_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT10M/learn.groundtruth.1M.k1000.fvecs",
        "processed_validation_gtdistances_filepath":    f"{DIRECTORY_TO_SAVE}/SIFT10M/validation.groundtruth.10K.k1000.fvecs",
        "processed_query_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT10M/query.groundtruth.10K.k1000.fvecs",
        
        "k": 1000,
        "d": 128,
        "read_function": read_bvecs
    },
    "SIFT100M": {
        "provided_base_filepath":   f"{DIRECTORY_TO_LOAD}/SIFT1B/bigann_base.bvecs",
        "provided_learn_filepath":  f"{DIRECTORY_TO_LOAD}/SIFT1B/bigann_learn.bvecs",
        "provided_query_filepath":  f"{DIRECTORY_TO_LOAD}/SIFT1B/bigann_query.bvecs",
        
        "base_vectors_to_load": 100000000,
        "learn_vectors_to_load": 1000000,
        "query_vectors_to_load": 10000,
        
        "validation_vectors_to_generate": 10000,
        
        "processed_base_filepath":          f"{DIRECTORY_TO_SAVE}/SIFT100M/base.100M.fvecs",
        "processed_learn_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT100M/learn.1M.fvecs",
        "processed_validation_filepath":    f"{DIRECTORY_TO_SAVE}/SIFT100M/validation.10K.fvecs",
        "processed_query_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT100M/query.10K.fvecs",
        
        "processed_learn_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT100M/learn.groundtruth.1M.k1000.ivecs",
        "processed_validation_groundtruth_filepath":    f"{DIRECTORY_TO_SAVE}/SIFT100M/validation.groundtruth.10K.k1000.ivecs",
        "processed_query_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT100M/query.groundtruth.10K.k1000.ivecs",
        
        "processed_learn_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT100M/learn.groundtruth.1M.k1000.fvecs",
        "processed_validation_gtdistances_filepath":    f"{DIRECTORY_TO_SAVE}/SIFT100M/validation.groundtruth.10K.k1000.fvecs",
        "processed_query_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT100M/query.groundtruth.10K.k1000.fvecs",
        
        "k": 1000,
        "d": 128,
        "read_function": read_bvecs
    },
    "SIFT50M": {
        "provided_base_filepath":   f"{DIRECTORY_TO_LOAD}/SIFT1B/bigann_base.bvecs",
        "provided_learn_filepath":  f"{DIRECTORY_TO_LOAD}/SIFT1B/bigann_learn.bvecs",
        "provided_query_filepath":  f"{DIRECTORY_TO_LOAD}/SIFT1B/bigann_query.bvecs",
        
        "base_vectors_to_load": 50000000,
        "learn_vectors_to_load": 1000000,
        "query_vectors_to_load": 10000,
        
        "validation_vectors_to_generate": 10000,
        
        "processed_base_filepath":          f"{DIRECTORY_TO_SAVE}/SIFT50M/base.50M.fvecs",
        "processed_learn_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT50M/learn.1M.fvecs",
        "processed_validation_filepath":    f"{DIRECTORY_TO_SAVE}/SIFT50M/validation.10K.fvecs",
        "processed_query_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT50M/query.10K.fvecs",
        
        "processed_learn_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT50M/learn.groundtruth.1M.k1000.ivecs",
        "processed_validation_groundtruth_filepath":    f"{DIRECTORY_TO_SAVE}/SIFT50M/validation.groundtruth.10K.k1000.ivecs",
        "processed_query_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT50M/query.groundtruth.10K.k1000.ivecs",
        
        "processed_learn_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT50M/learn.groundtruth.1M.k1000.fvecs",
        "processed_validation_gtdistances_filepath":    f"{DIRECTORY_TO_SAVE}/SIFT50M/validation.groundtruth.10K.k1000.fvecs",
        "processed_query_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT50M/query.groundtruth.10K.k1000.fvecs",
        
        "k": 1000,
        "d": 128,
        "read_function": read_bvecs
    },
    "DEEP10M": {        
        "provided_base_filepath":  f"{DIRECTORY_TO_LOAD}/DEEP1B/base.1B.fbin",
        "provided_learn_filepath": f"{DIRECTORY_TO_LOAD}/DEEP1B/learn.350M.fbin",
        "provided_query_filepath": f"{DIRECTORY_TO_LOAD}/DEEP1B/query.public.10K.fbin",
        
        "base_vectors_to_load": 10000000,
        "learn_vectors_to_load": 1000000,
        "query_vectors_to_load": 10000,
        
        "validation_vectors_to_generate": 10000,
        
        "processed_base_filepath":          f"{DIRECTORY_TO_SAVE}/DEEP10M/base.10M.fvecs",
        "processed_learn_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP10M/learn.1M.fvecs",
        "processed_validation_filepath":    f"{DIRECTORY_TO_SAVE}/DEEP10M/validation.10K.fvecs",
        "processed_query_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP10M/query.10K.fvecs",
        
        "processed_learn_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP10M/learn.groundtruth.1M.k1000.ivecs",
        "processed_validation_groundtruth_filepath":    f"{DIRECTORY_TO_SAVE}/DEEP10M/validation.groundtruth.10K.k1000.ivecs",
        "processed_query_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP10M/query.groundtruth.10K.k1000.ivecs",
        
        "processed_learn_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP10M/learn.groundtruth.1M.k1000.fvecs",
        "processed_validation_gtdistances_filepath":    f"{DIRECTORY_TO_SAVE}/DEEP10M/validation.groundtruth.10K.k1000.fvecs",
        "processed_query_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP10M/query.groundtruth.10K.k1000.fvecs",
        
        "k": 1000,
        "d": 96,
        "read_function": read_fbin_vecs
    },
    "DEEP50M": {        
        "provided_base_filepath":  f"{DIRECTORY_TO_LOAD}/DEEP1B/base.1B.fbin",
        "provided_learn_filepath": f"{DIRECTORY_TO_LOAD}/DEEP1B/learn.350M.fbin",
        "provided_query_filepath": f"{DIRECTORY_TO_LOAD}/DEEP1B/query.public.10K.fbin",
        
        "base_vectors_to_load": 50000000,
        "learn_vectors_to_load": 1000000,
        "query_vectors_to_load": 10000,
        
        "validation_vectors_to_generate": 10000,
        
        "processed_base_filepath":          f"{DIRECTORY_TO_SAVE}/DEEP50M/base.50M.fvecs",
        "processed_learn_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP50M/learn.1M.fvecs",
        "processed_validation_filepath":    f"{DIRECTORY_TO_SAVE}/DEEP50M/validation.10K.fvecs",
        "processed_query_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP50M/query.10K.fvecs",
        
        "processed_learn_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP50M/learn.groundtruth.1M.k1000.ivecs",
        "processed_validation_groundtruth_filepath":    f"{DIRECTORY_TO_SAVE}/DEEP50M/validation.groundtruth.10K.k1000.ivecs",
        "processed_query_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP50M/query.groundtruth.10K.k1000.ivecs",
        
        "processed_learn_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP50M/learn.groundtruth.1M.k1000.fvecs",
        "processed_validation_gtdistances_filepath":    f"{DIRECTORY_TO_SAVE}/DEEP50M/validation.groundtruth.10K.k1000.fvecs",
        "processed_query_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP50M/query.groundtruth.10K.k1000.fvecs",
        
        "k": 1000,
        "d": 96,
        "read_function": read_fbin_vecs
    },
    "DEEP100M": {        
        "provided_base_filepath":  f"{DIRECTORY_TO_LOAD}/DEEP1B/base.1B.fbin",
        "provided_learn_filepath": f"{DIRECTORY_TO_LOAD}/DEEP1B/learn.350M.fbin",
        "provided_query_filepath": f"{DIRECTORY_TO_LOAD}/DEEP1B/query.public.10K.fbin",
        
        "base_vectors_to_load": 100000000,
        "learn_vectors_to_load": 1000000,
        "query_vectors_to_load": 10000,
        
        "validation_vectors_to_generate": 10000,
        
        "processed_base_filepath":          f"{DIRECTORY_TO_SAVE}/DEEP100M/base.100M.fvecs",
        "processed_learn_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP100M/learn.1M.fvecs",
        "processed_validation_filepath":    f"{DIRECTORY_TO_SAVE}/DEEP100M/validation.10K.fvecs",
        "processed_query_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP100M/query.10K.fvecs",
        
        "processed_learn_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP100M/learn.groundtruth.1M.k1000.ivecs",
        "processed_validation_groundtruth_filepath":    f"{DIRECTORY_TO_SAVE}/DEEP100M/validation.groundtruth.10K.k1000.ivecs",
        "processed_query_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP100M/query.groundtruth.10K.k1000.ivecs",
        
        "processed_learn_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP100M/learn.groundtruth.1M.k1000.fvecs",
        "processed_validation_gtdistances_filepath":    f"{DIRECTORY_TO_SAVE}/DEEP100M/validation.groundtruth.10K.k1000.fvecs",
        "processed_query_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP100M/query.groundtruth.10K.k1000.fvecs",
        
        "k": 1000,
        "d": 96,
        "read_function": read_fbin_vecs
    },
    "GIST1M": {
        "provided_base_filepath":   f"{DIRECTORY_TO_LOAD}/GIST1M/gist/gist_base.fvecs",
        "provided_learn_filepath":  f"{DIRECTORY_TO_LOAD}/GIST1M/gist/gist_learn.fvecs",
        "provided_query_filepath":  f"{DIRECTORY_TO_LOAD}/GIST1M/gist/gist_query.fvecs",
        
        "base_vectors_to_load": 1000000,
        "learn_vectors_to_load": 100000,
        "query_vectors_to_load": 1000,
        
        "validation_vectors_to_generate": 1000,
        
        "processed_base_filepath":          f"{DIRECTORY_TO_SAVE}/GIST1M/base.1M.fvecs",
        "processed_learn_filepath":         f"{DIRECTORY_TO_SAVE}/GIST1M/learn.100K.fvecs",
        "processed_validation_filepath":    f"{DIRECTORY_TO_SAVE}/GIST1M/validation.1K.fvecs",
        "processed_query_filepath":         f"{DIRECTORY_TO_SAVE}/GIST1M/query.1K.fvecs",
        
        "processed_learn_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/GIST1M/learn.groundtruth.100K.k1000.ivecs",
        "processed_validation_groundtruth_filepath":    f"{DIRECTORY_TO_SAVE}/GIST1M/validation.groundtruth.1K.k1000.ivecs",
        "processed_query_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/GIST1M/query.groundtruth.1K.k1000.ivecs",
        
        "processed_learn_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/GIST1M/learn.groundtruth.100K.k1000.fvecs",
        "processed_validation_gtdistances_filepath":    f"{DIRECTORY_TO_SAVE}/GIST1M/validation.groundtruth.1K.k1000.fvecs",
        "processed_query_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/GIST1M/query.groundtruth.1K.k1000.fvecs",
        
        "k": 1000,
        "d": 960,
        "read_function": read_fvecs
    }, 
    "GLOVE100": {
        "provided_base_filepath":  f"{DIRECTORY_TO_LOAD}/GloVe/glove-100_base.fvecs",
        "provided_learn_filepath": f"{DIRECTORY_TO_LOAD}/GloVe/glove-100_learn_350K.fvecs",
        "provided_query_filepath": f"{DIRECTORY_TO_LOAD}/GloVe/glove-100_query_10K.fvecs",
        
        "base_vectors_to_load": 1183514,
        "learn_vectors_to_load": 100000,
        "query_vectors_to_load": 10000,
        
        "validation_vectors_to_generate": 10000,
        
        "processed_base_filepath":          f"{DIRECTORY_TO_SAVE}/GLOVE100/base.1183514.fvecs",
        "processed_learn_filepath":         f"{DIRECTORY_TO_SAVE}/GLOVE100/learn.100K.fvecs",
        "processed_validation_filepath":    f"{DIRECTORY_TO_SAVE}/GLOVE100/validation.10K.fvecs",
        "processed_query_filepath":         f"{DIRECTORY_TO_SAVE}/GLOVE100/query.10K.fvecs",
        
        "processed_learn_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/GLOVE100/learn.groundtruth.100K.k1000.ivecs",
        "processed_validation_groundtruth_filepath":    f"{DIRECTORY_TO_SAVE}/GLOVE100/validation.groundtruth.10K.k1000.ivecs",
        "processed_query_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/GLOVE100/query.groundtruth.10K.k1000.ivecs",
        
        "processed_learn_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/GLOVE100/learn.groundtruth.100K.k1000.fvecs",
        "processed_validation_gtdistances_filepath":    f"{DIRECTORY_TO_SAVE}/GLOVE100/validation.groundtruth.10K.k1000.fvecs",
        "processed_query_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/GLOVE100/query.groundtruth.10K.k1000.fvecs",
        
        "k": 1000,
        "d": 100,
        "read_function": read_fvecs
    },
    "T2I100M": {
        "provided_base_filepath":  f"{DIRECTORY_TO_LOAD}/T2I1B/base.1B.fbin",
        "provided_learn_filepath": f"{DIRECTORY_TO_LOAD}/T2I1B/query.learn.50M.fbin",
        "provided_query_filepath": f"{DIRECTORY_TO_LOAD}/T2I1B/query.public.100K.fbin",
        
        "base_vectors_to_load": 100000000,
        "learn_vectors_to_load": 1000000,
        "query_vectors_to_load": 10000,
        
        "validation_vectors_to_generate": 10000,
        
        "processed_base_filepath":          f"{DIRECTORY_TO_SAVE}/T2I100M/base.100M.fvecs",
        "processed_learn_filepath":         f"{DIRECTORY_TO_SAVE}/T2I100M/learn.1M.fvecs",
        "processed_validation_filepath":    f"{DIRECTORY_TO_SAVE}/T2I100M/validation.10K.fvecs",
        "processed_query_filepath":         f"{DIRECTORY_TO_SAVE}/T2I100M/query.10K.fvecs",
        
        "processed_learn_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/T2I100M/learn.groundtruth.1M.k1000.ivecs",
        "processed_validation_groundtruth_filepath":    f"{DIRECTORY_TO_SAVE}/T2I100M/validation.groundtruth.10K.k1000.ivecs",
        "processed_query_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/T2I100M/query.groundtruth.10K.k1000.ivecs",
        
        "processed_learn_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/T2I100M/learn.groundtruth.1M.k1000.fvecs",
        "processed_validation_gtdistances_filepath":    f"{DIRECTORY_TO_SAVE}/T2I100M/validation.groundtruth.10K.k1000.fvecs",
        "processed_query_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/T2I100M/query.groundtruth.10K.k1000.fvecs",
        
        "k": 1000,
        "d": 200,
        "read_function": read_fbin_vecs
    },
    "T2I50M": {
        "provided_base_filepath":  f"{DIRECTORY_TO_LOAD}/T2I1B/base.1B.fbin",
        "provided_learn_filepath": f"{DIRECTORY_TO_LOAD}/T2I1B/query.learn.50M.fbin",
        "provided_query_filepath": f"{DIRECTORY_TO_LOAD}/T2I1B/query.public.100K.fbin",
        
        "base_vectors_to_load": 50000000,
        "learn_vectors_to_load": 1000000,
        "query_vectors_to_load": 10000,
        
        "validation_vectors_to_generate": 10000,
        
        "processed_base_filepath":          f"{DIRECTORY_TO_SAVE}/T2I50M/base.50M.fvecs",
        "processed_learn_filepath":         f"{DIRECTORY_TO_SAVE}/T2I50M/learn.1M.fvecs",
        "processed_validation_filepath":    f"{DIRECTORY_TO_SAVE}/T2I50M/validation.10K.fvecs",
        "processed_query_filepath":         f"{DIRECTORY_TO_SAVE}/T2I50M/query.10K.fvecs",
        
        "processed_learn_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/T2I50M/learn.groundtruth.1M.k1000.ivecs",
        "processed_validation_groundtruth_filepath":    f"{DIRECTORY_TO_SAVE}/T2I50M/validation.groundtruth.10K.k1000.ivecs",
        "processed_query_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/T2I50M/query.groundtruth.10K.k1000.ivecs",
        
        "processed_learn_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/T2I50M/learn.groundtruth.1M.k1000.fvecs",
        "processed_validation_gtdistances_filepath":    f"{DIRECTORY_TO_SAVE}/T2I50M/validation.groundtruth.10K.k1000.fvecs",
        "processed_query_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/T2I50M/query.groundtruth.10K.k1000.fvecs",
        
        "k": 1000,
        "d": 200,
        "read_function": read_fbin_vecs
    },
    "T2I10M": {
        "provided_base_filepath":  f"{DIRECTORY_TO_LOAD}/T2I1B/base.1B.fbin",
        "provided_learn_filepath": f"{DIRECTORY_TO_LOAD}/T2I1B/query.learn.50M.fbin",
        "provided_query_filepath": f"{DIRECTORY_TO_LOAD}/T2I1B/query.public.100K.fbin",
        
        "base_vectors_to_load": 10000000,
        "learn_vectors_to_load": 1000000,
        "query_vectors_to_load": 10000,
        
        "validation_vectors_to_generate": 10000,
        
        "processed_base_filepath":          f"{DIRECTORY_TO_SAVE}/T2I10M/base.10M.fvecs",
        "processed_learn_filepath":         f"{DIRECTORY_TO_SAVE}/T2I10M/learn.1M.fvecs",
        "processed_validation_filepath":    f"{DIRECTORY_TO_SAVE}/T2I10M/validation.10K.fvecs",
        "processed_query_filepath":         f"{DIRECTORY_TO_SAVE}/T2I10M/query.10K.fvecs",
        
        "processed_learn_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/T2I10M/learn.groundtruth.1M.k1000.ivecs",
        "processed_validation_groundtruth_filepath":    f"{DIRECTORY_TO_SAVE}/T2I10M/validation.groundtruth.10K.k1000.ivecs",
        "processed_query_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/T2I10M/query.groundtruth.10K.k1000.ivecs",
        
        "processed_learn_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/T2I10M/learn.groundtruth.1M.k1000.fvecs",
        "processed_validation_gtdistances_filepath":    f"{DIRECTORY_TO_SAVE}/T2I10M/validation.groundtruth.10K.k1000.fvecs",
        "processed_query_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/T2I10M/query.groundtruth.10K.k1000.fvecs",
        
        "k": 1000,
        "d": 200,
        "read_function": read_fbin_vecs
    }
}

def organize_dataset(ds_name):
    os.makedirs(f"{DIRECTORY_TO_SAVE}/{ds_name}/", exist_ok=True)
    
    print(f">> Organizing dataset: {ds_name}")
    
    k = dataset_info[ds_name]["k"]
    d = dataset_info[ds_name]["d"]
    read_function = dataset_info[ds_name]["read_function"]
    
    # Save the processed vectors
    base_vectors_num = dataset_info[ds_name]["base_vectors_to_load"]
    base_vectors, dim = read_function(dataset_info[ds_name]["provided_base_filepath"], limit=base_vectors_num)
    assert dim == d and len(base_vectors) == base_vectors_num, f"Base vectors shape mismatch for {ds_name}"
    print(f">> ds_name: {ds_name}, base_vectors.shape: {base_vectors.shape}")
    
    query_vectors_num = dataset_info[ds_name]["query_vectors_to_load"]
    query_vectors, dim = read_function(dataset_info[ds_name]["provided_query_filepath"], limit=query_vectors_num)
    assert dim == d and len(query_vectors) == query_vectors_num, f"Query vectors shape mismatch for {ds_name}"
    print(f">> ds_name: {ds_name}, query_vectors.shape: {query_vectors.shape}")
    
    learn_vectors_num = dataset_info[ds_name]["learn_vectors_to_load"]
    validation_vectors_num = dataset_info[ds_name]["validation_vectors_to_generate"]
    learn_and_validation_vectors_num = learn_vectors_num + validation_vectors_num
    learn_and_validation_vectors, dim = read_function(dataset_info[ds_name]["provided_learn_filepath"], limit=learn_and_validation_vectors_num)
    assert dim == d and len(learn_and_validation_vectors) == learn_and_validation_vectors_num, f"Learn and validation vectors shape mismatch for {ds_name}"
    print(f">> ds_name: {ds_name}, learn_and_validation_vectors.shape: {learn_and_validation_vectors.shape}")
    
    learn_vectors = learn_and_validation_vectors[:learn_vectors_num]
    validation_vectors = learn_and_validation_vectors[learn_vectors_num:]
    
    write_fvecs(dataset_info[ds_name]["processed_base_filepath"], base_vectors)
    write_fvecs(dataset_info[ds_name]["processed_query_filepath"], query_vectors)
    write_fvecs(dataset_info[ds_name]["processed_learn_filepath"], learn_vectors)
    write_fvecs(dataset_info[ds_name]["processed_validation_filepath"], validation_vectors)
    
    # Calculate the groundtuth using FAISS
    index = faiss.IndexFlatL2(d)  
    index.add(base_vectors)

    distances, indices = index.search(learn_vectors, k)
    assert indices.shape == (learn_vectors_num, k), f"Learn groundtruth shape mismatch for {ds_name}"
    assert distances.shape == (learn_vectors_num, k), f"Learn groundtruth distances shape mismatch for {ds_name}"
    write_ivecs(dataset_info[ds_name]["processed_learn_groundtruth_filepath"], indices)
    write_fvecs(dataset_info[ds_name]["processed_learn_gtdistances_filepath"], distances)
    
    distances, indices = index.search(validation_vectors, k)
    assert indices.shape == (validation_vectors_num, k), f"Validation groundtruth shape mismatch for {ds_name}"
    assert distances.shape == (validation_vectors_num, k), f"Validation groundtruth distances shape mismatch for {ds_name}"
    write_ivecs(dataset_info[ds_name]["processed_validation_groundtruth_filepath"], indices)
    write_fvecs(dataset_info[ds_name]["processed_validation_gtdistances_filepath"], distances)

    distances, indices = index.search(query_vectors, k)
    assert indices.shape == (query_vectors_num, k), f"Query groundtruth shape mismatch for {ds_name}"
    assert distances.shape == (query_vectors_num, k), f"Query groundtruth distances shape mismatch for {ds_name}"
    write_ivecs(dataset_info[ds_name]["processed_query_groundtruth_filepath"], indices)
    write_fvecs(dataset_info[ds_name]["processed_query_gtdistances_filepath"], distances)

    print(">> Done organizing dataset: ", ds_name)


organize_dataset("DEEP100M")
organize_dataset("SIFT100M")

datasets = ["SIFT10M", "DEEP10M", "GIST1M"]
with ThreadPoolExecutor() as executor:
    executor.map(organize_dataset, datasets)
    
datasets = ["GLOVE100", "T2I10M", "T2I50M"]
with ThreadPoolExecutor() as executor:
    executor.map(organize_dataset, datasets)
    
organize_dataset("T2I100M")