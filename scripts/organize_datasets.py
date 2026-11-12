import numpy as np
import faiss
import sys
import os
from tqdm.auto import tqdm

DIRECTORY_TO_LOAD = os.environ.get("RAW_DATASET_DIRECTORY", "data/raw")
DIRECTORY_TO_SAVE = os.environ.get("DATASET_DIRECTORY", os.environ.get("DARTH_DATASET_DIR", "data"))

os.makedirs(f"{DIRECTORY_TO_SAVE}", exist_ok=True)

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

def faiss_batched_search(index, vectors, k, batch_size=100, desc="Searching"):
    n = vectors.shape[0]
    distances = np.empty((n, k), dtype=np.float32)
    indices = np.empty((n, k), dtype=np.int64)

    total_batches = (n + batch_size - 1) // batch_size
    for start in tqdm(range(0, n, batch_size), total=total_batches, desc=desc):
        end = min(start + batch_size, n)
        D, I = index.search(vectors[start:end], k)
        distances[start:end] = D
        indices[start:end] = I

    return distances, indices

def check_written_vecs(file_path, expected_num, expected_dim, dtype='float32'):
    """Verify a .fvecs/.ivecs file matches expected (n, d) without loading all data.

    - Reads the first int32 header to get `dim`.
    - Uses file size to infer `num_vectors`.
    - Asserts `dim == expected_dim` and `num_vectors == expected_num`.
    """
    try:
        file_size = os.path.getsize(file_path)
    except OSError as e:
        raise FileNotFoundError(f"Cannot stat file {file_path}: {e}")

    with open(file_path, 'rb') as f:
        header = np.fromfile(f, dtype=np.int32, count=1)
        if header.size == 0:
            raise ValueError(f"File {file_path} is empty or malformed.")
        dim = int(header[0])

    # For both .fvecs and .ivecs, record size is (dim + 1) int32s
    record_size_bytes = (dim + 1) * 4
    if file_size % record_size_bytes != 0:
        raise ValueError(
            f"File size mismatch for {file_path}: size={file_size} not divisible by record={record_size_bytes}")

    num_vectors = file_size // record_size_bytes

    assert dim == expected_dim, (
        f"Dimension mismatch for {file_path}: got {dim}, expected {expected_dim}")
    assert num_vectors == expected_num, (
        f"Count mismatch for {file_path}: got {num_vectors}, expected {expected_num}")

    print(f">> Verified {os.path.basename(file_path)}: {num_vectors} x {dim} ({dtype})")

dataset_info = {
    "SIFT10M": {
        "provided_base_filepath":   f"{DIRECTORY_TO_LOAD}/SIFT1B/bigann_base.bvecs",
        "provided_learn_filepath":  f"{DIRECTORY_TO_LOAD}/SIFT1B/bigann_learn.bvecs",
        "provided_query_filepath":  f"{DIRECTORY_TO_LOAD}/SIFT1B/bigann_query.bvecs",
        
        "base_vectors_to_load": 10000000,
        "learn_vectors_to_load": 10000,
        "query_vectors_to_load": 10000,
        
        "validation_vectors_to_generate": 10000,
        "calibration_vectors_to_generate": 10000,
        
        "processed_base_filepath":          f"{DIRECTORY_TO_SAVE}/SIFT10M/base.10M.fvecs",
        "processed_learn_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT10M/learn.1M.fvecs",
        "processed_validation_filepath":    f"{DIRECTORY_TO_SAVE}/SIFT10M/validation.10K.fvecs",
        "processed_calibration_filepath":   f"{DIRECTORY_TO_SAVE}/SIFT10M/calibration.10K.fvecs",
        "processed_query_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT10M/query.10K.fvecs",
        
        "processed_learn_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT10M/learn.groundtruth.1M.k1000.ivecs",
        "processed_validation_groundtruth_filepath":    f"{DIRECTORY_TO_SAVE}/SIFT10M/validation.groundtruth.10K.k1000.ivecs",
        "processed_calibration_groundtruth_filepath":   f"{DIRECTORY_TO_SAVE}/SIFT10M/calibration.groundtruth.10K.k1000.ivecs",
        "processed_query_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT10M/query.groundtruth.10K.k1000.ivecs",
        
        "processed_learn_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT10M/learn.groundtruth.1M.k1000.fvecs",
        "processed_validation_gtdistances_filepath":    f"{DIRECTORY_TO_SAVE}/SIFT10M/validation.groundtruth.10K.k1000.fvecs",
        "processed_calibration_gtdistances_filepath":   f"{DIRECTORY_TO_SAVE}/SIFT10M/calibration.groundtruth.10K.k1000.fvecs",
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
        "calibration_vectors_to_generate": 10000,
        
        "processed_base_filepath":          f"{DIRECTORY_TO_SAVE}/SIFT100M/base.100M.fvecs",
        "processed_learn_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT100M/learn.1M.fvecs",
        "processed_validation_filepath":    f"{DIRECTORY_TO_SAVE}/SIFT100M/validation.10K.fvecs",
        "processed_calibration_filepath":   f"{DIRECTORY_TO_SAVE}/SIFT100M/calibration.10K.fvecs",
        "processed_query_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT100M/query.10K.fvecs",
        
        "processed_learn_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT100M/learn.groundtruth.1M.k1000.ivecs",
        "processed_validation_groundtruth_filepath":    f"{DIRECTORY_TO_SAVE}/SIFT100M/validation.groundtruth.10K.k1000.ivecs",
        "processed_calibration_groundtruth_filepath":   f"{DIRECTORY_TO_SAVE}/SIFT100M/calibration.groundtruth.10K.k1000.ivecs",
        "processed_query_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT100M/query.groundtruth.10K.k1000.ivecs",
        
        "processed_learn_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT100M/learn.groundtruth.1M.k1000.fvecs",
        "processed_validation_gtdistances_filepath":    f"{DIRECTORY_TO_SAVE}/SIFT100M/validation.groundtruth.10K.k1000.fvecs",
        "processed_calibration_gtdistances_filepath":   f"{DIRECTORY_TO_SAVE}/SIFT100M/calibration.groundtruth.10K.k1000.fvecs",
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
        "calibration_vectors_to_generate": 10000,
        
        "processed_base_filepath":          f"{DIRECTORY_TO_SAVE}/SIFT50M/base.50M.fvecs",
        "processed_learn_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT50M/learn.1M.fvecs",
        "processed_validation_filepath":    f"{DIRECTORY_TO_SAVE}/SIFT50M/validation.10K.fvecs",
        "processed_calibration_filepath":   f"{DIRECTORY_TO_SAVE}/SIFT50M/calibration.10K.fvecs",
        "processed_query_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT50M/query.10K.fvecs",
        
        "processed_learn_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT50M/learn.groundtruth.1M.k1000.ivecs",
        "processed_validation_groundtruth_filepath":    f"{DIRECTORY_TO_SAVE}/SIFT50M/validation.groundtruth.10K.k1000.ivecs",
        "processed_calibration_groundtruth_filepath":   f"{DIRECTORY_TO_SAVE}/SIFT50M/calibration.groundtruth.10K.k1000.ivecs",
        "processed_query_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT50M/query.groundtruth.10K.k1000.ivecs",
        
        "processed_learn_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/SIFT50M/learn.groundtruth.1M.k1000.fvecs",
        "processed_validation_gtdistances_filepath":    f"{DIRECTORY_TO_SAVE}/SIFT50M/validation.groundtruth.10K.k1000.fvecs",
        "processed_calibration_gtdistances_filepath":   f"{DIRECTORY_TO_SAVE}/SIFT50M/calibration.groundtruth.10K.k1000.fvecs",
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
        "learn_vectors_to_load": 10000,
        "query_vectors_to_load": 10000,
        
        "validation_vectors_to_generate": 10000,
        "calibration_vectors_to_generate": 10000,
        
        "processed_base_filepath":          f"{DIRECTORY_TO_SAVE}/DEEP10M/base.10M.fvecs",
        "processed_learn_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP10M/learn.1M.fvecs",
        "processed_validation_filepath":    f"{DIRECTORY_TO_SAVE}/DEEP10M/validation.10K.fvecs",
        "processed_calibration_filepath":   f"{DIRECTORY_TO_SAVE}/DEEP10M/calibration.10K.fvecs",
        "processed_query_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP10M/query.10K.fvecs",
        
        "processed_learn_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP10M/learn.groundtruth.1M.k1000.ivecs",
        "processed_validation_groundtruth_filepath":    f"{DIRECTORY_TO_SAVE}/DEEP10M/validation.groundtruth.10K.k1000.ivecs",
        "processed_calibration_groundtruth_filepath":   f"{DIRECTORY_TO_SAVE}/DEEP10M/calibration.groundtruth.10K.k1000.ivecs",
        "processed_query_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP10M/query.groundtruth.10K.k1000.ivecs",
        
        "processed_learn_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP10M/learn.groundtruth.1M.k1000.fvecs",
        "processed_validation_gtdistances_filepath":    f"{DIRECTORY_TO_SAVE}/DEEP10M/validation.groundtruth.10K.k1000.fvecs",
        "processed_calibration_gtdistances_filepath":   f"{DIRECTORY_TO_SAVE}/DEEP10M/calibration.groundtruth.10K.k1000.fvecs",
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
        "calibration_vectors_to_generate": 10000,
        
        "processed_base_filepath":          f"{DIRECTORY_TO_SAVE}/DEEP50M/base.50M.fvecs",
        "processed_learn_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP50M/learn.1M.fvecs",
        "processed_validation_filepath":    f"{DIRECTORY_TO_SAVE}/DEEP50M/validation.10K.fvecs",
        "processed_calibration_filepath":   f"{DIRECTORY_TO_SAVE}/DEEP50M/calibration.10K.fvecs",
        "processed_query_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP50M/query.10K.fvecs",
        
        "processed_learn_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP50M/learn.groundtruth.1M.k1000.ivecs",
        "processed_validation_groundtruth_filepath":    f"{DIRECTORY_TO_SAVE}/DEEP50M/validation.groundtruth.10K.k1000.ivecs",
        "processed_calibration_groundtruth_filepath":   f"{DIRECTORY_TO_SAVE}/DEEP50M/calibration.groundtruth.10K.k1000.ivecs",
        "processed_query_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP50M/query.groundtruth.10K.k1000.ivecs",
        
        "processed_learn_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP50M/learn.groundtruth.1M.k1000.fvecs",
        "processed_validation_gtdistances_filepath":    f"{DIRECTORY_TO_SAVE}/DEEP50M/validation.groundtruth.10K.k1000.fvecs",
        "processed_calibration_gtdistances_filepath":   f"{DIRECTORY_TO_SAVE}/DEEP50M/calibration.groundtruth.10K.k1000.fvecs",
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
        "calibration_vectors_to_generate": 10000,
        
        "processed_base_filepath":          f"{DIRECTORY_TO_SAVE}/DEEP100M/base.100M.fvecs",
        "processed_learn_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP100M/learn.1M.fvecs",
        "processed_validation_filepath":    f"{DIRECTORY_TO_SAVE}/DEEP100M/validation.10K.fvecs",
        "processed_calibration_filepath":   f"{DIRECTORY_TO_SAVE}/DEEP100M/calibration.10K.fvecs",
        "processed_query_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP100M/query.10K.fvecs",
        
        "processed_learn_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP100M/learn.groundtruth.1M.k1000.ivecs",
        "processed_validation_groundtruth_filepath":    f"{DIRECTORY_TO_SAVE}/DEEP100M/validation.groundtruth.10K.k1000.ivecs",
        "processed_calibration_groundtruth_filepath":   f"{DIRECTORY_TO_SAVE}/DEEP100M/calibration.groundtruth.10K.k1000.ivecs",
        "processed_query_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP100M/query.groundtruth.10K.k1000.ivecs",
        
        "processed_learn_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/DEEP100M/learn.groundtruth.1M.k1000.fvecs",
        "processed_validation_gtdistances_filepath":    f"{DIRECTORY_TO_SAVE}/DEEP100M/validation.groundtruth.10K.k1000.fvecs",
        "processed_calibration_gtdistances_filepath":   f"{DIRECTORY_TO_SAVE}/DEEP100M/calibration.groundtruth.10K.k1000.fvecs",
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
        "calibration_vectors_to_generate": 10000,
        
        "processed_base_filepath":          f"{DIRECTORY_TO_SAVE}/GIST1M/base.1M.fvecs",
        "processed_learn_filepath":         f"{DIRECTORY_TO_SAVE}/GIST1M/learn.100K.fvecs",
        "processed_validation_filepath":    f"{DIRECTORY_TO_SAVE}/GIST1M/validation.1K.fvecs",
        "processed_calibration_filepath":   f"{DIRECTORY_TO_SAVE}/GIST1M/calibration.10K.fvecs",
        "processed_query_filepath":         f"{DIRECTORY_TO_SAVE}/GIST1M/query.1K.fvecs",
        
        "processed_learn_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/GIST1M/learn.groundtruth.100K.k1000.ivecs",
        "processed_validation_groundtruth_filepath":    f"{DIRECTORY_TO_SAVE}/GIST1M/validation.groundtruth.1K.k1000.ivecs",
        "processed_calibration_groundtruth_filepath":   f"{DIRECTORY_TO_SAVE}/GIST1M/calibration.groundtruth.10K.k1000.ivecs",
        "processed_query_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/GIST1M/query.groundtruth.1K.k1000.ivecs",
        
        "processed_learn_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/GIST1M/learn.groundtruth.100K.k1000.fvecs",
        "processed_validation_gtdistances_filepath":    f"{DIRECTORY_TO_SAVE}/GIST1M/validation.groundtruth.1K.k1000.fvecs",
        "processed_calibration_gtdistances_filepath":   f"{DIRECTORY_TO_SAVE}/GIST1M/calibration.groundtruth.10K.k1000.fvecs",
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
        "calibration_vectors_to_generate": 10000,
        
        "processed_base_filepath":          f"{DIRECTORY_TO_SAVE}/GLOVE100/base.1183514.fvecs",
        "processed_learn_filepath":         f"{DIRECTORY_TO_SAVE}/GLOVE100/learn.100K.fvecs",
        "processed_validation_filepath":    f"{DIRECTORY_TO_SAVE}/GLOVE100/validation.10K.fvecs",
        "processed_calibration_filepath":   f"{DIRECTORY_TO_SAVE}/GLOVE100/calibration.10K.fvecs",
        "processed_query_filepath":         f"{DIRECTORY_TO_SAVE}/GLOVE100/query.10K.fvecs",
        
        "processed_learn_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/GLOVE100/learn.groundtruth.100K.k1000.ivecs",
        "processed_validation_groundtruth_filepath":    f"{DIRECTORY_TO_SAVE}/GLOVE100/validation.groundtruth.10K.k1000.ivecs",
        "processed_calibration_groundtruth_filepath":   f"{DIRECTORY_TO_SAVE}/GLOVE100/calibration.groundtruth.10K.k1000.ivecs",
        "processed_query_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/GLOVE100/query.groundtruth.10K.k1000.ivecs",
        
        "processed_learn_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/GLOVE100/learn.groundtruth.100K.k1000.fvecs",
        "processed_validation_gtdistances_filepath":    f"{DIRECTORY_TO_SAVE}/GLOVE100/validation.groundtruth.10K.k1000.fvecs",
        "processed_calibration_gtdistances_filepath":   f"{DIRECTORY_TO_SAVE}/GLOVE100/calibration.groundtruth.10K.k1000.fvecs",
        "processed_query_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/GLOVE100/query.groundtruth.10K.k1000.fvecs",
        
        "k": 1000,
        "d": 100,
        "read_function": read_fvecs
    },
    "T2I100M": {
        #"provided_base_filepath":  f"{DIRECTORY_TO_LOAD}/T2I1B/base.1B.fbin",
        "provided_base_filepath":  f"{DIRECTORY_TO_SAVE}/T2I100M/base.100M.fvecs", # tmp
        "provided_learn_filepath": f"{DIRECTORY_TO_LOAD}/T2I1B/query.learn.50M.fbin",
        "provided_query_filepath": f"{DIRECTORY_TO_LOAD}/T2I1B/query.public.100K.fbin",
        
        "base_vectors_to_load": 100000000,
        "learn_vectors_to_load": 1000000,
        "query_vectors_to_load": 10000,
        
        "validation_vectors_to_generate": 10000,
        "calibration_vectors_to_generate": 10000,
        
        "processed_base_filepath":          f"{DIRECTORY_TO_SAVE}/T2I100M/base.100M.fvecs",
        "processed_learn_filepath":         f"{DIRECTORY_TO_SAVE}/T2I100M/learn.1M.fvecs",
        "processed_validation_filepath":    f"{DIRECTORY_TO_SAVE}/T2I100M/validation.10K.fvecs",
        "processed_calibration_filepath":   f"{DIRECTORY_TO_SAVE}/T2I100M/calibration.10K.fvecs",
        "processed_query_filepath":         f"{DIRECTORY_TO_SAVE}/T2I100M/query.10K.fvecs",
        
        "processed_learn_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/T2I100M/learn.groundtruth.1M.k1000.ivecs",
        "processed_validation_groundtruth_filepath":    f"{DIRECTORY_TO_SAVE}/T2I100M/validation.groundtruth.10K.k1000.ivecs",
        "processed_calibration_groundtruth_filepath":   f"{DIRECTORY_TO_SAVE}/T2I100M/calibration.groundtruth.10K.k1000.ivecs",
        "processed_query_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/T2I100M/query.groundtruth.10K.k1000.ivecs",
        
        "processed_learn_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/T2I100M/learn.groundtruth.1M.k1000.fvecs",
        "processed_validation_gtdistances_filepath":    f"{DIRECTORY_TO_SAVE}/T2I100M/validation.groundtruth.10K.k1000.fvecs",
        "processed_calibration_gtdistances_filepath":   f"{DIRECTORY_TO_SAVE}/T2I100M/calibration.groundtruth.10K.k1000.fvecs",
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
        "calibration_vectors_to_generate": 10000,
        
        "processed_base_filepath":          f"{DIRECTORY_TO_SAVE}/T2I50M/base.50M.fvecs",
        "processed_learn_filepath":         f"{DIRECTORY_TO_SAVE}/T2I50M/learn.1M.fvecs",
        "processed_validation_filepath":    f"{DIRECTORY_TO_SAVE}/T2I50M/validation.10K.fvecs",
        "processed_calibration_filepath":   f"{DIRECTORY_TO_SAVE}/T2I50M/calibration.10K.fvecs",
        "processed_query_filepath":         f"{DIRECTORY_TO_SAVE}/T2I50M/query.10K.fvecs",
        
        "processed_learn_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/T2I50M/learn.groundtruth.1M.k1000.ivecs",
        "processed_validation_groundtruth_filepath":    f"{DIRECTORY_TO_SAVE}/T2I50M/validation.groundtruth.10K.k1000.ivecs",
        "processed_calibration_groundtruth_filepath":   f"{DIRECTORY_TO_SAVE}/T2I50M/calibration.groundtruth.10K.k1000.ivecs",
        "processed_query_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/T2I50M/query.groundtruth.10K.k1000.ivecs",
        
        "processed_learn_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/T2I50M/learn.groundtruth.1M.k1000.fvecs",
        "processed_validation_gtdistances_filepath":    f"{DIRECTORY_TO_SAVE}/T2I50M/validation.groundtruth.10K.k1000.fvecs",
        "processed_calibration_gtdistances_filepath":   f"{DIRECTORY_TO_SAVE}/T2I50M/calibration.groundtruth.10K.k1000.fvecs",
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
        "learn_vectors_to_load": 10000,
        "query_vectors_to_load": 10000,
        
        "validation_vectors_to_generate": 10000,
        "calibration_vectors_to_generate": 10000,
        
        "processed_base_filepath":          f"{DIRECTORY_TO_SAVE}/T2I10M/base.10M.fvecs",
        "processed_learn_filepath":         f"{DIRECTORY_TO_SAVE}/T2I10M/learn.10K.fvecs",
        "processed_validation_filepath":    f"{DIRECTORY_TO_SAVE}/T2I10M/validation.10K.fvecs",
        "processed_calibration_filepath":   f"{DIRECTORY_TO_SAVE}/T2I10M/calibration.10K.fvecs",
        "processed_query_filepath":         f"{DIRECTORY_TO_SAVE}/T2I10M/query.10K.fvecs",
        
        "processed_learn_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/T2I10M/learn.groundtruth.10K.k1000.ivecs",
        "processed_validation_groundtruth_filepath":    f"{DIRECTORY_TO_SAVE}/T2I10M/validation.groundtruth.10K.k1000.ivecs",
        "processed_calibration_groundtruth_filepath":   f"{DIRECTORY_TO_SAVE}/T2I10M/calibration.groundtruth.10K.k1000.ivecs",
        "processed_query_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/T2I10M/query.groundtruth.10K.k1000.ivecs",
        
        "processed_learn_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/T2I10M/learn.groundtruth.10K.k1000.fvecs",
        "processed_validation_gtdistances_filepath":    f"{DIRECTORY_TO_SAVE}/T2I10M/validation.groundtruth.10K.k1000.fvecs",
        "processed_calibration_gtdistances_filepath":   f"{DIRECTORY_TO_SAVE}/T2I10M/calibration.groundtruth.10K.k1000.fvecs",
        "processed_query_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/T2I10M/query.groundtruth.10K.k1000.fvecs",
        
        "k": 1000,
        "d": 200,
        "read_function": read_fbin_vecs
    }
}

def organize_dataset(ds_name, metric="L2", generate_base=False, generate_learn=False, generate_validation=False, generate_calibration=False, generate_query=False):
    os.makedirs(f"{DIRECTORY_TO_SAVE}/{ds_name}/", exist_ok=True)
    
    metric_prefix_map = {
        "L2": "l2.",
        "inner-product": "innprod."
    }
    
    gt_prefix = metric_prefix_map.get(metric, "")
    
    print(f">> Organizing dataset: {ds_name} with metric: {metric}")
    
    k = dataset_info[ds_name]["k"]
    d = dataset_info[ds_name]["d"]
    read_function = dataset_info[ds_name]["read_function"]
    
    # Load base vectors - needed for all groundtruth calculations
    base_vectors_num = dataset_info[ds_name]["base_vectors_to_load"]
    base_vectors, dim = read_function(dataset_info[ds_name]["provided_base_filepath"], limit=base_vectors_num)
    assert dim == d and len(base_vectors) == base_vectors_num, f"Base vectors shape mismatch for {ds_name}"
    print(f">> ds_name: {ds_name}, base_vectors.shape: {base_vectors.shape}")
    
    if generate_base:
        base_filepath = dataset_info[ds_name]["processed_base_filepath"]
        write_fvecs(base_filepath, base_vectors)
        check_written_vecs(base_filepath, base_vectors_num, d, dtype='float32')
        print(f">> Generated base vectors: {base_filepath}")
    
    # Create index once for all groundtruth calculations
    if metric == "L2":
        index = faiss.IndexFlatL2(d)
    elif metric == "inner-product":
        index = faiss.IndexFlatIP(d)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    index.add(base_vectors)
    
    # Process query vectors if requested
    if generate_query:
        query_vectors_num = dataset_info[ds_name]["query_vectors_to_load"]
        query_vectors, dim = read_function(dataset_info[ds_name]["provided_query_filepath"], limit=query_vectors_num)
        assert dim == d and len(query_vectors) == query_vectors_num, f"Query vectors shape mismatch for {ds_name}"
        print(f">> ds_name: {ds_name}, query_vectors.shape: {query_vectors.shape}")
        write_fvecs(dataset_info[ds_name]["processed_query_filepath"], query_vectors)
        check_written_vecs(dataset_info[ds_name]["processed_query_filepath"], query_vectors_num, d, dtype='float32')
        
        # Calculate query groundtruth using the pre-built index (batched)
        distances, indices = faiss_batched_search(index, query_vectors, k, batch_size=100, desc=f"{ds_name} query groundtruth")
        assert indices.shape == (query_vectors_num, k), f"Query groundtruth shape mismatch for {ds_name}"
        assert distances.shape == (query_vectors_num, k), f"Query groundtruth distances shape mismatch for {ds_name}"
        
        query_gt_filepath = dataset_info[ds_name]["processed_query_groundtruth_filepath"].replace("query.groundtruth", f"{gt_prefix}query.groundtruth")
        query_gtdist_filepath = dataset_info[ds_name]["processed_query_gtdistances_filepath"].replace("query.groundtruth", f"{gt_prefix}query.groundtruth")
        
        write_ivecs(query_gt_filepath, indices)
        write_fvecs(query_gtdist_filepath, distances)
        check_written_vecs(query_gt_filepath, query_vectors_num, k, dtype='int32')
        check_written_vecs(query_gtdist_filepath, query_vectors_num, k, dtype='float32')
        print(f">> Generated query vectors: {dataset_info[ds_name]['processed_query_filepath']}")
        print(f">> Generated query groundtruth: {query_gt_filepath}, {query_gtdist_filepath}")
    
    # Process learn, validation, and calibration vectors if any are requested
    if any([generate_learn, generate_validation, generate_calibration]):
        learn_vectors_num = dataset_info[ds_name]["learn_vectors_to_load"]
        validation_vectors_num = dataset_info[ds_name]["validation_vectors_to_generate"]
        calibration_vectors_num = dataset_info[ds_name]["calibration_vectors_to_generate"]
        total_vectors_needed = learn_vectors_num + validation_vectors_num + calibration_vectors_num
        
        if total_vectors_needed > 0:
            learn_val_cal_vectors, dim = read_function(dataset_info[ds_name]["provided_learn_filepath"], limit=total_vectors_needed)
            assert dim == d and len(learn_val_cal_vectors) == total_vectors_needed, f"Learn, validation, and calibration vectors shape mismatch for {ds_name}"
            print(f">> ds_name: {ds_name}, learn_val_cal_vectors.shape: {learn_val_cal_vectors.shape}")
                        
            # Process learn vectors
            if generate_learn:
                print(">> Generating learn vectors and groundtruth...")
                learn_vectors = learn_val_cal_vectors[0:learn_vectors_num]
                write_fvecs(dataset_info[ds_name]["processed_learn_filepath"], learn_vectors)
                check_written_vecs(dataset_info[ds_name]["processed_learn_filepath"], learn_vectors_num, d, dtype='float32')
                
                # Calculate learn groundtruth using the pre-built index (batched)
                distances, indices = faiss_batched_search(index, learn_vectors, k, batch_size=100, desc=f"{ds_name} learn groundtruth")
                assert indices.shape == (learn_vectors_num, k), f"Learn groundtruth shape mismatch for {ds_name}"
                assert distances.shape == (learn_vectors_num, k), f"Learn groundtruth distances shape mismatch for {ds_name}"
                
                learn_gt_filepath = dataset_info[ds_name]["processed_learn_groundtruth_filepath"].replace("learn.groundtruth", f"{gt_prefix}learn.groundtruth")
                learn_gtdist_filepath = dataset_info[ds_name]["processed_learn_gtdistances_filepath"].replace("learn.groundtruth", f"{gt_prefix}learn.groundtruth")
                
                write_ivecs(learn_gt_filepath, indices)
                write_fvecs(learn_gtdist_filepath, distances)
                check_written_vecs(learn_gt_filepath, learn_vectors_num, k, dtype='int32')
                check_written_vecs(learn_gtdist_filepath, learn_vectors_num, k, dtype='float32')
                print(f">> Generated learn vectors: {dataset_info[ds_name]['processed_learn_filepath']}")
                print(f">> Generated learn groundtruth: {learn_gt_filepath}, {learn_gtdist_filepath}")
            
            # Process validation vectors
            if generate_validation:
                print(">> Generating validation vectors and groundtruth...")
                validation_vectors = learn_val_cal_vectors[learn_vectors_num:learn_vectors_num + validation_vectors_num]
                write_fvecs(dataset_info[ds_name]["processed_validation_filepath"], validation_vectors)
                check_written_vecs(dataset_info[ds_name]["processed_validation_filepath"], validation_vectors_num, d, dtype='float32')
                
                # Calculate validation groundtruth using the pre-built index (batched)
                distances, indices = faiss_batched_search(index, validation_vectors, k, batch_size=100, desc=f"{ds_name} validation groundtruth")
                assert indices.shape == (validation_vectors_num, k), f"Validation groundtruth shape mismatch for {ds_name}"
                assert distances.shape == (validation_vectors_num, k), f"Validation groundtruth distances shape mismatch for {ds_name}"
                
                val_gt_filepath = dataset_info[ds_name]["processed_validation_groundtruth_filepath"].replace("validation.groundtruth", f"{gt_prefix}validation.groundtruth")
                val_gtdist_filepath = dataset_info[ds_name]["processed_validation_gtdistances_filepath"].replace("validation.groundtruth", f"{gt_prefix}validation.groundtruth")
                
                write_ivecs(val_gt_filepath, indices)
                write_fvecs(val_gtdist_filepath, distances)
                check_written_vecs(val_gt_filepath, validation_vectors_num, k, dtype='int32')
                check_written_vecs(val_gtdist_filepath, validation_vectors_num, k, dtype='float32')
                print(f">> Generated validation vectors: {dataset_info[ds_name]['processed_validation_filepath']}")
                print(f">> Generated validation groundtruth: {val_gt_filepath}, {val_gtdist_filepath}")
            
            # Process calibration vectors
            if generate_calibration:
                print(">> Generating calibration vectors and groundtruth...")
                calibration_vectors = learn_val_cal_vectors[learn_vectors_num + validation_vectors_num : total_vectors_needed]
                write_fvecs(dataset_info[ds_name]["processed_calibration_filepath"], calibration_vectors)
                check_written_vecs(dataset_info[ds_name]["processed_calibration_filepath"], calibration_vectors_num, d, dtype='float32')
                
                # Calculate calibration groundtruth using the pre-built index (batched)
                distances, indices = faiss_batched_search(index, calibration_vectors, k, batch_size=100, desc=f"{ds_name} calibration groundtruth")
                assert indices.shape == (calibration_vectors_num, k), f"Calibration groundtruth shape mismatch for {ds_name}"
                assert distances.shape == (calibration_vectors_num, k), f"Calibration groundtruth distances shape mismatch for {ds_name}"
                
                cal_gt_filepath = dataset_info[ds_name]["processed_calibration_groundtruth_filepath"].replace("calibration.groundtruth", f"{gt_prefix}calibration.groundtruth")
                cal_gtdist_filepath = dataset_info[ds_name]["processed_calibration_gtdistances_filepath"].replace("calibration.groundtruth", f"{gt_prefix}calibration.groundtruth")
                
                write_ivecs(cal_gt_filepath, indices)
                write_fvecs(cal_gtdist_filepath, distances)
                check_written_vecs(cal_gt_filepath, calibration_vectors_num, k, dtype='int32')
                check_written_vecs(cal_gtdist_filepath, calibration_vectors_num, k, dtype='float32')
                print(f">> Generated calibration vectors: {dataset_info[ds_name]['processed_calibration_filepath']}")
                print(f">> Generated calibration groundtruth: {cal_gt_filepath}, {cal_gtdist_filepath}")

    print(">> Done organizing dataset: ", ds_name)


# Example usage with all flags set to True for full processing
def process_dataset_fully(ds_name, metric="L2"):
    organize_dataset(ds_name, metric=metric,
                    generate_base=True, 
                    generate_learn=True, 
                    generate_validation=True, 
                    generate_calibration=True, 
                    generate_query=True)

def generate_calibration_only(ds_name, metric="L2"):
    organize_dataset(ds_name, metric=metric,
                    generate_base=False, 
                    generate_learn=False, 
                    generate_validation=False, 
                    generate_calibration=True, 
                    generate_query=False)

def main():
    #process_dataset_fully("GLOVE100", metric="inner-product")
    #process_dataset_fully("GIST1M", metric="inner-product")
    process_dataset_fully("T2I10M", metric="inner-product")
    #process_dataset_fully("DEEP10M", metric="inner-product")
    #process_dataset_fully("DEEP10M", metric="inner-product")
    
    
    #process_dataset_fully("GIST1M", metric="inner-product")
    #process_dataset_fully("SIFT100M")

    # Process smaller datasets in parallel
    #datasets = ["SIFT10M", "DEEP10M", "GIST1M"]
    #with ThreadPoolExecutor() as executor:
    #    executor.map(process_dataset_fully, datasets)
        
    #datasets = ["GLOVE100", "T2I10M", "T2I50M"]
    #with ThreadPoolExecutor() as executor:
    #    executor.map(process_dataset_fully, datasets)
        
    #process_dataset_fully("T2I100M")
    
    #generate_calibration_only("SIFT100M")
    #generate_calibration_only("DEEP100M")
    #generate_calibration_only("GLOVE100")
    #generate_calibration_only("GIST1M")
    #generate_calibration_only("T2I100M")

# Example of selective processing:
# organize_dataset("DEEP100M", generate_calibration=True)  # Only generate calibration vectors
# organize_dataset("SIFT10M", generate_validation=True, generate_calibration=True)  # Only validation and calibration


if __name__ == "__main__":
    main()