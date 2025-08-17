import numpy as np
import faiss
from tqdm import tqdm

def read_bvecs(file_path, limit=None):
    vectors = None
    dim = None
    with open(file_path, 'rb') as f:
        first_entry = np.fromfile(f, dtype=np.int32, count=1)
        if len(first_entry) == 0:
            raise ValueError("The file is empty or not in the expected bvecs format.")
        dim = first_entry[0]

        f.seek(0)

        vector_size = 4 + dim

        file_size = f.seek(0, 2)
        file_size = f.tell()
        f.seek(0)
        num_vectors = file_size // vector_size
        
        print(f">>Dimension: {dim}, Total Num vectors: {num_vectors}")
        
        if (limit is not None) and limit < num_vectors:
            num_vectors = limit
            print(f">>Limiting to {num_vectors} vectors") 

        data = np.fromfile(f, dtype=np.uint8, count=num_vectors * vector_size)
        data = data.reshape(-1, vector_size)
        
        vectors = data[:, 4:]
        
        print(f">>Vectors shape: {vectors.shape}")
        assert vectors.shape == (num_vectors, dim)
    
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

        vector_size = (dim + 1) * 4  # 4 bytes per float

        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0)

        num_vectors = file_size // vector_size

        print(f">>Dimension: {dim}, Total Num vectors: {num_vectors}")

        if limit is not None and limit < num_vectors:
            num_vectors = limit
            print(f">>Limiting to {num_vectors} vectors")

        data = np.fromfile(f, dtype=np.float32, count=num_vectors * (dim + 1))
        data = data.reshape(-1, dim + 1)

        vectors = data[:, 1:]

        print(f">>Vectors shape: {vectors.shape}")
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
            print(f">> Limiting dataset {file_path} to {n} vectors")
        
        vectors = np.fromfile(f, count=n * dim, dtype=np.float32)
        vectors = vectors.reshape(n, dim)
    
    return vectors, int(dim)

def write_ivecs(file_name, indices, k):
    with open(file_name, 'wb') as f:
        for vec in indices:
            dim = len(vec)
            f.write(np.int32(dim).tobytes())
            f.write(np.array(vec, dtype=np.int32).tobytes())

dataset_info = {
    "SIFT10M": {
        "base_filepath": "/data/mchatzakis/datasets/SIFT1B/bigann_base.bvecs",
        "index_vectors_num": 10000000,
        "train_filepath": "/data/mchatzakis/datasets/SIFT1B/bigann_learn.bvecs",
        "num_queries": 1000000,
        "gt_dir": "/data/mchatzakis/datasets/SIFT1B/bigann_learn_gt",
        "k": 1000,
        "d": 128,
        "read_function": read_bvecs
    },
    "DEEP10M": {
        "base_filepath": "/data/mchatzakis/datasets/DEEP1B/base.1B.fbin",
        "train_filepath": "/data/mchatzakis/datasets/DEEP1B/learn.350M.fbin",
        "gt_dir": "/data/mchatzakis/datasets/DEEP1B/groundtruth_learn",
        "index_vectors_num": 10000000,
        "num_queries": 1000000,
        "k": 1000,
        "d": 96,
        "read_function": read_fbin_vecs
    },
    "GIST1M": {
        "base_filepath": "/data/mchatzakis/datasets/GIST1M/gist/gist_base.fvecs",
        "train_filepath": "/data/mchatzakis/datasets/GIST1M/gist/gist_learn.fvecs",
        "gt_dir": "/data/mchatzakis/datasets/GIST1M/gist/gist_groundtruth_learn",
        "index_vectors_num": 1000000,
        "num_queries": 500000,
        "k": 1000,
        "d": 960,
        "read_function": read_fvecs
    }, 
    "GLOVE100": {
        "base_filepath": "/data/mchatzakis/datasets/GloVe/glove-100_base.fvecs",
        "train_filepath": "/data/mchatzakis/datasets/GloVe/glove-100_learn_350K.fvecs",#",
        "gt_dir": "/data/mchatzakis/datasets/GloVe/glove-100_groundtruth_learn",
        "index_vectors_num": 1183514,
        "num_queries": 350000,
        "k": 1000,
        "d": 100,
        "read_function": read_fvecs
    }
}

for ds_name in ["GLOVE100"]:
    #####################################################
    #####################################################
    read_function = dataset_info[ds_name]["read_function"]
    base_filepath = dataset_info[ds_name]["base_filepath"]
    index_vectors_num = dataset_info[ds_name]["index_vectors_num"]
    train_filepath = dataset_info[ds_name]["train_filepath"]
    num_queries = dataset_info[ds_name]["num_queries"]
    k = dataset_info[ds_name]["k"]
    gt_dir = dataset_info[ds_name]["gt_dir"]

    print(f">>Dataset: {ds_name}")
    print(f">>Base filepath: {base_filepath}")
    print(f">>Index vectors num: {index_vectors_num}")
    print(f">>Train filepath: {train_filepath}")
    print(f">>Num queries: {num_queries}")
    print(f">>K: {k}")
    print(f">>Ground truth directory: {gt_dir}")
    print(f">>Read function: {read_function}")
    print()
    #####################################################
    #####################################################

    index_vectors, dimensionality = read_function(base_filepath, limit=index_vectors_num)
    query_vectors, _ = read_function(train_filepath, limit=num_queries)

    index = faiss.IndexFlatL2(dimensionality)  
    index.add(index_vectors)

    batch_size = 1000
    num_batches = num_queries // batch_size
    print(f">>Number of batches: {num_batches}")

    all_distances = []
    all_indices = []
    for i in tqdm(range(num_batches)):
        start = i * batch_size
        end = start + batch_size
        distances, indices = index.search(query_vectors[start:end], k)
        all_distances.append(distances)
        all_indices.append(indices)

    if num_queries % batch_size != 0:
        distances, indices = index.search(query_vectors[num_batches * batch_size:], k)
        all_distances.append(distances)
        all_indices.append(indices)

    all_distances = np.vstack(all_distances)
    all_indices = np.vstack(all_indices)

    #gt_file_name = gt_dir + f"_nq{num_queries}_DB{index_vectors_num}_k{k}.ivecs"
    gt_file_name = gt_dir + f"_nq{num_queries}_k{k}.ivecs"
    write_ivecs(gt_file_name, all_indices, k)
    print(f">>Ground truth file written to {gt_file_name}")
    print("\n\n\n")