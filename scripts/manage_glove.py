# http://ann-benchmarks.com/glove-100-angular.hdf5

import h5py
import numpy as np

GLOVE_DIR = "/mnt/hddhelp/mchatzakis/datasets/GLOVE100"

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
            

with h5py.File(f"{GLOVE_DIR}/misc/glove-100-angular.hdf5", "r") as f:
    print("Keys in the HDF5 file:", list(f.keys()))
    
    test = f["test"]
    train = f["train"]
    neighbors = f["neighbors"]
    distances = f["distances"]
    
    print("Test data shape:", test.shape)
    print("Train data shape:", train.shape)
    print("Neighbors data shape:", neighbors.shape)
    print("Distances data shape:", distances.shape)
    
    dimension = test.shape[1]
    print("Dimension: ", dimension)
    
    write_fvecs(f"{GLOVE_DIR}/glove-100_base.fvecs", train)    


with open(f"{GLOVE_DIR}/misc/glove.6B.100d.txt", "r") as f:
    raw_dataset_vectors = []
    for line in f:
        parts = line.split()
        word = parts[0]
        vector = np.array([float(x) for x in parts[1:]])
        raw_dataset_vectors.append(vector)
    
    np.random.shuffle(raw_dataset_vectors)
    
    training_vectors_num = 350000
    query_vectors_num = 10000
    
    training_vectors = []
    query_vectors = []
    
    for i in range(training_vectors_num):
        training_vectors.append(raw_dataset_vectors[i])
    
    for i in range(training_vectors_num, training_vectors_num + query_vectors_num):
        query_vectors.append(raw_dataset_vectors[i])
    
    write_fvecs(f"{GLOVE_DIR}/glove-100_learn_350K.fvecs", training_vectors)
    write_fvecs(f"{GLOVE_DIR}/glove-100_query_10K.fvecs", query_vectors)

    