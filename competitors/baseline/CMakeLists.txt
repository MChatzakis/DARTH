add_executable(hnsw_baseline hnsw_baseline.cpp ../../common/ComUtil.cpp ../../common/VectorDataLoader.cpp)
target_link_libraries(hnsw_baseline PRIVATE faiss)
add_executable(ivf_baseline ivf_baseline.cpp ../../common/ComUtil.cpp ../../common/VectorDataLoader.cpp)
target_link_libraries(ivf_baseline PRIVATE faiss)
