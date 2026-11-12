add_executable(rem_pipeline rem_pipeline.cpp ../../common/ComUtil.cpp ../../common/VectorDataLoader.cpp)
target_link_libraries(rem_pipeline PRIVATE faiss)
