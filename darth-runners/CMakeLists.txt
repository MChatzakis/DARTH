add_executable(darth_pipeline darth_pipeline.cpp ../common/ComUtil.cpp ../common/VectorDataLoader.cpp)
target_link_libraries(darth_pipeline PRIVATE faiss)
