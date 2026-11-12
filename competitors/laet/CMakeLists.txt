add_executable(laet_pipeline laet_pipeline.cpp ../../common/ComUtil.cpp ../../common/VectorDataLoader.cpp)
target_link_libraries(laet_pipeline PRIVATE faiss)
