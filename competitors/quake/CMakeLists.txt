cmake_minimum_required(VERSION 3.24)
project(darth_quake_evaluation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# This standalone build uses the vendored C++ implementation directly. It does
# not invoke upstream setup.py (which auto-enables GPU/NUMA and all CPU cores),
# nor its git-submodule bootstrap. No Python bindings are needed by the runner.
find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(FAISS_ENABLE_GPU OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_RAFT OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_C_API OFF CACHE BOOL "" FORCE)
set(FAISS_OPT_LEVEL generic CACHE STRING "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

set(QUAKE_CPP "${CMAKE_CURRENT_SOURCE_DIR}/quake/src/cpp")
add_subdirectory("${QUAKE_CPP}/third_party/faiss" vendored_faiss EXCLUDE_FROM_ALL)
file(GLOB QUAKE_SOURCES CONFIGURE_DEPENDS "${QUAKE_CPP}/src/*.cpp")
add_library(quake_static STATIC ${QUAKE_SOURCES})
target_include_directories(quake_static PUBLIC
    "${QUAKE_CPP}/include"
    "${QUAKE_CPP}/third_party/concurrentqueue"
    ${TORCH_INCLUDE_DIRS})
target_link_libraries(quake_static PUBLIC faiss ${TORCH_LIBRARIES} OpenMP::OpenMP_CXX Threads::Threads)

add_executable(darth_quake_eval darth_quake_eval.cpp)
target_link_libraries(darth_quake_eval PRIVATE quake_static)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/UPSTREAM_COMMIT" QUAKE_COMMIT LIMIT_COUNT 1)
target_compile_definitions(darth_quake_eval PRIVATE QUAKE_UPSTREAM_COMMIT="${QUAKE_COMMIT}")
