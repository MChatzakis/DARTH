cmake_minimum_required(VERSION 3.16)
project(darth_auncel_evaluation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Auncel is a fork of FAISS 1.5.2. Its upstream autoconf build compiles every
# top-level source into libfaiss; this CMake build does the same, CPU only.
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

set(AUNCEL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Auncel")
file(GLOB AUNCEL_SOURCES CONFIGURE_DEPENDS "${AUNCEL_SOURCE_DIR}/*.cpp")
add_library(auncel_faiss STATIC ${AUNCEL_SOURCES})
target_include_directories(auncel_faiss PUBLIC "${AUNCEL_SOURCE_DIR}")
target_compile_definitions(auncel_faiss PUBLIC FINTEGER=int)
target_compile_options(auncel_faiss PRIVATE -O3 -march=native -Wno-sign-compare -w)
target_link_libraries(auncel_faiss PUBLIC OpenMP::OpenMP_CXX ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/darth_auncel_eval.cpp")
    add_executable(darth_auncel_eval darth_auncel_eval.cpp)
    set_target_properties(darth_auncel_eval PROPERTIES CXX_STANDARD 17)
    target_compile_options(darth_auncel_eval PRIVATE -O3 -march=native)
    target_link_libraries(darth_auncel_eval PRIVATE auncel_faiss)
endif()
