cmake_minimum_required(VERSION 3.16)
project(darth_adaef_evaluation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Ada-ef is header-only on top of hnswlib. It needs Eigen 3.4 and the
# header-only Boost.Math normal distribution.
set(EIGEN3_INCLUDE_DIR "" CACHE PATH "Directory that contains Eigen/Core")
set(BOOST_ROOT "" CACHE PATH "Directory that contains boost/math/")
if(NOT EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen/Core")
    message(FATAL_ERROR "Set EIGEN3_INCLUDE_DIR to an Eigen 3.4 include directory")
endif()
if(NOT EXISTS "${BOOST_ROOT}/boost/math/distributions/normal.hpp")
    message(FATAL_ERROR "Set BOOST_ROOT to a Boost source/include directory")
endif()

find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

add_executable(darth_adaef_eval darth_adaef_eval.cpp)
target_include_directories(darth_adaef_eval SYSTEM PRIVATE "${EIGEN3_INCLUDE_DIR}" "${BOOST_ROOT}")
target_include_directories(darth_adaef_eval PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
# The upstream experiments were built with -O3 -march=native.
target_compile_options(darth_adaef_eval PRIVATE -O3 -march=native -w)
target_link_libraries(darth_adaef_eval PRIVATE OpenMP::OpenMP_CXX Threads::Threads)
