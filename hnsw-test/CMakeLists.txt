# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_executable(hnsw_test EXCLUDE_FROM_ALL hnsw_test.cpp ComUtil.cpp VectorDataLoader.cpp)
target_link_libraries(hnsw_test PRIVATE faiss)

add_executable(ivf_test EXCLUDE_FROM_ALL ivf_test.cpp ComUtil.cpp VectorDataLoader.cpp)
target_link_libraries(ivf_test PRIVATE faiss)

