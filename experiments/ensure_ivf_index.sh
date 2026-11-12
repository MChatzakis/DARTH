#!/usr/bin/env bash
# Build the shared L2 IVF index expected by the comparison runners, if missing.
set -eo pipefail
if (($# != 3)); then echo 'Usage: ensure_ivf_index.sh DATASET NLIST INDEX_PATH' >&2; exit 2; fi
DATASET=$1
NLIST=$2
INDEX_PATH=$3
if [[ -e "$INDEX_PATH" ]]; then exit 0; fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
BUILD_DIRECTORY=${DARTH_BUILD_DIRECTORY:-"$REPO_ROOT/build-native"}
DATASET_DIRECTORY=${DATASET_DIRECTORY:-${DARTH_DATASET_DIR:-}}
if [[ -z "$DATASET_DIRECTORY" ]]; then echo 'Set DATASET_DIRECTORY.' >&2; exit 2; fi
if [[ ! -x "$BUILD_DIRECTORY/common/ivf_index_builder" ]]; then
    cmake_options=(-DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF)
    if [[ -n "${LIGHTGBM_ROOT:-}" ]]; then cmake_options+=("-DLIGHTGBM_ROOT=$LIGHTGBM_ROOT"); fi
    cmake -S "$REPO_ROOT" -B "$BUILD_DIRECTORY" "${cmake_options[@]}"
    cmake --build "$BUILD_DIRECTORY" --target ivf_index_builder -j "${BUILD_JOBS:-1}"
fi
"$BUILD_DIRECTORY/common/ivf_index_builder" "$DATASET" "$DATASET_DIRECTORY" "$NLIST" "$INDEX_PATH" "${THREADS:-1}"
