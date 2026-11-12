#!/usr/bin/env bash
# Plain HNSW/IVF search. Target recall affects reported RQUT, not search effort.
set -eo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIRECTORY=${BUILD_DIRECTORY:-"$SCRIPT_DIR/../build-native"}
source "$SCRIPT_DIR/experiment_paths.sh"
INDEX=${INDEX:-hnsw}
BASELINE_MODE=${BASELINE_MODE:-plain} # plain or naive (HNSW only)
DISTANCE_THRESHOLD=${DISTANCE_THRESHOLD:-1000}
K=${K:-10}
QUERY_NUM=${QUERY_NUM:-1000}
THREADS=${THREADS:-8}
TARGET_RECALLS=${TARGET_RECALLS:-${TARGET_RECALL:-"0.80 0.85 0.90 0.95 0.99"}}
M=${M:-32}; EF_CONSTRUCTION=${EF_CONSTRUCTION:-500}; EF_SEARCH=${EF_SEARCH:-500}
NLIST=${NLIST:-1000}; NPROBE=${NPROBE:-100}
datasets=(SIFT10M SIFT50M SIFT100M DEEP10M DEEP50M DEEP100M GIST1M GLOVE100 T2I10M T2I100M)
if [[ -n "${DATASETS:-}" ]]; then read -r -a datasets <<< "$DATASETS"; fi
if [[ "$INDEX" == hnsw ]]; then
    index_name="hnsw_L2_M${M}_efC${EF_CONSTRUCTION}.index"
    options=(--M "$M" --efConstruction "$EF_CONSTRUCTION" --efSearch "$EF_SEARCH")
    if [[ "$BASELINE_MODE" == naive ]]; then
        options+=(--mode naive-early-stop-testing --initial-prediction-interval "$DISTANCE_THRESHOLD")
    elif [[ "$BASELINE_MODE" == plain ]]; then
        options+=(--mode no-early-stop)
    else
        echo "BASELINE_MODE must be plain or naive." >&2; exit 1
    fi
elif [[ "$INDEX" == ivf ]]; then
    if [[ "$BASELINE_MODE" != plain ]]; then echo "The naive baseline requires HNSW." >&2; exit 1; fi
    index_name="ivf_L2_nlist${NLIST}.index"
    options=(--nlist "$NLIST" --nprobe "$NPROBE" --mode plain-ivf)
else
    echo "INDEX must be hnsw or ivf." >&2; exit 1
fi
if [[ ! -f "$BUILD_DIRECTORY/CMakeCache.txt" ]]; then
    cmake_options=(-DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF)
    if [[ -n "${LIGHTGBM_ROOT:-}" ]]; then cmake_options+=("-DLIGHTGBM_ROOT=$LIGHTGBM_ROOT"); fi
    run_experiment cmake -S "$REPO_ROOT" -B "$BUILD_DIRECTORY" "${cmake_options[@]}"
fi
run_experiment cmake --build "$BUILD_DIRECTORY" --target "${INDEX}_baseline" -j "$BUILD_JOBS"
failed=()
for dataset in "${datasets[@]}"; do
    for target in $TARGET_RECALLS; do
        echo "--------------- baseline: $dataset, recall $target ---------------"
        if ! run_experiment mkdir -p "$RESULTS_DIRECTORY/baseline/$dataset"; then failed+=("$dataset/$target"); continue; fi
        if ! run_experiment "$BUILD_DIRECTORY/competitors/baseline/${INDEX}_baseline" \
            --dataset "$dataset" --dataset-dir-prefix "$DATASET_DIRECTORY" --query-type testing \
            --query-num "$QUERY_NUM" --k "$K" --threads "$THREADS" --target-recall "$target" \
            --index-filepath "$INDEX_DIRECTORY/$dataset/$index_name" \
            --output "$RESULTS_DIRECTORY/baseline/$dataset/$INDEX-$BASELINE_MODE-target$target.csv" "${options[@]}" "$@"; then
            failed+=("$dataset/$target")
        fi
    done
done
if ((${#failed[@]})); then echo "Failed runs: ${failed[*]}" >&2; exit 1; fi
