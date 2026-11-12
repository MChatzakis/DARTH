#!/usr/bin/env bash
# Shared settings for run_laet.sh and run_rem.sh.
set -eo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIRECTORY=${BUILD_DIRECTORY:-"$SCRIPT_DIR/../build-native"}
source "$SCRIPT_DIR/experiment_paths.sh"

if [[ ! -f "$BUILD_DIRECTORY/CMakeCache.txt" ]]; then
    cmake_options=(-DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF)
    if [[ -n "${LIGHTGBM_ROOT:-}" ]]; then cmake_options+=("-DLIGHTGBM_ROOT=$LIGHTGBM_ROOT"); fi
    run_experiment cmake -S "$REPO_ROOT" -B "$BUILD_DIRECTORY" "${cmake_options[@]}"
fi
run_experiment cmake --build "$BUILD_DIRECTORY" --target "${NATIVE_METHOD}_pipeline" -j "$BUILD_JOBS"

INDEX=${INDEX:-hnsw}                 # hnsw or ivf
METRIC=${METRIC:-L2}
K=${K:-10}
TARGET_RECALLS=${TARGET_RECALLS:-${TARGET_RECALL:-"0.80 0.85 0.90 0.95 0.99"}}
THREADS=${THREADS:-8}
M=${M:-32}
EF_CONSTRUCTION=${EF_CONSTRUCTION:-500}
EF_SEARCH=${EF_SEARCH:-500}
NLIST=${NLIST:-1000}
NPROBE=${NPROBE:-100}
FIXED_AMOUNT_OF_SEARCH=${FIXED_AMOUNT_OF_SEARCH:-100} # LAET HNSW observation distance
SAVE_RESULTS=${SAVE_RESULTS:-0}
SAVE_MODELS=${SAVE_MODELS:-0}         # LAET only
MODELS_DIRECTORY=${MODELS_DIRECTORY:-"$REPO_ROOT/models"}

datasets=(SIFT10M SIFT50M SIFT100M DEEP10M DEEP50M DEEP100M GIST1M GLOVE100 T2I10M T2I100M)
if [[ -n "${DATASETS:-}" ]]; then read -r -a datasets <<< "$DATASETS"; fi
if [[ "$INDEX" == hnsw ]]; then
    index_name="hnsw_${METRIC}_M${M}_efC${EF_CONSTRUCTION}.index"
    index_options=(--M "$M" --efConstruction "$EF_CONSTRUCTION" --efSearch "$EF_SEARCH")
elif [[ "$INDEX" == ivf ]]; then
    index_name="ivf_${METRIC}_nlist${NLIST}.index"
    index_options=(--nlist "$NLIST" --nprobe "$NPROBE")
else
    echo "INDEX must be hnsw or ivf." >&2; exit 1
fi
method_options=()
if [[ "$NATIVE_METHOD" == laet ]]; then
    method_options+=(--fixed-amount-of-search "$FIXED_AMOUNT_OF_SEARCH")
fi
read -r -a targets <<< "$TARGET_RECALLS"
if ((${#targets[@]} == 0)); then echo "TARGET_RECALLS must contain at least one target." >&2; exit 1; fi
target_list=$(IFS=,; echo "${targets[*]}")
failed=()
for dataset in "${datasets[@]}"; do
    echo "--------------- $NATIVE_METHOD: $dataset, recalls $TARGET_RECALLS ---------------"
    output_options=()
    if [[ "$SAVE_RESULTS" == 1 ]]; then
        if ! run_experiment mkdir -p "$RESULTS_DIRECTORY/$dataset"; then failed+=("$dataset"); continue; fi
        output_options+=(--results "$RESULTS_DIRECTORY/$dataset/$INDEX-$NATIVE_METHOD-$METRIC")
    fi
    if [[ "$NATIVE_METHOD" == laet && "$SAVE_MODELS" == 1 ]]; then
        output_options+=(--save-models "$MODELS_DIRECTORY/$dataset/$INDEX-laet-$METRIC")
    fi
    if ! run_experiment "$BUILD_DIRECTORY/competitors/$NATIVE_METHOD/${NATIVE_METHOD}_pipeline" \
        --dataset "$dataset" --dataset-dir-prefix "$DATASET_DIRECTORY" \
        --index "$INDEX" --metric "$METRIC" --k "$K" --target-recalls "$target_list" --threads "$THREADS" \
        --index-filepath "$INDEX_DIRECTORY/$dataset/$index_name" \
        "${index_options[@]}" "${method_options[@]}" "${output_options[@]}" "$@"; then
        failed+=("$dataset")
    fi
done
if ((${#failed[@]})); then echo "Failed runs: ${failed[*]}" >&2; exit 1; fi
