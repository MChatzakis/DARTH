#!/usr/bin/env bash
set -eo pipefail

# Paths are shared with the other experiment scripts.
# Set DATASET_DIRECTORY to the directory containing the processed datasets.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIRECTORY=${BUILD_DIRECTORY:-"$SCRIPT_DIR/../build-native"}
source "$SCRIPT_DIR/experiment_paths.sh"

if [[ ! -f "$BUILD_DIRECTORY/CMakeCache.txt" ]]; then
    cmake_options=(-DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF)
    if [[ -n "${LIGHTGBM_ROOT:-}" ]]; then cmake_options+=("-DLIGHTGBM_ROOT=$LIGHTGBM_ROOT"); fi
    run_experiment cmake -S "$REPO_ROOT" -B "$BUILD_DIRECTORY" "${cmake_options[@]}"
fi
run_experiment cmake --build "$BUILD_DIRECTORY" --target darth_pipeline -j "$BUILD_JOBS"

# Edit these settings, or override them with environment variables.
INDEX=${INDEX:-hnsw}             # hnsw or ivf
METHOD=${METHOD:-darth}         # darth, darth-plus, or darth-plus-lb
METRIC=${METRIC:-L2}                 # L2 or inner-product
K=${K:-10}
# Space-separated targets; TARGET_RECALL remains a shortcut for one target.
TARGET_RECALLS=${TARGET_RECALLS:-${TARGET_RECALL:-"0.80 0.85 0.90 0.95 0.99"}}
THREADS=${THREADS:-8}
TRAINING_THREADS=${TRAINING_THREADS:-$THREADS}

# HNSW settings
M=${M:-32}
EF_CONSTRUCTION=${EF_CONSTRUCTION:-500}
EF_SEARCH=${EF_SEARCH:-500}

# IVF settings
NLIST=${NLIST:-1000}
NPROBE=${NPROBE:-100}

# Set either to 1 to save the corresponding files. Both are off by default.
SAVE_RESULTS=${SAVE_RESULTS:-0}
SAVE_MODELS=${SAVE_MODELS:-0}
MODELS_DIRECTORY=${MODELS_DIRECTORY:-"$REPO_ROOT/models"}

# Edit this list to choose datasets.
datasets=(
    SIFT10M SIFT50M SIFT100M
    DEEP10M DEEP50M DEEP100M
    GIST1M GLOVE100 T2I10M T2I100M
)
# Alternatively, set DATASETS to a space-separated list.
if [[ -n "${DATASETS:-}" ]]; then
    read -r -a datasets <<< "$DATASETS"
fi
if ((${#datasets[@]} == 0)); then
    echo "DATASETS must contain at least one dataset name." >&2
    exit 1
fi

if [[ "$INDEX" == hnsw ]]; then
    index_name="hnsw_${METRIC}_M${M}_efC${EF_CONSTRUCTION}.index"
    index_options=(--M "$M" --efConstruction "$EF_CONSTRUCTION" --efSearch "$EF_SEARCH")
elif [[ "$INDEX" == ivf ]]; then
    index_name="ivf_${METRIC}_nlist${NLIST}.index"
    index_options=(--nlist "$NLIST" --nprobe "$NPROBE")
else
    echo "INDEX must be hnsw or ivf." >&2
    exit 1
fi

failed_datasets=()
# The pipeline trains once per dataset and searches all requested targets.
read -r -a targets <<< "$TARGET_RECALLS"
if ((${#targets[@]} == 0)); then
    echo "TARGET_RECALLS must contain at least one target." >&2
    exit 1
fi
target_list=$(IFS=,; echo "${targets[*]}")
for dataset in "${datasets[@]}"; do
    echo ""
    echo "--------------- $dataset: recalls $TARGET_RECALLS ---------------"
    output_options=()
    if [[ "$SAVE_RESULTS" == 1 ]]; then
        if ! run_experiment mkdir -p "$RESULTS_DIRECTORY/$dataset"; then
            failed_datasets+=("$dataset")
            continue
        fi
        output_options+=(--results "$RESULTS_DIRECTORY/$dataset/$INDEX-$METHOD-$METRIC")
    fi
    if [[ "$SAVE_MODELS" == 1 ]]; then
        output_options+=(--save-models "$MODELS_DIRECTORY/$dataset/$INDEX-$METHOD-$METRIC")
    fi
    # The pipeline defaults to 10,000 training, 1,000 validation, and 1,000 test queries.
    # Pass query-count options after this script to override those defaults.
    if ! run_experiment "$BUILD_DIRECTORY/darth-runners/darth_pipeline" \
        --dataset "$dataset" --dataset-dir-prefix "$DATASET_DIRECTORY" \
        --index "$INDEX" --method "$METHOD" --metric "$METRIC" \
        --k "$K" --target-recalls "$target_list" \
        --threads "$THREADS" --training-threads "$TRAINING_THREADS" \
        --index-filepath "$INDEX_DIRECTORY/$dataset/$index_name" \
        "${index_options[@]}" "${output_options[@]}" "$@"; then
        failed_datasets+=("$dataset")
    fi
done

if ((${#failed_datasets[@]})); then
    echo "Failed datasets: ${failed_datasets[*]}" >&2
    exit 1
fi
