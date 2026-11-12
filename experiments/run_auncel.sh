#!/usr/bin/env bash
# Build Auncel, then evaluate it on DARTH's prebuilt IVF indexes.
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: ./experiments/run_auncel.sh [--check]

Default: build the evaluator, then for every dataset load DARTH's IndexIVFFlat,
train Auncel on the first learn queries, tune its hyperparameters on the first
validation queries for each recall target, and answer the first test queries
one at a time with the dataset's nprobe as the list budget.

  --check   Build the evaluator and check every input file; do not run searches

Environment:
  DATASETS="GIST1M GLOVE100 DEEP100M SIFT100M T2I100M"
  TARGET_RECALLS="0.80 0.85 0.90 0.95 0.99"
  K=50                    Number of returned neighbors
  TESTING_QUERIES=1000    First N test queries, as in DARTH+
  TRAIN_QUERIES=10000     First N learn queries for Auncel's training
  TUNE_QUERIES=1000       First N validation queries for hyperparameter tuning
  THREADS=$(nproc)        Threads for training and tuning; test queries use one
  RESUME=1                Skip datasets whose settings match an audited run
  DATASET_DIRECTORY=${REPO_ROOT}/data
  DARTH_IVF_INDEX_DIR=${REPO_ROOT}/indexes/ivf
  AUNCEL_RESULTS_DIR=<repo>/experiments/ivf-results/auncel
  PYTHON=python3          Python with numpy, for the audit

Examples:
  nohup ./experiments/run_auncel.sh > auncel-sweep.log 2>&1 &
  DATASETS=GLOVE100 AUNCEL_RESULTS_DIR=/tmp/auncel-pilot \
    ./experiments/run_auncel.sh
EOF
}

MODE=run
for argument in "$@"; do
    case "${argument}" in
        --check) MODE=check ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown argument: ${argument}" >&2; usage >&2; exit 2 ;;
    esac
done

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ROOT_DIR="${REPO_ROOT}/competitors/Auncel"
source "${REPO_ROOT}/experiments/experiment_logging.sh"

DATASET_DIRECTORY=${DATASET_DIRECTORY:-${DARTH_DATASET_DIR:-${REPO_ROOT}/data}}
INDEX_DIRECTORY=${DARTH_IVF_INDEX_DIR:-${INDEX_DIRECTORY:-${DARTH_INDEX_DIR:-${REPO_ROOT}/indexes/ivf}}}
RESULTS_DIRECTORY=${AUNCEL_RESULTS_DIR:-${REPO_ROOT}/experiments/ivf-results/auncel}
BUILD_DIRECTORY=${AUNCEL_BUILD_DIRECTORY:-${ROOT_DIR}/build}
PYTHON=${PYTHON:-python3}
EXECUTABLE=${BUILD_DIRECTORY}/darth_auncel_eval
K=${K:-50}
TESTING_QUERIES=${TESTING_QUERIES:-1000}
TRAIN_QUERIES=${TRAIN_QUERIES:-10000}
TUNE_QUERIES=${TUNE_QUERIES:-1000}
THREADS=${THREADS:-$(nproc)}
RESUME=${RESUME:-1}
read -r -a datasets <<< "${DATASETS:-GIST1M GLOVE100 DEEP100M SIFT100M T2I100M}"
read -r -a targets <<< "${TARGET_RECALLS:-${TARGET_RECALL:-0.80 0.85 0.90 0.95 0.99}}"

for setting in K TESTING_QUERIES TRAIN_QUERIES TUNE_QUERIES THREADS; do
    if [[ ! "${!setting}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${setting} must be a positive integer" >&2; exit 2
    fi
done
if [[ "${RESUME}" != 0 && "${RESUME}" != 1 ]]; then
    echo "RESUME must be 0 or 1" >&2; exit 2
fi
if ((${#datasets[@]} == 0 || ${#targets[@]} == 0)); then
    echo "DATASETS and TARGET_RECALLS must be nonempty" >&2; exit 2
fi
for target in "${targets[@]}"; do
    if [[ ! "${target}" =~ ^0\.[0-9]+$ ]]; then
        echo "Invalid recall target: ${target}" >&2; exit 2
    fi
done

# A stray Ctrl-S (or C-a s inside screen) pauses terminal output and silently
# stalls the sweep on its next write. Disable XON/XOFF while the sweep runs.
if [[ -t 0 ]]; then
    saved_stty="$(stty -g)"
    trap 'stty "${saved_stty}"' EXIT
    stty -ixon
fi
trap 'status=$?; echo "Auncel sweep failed (exit ${status}); see the error above." >&2; exit "${status}"' ERR

# nlist and nprobe match the original DARTH+ IVF experiments and
# the native LAET and REM pipelines; file names match common/VectorDataLoader.cpp.
dataset_config() {
    local dataset=$1
    local d=${DATASET_DIRECTORY}/${dataset}
    case "${dataset}" in
        GLOVE100) nlist=1000 nprobe=100 learn=learn.100K validation=validation.10K query=query.10K ;;
        GIST1M) nlist=1000 nprobe=200 learn=learn.100K validation=validation.1K query=query.1K ;;
        DEEP100M|SIFT100M) nlist=10000 nprobe=150 learn=learn.1M validation=validation.10K query=query.10K ;;
        T2I100M) nlist=20000 nprobe=500 learn=learn.1M validation=validation.10K query=query.10K ;;
        *) echo "Unknown dataset: ${dataset}" >&2; exit 2 ;;
    esac
    learn_gt=$d/${learn/./.groundtruth.}.k1000
    validation_gt=$d/${validation/./.groundtruth.}.k1000.ivecs
    query_gt=$d/${query/./.groundtruth.}.k1000.ivecs
    learn=$d/${learn}.fvecs validation=$d/${validation}.fvecs query=$d/${query}.fvecs
    index_file=${INDEX_DIRECTORY}/${dataset}/${dataset}.nlist${nlist}.index
    output_directory=${RESULTS_DIRECTORY}/${dataset}/k${K}
}

result_stem() {
    echo "${output_directory}/nlist${nlist}_nprobe${nprobe}_qs${TESTING_QUERIES}_tr$1"
}

results_are_valid() {
    local target stem rows
    for target in "${targets[@]}"; do
        stem=$(result_stem "${target}")
        [[ -s "${stem}.csv" && -s "${stem}_preparation.txt" ]] || return 1
        [[ "$(head -n 1 "${stem}.csv")" == "qid,dists,elaps_ms,r,clusters_searched,hits,neighbor_ids" ]] || return 1
        rows=$(($(wc -l < "${stem}.csv") - 1))
        ((rows == TESTING_QUERIES)) || return 1
        grep -q "^target_recall=${target}$" "${stem}_preparation.txt" || return 1
    done
}

run_configuration() {
    printf '%s\n' "$("${EXECUTABLE}" --version)" "dataset=${dataset}" "nlist=${nlist}" "nprobe=${nprobe}" \
        "k=${K}" "testing_queries=${TESTING_QUERIES}" "train_queries=${TRAIN_QUERIES}" \
        "tune_queries=${TUNE_QUERIES}" "targets=${targets[*]}" "index=${index_file}" "learn=${learn}" \
        "learn_gt=${learn_gt}" "validation=${validation}" "validation_gt=${validation_gt}" \
        "query=${query}" "query_gt=${query_gt}"
}

experiment_step "Auncel configure" cmake -S "${ROOT_DIR}" -B "${BUILD_DIRECTORY}" -DCMAKE_BUILD_TYPE=Release
experiment_step "Auncel build" cmake --build "${BUILD_DIRECTORY}" --target darth_auncel_eval -j "${THREADS}"
"${EXECUTABLE}" --version
"${PYTHON}" -c 'import numpy' 2> /dev/null || { echo "${PYTHON} needs numpy to audit results" >&2; exit 1; }

for dataset in "${datasets[@]}"; do
    dataset_config "${dataset}"
    if [[ "${MODE}" == run && ! -e "${index_file}" ]]; then
        DATASET_DIRECTORY="${DATASET_DIRECTORY}" bash "${REPO_ROOT}/experiments/ensure_ivf_index.sh" "${dataset}" "${nlist}" "${index_file}"
    fi
    for file in "${index_file}" "${learn}" "${learn_gt}.ivecs" "${learn_gt}.fvecs" "${validation}" \
        "${validation_gt}" "${query}" "${query_gt}"; do
        [[ -f "${file}" ]] || { echo "Missing ${dataset} input: ${file}" >&2; exit 1; }
    done
    experiment_progress "${dataset}: nlist=${nlist}, nprobe=${nprobe}, inputs present"
    [[ "${MODE}" == run ]] || continue

    mkdir -p "${output_directory}/training_traces"
    configuration_file=${output_directory}/run_config.txt
    if [[ "${RESUME}" == 1 && -f "${configuration_file}" ]] &&
        [[ "$(run_configuration)" == "$(cat "${configuration_file}")" ]] && results_are_valid; then
        experiment_progress "${dataset}: complete, matching run found; skipping (RESUME=0 reruns)"
        continue
    fi
    rm -f "${configuration_file}"

    # Auncel writes its training-trace logs to the working directory.
    (
        cd "${output_directory}/training_traces"
        experiment_step "Auncel ${dataset}" "${EXECUTABLE}" --dataset "${dataset}" \
            --index "${index_file}" --nprobe "${nprobe}" \
            --learn "${learn}" --learn-gt "${learn_gt}.ivecs" --learn-gt-dist "${learn_gt}.fvecs" \
            --validation "${validation}" --validation-gt "${validation_gt}" \
            --query "${query}" --query-gt "${query_gt}" \
            --targets "$(IFS=,; echo "${targets[*]}")" --k "${K}" --train-queries "${TRAIN_QUERIES}" \
            --tune-queries "${TUNE_QUERIES}" --test-queries "${TESTING_QUERIES}" \
            --output-dir "${output_directory}" --threads "${THREADS}"
    ) 2>&1 | sed -u -E '/^(SB\(\) [0-9]+|System search time: .*)$/d' | tee "${output_directory}/evaluation.log"

    results_are_valid || { echo "Invalid Auncel results in ${output_directory}" >&2; exit 1; }
    csvs=()
    for target in "${targets[@]}"; do csvs+=("$(result_stem "${target}").csv"); done
    experiment_step "${dataset} audit" "${PYTHON}" -I "${ROOT_DIR}/audit_results.py" --gt "${query_gt}" \
        --k "${K}" --queries "${TESTING_QUERIES}" --nprobe "${nprobe}" "${csvs[@]}" \
        2>&1 | tee -a "${output_directory}/evaluation.log"
    run_configuration > "${configuration_file}"
done

if [[ "${MODE}" == run ]]; then
    echo "Auncel sweep completed and CSVs audited. Results: ${RESULTS_DIRECTORY}/<DATASET>/k${K}/"
else
    echo "Auncel build and input checks passed; no searches were run."
fi
