#!/usr/bin/env bash
# Build Ada-ef, then evaluate it on DARTH+'s inner-product HNSW protocol.
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: ./experiments/run_ada_ef.sh [--check]

Default: build the evaluator, then for every dataset build (or reuse) an
hnswlib inner-product index with DARTH+'s M/efConstruction, compute Ada-ef's
dataset statistics, build its ef-estimation table from the first learn queries
for each recall target, and answer the first test queries one at a time.

  --check   Build the evaluator and check every input file; do not run searches

Environment:
  DATASETS="GLOVE100 GIST1M DEEP10M SIFT10M T2I10M"
  TARGET_RECALLS="0.80 0.85 0.90 0.95 0.99"
  K=50                    Number of returned neighbors
  TESTING_QUERIES=1000    First N test queries, as in DARTH+
  LEARN_QUERIES=200       First N learn queries for Ada-ef's ef-estimation table
  EF_UPPER_BOUND=<DARTH+ efSearch of each dataset>   Ada-ef's ef cap
  DISTANCE_LIST_SIZE=1057 Distances collected before ef is chosen (upstream default)
  BUILD_THREADS=$(nproc)  Threads for index construction and ground truth only
  RESUME=1                Skip datasets whose settings match a validated run
  DATASET_DIRECTORY=${REPO_ROOT}/data
  ADAEF_INDEX_DIR=${REPO_ROOT}/indexes/adaef
  ADAEF_RESULTS_DIR=<repo>/experiments/hnsw-results/adaef
  ADAEF_DATA_DIR=<repo>/competitors/Ada-ef/data          Recomputed T2I10M ground truth
  EIGEN3_INCLUDE_DIR=/usr/include/eigen3
  BOOST_ROOT=/usr/include
  PYTHON=python3          Python with numpy (audit) and faiss (T2I10M ground truth)

Examples:
  nohup ./experiments/run_ada_ef.sh > adaef-sweep.log 2>&1 &
  DATASETS=GLOVE100 TESTING_QUERIES=50 ADAEF_RESULTS_DIR=/tmp/adaef-pilot \
    ./experiments/run_ada_ef.sh
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
ROOT_DIR="${REPO_ROOT}/competitors/Ada-ef"
source "${REPO_ROOT}/experiments/experiment_logging.sh"

DATASET_DIRECTORY=${DATASET_DIRECTORY:-${DARTH_DATASET_DIR:-${REPO_ROOT}/data}}
INDEX_DIRECTORY=${ADAEF_INDEX_DIR:-${INDEX_DIRECTORY:-${DARTH_INDEX_DIR:-${REPO_ROOT}/indexes/adaef}}}
RESULTS_DIRECTORY=${ADAEF_RESULTS_DIR:-${REPO_ROOT}/experiments/hnsw-results/adaef}
DATA_DIRECTORY=${ADAEF_DATA_DIR:-${ROOT_DIR}/data}
BUILD_DIRECTORY=${ADAEF_BUILD_DIRECTORY:-${ROOT_DIR}/build}
EIGEN3_INCLUDE_DIR=${EIGEN3_INCLUDE_DIR:-/usr/include/eigen3}
BOOST_ROOT=${BOOST_ROOT:-/usr/include}
PYTHON=${PYTHON:-python3}
EXECUTABLE=${BUILD_DIRECTORY}/darth_adaef_eval
K=${K:-50}
TESTING_QUERIES=${TESTING_QUERIES:-1000}
LEARN_QUERIES=${LEARN_QUERIES:-200}
DISTANCE_LIST_SIZE=${DISTANCE_LIST_SIZE:-1057}
BUILD_THREADS=${BUILD_THREADS:-$(nproc)}
RESUME=${RESUME:-1}
read -r -a datasets <<< "${DATASETS:-GLOVE100 GIST1M DEEP10M SIFT10M T2I10M}"
read -r -a targets <<< "${TARGET_RECALLS:-${TARGET_RECALL:-0.80 0.85 0.90 0.95 0.99}}"

for setting in K TESTING_QUERIES LEARN_QUERIES DISTANCE_LIST_SIZE BUILD_THREADS; do
    if [[ ! "${!setting}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${setting} must be a positive integer" >&2; exit 2
    fi
done
if [[ -n "${EF_UPPER_BOUND:-}" && ! "${EF_UPPER_BOUND}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EF_UPPER_BOUND must be a positive integer" >&2; exit 2
fi
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
trap 'status=$?; echo "Ada-ef sweep failed (exit ${status}); see the error above." >&2; exit "${status}"' ERR

# M, efConstruction, efSearch, and file names follow DARTH+'s inner-product
# HNSW experiments and common/VectorDataLoader.cpp. T2I10M's inner-product ground truth
# is no longer on disk, so it is recomputed exactly for the needed prefixes.
dataset_config() {
    local dataset=$1
    local d=${DATASET_DIRECTORY}/${dataset}
    compute_gt=0
    case "${dataset}" in
        GLOVE100)
            M=32 efC=500 efS=1000
            base=$d/base.1183514.fvecs learn=$d/learn.100K.fvecs query=$d/query.10K.fvecs
            learn_gt=$d/innprod.learn.groundtruth.100K.k1000.ivecs
            query_gt=$d/innprod.query.groundtruth.10K.k1000.ivecs ;;
        GIST1M)
            M=64 efC=500 efS=1000
            base=$d/base.1M.fvecs learn=$d/learn.100K.fvecs query=$d/query.1K.fvecs
            learn_gt=$d/innprod.learn.groundtruth.100K.k1000.ivecs
            query_gt=$d/innprod.query.groundtruth.1K.k1000.ivecs ;;
        DEEP10M|SIFT10M)
            M=16 efC=500
            if [[ "${dataset}" == DEEP10M ]]; then efS=750; else efS=500; fi
            base=$d/base.10M.fvecs learn=$d/learn.1M.fvecs query=$d/query.10K.fvecs
            learn_gt=$d/innprod.learn.groundtruth.1M.k1000.ivecs
            query_gt=$d/innprod.query.groundtruth.10K.k1000.ivecs ;;
        T2I10M)
            M=40 efC=1000 efS=1500
            base=$d/base.10M.fvecs learn=$d/learn.10K.fvecs query=$d/query.10K.fvecs
            learn_gt=${DATA_DIRECTORY}/${dataset}/innprod.learn.groundtruth.first${LEARN_QUERIES}.k1000.ivecs
            query_gt=${DATA_DIRECTORY}/${dataset}/innprod.query.groundtruth.first${TESTING_QUERIES}.k1000.ivecs
            compute_gt=1 ;;
        *) echo "Unknown dataset: ${dataset}" >&2; exit 2 ;;
    esac
    ef_upper_bound=${EF_UPPER_BOUND:-${efS}}
    index_file=${INDEX_DIRECTORY}/${dataset}/${dataset}.M${M}.efC${efC}.inner-product.hnswlib
    statistics_file=${INDEX_DIRECTORY}/${dataset}/${dataset}.inner-product.adaef-statistics.bin
    output_directory=${RESULTS_DIRECTORY}/${dataset}/k${K}
}

result_stem() {
    echo "${output_directory}/M${M}_efC${efC}_qs${TESTING_QUERIES}_tr$1_inner-product"
}

results_are_valid() {
    local target stem rows
    for target in "${targets[@]}"; do
        stem=$(result_stem "${target}")
        [[ -s "${stem}.csv" && -s "${stem}_preparation.txt" ]] || return 1
        [[ "$(head -n 1 "${stem}.csv")" == "qid,dists,elaps_ms,r,hits,ef,score,neighbor_ids" ]] || return 1
        rows=$(($(wc -l < "${stem}.csv") - 1))
        ((rows == TESTING_QUERIES)) || return 1
        grep -q "^target_recall=${target}$" "${stem}_preparation.txt" || return 1
    done
}

run_configuration() {
    printf '%s\n' "$("${EXECUTABLE}" --version)" "dataset=${dataset}" "M=${M}" "efC=${efC}" "k=${K}" \
        "testing_queries=${TESTING_QUERIES}" "learn_queries=${LEARN_QUERIES}" \
        "ef_upper_bound=${ef_upper_bound}" "distance_list_size=${DISTANCE_LIST_SIZE}" \
        "targets=${targets[*]}" "base=${base}" "learn=${learn}" "learn_gt=${learn_gt}" \
        "query=${query}" "query_gt=${query_gt}" "index=${index_file}"
}

experiment_step "Ada-ef configure" cmake -S "${ROOT_DIR}" -B "${BUILD_DIRECTORY}" -DCMAKE_BUILD_TYPE=Release \
    "-DEIGEN3_INCLUDE_DIR=${EIGEN3_INCLUDE_DIR}" "-DBOOST_ROOT=${BOOST_ROOT}"
experiment_step "Ada-ef build" cmake --build "${BUILD_DIRECTORY}" --target darth_adaef_eval -j "${BUILD_THREADS}"
"${EXECUTABLE}" --version

for dataset in "${datasets[@]}"; do
    dataset_config "${dataset}"
    required=("${base}" "${learn}" "${query}")
    "${PYTHON}" -c 'import numpy' 2> /dev/null || {
        echo "${PYTHON} needs numpy to audit results" >&2; exit 1; }
    if ((compute_gt)); then
        "${PYTHON}" -c 'import faiss, numpy' 2> /dev/null || {
            echo "${PYTHON} needs faiss and numpy to compute ${dataset} ground truth" >&2; exit 1; }
    else
        required+=("${learn_gt}" "${query_gt}")
    fi
    for file in "${required[@]}"; do
        [[ -f "${file}" ]] || { echo "Missing ${dataset} input: ${file}" >&2; exit 1; }
    done
    experiment_progress "${dataset}: M=${M}, efC=${efC}, ef_upper_bound=${ef_upper_bound}, inputs present"
    [[ "${MODE}" == run ]] || continue

    if ((compute_gt)); then
        for spec in "query ${query} ${TESTING_QUERIES} ${query_gt}" "learn ${learn} ${LEARN_QUERIES} ${learn_gt}"; do
            read -r name queries count gt_file <<< "${spec}"
            if [[ ! -s "${gt_file}" ]]; then
                OMP_NUM_THREADS=${BUILD_THREADS} experiment_step "${dataset} ${name} ground truth" \
                    "${PYTHON}" -I "${ROOT_DIR}/prepare_groundtruth.py" --base "${base}" --queries "${queries}" \
                    --count "${count}" --k 1000 --out-prefix "${gt_file%.ivecs}"
            fi
        done
    fi

    mkdir -p "${output_directory}"
    configuration_file=${output_directory}/run_config.txt
    if [[ "${RESUME}" == 1 && -f "${configuration_file}" ]] &&
        [[ "$(run_configuration)" == "$(cat "${configuration_file}")" ]] && results_are_valid; then
        experiment_progress "${dataset}: complete, matching run found; skipping (RESUME=0 reruns)"
        continue
    fi
    rm -f "${configuration_file}"

    experiment_step "Ada-ef ${dataset}" "${EXECUTABLE}" --dataset "${dataset}" \
        --base "${base}" --learn "${learn}" --learn-gt "${learn_gt}" \
        --query "${query}" --query-gt "${query_gt}" --M "${M}" --ef-construction "${efC}" \
        --index "${index_file}" --statistics "${statistics_file}" --output-dir "${output_directory}" \
        --targets "$(IFS=,; echo "${targets[*]}")" --k "${K}" \
        --learn-queries "${LEARN_QUERIES}" --test-queries "${TESTING_QUERIES}" \
        --ef-upper-bound "${ef_upper_bound}" --distance-list-size "${DISTANCE_LIST_SIZE}" \
        --build-threads "${BUILD_THREADS}" 2>&1 | tee "${output_directory}/evaluation.log"

    results_are_valid || { echo "Invalid Ada-ef results in ${output_directory}" >&2; exit 1; }
    csvs=()
    for target in "${targets[@]}"; do csvs+=("$(result_stem "${target}").csv"); done
    experiment_step "${dataset} audit" "${PYTHON}" -I "${ROOT_DIR}/audit_results.py" --gt "${query_gt}" \
        --k "${K}" --queries "${TESTING_QUERIES}" --base "${base}" --ef-cap "${ef_upper_bound}" "${csvs[@]}" \
        2>&1 | tee -a "${output_directory}/evaluation.log"
    run_configuration > "${configuration_file}"
done

if [[ "${MODE}" == run ]]; then
    echo "Ada-ef sweep completed and CSVs audited. Results: ${RESULTS_DIRECTORY}/<DATASET>/k${K}/"
else
    echo "Ada-ef build and input checks passed; no searches were run."
fi
