#!/usr/bin/env bash
# Set up CPU-only Quake, then evaluate fixed IVF indexes.
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: ./experiments/run_quake.sh [--dry-run | --check | --setup-only]

Default: install local CPU dependencies, build, then evaluate all five datasets at k=50, Rt={0.80,0.85,0.90,0.95,0.99}.
Search is always single-threaded, without maintenance or background workers.
Indexes preserve the exact original DARTH IVF centroids, lists, and vector IDs.

  --dry-run     Print commands without installing, building, or searching
  --check       Set up and check all data paths;
                do not run real-dataset searches
  --setup-only  Set up without requiring real datasets

Environment:
  DATASETS="SIFT100M DEEP100M GLOVE100 GIST1M T2I100M"
  TARGET_RECALLS="0.80 0.85 0.90 0.95 0.99"
  K=50                         Number of returned neighbors
  TESTING_QUERIES=1000          First N test queries, in their original order
  BUILD_JOBS=1                 Compiler processes; does not change search threads
  SETUP=1                      0 reuses the existing environment and executable
  RESUME=1                     Reuse only complete, matching, audited datasets
  PYTHON=python3               Python 3.10-3.12 for local venv
  DATASET_DIRECTORY=${REPO_ROOT}/data
  DARTH_IVF_INDEX_DIR=${REPO_ROOT}/indexes/ivf
  QUAKE_RESULTS_DIR=<repo>/experiments/ivf-results/quake
  QUAKE_ENV_DIRECTORY=<repo>/competitors/quake/.venv
  QUAKE_BUILD_DIRECTORY=<repo>/competitors/quake/build
  QUAKE_LOG_DIRECTORY=<repo>/competitors/quake/logs/<timestamp>

Examples:
  ./experiments/run_quake.sh --check
  nohup ./experiments/run_quake.sh > quake-sweep.log 2>&1 &
  DATASETS=GLOVE100 TESTING_QUERIES=20 QUAKE_RESULTS_DIR=/tmp/quake-pilot \
    SETUP=0 ./experiments/run_quake.sh
EOF
}

MODE=run
DRY_RUN=0
for argument in "$@"; do
    case "${argument}" in
        --dry-run) DRY_RUN=1 ;;
        --check) MODE=check ;;
        --setup-only) MODE=setup ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown argument: ${argument}" >&2; usage >&2; exit 2 ;;
    esac
done
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ROOT_DIR="${REPO_ROOT}/competitors/quake"
source "${REPO_ROOT}/experiments/experiment_logging.sh"
PYTHON=${PYTHON:-python3}
ENV_DIRECTORY=${QUAKE_ENV_DIRECTORY:-${ROOT_DIR}/.venv}
ENV_PYTHON=${ENV_DIRECTORY}/bin/python
BUILD_DIRECTORY=${QUAKE_BUILD_DIRECTORY:-${ROOT_DIR}/build}
EXECUTABLE=${BUILD_DIRECTORY}/darth_quake_eval
LOG_DIRECTORY=${QUAKE_LOG_DIRECTORY:-${ROOT_DIR}/logs/$(date +%Y%m%d-%H%M%S)}
BUILD_JOBS=${BUILD_JOBS:-1}
K=${K:-50}
TESTING_QUERIES=${TESTING_QUERIES:-1000}
SETUP=${SETUP:-1}
RESUME=${RESUME:-1}
for setting in BUILD_JOBS K TESTING_QUERIES; do
    if [[ ! "${!setting}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${setting} must be a positive integer" >&2; exit 2
    fi
done
for setting in SETUP RESUME; do
    if [[ "${!setting}" != 0 && "${!setting}" != 1 ]]; then
        echo "${setting} must be 0 or 1" >&2; exit 2
    fi
done
read -r -a datasets <<< "${DATASETS:-SIFT100M DEEP100M GLOVE100 GIST1M T2I100M}"
read -r -a targets <<< "${TARGET_RECALLS:-${TARGET_RECALL:-0.80 0.85 0.90 0.95 0.99}}"
if ((${#datasets[@]} == 0 || ${#targets[@]} == 0)); then
    echo "DATASETS and TARGET_RECALLS must be nonempty" >&2; exit 2
fi

# Runtime library limits apply even when the invoking shell has larger values.
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_MAX_ACTIVE_LEVELS=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PATH="${ENV_DIRECTORY}/bin:${PATH}"
CURRENT_LOG=""
trap 'status=$?; echo "Quake sweep failed (exit ${status}); see ${CURRENT_LOG:-the error above}." >&2; exit "${status}"' ERR

run_step() {
    local name=$1
    shift
    CURRENT_LOG="${LOG_DIRECTORY}/${name}.log"
    printf '\n[%s] ' "${name}"
    printf '%q ' "$@"
    printf '\n'
    if [[ "${DRY_RUN}" == 0 ]]; then
        mkdir -p "${LOG_DIRECTORY}"
        experiment_step "Quake ${name}" "$@" 2>&1 | tee -a "${CURRENT_LOG}"
    fi
}

evaluation_args=("${ROOT_DIR}/evaluate.py"
    --datasets "${datasets[@]}" --k "${K}" --query-count "${TESTING_QUERIES}"
    --target-recalls "${targets[@]}"
    --dataset-directory "${DATASET_DIRECTORY:-${DARTH_DATASET_DIR:-${REPO_ROOT}/data}}"
    --index-directory "${DARTH_IVF_INDEX_DIR:-${INDEX_DIRECTORY:-${DARTH_INDEX_DIR:-${REPO_ROOT}/indexes/ivf}}}"
    --output-directory "${QUAKE_RESULTS_DIR:-${REPO_ROOT}/experiments/ivf-results/quake}"
    --executable "${EXECUTABLE}")
run_step python_version "${PYTHON}" -c \
    'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "Use Python 3.10-3.12 with the pinned CPU dependencies"'
if [[ "${MODE}" == run ]]; then
    index_directory=${DARTH_IVF_INDEX_DIR:-${INDEX_DIRECTORY:-${DARTH_INDEX_DIR:-${REPO_ROOT}/indexes/ivf}}}
    for dataset in "${datasets[@]}"; do
        case "$dataset" in
            GIST1M|GLOVE100) nlist=1000 ;;
            DEEP100M|SIFT100M) nlist=10000 ;;
            T2I100M) nlist=20000 ;;
            *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
        esac
        index_file="$index_directory/$dataset/$dataset.nlist${nlist}.index"
        if [[ ! -e "$index_file" ]]; then
            run_step "index_${dataset}" env DATASET_DIRECTORY="${DATASET_DIRECTORY:-${DARTH_DATASET_DIR:-${REPO_ROOT}/data}}" \
                bash "${REPO_ROOT}/experiments/ensure_ivf_index.sh" "$dataset" "$nlist" "$index_file"
        fi
    done
fi
if [[ "${MODE}" != setup ]]; then
    run_step preflight "${PYTHON}" "${evaluation_args[@]}" --preflight-only
fi
if [[ "${SETUP}" == 1 ]]; then
    if [[ ! -x "${ENV_PYTHON}" ]]; then
        run_step environment "${PYTHON}" -m venv "${ENV_DIRECTORY}"
    fi
    run_step pip "${ENV_PYTHON}" -m pip install --disable-pip-version-check 'pip==25.3'
    run_step dependencies "${ENV_PYTHON}" -m pip install --disable-pip-version-check -r "${ROOT_DIR}/requirements-cpu.txt"
    # Use an existing recent CMake, otherwise install one inside this venv.
    if [[ "${DRY_RUN}" == 0 ]] && ! "${ENV_PYTHON}" -c \
        'import re, subprocess; v=subprocess.check_output(["cmake","--version"],text=True); assert tuple(map(int,re.search(r"(\d+)\.(\d+)",v).groups())) >= (3,24)' >/dev/null 2>&1; then
        run_step cmake_dependency "${ENV_PYTHON}" -m pip install 'cmake==3.29.6'
    fi
    if [[ "${DRY_RUN}" == 1 ]]; then
        TORCH_CMAKE='<CPU PyTorch CMake directory>'
    else
        TORCH_CMAKE=$("${ENV_PYTHON}" -c 'import torch; assert torch.version.cuda is None, "CPU-only PyTorch required"; print(torch.utils.cmake_prefix_path)')
    fi
    run_step configure cmake -S "${ROOT_DIR}" -B "${BUILD_DIRECTORY}" \
        -DCMAKE_BUILD_TYPE=Release "-DCMAKE_PREFIX_PATH=${TORCH_CMAKE}"
    run_step build cmake --build "${BUILD_DIRECTORY}" --target darth_quake_eval -j "${BUILD_JOBS}"
fi
run_step version "${EXECUTABLE}" --version
if [[ "${MODE}" == run ]]; then
    if [[ "${RESUME}" == 1 ]]; then evaluation_args+=(--resume); fi
    run_step evaluation "${ENV_PYTHON}" "${evaluation_args[@]}"
fi
if [[ "${DRY_RUN}" == 1 ]]; then
    echo 'Dry run complete; no installation, builds, or searches were started.'
elif [[ "${MODE}" == run ]]; then
    echo "Quake sweep completed and CSVs verified. Results: ${QUAKE_RESULTS_DIR:-${REPO_ROOT}/experiments/ivf-results/quake}"
else
    echo 'Quake setup and input checks passed; no real-dataset searches were run.'
fi
