#!/usr/bin/env bash
# Shared paths for the retained baseline experiments.
EXPERIMENT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$EXPERIMENT_DIR/.." && pwd)
: "${DATASET_DIRECTORY:=${DARTH_DATASET_DIR:-}}"
if [[ -z "$DATASET_DIRECTORY" ]]; then
  echo 'Set DATASET_DIRECTORY (or DARTH_DATASET_DIR) to the dataset base directory.' >&2
  return 1 2>/dev/null || exit 1
fi
DATASET_DIRECTORY=$(realpath -m -- "$DATASET_DIRECTORY")/
INDEX_DIRECTORY=${INDEX_DIRECTORY:-${DARTH_INDEX_DIR:-"$REPO_ROOT/indexes"}}
RESULTS_DIRECTORY=${RESULTS_DIRECTORY:-${DARTH_RESULTS_DIR:-"$REPO_ROOT/experiments/results"}}
BUILD_DIRECTORY=${BUILD_DIRECTORY:-${DARTH_BUILD_DIR:-"$REPO_ROOT/build"}}
PYTHON=${PYTHON:-python3}
for path_variable in INDEX_DIRECTORY RESULTS_DIRECTORY BUILD_DIRECTORY; do
  printf -v "$path_variable" '%s' "$(realpath -m -- "${!path_variable}")"
done
if [[ "$PYTHON" == */* ]]; then PYTHON=$(realpath -m -- "$PYTHON"); fi
BUILD_JOBS=${BUILD_JOBS:-1}
DARTH_DRY_RUN=${DARTH_DRY_RUN:-0}
export DATASET_DIRECTORY INDEX_DIRECTORY RESULTS_DIRECTORY BUILD_DIRECTORY PYTHON
run_experiment() {
  if [[ "$DARTH_DRY_RUN" == 1 ]]; then
    printf '[dry-run]'; printf ' %q' "$@"; printf '\n'
  else
    "$@"
  fi
}
