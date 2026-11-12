#!/usr/bin/env bash
set -euo pipefail

# Calibrate and evaluate ConANN using the exact DARTH+ IVF query protocol:
#   - first 8,333 learn queries: calibration
#   - next 1,667 learn queries: tuning
#   - first 1,000 query vectors: testing
#   - k in {10, 25, 50, 75, 100}
#   - target recall in {0.80, 0.85, 0.90, 0.95, 0.99}
#   - the shared DARTH IVF index, nlist, and nprobe
#
# The script builds the evaluator, index, and missing query splits as needed.
# The evaluator accepts exact counts so query splits have exact boundaries.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="$REPO_ROOT/competitors/conann"
source "$REPO_ROOT/experiments/darth_threading.sh"
PYTHON=${PYTHON:-python3}
# Numeric CLI arguments and output names must remain locale independent.
export LC_NUMERIC=C
MODE=run
for argument in "$@"; do
  case "$argument" in
    --dry-run) MODE=dry ;;
    --check) MODE=check ;;
    --help|-h)
      echo 'Usage: bash experiments/run_conann.sh [--dry-run | --check]'
      echo 'Overrides: CONANN_K_VALUES, DATASETS, TARGET_RECALLS, CONANN_RESULTS_DIR, DARTH_IVF_INDEX_DIR.'
      echo 'Uses prepared exact splits; timed evaluation always uses one thread.'
      exit 0 ;;
    *) echo "Unknown argument: $argument" >&2; exit 2 ;;
  esac
done
CONANN_SOURCE_DIR="$ROOT_DIR/conann"
CONANN_BUILD_DIR="${CONANN_BUILD_DIR:-$CONANN_SOURCE_DIR/build}"
RESULTS_DIR="${CONANN_RESULTS_DIR:-$REPO_ROOT/experiments/ivf-results/conann}"
INDEX_DIR="${DARTH_IVF_INDEX_DIR:-${INDEX_DIRECTORY:-${DARTH_INDEX_DIR:-${REPO_ROOT}/indexes/ivf}}}"
DATASET_DIRECTORY=${DATASET_DIRECTORY:-${DARTH_DATASET_DIR:-${REPO_ROOT}/data}}
ERROR_EXECUTABLE="$CONANN_BUILD_DIR/eval/error"

CALIBRATION_QUERIES=8333
TUNING_QUERIES=1667
TESTING_QUERIES=1000
TOTAL_QUERIES=$((CALIBRATION_QUERIES + TUNING_QUERIES + TESTING_QUERIES))
read -r -a K_VALUES <<< "${CONANN_K_VALUES:-${K_VALUES:-10 25 50 75 100}}"
read -r -a RECALL_TARGETS <<< "${TARGET_RECALLS:-${TARGET_RECALL:-0.80 0.85 0.90 0.95 0.99}}"
SELECTED_DATASETS=${DATASETS:-"GIST1M GLOVE100 DEEP100M SIFT100M T2I100M"}
for dataset in ${SELECTED_DATASETS}; do
  case "$dataset" in GIST1M|GLOVE100|DEEP100M|SIFT100M|T2I100M) ;; *) echo "Unknown dataset: $dataset" >&2; exit 2 ;; esac
done
for target in "${RECALL_TARGETS[@]}"; do
  if ! [[ "$target" =~ ^(0\.[0-9]+|1(\.0+)?)$ ]] || ! LC_ALL=C awk -v x="$target" 'BEGIN {exit !(x>0 && x<=1)}'; then
    echo "Invalid recall target: $target" >&2; exit 2
  fi
done

# Dataset names, nlist, and nprobe match the original DARTH+ IVF experiments.
DATASET_PARAMETERS=(
  "GIST1M    1000  200"
  "GLOVE100  1000  100"
  "DEEP100M 10000  150"
  "SIFT100M 10000  150"
  "T2I100M  20000  500"
)

vecs_metadata() {
  local path="$1"
  local dimension bytes row_bytes
  dimension=$(od -An -td4 -N4 "$path" | tr -d '[:space:]')
  bytes=$(stat -c '%s' "$path")
  row_bytes=$((4 * (dimension + 1)))
  if ((dimension <= 0 || bytes % row_bytes != 0)); then
    echo "Invalid vecs file: $path" >&2
    exit 1
  fi
  printf '%s %s\n' "$((bytes / row_bytes))" "$dimension"
}

if [[ "$MODE" == run ]]; then
  if [[ ! -f "$CONANN_BUILD_DIR/CMakeCache.txt" ]]; then
    cmake -S "$CONANN_SOURCE_DIR" -B "$CONANN_BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
      -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_C_API=OFF
  fi
  cmake --build "$CONANN_BUILD_DIR" --target error -j "$BUILD_JOBS"
fi

if [[ "$MODE" == run ]]; then mkdir -p "$RESULTS_DIR"; fi

# A stray Ctrl-S (or C-a s inside screen) sends XOFF, which pauses terminal
# output. The next write to the terminal then blocks forever, silently stalling
# the sweep between evaluations. Disable XON/XOFF flow control while the sweep
# runs and restore the terminal settings on exit.
if [[ "$MODE" == run && -t 0 ]]; then
  saved_stty="$(stty -g)"
  trap 'stty "$saved_stty"' EXIT
  stty -ixon
fi

if ((${#K_VALUES[@]} == 0)); then
  echo "CONANN_K_VALUES must contain at least one k value." >&2
  exit 1
fi
for k in "${K_VALUES[@]}"; do
  if [[ ! "$k" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid k value: $k" >&2
    exit 1
  fi
done

for entry in "${DATASET_PARAMETERS[@]}"; do
  read -r dataset nlist nprobe <<< "$entry"
  if [[ " ${SELECTED_DATASETS} " != *" ${dataset} "* ]]; then continue; fi
  data_dir="$ROOT_DIR/data/$dataset"
  index_file="$INDEX_DIR/$dataset/${dataset}.nlist${nlist}.index"

  if [[ "$MODE" == run ]]; then
    if [[ ! -e "$index_file" ]]; then
      DATASET_DIRECTORY="$DATASET_DIRECTORY" bash "$REPO_ROOT/experiments/ensure_ivf_index.sh" "$dataset" "$nlist" "$index_file"
    fi
    if [[ ! -e "$data_dir/split.ini" ]]; then
      if [[ -d "$data_dir" ]] && [[ -n "$(ls -A "$data_dir")" ]]; then
        echo "Incomplete existing preparation in $data_dir; inspect it before preparing again." >&2; exit 1
      fi
      "$PYTHON" "$ROOT_DIR/prepare_darth_data.py" --datasets "$dataset" --k-values "${K_VALUES[@]}" --darth-data-dir "$DATASET_DIRECTORY"
    fi
  fi

  if [[ "$MODE" != dry ]]; then
  for required in db.fvecs queries.fvecs split.ini; do
    if [[ ! -e "$data_dir/$required" ]]; then
      echo "Missing $data_dir/$required; run prepare_darth_data.py first." >&2
      exit 1
    fi
  done
  if [[ ! -f "$index_file" ]]; then
    echo "Missing prebuilt DARTH IVF index: $index_file" >&2
    exit 1
  fi
  read -r query_rows _ < <(vecs_metadata "$data_dir/queries.fvecs")
  if ((query_rows != TOTAL_QUERIES)); then
    echo "Expected $TOTAL_QUERIES rows in $data_dir/queries.fvecs, found $query_rows; run prepare_darth_data.py." >&2
    exit 1
  fi
  PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" - "$data_dir/split.ini" "$DATASET_DIRECTORY" "$dataset" <<'PY'
import sys
from pathlib import Path
from split_metadata import read_split
meta = read_split(sys.argv[1])
expected = {"calibration_queries": 8333, "tuning_queries": 1667,
            "testing_queries": 1000, "total_queries": 11000,
            "query_order": ["calibration", "tuning", "testing"], "dataset": sys.argv[3]}
if any(meta.get(key) != value for key, value in expected.items()):
    raise ValueError("ConANN split metadata mismatch; run prepare_darth_data.py")
source = (Path(sys.argv[2]) / sys.argv[3]).resolve()
for key in ["training_query_source", "testing_query_source"]:
    if Path(meta[key]).resolve().parent != source:
        raise ValueError("ConANN inputs use a different dataset directory; run prepare_darth_data.py --darth-data-dir for this source")
PY
  fi

  for k in "${K_VALUES[@]}"; do
    if [[ "$MODE" != dry ]]; then
    for required in "indices-${k}.fvecs" "distances-${k}.fvecs"; do
      if [[ ! -e "$data_dir/$required" ]]; then
        echo "Missing $data_dir/$required; run prepare_darth_data.py first." >&2
        exit 1
      fi
    done
    read -r id_rows id_k < <(vecs_metadata "$data_dir/indices-${k}.fvecs")
    read -r distance_rows distance_k < <(vecs_metadata "$data_dir/distances-${k}.fvecs")
    if ((id_rows != TOTAL_QUERIES || distance_rows != TOTAL_QUERIES || id_k != k || distance_k != k)); then
      echo "Invalid k=$k ground truth in $data_dir; run prepare_darth_data.py." >&2
      exit 1
    fi
    fi

    if [[ "$MODE" == check ]]; then
      echo "CHECK OK: $dataset k=$k, exact split counts and ground-truth shape"
      continue
    fi

    est_gb=$(awk -v nq="$TOTAL_QUERIES" -v nprobe="$nprobe" -v k="$k" \
      'BEGIN {printf "%.1f", (nq*nprobe*k*8)/1e9}')
    echo "=== $dataset: nlist=$nlist, nprobe=$nprobe, k=$k, prediction payload ~${est_gb} GB ==="

    dataset_results="$RESULTS_DIR/$dataset/k$k"
    if [[ "$MODE" == run ]]; then mkdir -p "$dataset_results"; fi

    for target_recall in "${RECALL_TARGETS[@]}"; do
      target_recall=$(LC_ALL=C awk -v recall="$target_recall" 'BEGIN {printf "%.2f", recall}')
      alpha=$(LC_ALL=C awk -v recall="$target_recall" 'BEGIN {printf "%.2f", 1-recall}')
      result_stem="nlist${nlist}_nprobe${nprobe}_qs${TESTING_QUERIES}_tr${target_recall}"
      result_csv="$dataset_results/${result_stem}.csv"
      preparation_report="$dataset_results/${result_stem}_preparation.txt"
      run_log="$dataset_results/${result_stem}.log"

      echo "--- $dataset k=$k target_recall=$target_recall alpha=$alpha ---"
      if [[ "$MODE" == dry ]]; then
        printf '%q ' "$ERROR_EXECUTABLE" "$dataset" --counts "$CALIBRATION_QUERIES" "$TUNING_QUERIES" \
          "$alpha" "$nlist" "$nprobe" "$k" "$index_file" "$result_csv"
        printf '\n'
        continue
      fi
      (
        cd "$CONANN_SOURCE_DIR"
        "$ERROR_EXECUTABLE" \
          "$dataset" --counts "$CALIBRATION_QUERIES" "$TUNING_QUERIES" \
          "$alpha" "$nlist" "$nprobe" "$k" "$index_file" "$result_csv"
      ) 2>&1 | tee "$run_log"

      expected_header="qid,dists,elaps_ms,r,clusters_searched"
      actual_header=$(head -n 1 "$result_csv")
      result_rows=$(($(wc -l < "$result_csv") - 1))
      if [[ "$actual_header" != "$expected_header" || "$result_rows" -ne "$TESTING_QUERIES" ]]; then
        echo "Invalid result CSV: $result_csv (header='$actual_header', rows=$result_rows)" >&2
        exit 1
      fi
      if [[ ! -s "$preparation_report" ]] ||
         ! grep -q '^calibration_time_s=' "$preparation_report" ||
         ! grep -q '^ready_to_query_time_s=' "$preparation_report"; then
        echo "Invalid preparation report: $preparation_report" >&2
        exit 1
      fi

    done

    if [[ "$MODE" == run ]]; then
    shopt -s nullglob
    for diagnostic in "$ROOT_DIR"/ConANN-*"${dataset}-${nlist}-${k}-"*"-np${nprobe}-"*; do
      mv "$diagnostic" "$dataset_results/"
    done
    shopt -u nullglob
    fi
  done
done

echo "ConANN $MODE completed. Per-query CSV destination: $RESULTS_DIR/<DATASET>/k<K>/."
