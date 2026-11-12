#!/usr/bin/env bash
# Run HNSW with a fixed distance threshold across datasets and targets.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export INDEX=hnsw BASELINE_MODE=naive
exec bash "$SCRIPT_DIR/run_baseline.sh" "$@"
