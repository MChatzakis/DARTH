#!/usr/bin/env bash
# Run darth-plus across datasets and recall targets.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export METHOD=darth-plus
exec bash "$SCRIPT_DIR/run_darth.sh" "$@"
