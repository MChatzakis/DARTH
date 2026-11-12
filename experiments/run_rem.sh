#!/usr/bin/env bash
# Run REM end to end across datasets and recall targets.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export NATIVE_METHOD=rem
exec bash "$SCRIPT_DIR/run_native_competitor.sh" "$@"
