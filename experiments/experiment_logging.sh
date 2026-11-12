# Progress helpers shared by the revision launchers. Source this file.
# Keep Python messages visible immediately through the launchers' tee pipes.
export PYTHONUNBUFFERED=1

experiment_progress() {
    printf '[%s] %s\n' "$(date -Is)" "$*"
}

experiment_step() (
    set -Ee
    label=$1
    shift
    started=${SECONDS}
    trap 'status=$?; experiment_progress "FAILED ${label}; exit=${status}; elapsed=$((SECONDS - started))s" >&2; exit "${status}"' ERR
    experiment_progress "START ${label}"
    "$@"
    experiment_progress "DONE ${label}; elapsed=$((SECONDS - started))s"
)
