# Shared worker policy for the revision launchers and DARTH+ study drivers.
# Source this file; timed evaluation always uses one thread.

if [[ -n "${THREADS:-}" && ! "${THREADS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "THREADS must be a positive integer (training/build workers only)." >&2
    exit 2
fi
export GENERATION_JOBS=${GENERATION_JOBS:-24}
export TRAIN_JOBS=${TRAIN_JOBS:-${THREADS:-8}}
export BUILD_JOBS=${BUILD_JOBS:-${THREADS:-8}}
for darth_worker_setting in GENERATION_JOBS TRAIN_JOBS BUILD_JOBS; do
    if [[ ! "${!darth_worker_setting}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${darth_worker_setting} must be a positive integer." >&2
        exit 2
    fi
done
unset darth_worker_setting

darth_set_runtime_threads() {
    export OMP_NUM_THREADS=$1 OMP_THREAD_LIMIT=$1 OMP_MAX_ACTIVE_LEVELS=1
    export OPENBLAS_NUM_THREADS=$1 MKL_NUM_THREADS=$1 BLIS_NUM_THREADS=$1
    export VECLIB_MAXIMUM_THREADS=$1 NUMEXPR_NUM_THREADS=$1 NUMEXPR_MAX_THREADS=$1
}

# OMP_THREAD_LIMIT also caps predictors loaded from multithreaded training runs.
# Override inherited library settings before any native search or Python import.
darth_set_runtime_threads 1

darth_run_generation() (
    # Parallelize independent training/validation queries only. In particular,
    # the timed sensitivity baseline must not use this environment.
    # Keep BLAS serial so each OpenMP worker cannot spawn another BLAS team.
    darth_set_runtime_threads 1
    export OMP_NUM_THREADS=${GENERATION_JOBS} OMP_THREAD_LIMIT=${GENERATION_JOBS}
    "$@"
)

darth_run_training() (
    # The subprocess gets its own OpenMP/BLAS limits, including THREAD_LIMIT.
    # Its settings cannot leak into subsequent generation or timed searches.
    darth_set_runtime_threads "${TRAIN_JOBS}"
    "$@"
)
