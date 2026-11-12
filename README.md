<img src="assets/darth-logo.png" alt="DARTH logo" width="500">

# DARTH: Declarative Recall Through Early Termination

DARTH uses a learned recall predictor to stop approximate nearest-neighbor
search early. This repository integrates DARTH and DARTH+ with FAISS HNSW and
IVF indexes. See the [SIGMOD 2026 paper](https://arxiv.org/abs/2505.19001v1)
for the method and evaluation.

The C++ pipeline builds or loads an index, generates training observations,
trains and validates LightGBM models, selects prediction intervals, and searches
queries in one run. It does not need Python, intermediate CSV files, or
pretrained models.

## 1. Install dependencies

The following is a CPU build on Ubuntu/Debian. You need a C++17 compiler,
CMake **3.24 or newer**, Git, OpenMP, BLAS/LAPACK, and the LightGBM C++ shared
library and headers. For example:

```bash
sudo apt update
sudo apt install build-essential cmake git libopenblas-dev liblapack-dev
cmake --version
```

If your distribution supplies an older CMake, install a newer version before
continuing. GCC's OpenMP runtime is included with the usual `build-essential`
packages. To build LightGBM 4.5.0 in your home directory:

```bash
git clone --recursive --branch v4.5.0 https://github.com/microsoft/LightGBM.git "$HOME/LightGBM"
cmake -S "$HOME/LightGBM" -B "$HOME/LightGBM/build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/lightgbm-install" -DBUILD_CLI=OFF -DINSTALL_HEADERS=ON
cmake --build "$HOME/LightGBM/build" -j 4
cmake --install "$HOME/LightGBM/build"
```

The installation should contain `include/LightGBM/c_api.h` and
`lib/lib_lightgbm.so`. See the [LightGBM installation guide](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html)
for other platforms. A Python LightGBM package alone does not provide the C++
headers required by this build.

## 2. Build DARTH

From the root of this repository:

```bash
cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF \
  -DLIGHTGBM_ROOT="$HOME/lightgbm-install"
cmake --build build-native --target darth_pipeline -j 4
```

Set `LIGHTGBM_ROOT` to your installation prefix if it is elsewhere. CMake can
also take `LIGHTGBM_INCLUDE_DIR` and `LIGHTGBM_LIB` explicitly. If the runtime
loader cannot find LightGBM, add its library directory to `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH="$HOME/lightgbm-install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## 3. Prepare data

The pipeline reads FAISS `.fvecs` vectors and optional `.ivecs` exact-neighbor
labels. Use separate training, validation, and search query sets. The dataset
loader expects a base directory containing a folder for each dataset; for
example, SIFT100M uses:

```text
DATASET_DIRECTORY/SIFT100M/
  base.100M.fvecs
  learn.1M.fvecs
  validation.10K.fvecs
  query.10K.fvecs
  learn.groundtruth.1M.k1000.ivecs       # optional
  validation.groundtruth.10K.k1000.ivecs # optional
```

Other expected names are in [VectorDataLoader.cpp](common/VectorDataLoader.cpp).
Missing training or validation ground truth is computed exactly, which can be
expensive for large datasets. Search ground truth is needed only to evaluate
recall. Optional Python data-preparation utilities are in [scripts/](scripts/);
they use the separate dependencies in
[requirements.txt](requirements.txt). Use Python 3.9–3.12 and install them
with `python -m pip install -r requirements.txt`. The organizer's `main()`
currently selects datasets in the source, so inspect its configuration before
running it.

Dataset sources: [SIFT and GIST](http://corpus-texmex.irisa.fr/),
[DEEP and Text2Image](https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search),
and [GloVe](https://nlp.stanford.edu/projects/glove/).

## 4. Run DARTH

For one supported dataset:

```bash
build-native/darth-runners/darth_pipeline \
  --dataset SIFT100M --dataset-dir-prefix /path/to/processed-data \
  --index hnsw --method darth-plus --k 50 \
  --index-filepath /path/to/indexes/sift-hnsw.index \
  --threads 8 --training-threads 8
```

The index is loaded if it exists; otherwise it is built and saved. Cached indexes
are checked against the base vectors and any supplied construction settings. Omit
`--index-filepath` to keep the index in memory. Use `--index ivf --nlist 1000
--nprobe 100` for IVF, or `--method darth` / `--method darth-plus-lb` for the
other methods. Add `--evaluate` to report recall using the dataset loader's
test ground truth, or compute exact test neighbors if that file is absent. Use
`--results /path/to/prefix` to save neighbors and distances, and
`--save-models /path/to/prefix` to export the models.
Create the parent directory for `--results` before running.

The pipeline prints training and validation MSE, MAE, and R², its selected
initial and minimum prediction intervals (IPI and MPI), and search timing. By
default it uses 10,000 training queries, 1,000 validation queries, and 1,000
test queries. Use `--training-query-num`, `--validation-query-num`, and
`--query-num` to change those counts; set a count to `0` to use every query in
that file. Optional calibration uses all calibration queries by default and
can be limited with `--calibration-query-num`.
When search ground truth is supplied, `rqut` is the fraction of queries whose
recall is strictly below the target; `queries_below_target` is the corresponding
number of queries.

To use another dataset, add its vector and ground-truth filenames to
[VectorDataLoader.cpp](common/VectorDataLoader.cpp), then supply its name and
base directory. Training, validation, and testing must use distinct files.
See `build-native/darth-runners/darth_pipeline --help` for all options,
including index settings, metrics, and optional conformal calibration.

## Run across datasets

The sweep script runs the pipeline once per dataset: it trains the models once
and searches each requested recall target. By default it selects
all ten names understood by the loader: `SIFT10M`, `SIFT50M`, `SIFT100M`,
`DEEP10M`, `DEEP50M`, `DEEP100M`, `GIST1M`, `GLOVE100`, `T2I10M`, and
`T2I100M`. Prepare each selected dataset first. It caches one index per dataset,
index type, metric, and construction settings; it reports failures and continues
with the remaining datasets.

```bash
DATASET_DIRECTORY=/path/to/processed-data INDEX_DIRECTORY=/path/to/indexes \
  K=50 THREADS=8 bash experiments/run_darth.sh
```

Edit the settings and dataset list near the top of the script, or override them
with environment variables. For example, `DATASETS="SIFT100M GIST1M"` selects
two datasets, `INDEX=ivf NLIST=1000 NPROBE=100` selects IVF, and
`DARTH_DRY_RUN=1` prints commands without running them. Additional pipeline
options, such as `--training-query-num 20000` or `--evaluate`, can be passed
after the script name.

The default recall targets are `0.80, 0.85, 0.90, 0.95, 0.99` for both the
DARTH, LAET and REM C++ runners and the experiment scripts. To override them,
use `--target-recall R` for one target or `--target-recalls R,R,...` for several
in those runners; use `TARGET_RECALL` or space-separated `TARGET_RECALLS` for
a script.
The models are shared by all targets, while each target has its own intervals
and search results. The script uses common settings for all selected datasets, not the paper's
per-dataset index configurations. Set `SAVE_RESULTS=1` and optionally `RESULTS_DIRECTORY` to save
neighbors, or set `SAVE_MODELS=1` and optionally `MODELS_DIRECTORY` to export
models. Result files include the target; exported model files are shared across
targets. Repeated runs with the same settings overwrite their previous outputs.

## Other approaches

The launchers below run each method across targets. DARTH+, LAET, and REM use
the same dataset and index settings as `run_darth.sh`. Pass `--evaluate` to
the native launchers to compute test recall.

| Method | Launcher |
| --- | --- |
| DARTH | [run_darth.sh](experiments/run_darth.sh) |
| DARTH+ | [run_darth_plus.sh](experiments/run_darth_plus.sh) |
| DARTH+ lower bound | [run_darth_plus_lb.sh](experiments/run_darth_plus_lb.sh) |
| LAET | [run_laet.sh](experiments/run_laet.sh) |
| REM | [run_rem.sh](experiments/run_rem.sh) |
| Plain HNSW/IVF baseline | [run_baseline.sh](experiments/run_baseline.sh) |
| Fixed-distance HNSW baseline | [run_naive.sh](experiments/run_naive.sh) |
| Ada-ef | [run_ada_ef.sh](experiments/run_ada_ef.sh) |
| Auncel | [run_auncel.sh](experiments/run_auncel.sh) |
| ConANN | [run_conann.sh](experiments/run_conann.sh) |
| Quake | [run_quake.sh](experiments/run_quake.sh) |

The plain and fixed-distance baselines use the same search budget for every
target; only their target-dependent RQUT statistic changes. Set
`DISTANCE_THRESHOLD` for the fixed-distance baseline. Ada-ef, Auncel, ConANN,
and Quake use their own comparison protocols and dependencies. Auncel and
ConANN need BLAS/LAPACK, OpenMP, and Python with NumPy; ConANN also needs GSL.
Ada-ef needs Eigen 3.4, Boost, OpenMP, Python and NumPy. Quake needs Python
3.10–3.12, CMake 3.24+, and BLAS/LAPACK. Run each launcher with `--help` for
its dataset, ground-truth and configuration requirements. Auncel, ConANN and
Quake can create missing shared IVF indexes; `DARTH_IVF_INDEX_DIR` selects
where to store them.

## Citation

```bibtex
@article{chatzakis2025darth,
  title={DARTH: Declarative Recall Through Early Termination for Approximate Nearest Neighbor Search},
  author={Manos Chatzakis and Yannis Papakonstantinou and Themis Palpanas},
  year={2025},
  eprint={2505.19001},
  archivePrefix={arXiv},
  primaryClass={cs.DB},
  url={https://arxiv.org/abs/2505.19001}
}
```

This repository includes FAISS code, which remains subject to its original
license. See [LICENSE](LICENSE). For questions, contact
[the authors](mailto:manos.chatzaki@gmail.com).
