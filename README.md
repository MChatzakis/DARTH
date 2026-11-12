<p align="center">
  <img src="assets/darth-logo.png" alt="DARTH logo" width="500">
</p>

# DARTH+: Approximate Nearest Neighbor Search with Declarative Recall and Quality Guarantees
## Manos Chatzakis, Yannis Papakonstantinou, Themis Palpanas

DARTH and DARTH+ enable declarative recall for approximate nearest-neighbor
search: specify a target recall, and a learned predictor estimates when to stop
each query. Their adaptive early termination adjusts to query difficulty and
reduces unnecessary search work.

DARTH+ extends DARTH with more efficient training data collection and
query-dimension-aware prediction features. Its lower-bound variant can also
provide optional probabilistic quality guarantees on individual queries. This
repository integrates the methods with FAISS HNSW and IVF indexes, supporting
L2 and inner-product search.

The C++ pipeline builds or loads an index, trains and validates LightGBM
models, selects prediction intervals, and searches queries in one run.

## Installation

For a CPU build on Ubuntu/Debian, install a C++17 compiler, CMake 3.24 or newer,
Git, OpenMP, BLAS/LAPACK, and the LightGBM C++ library and headers:

```bash
sudo apt update
sudo apt install build-essential cmake git libopenblas-dev liblapack-dev
```

For example, build and install LightGBM 4.5.0:

```bash
git clone --recursive --branch v4.5.0 https://github.com/microsoft/LightGBM.git "$HOME/LightGBM"
cmake -S "$HOME/LightGBM" -B "$HOME/LightGBM/build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/lightgbm-install" -DBUILD_CLI=OFF -DINSTALL_HEADERS=ON
cmake --build "$HOME/LightGBM/build" -j 4
cmake --install "$HOME/LightGBM/build"
```

Build the pipeline from the repository root:

```bash
cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF \
  -DLIGHTGBM_ROOT="$HOME/lightgbm-install"
cmake --build build-native --target darth_pipeline -j 4
```

Adjust `LIGHTGBM_ROOT` for your installation. If the runtime loader cannot
find LightGBM, add its library directory to `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH="$HOME/lightgbm-install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## Prepare data

The pipeline reads FAISS `.fvecs` vectors and optional `.ivecs` exact-neighbor
labels. Training, validation, and search queries must use separate files. For
example, SIFT100M expects:

```text
DATASET_DIRECTORY/SIFT100M/
  base.100M.fvecs
  learn.1M.fvecs
  validation.10K.fvecs
  query.10K.fvecs
  learn.groundtruth.1M.k1000.ivecs       # optional
  validation.groundtruth.10K.k1000.ivecs # optional
```

See [VectorDataLoader.cpp](common/VectorDataLoader.cpp) for more information.
Dataset sources: [SIFT and GIST](http://corpus-texmex.irisa.fr/),
[DEEP and Text2Image](https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search),
and [GloVe](https://nlp.stanford.edu/projects/glove/).

## Run

For example, run DARTH+ on SIFT100M with HNSW:

```bash
build-native/darth-runners/darth_pipeline \
  --dataset SIFT100M --dataset-dir-prefix /path/to/processed-data \
  --index hnsw --method darth-plus --k 50 \
  --index-filepath /path/to/indexes/sift-hnsw.index \
  --threads 8 --training-threads 8
```

The pipeline loads the index if it exists, or builds and saves it otherwise.
Omit `--index-filepath` to keep it in memory. Use `--index ivf --nlist 1000
--nprobe 100` for IVF, or `--method darth` / `--method darth-plus-lb` for
DARTH or the lower-bound variant. Add `--evaluate` to report recall; missing
test ground truth is computed exactly. Use `--target-recall R` or
`--target-recalls R,R,...` to set recall targets, and `--results` or
`--save-models` with a path prefix to save outputs. Create the results parent
directory first.

## Experiments
The experiment scripts run methods across datasets and recall targets. Prepare
the selected datasets first. For example:

```bash
DATASET_DIRECTORY=/path/to/processed-data INDEX_DIRECTORY=/path/to/indexes \
  K=50 THREADS=8 bash experiments/run_darth.sh
```

Set `DATASETS="SIFT100M GIST1M"` to select datasets, `INDEX=ivf NLIST=1000
NPROBE=100` for IVF, or `DARTH_DRY_RUN=1` to print commands. The scripts use
common settings across datasets, rather than the paper's per-dataset index
configurations. Pass additional pipeline options after the script name. See
each launcher and its `--help` for settings and dependencies.

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

## References
If you use DARTH or DARTH+, please cite:

```bibtex
@article{chatzakis2025darth,
  title={Darth: Declarative recall through early termination for approximate nearest neighbor search},
  author={Chatzakis, Manos and Papakonstantinou, Yannis and Palpanas, Themis},
  journal={Proceedings of the ACM on Management of Data},
  volume={3},
  number={4},
  pages={1--26},
  year={2025},
  publisher={ACM, New York, NY, USA}
}

@article{chatzakis2026darth+,
  title={DARTH+: Approximate Nearest Neighbor Search with Declarative Recall and Quality Guarantees},
  author={Chatzakis, Manos and Papakonstantinou, Yannis and Palpanas, Themis},
  journal={HAL Preprint},
  year={2026}
}

@article{chatzakis2026declarativephd,
  title={Declarative Recall for Approximate Vector Search},
  author={Chatzakis, Manos},
  journal={Proceedings of the VLDB Endowment. ISSN},
  volume={2150},
  pages={8097}
}

@inproceedings{chatzakis2026darth+workshop,
  title={DARTH+: Vector Search with Declarative Recall and Guarantees},
  author={Chatzakis, Manos and Papakonstantinou, Yannis and Palpanas, Themis},
  booktitle={The 2nd Workshop on Vector Databases},
  year={2026},
}
```

## About
This repository includes FAISS code, which remains subject to its original
license. See [LICENSE](LICENSE). For questions, contact
[the authors](mailto:manos.chatzaki@gmail.com).
