
<!-- <h2 align="center">⚠️ Repository under construction. Source code is being cleaned. ⚠️ </h2> -->

<p align="center">
<img width="600" src="./assets/darth-logo.png"/>
</p>

<h2 align="center">Declarative Recall Through Early Termination for Approximate Nearest Neighbor Search</h2>

Approximate Nearest Neighbor Search (ANNS) presents an inherent tradeoff between performance and recall (i.e., result quality). Each ANNS algorithm provides its own algorithm-dependent parameters to allow applications to influence the recall/performance tradeoff of their searches. This situation is doubly problematic. 
First, the application developers have to experiment with these algorithm-dependent parameters to fine-tune the parameters that produce the desired recall for each use case. 
This process usually takes a lot of effort. Even worse, the chosen parameters may produce good recall for some queries, but bad recall for hard queries. 
To solve these problems, we present DARTH, a method that uses target declarative recall. DARTH uses a novel method for providing target declarative recall on top of an ANNS index by employing an adaptive early termination strategy integrated into the search algorithm. 
Through a wide range of experiments, we demonstrate that DARTH effectively meets user-defined recall targets while achieving significant speedups, up to 14.6x (average: 6.8x; median: 5.7x) faster than the search without early termination for HNSW and up to 41.8x (average: 13.6x; median: 8.1x) for IVF. 

<b>This paper appeared in [SIGMOD2026](https://2026.sigmod.org/).</b> A preprint is available on [arXiv](https://arxiv.org/abs/2505.19001v1#).


## Installation
The installation prerequisites are can be found in the FAISS install instructions, which can be found [here](./faiss_docs/INSTALL.md).
To run DARTH, it is required that LightGBM is installed and be put in /HOME/lightgbm-install.
To compile FAISS with DARTH, use:
```bash
cmake -B build -S . # For the paper, we used: -DFAISS_ENABLE_GPU=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
make -C build -j faiss # Compiles FAISS with DARTH
make -C build -j hnsw_test # For the driver code of HNSW-based experiments
make -C build -j ivf_test # For the driver code of IVF-based experiments
```

<!-- cmake -DFAISS_ENABLE_GPU=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -B build -S . -->

To use the python scripts and notebooks, install the required Python packages:
```bash
pip install -r requirements.txt
```

To reproduce experiments, see scripts in the `experiments` directory. 

This version of the code is refactored to be easier to use. For any questions, bugs or suggestions, please [contact the authors](mailto:manos.chatzaki@gmail.com).

## Datasets
Our evaluation utilized the following datasets: 
* [SIFT and GIST](http://corpus-texmex.irisa.fr/)
* [DEEP and Text2Image](https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search)
* [Glove](https://nlp.stanford.edu/projects/glove/)

The datasets need to go through a minor preprocessing step to be used with DARTH. For details, refer to the scripts of [utils](./notebooks_scripts/utils).

## Contributors
* [Manos (Emmanouil) Chatzakis](https://mchatzakis.github.io/) (LIPADE, Universite Paris Cite)
* [Yannis Papakonstantinou](https://www.linkedin.com/in/yannispapakonstantinou/) (Google Cloud)
* [Themis Palpanas](https://helios2.mi.parisdescartes.fr/~themisp/) (LIPADE, Universite Paris Cite)

## Reference
To cite our work, please use:
```
@article{chatzakis2025darth,
    title={DARTH: Declarative Recall Through Early Termination for Approximate Nearest Neighbor Search}, 
    author={Manos Chatzakis and Yannis Papakonstantinou and Themis Palpanas},
    year={2025},
    eprint={2505.19001},
    archivePrefix={arXiv},
    primaryClass={cs.DB},
    url={https://arxiv.org/abs/2505.19001}, 
}
```

## About
This repository contains the implementation and integration of DARTH in the FAISS library, developed by Facebook Research. 
All original FAISS code and components remain under their respective licenses and rules as selected by the developers. 
Please refer to the FAISS license for details regarding using the original library. 
We do not claim any ownership or rights over the original FAISS library: all rights and acknowledgments are retained by the original authors.

We thank [Eva Chamilaki](https://evachamilaki.github.io/index.html) for the DARTH logo.
