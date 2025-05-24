
<h2 align="center">⚠️ Repository under construction. Source code is coming soon ⚠️ </h2>

<p align="center">
<img width="600" src="./assets/darth-logo.png"/>
</p>


<!--<h1 align="center">DARTH</h1>-->
<h2 align="center">Declarative Recall Through Early Termination for Approximate Nearest Neighbor Search</h2>

Approximate Nearest Neighbor Search (ANNS) presents an inherent tradeoff between performance and recall (i.e., result quality). Each ANNS algorithm provides its own algorithm-dependent parameters to allow applications to influence the recall/performance tradeoff of their searches. This situation is doubly problematic. 
First, the application developers have to experiment with these algorithm-dependent parameters to fine-tune the parameters that produce the desired recall for each use case. 
This process usually takes a lot of effort. Even worse, the chosen parameters may produce good recall for some queries, but bad recall for hard queries. 
To solve these problems, we present DARTH, a method that uses target declarative recall. DARTH uses a novel method for providing target declarative recall on top of an ANNS index by employing an adaptive early termination strategy integrated into the search algorithm. 
Through a wide range of experiments, we demonstrate that DARTH effectively meets user-defined recall targets while achieving significant speedups, up to 14.6x (average: 6.8x; median: 5.7x) faster than the search without early termination for HNSW and up to 41.8x (average: 13.6x; median: 8.1x) for IVF. 

<b>This paper appeared in [SIGMOD2026](https://2026.sigmod.org/).</b> A preprint is available on [arXiv](todo).

## Reference
To cite our work, please use:
```
@article{chatzakis2025darth,
  title={DARTH: Declarative Recall Through Early Termination for Approximate Nearest Neighbor Search},
  author={Chatzakis, Manos and Papakonstantinou, Yannis and Palpanas, Themis},
  journal={Proceedings of the ACM on Management of Data},
  volume={},
  number={},
  pages={},
  year={2026},
  publisher={ACM New York, NY, USA}
}
```

## Installation and Usage
To use DARTH for C++, FAISS and its corresponding dependencies (e.g., CMake) should be installed. Please refer to [FAISS installation manual](todo) for this.
On top of FAISS, DARTH requires an active installation of [LightGBM](todo) library visible in the PATH.

If you are interested in reproducing the graphs or training the models from scratch, please install the required Python packages:
```bash
pip install -r requirements.txt
```

Pretrained models from the paper are provided in the [models](./todo) directory.

To compile DARTH with FAISS, use:
```bash
cmake -B build -S . # You may include/exclude any modular part of FAISS in the compilation, e.g., -DFAISS_ENABLE_GPU=OFF -DBUILD_SHARED_LIBS=ON
make -C build -j faiss # Build FAISS with DARTH
make -C build -j darth-demos # To compile some demo scripts for DARTH
```

Usage examples and demos are located under the [darth-demos](todo). The scripts for the experiments we performed in the original paper can be found under the [experiments](todo) directory. 

## Datasets
Due to space constraints, we cannot include the datasets, training and testing queries. To use DARTH or reproduce the experiments, please refer to the original dataset repositories. 
* [SIFT and GIST](http://corpus-texmex.irisa.fr/)
* [DEEP and Text2Image](https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search)
* [Glove](https://nlp.stanford.edu/projects/glove/)

## Contributors
* [Manos (Emmanouil) Chatzakis](https://mchatzakis.github.io/) (Universite Paris Cite, LIPADE)
* [Yannis Papakonstantinou](https://www.linkedin.com/in/yannispapakonstantinou/) (Google Cloud & University of California San Diego)
* [Themis Palpanas](https://helios2.mi.parisdescartes.fr/~themisp/) (Universite Paris Cite, LIPADE)


## About
This repository contains the implementation and integration of DARTH in the FAISS library, developed by Facebook Research. 
All original FAISS code and components remain under their respective licenses and rules as selected by the developers. 
Please refer to the FAISS license for details regarding using the original library. 
We do not claim any ownership or rights over the original FAISS library: all rights and acknowledgments are retained by the original authors.

We thank [Eva Chamilaki](https://evachamilaki.github.io/index.html) for the DARTH logo.
