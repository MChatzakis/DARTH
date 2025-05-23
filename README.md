
<p align="center">
<img width="300" src="./assets/darth-logo.png"/>
</p>


<!--<h1 align="center">DARTH</h1>-->
<h2 align="center">Declarative Recall Through Early Termination for Approximate Nearest Neighbor Search</h2>

Approximate Nearest Neighbor Search (ANNS) presents an inherent tradeoff between performance and recall (i.e., result quality). Each ANNS algorithm provides its own algorithm-dependent parameters to allow applications to influence the recall/performance tradeoff of their searches. This situation is doubly problematic. 
First, the application developers have to experiment with these algorithm-dependent parameters to fine-tune the parameters that produce the desired recall for each use case. 
This process usually takes a lot of effort. Even worse, the chosen parameters may produce good recall for some queries, but bad recall for hard queries. 
To solve these problems, we present DARTH, a method that uses target declarative recall. DARTH uses a novel method for providing target declarative recall on top of an ANNS index by employing an adaptive early termination strategy integrated into the search algorithm. 
Through a wide range of experiments, we demonstrate that DARTH effectively meets user-defined recall targets while achieving significant speedups, up to 14.6x (average: 6.8x; median: 5.7x) faster than the search without early termination for HNSW and up to 41.8x (average: 13.6x; median: 8.1x) for IVF. 
This paper appeared in SIGMOD2026.


## Contributors

* Manos (Emmanouil) Chatzakis (Universite Paris Cite)
* Yannis Papakonstantinou (Google)
* Themis Palpanas (Universite Paris Cite


## About
This repository contains the implementation and integration of DARTH in the FAISS library, developed by Facebook Research. 
All original FAISS code and components remain under their respective licenses and rules as selected by the developers. Please refer to the FAISS license for details regarding using the original library. 
We do not claim any ownership or rights over the original FAISS library: all rights and acknowledgments are retained by the original authors.
