
<p align="center">
<img width="130" src="./assets/darth_logo.png"/>
</p>


<h1 align="center">DARTH</h1>
<h2 align="center">Declarative Recall Through Early Termination for Approximate Nearest Neighbor Search</h2>

Approximate Nearest Neighbor Search (ANNS) presents an inherent tradeoff between performance and recall (i.e., result quality). Each ANNS algorithm provides its own algorithm-dependent parameters to allow applications to influence the recall/performance tradeoff of their searches. This situation is doubly problematic. 
First, the application developers have to experiment with these algorithm-dependent parameters to fine-tune the parameters that produce the desired recall for each use case. 
This process usually takes a lot of effort. Even worse, the chosen parameters may produce good recall for some queries, but bad recall for hard queries. 
To solve these problems, we present DARTH, a method that uses target declarative recall. DARTH uses a novel method for providing target declarative recall on top of an ANNS index by employing an adaptive early termination strategy integrated into the search algorithm. 
Through a wide range of experiments, we demonstrate that DARTH effectively meets user-defined recall targets while achieving significant speedups, up to 14.6x (average: 6.8x; median: 5.7x) faster than the search without early termination for HNSW and up to 41.8x (average: 13.6x; median: 8.1x) for IVF. 
This paper appeared in SIGMOD2026.

