Design of OpenMP-based Parallel Neighborhood-based [Link prediction] approaches.

<br>

Link prediction aids in rectifying inaccuracies within various graph algorithms resulting from unaccounted-for or overlooked links within networks. However, numerous existing works adopt a baseline approach, incurring unnecessary computational overheads due to its elevated time complexity. Moreover, many investigations concentrate on smaller graphs, potentially leading to erroneous conclusions.

This code repository presents two parallel approaches, named **IHub** and **LHub**, which predict links through neighborhood-based similarity metrics on large graphs. *LHub*, a heuristic approach, additionally disregards significant hubs, grounded on the notion that high-degree nodes contribute minimal similarity among their neighbors.

On a server equipped with dual 16-core Intel Xeon Gold 6226R processors, *LHub* demonstrates an average speed improvement of `1019×` over *IHub*, particularly evident in web graphs and social networks, while sustaining comparable prediction accuracy. Notably, *LHub* achieves a link prediction rate of `38.1𝑀` edges/s and enhances performance by a factor of `1.6×` for every doubling of threads.

<br>

Below we plot the time taken by the *IHub* and *LHub* approaches for link prediction using the best similarity measure, on 13 different graphs, with `10^-2|E|` and `0.1|E|` unobserved edges.

[![](https://i.imgur.com/ejcLHtE.png)][sheets01]

Below we plot the speedup of *LHub* wrt *IHub*, on 13 different graphs, with `10^-2|E|` and `0.1|E|` unobserved edges. *LHub* surpasses *IHub* by, on average, `1622×` and `415×`, on `10^-2|E|` and `0.1|E|` unobserved edges respectively. Further, on the *sk-2005* graph with `0.1|E|` edges removed, *LHub* achieves a link prediction rate of `38.1M` edges/s.

[![](https://i.imgur.com/XMYfnE9.png)][sheets01]

Finally, we plot the F1 score for links predicted with the *IHub* and *LHub* approaches. The *IHub* approach achieves an average F1 score of `1.8×10^−2` and `1.1×10^−1` when predicting `10^−2|E|` and `0.1|E|` edges, respectively. In comparison, the *LHub* approach achieves F1 scores, averaging `3.2×10^−2` and `9.8×10^−2` , respectively. Thus, the *LHub* approach predicts links with similar F1 scores, while being significantly faster than the *IHub* approach.

[![](https://i.imgur.com/poUxGH7.png)][sheets01]

<br>

Refer to our technical report for more details: \
[A Fast Parallel Approach for Neighborhood-based Link Prediction by Disregarding Large Hubs][report].

<br>

> [!NOTE]
> You can just copy `main.sh` to your system and run it. \
> For the code, refer to `main.cxx`.

[Link prediction]: https://en.wikipedia.org/wiki/Link_prediction
[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://faculty.iiit.ac.in/~kkishore/
[SuiteSparse Matrix Collection]: https://sparse.tamu.edu
[sheets01]: https://docs.google.com/spreadsheets/d/1Fq24lMsDhQetWeWio3NuM76v2wYHNT7DehQxT4_hAu0/edit?usp=sharing
[report]: https://arxiv.org/abs/2401.11415

<br>
<br>


### Code structure

The code structure of this repository is as follows:

```bash
- inc/_algorithm.hxx: Algorithm utility functions
- inc/_bitset.hxx: Bitset manipulation functions
- inc/_cmath.hxx: Math functions
- inc/_ctypes.hxx: Data type utility functions
- inc/_cuda.hxx: CUDA utility functions
- inc/_debug.hxx: Debugging macros (LOG, ASSERT, ...)
- inc/_iostream.hxx: Input/output stream functions
- inc/_iterator.hxx: Iterator utility functions
- inc/_main.hxx: Main program header
- inc/_mpi.hxx: MPI (Message Passing Interface) utility functions
- inc/_openmp.hxx: OpenMP utility functions
- inc/_queue.hxx: Queue utility functions
- inc/_random.hxx: Random number generation functions
- inc/_string.hxx: String utility functions
- inc/_utility.hxx: Runtime measurement functions
- inc/_vector.hxx: Vector utility functions
- inc/batch.hxx: Batch update generation functions
- inc/bfs.hxx: Breadth-first search algorithms
- inc/csr.hxx: Compressed Sparse Row (CSR) data structure functions
- inc/dfs.hxx: Depth-first search algorithms
- inc/duplicate.hxx: Graph duplicating functions
- inc/Graph.hxx: Graph data structure functions
- inc/main.hxx: Main header
- inc/mtx.hxx: Graph file reading functions
- inc/predict.hxx: Link prediction functions
- inc/properties.hxx: Graph Property functions
- inc/selfLoop.hxx: Graph Self-looping functions
- inc/symmetricize.hxx: Graph Symmetricization functions
- inc/transpose.hxx: Graph transpose functions
- inc/update.hxx: Update functions
- main.cxx: Experimentation code
- process.js: Node.js script for processing output logs
```

Note that each branch in this repository contains code for a specific experiment. The `main` branch contains code for the final experiment. If the intention of a branch in unclear, or if you have comments on our technical report, feel free to open an issue.

<br>
<br>


## References

- [Link prediction in social networks: the state-of-the-art | Wang et al. (2014)](https://arxiv.org/abs/1411.5118)
- [A comprehensive survey of link prediction methods | Arrar et al. (2023)](https://link.springer.com/article/10.1007/s11227-023-05591-8)
- [Progresses and challenges in link prediction | Zhou (2021)](https://www.cell.com/iscience/pdf/S2589-0042(21)01185-8.pdf)
- [LPCD: Incremental Approach for Dynamic Networks | Gatadi and Rani (2023)](https://link.springer.com/chapter/10.1007/978-3-031-36402-0_18)
- [The University of Florida Sparse Matrix Collection; Timothy A. Davis et al. (2011)](https://doi.org/10.1145/2049662.2049663)

<br>
<br>


[![](https://i.imgur.com/xCTUbVR.jpg)](https://www.youtube.com/watch?v=yqO7wVBTuLw&pp)<br>
[![ORG](https://img.shields.io/badge/org-puzzlef-green?logo=Org)](https://puzzlef.github.io)
[![DOI](https://zenodo.org/badge/689546501.svg)](https://zenodo.org/doi/10.5281/zenodo.10607304)


[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://faculty.iiit.ac.in/~kkishore/
[SuiteSparse Matrix Collection]: https://sparse.tamu.edu
