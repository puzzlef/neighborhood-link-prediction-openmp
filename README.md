Comparing dynamic approaches of various OpenMP-based algorithms for [community detection].

There are three different dynamic approaches we are trying out:
- **Dynamic Frontier Louvain**: We use [Dynamic Frontier approach with Louvain algorithm](https://github.com/puzzlef/louvain-communities-openmp-dynamic).
- **Dynamic Frontier RAK**: We use [Dynamic Frontier approach with RAK algorithm](https://github.com/puzzlef/rak-communities-openmp-dynamic).
- **Dynamic Frontier Hybrid Louvain-RAK**: This is a hybrid approach, where we use Static Louvain for the initial graph, and then switch to Dynamic Frontier RAK.

See [extended report] for details. For HIPC2023 submission, see [submission-hipc23].

The input data used for below experiments is available from the [SuiteSparse Matrix Collection].
The experiments were done with guidance from [Prof. Kishore Kothapalli] and
[Prof. Dip Sankar Banerjee].

[community detection]: https://en.wikipedia.org/wiki/Community_search
[extended report]: https://gist.github.com/wolfram77/91b2d2ac50b9aba6b203e88b291c7671
[submission-hipc23]: https://github.com/puzzlef/communities-openmp-dynamic/tree/submission-hipc23
[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://faculty.iiit.ac.in/~kkishore/
[SuiteSparse Matrix Collection]: https://sparse.tamu.edu

<br>


### Comparision on large graphs

In this experiment ([input-large]), we compare the performance of *Dynamic*
*Frontier* based *Louvain*, *RAK*, and *Hybrid Louvain-RAK*. We generate random
batch updates consisting of an equal mix of *deletions (-)* and  *insertions*
*(+)* of edges of size `10^-7 |E|` to `0.1 |E|` in multiples of `10` (where `|E|`
is the number of edges in the original graph after making it undirected). For
each batch size, we generate *five* different batches for the purpose of
*averaging*. Each batch of edges (insertion / deletion) is generated randomly
such that the selection of each vertex (as endpoint) is *equally probable*. This
is repeated for each input graph.

We make the following observations. The *highest modularity* is obtained by
*Dynamic Frontier* based *Hybrid Louvain-RAK* (and *Static Louvain*) on average,
which is immediately followed by *Dynamic Frontier* based *Louvain*. This is
followed by *Dynamic RAK* approaches on average, which is then followed by
*Static RAK*. In terms of time, *Dynamic Frontier* based *Hybrid Louvain-RAK* is
the fastest, followed by *Dynamic Frontier* based *RAK* and  other dynamic RAK
approaches, which is then followed by *Dynamic Frontier* based *Louvain* on
average.

> See
> [code](https://github.com/puzzlef/communities-openmp-dynamic/tree/input-large),
> [output](https://gist.github.com/wolfram77/a9f4b3a9c65ea26271ef062f19746bf6), or
> [sheets].

[![](https://i.imgur.com/HydmJjE.png)][sheets]
[![](https://i.imgur.com/DImj54W.png)][sheets]
[![](https://i.imgur.com/kp8201a.png)][sheets]
[![](https://i.imgur.com/h6nBkj3.png)][sheets]
[![](https://i.imgur.com/ylROvV3.png)][sheets]
[![](https://i.imgur.com/DErQ7Vn.png)][sheets]

[input-large]: https://github.com/puzzlef/communities-openmp-dynamic/tree/input-large
[sheets]: https://docs.google.com/spreadsheets/d/1R96ZRfAMvc8JsZcSFiUz2A4NBuJYIQoWIkAa0otHVKc/edit?usp=sharing

<br>


### Strong scaling behavior

In this experiment ([strong-scaling]), we investigated the **strong-scaling**
behavior of *Dynamic* *Frontier* based *Hybrid Louvain-RAK* and compared it with
*Dynamic Frontier* based *Louvain* and *LPA*. We fixed the batch size at `10^-3 |E|`
(`|E|` is the total number of edges in the original graph after making it
undirected) and varied the number of threads from `1` to `128`. The speedup of
each algorithm was measured as the ratio of the time taken by the algorithm
compared to the same algorithm running on one thread.

Our findings suggest that **Hybrid Louvain-LPA** exhibited a speedup rate of
`1.46x` than Louvain (`1.31x`) and LPA (`1.44x`). We observed a dip in speedup
when using 128 threads, likely due to hyper-threading effects (the system has 64
cores). *Hybrid Louvain-LPA* demonstrated the good speedup on *social networks*,
but did not scale well enough on *road networks*.

[strong-scaling]: https://github.com/puzzlef/communities-openmp-dynamic/tree/strong-scaling

<br>


### Multi-batch updates

In this experiment ([multi-batch]), we generate `5000` random **multi-batch**
**updates** consisting of *edge insertions* of size `10^-3 |E|` one after the
other on graphs `web-Stanford` and `web-BerkStan` and observe the performance
and modularity of communities obtained with *Dynamic Frontier* based *Louvain*,
*RAK*, and *Hybrid* *Louvain-RAK*. We do this to measure after how many batch
updates do we need to re-run the static algorithm.

Our results indicate that we need to rerun the static algorithm after `~1300`
batch updates with *Dynamic Frontier* based *Louvain*, and after `~600` batch
updates with *Dynamic Frontier* based *Hybrid Louvain-RAK*.

[multi-batch]: https://github.com/puzzlef/communities-openmp-dynamic/tree/multi-batch

<br>
<br>


## Build instructions

To run the [input-large] experiment, download this repository and run the
following. Note that input graphs must be placed in `~/Data` directory, and
output logs will be written to `~/Logs` directory.

```bash
# Perform comparision on large graphs
$ DOWNLOAD=0 ./mains.sh

# Perform comparision on large graphs with custom number of threads
$ DOWNLOAD=0 MAX_THREADS=4 ./mains.sh
```

To run the [strong-scaling] experiment, comment `./main.sh` in file `mains.sh`,
uncomment `./main.sh "--strong-scaling"`, and run the same as above.

<br>
<br>


## References

- [Near linear time algorithm to detect community structures in large-scale networks; Usha Nandini Raghavan et al. (2007)](https://arxiv.org/abs/0709.2938)
- [Delta-Screening: A Fast and Efficient Technique to Update Communities in Dynamic Graphs](https://ieeexplore.ieee.org/document/9384277)
- [Fast unfolding of communities in large networks; Vincent D. Blondel et al. (2008)](https://arxiv.org/abs/0803.0476)
- [Community Detection on the GPU; Md. Naim et al. (2017)](https://arxiv.org/abs/1305.2006)
- [Scalable Static and Dynamic Community Detection Using Grappolo; Mahantesh Halappanavar et al. (2017)](https://ieeexplore.ieee.org/document/8091047)
- [From Louvain to Leiden: guaranteeing well-connected communities; V.A. Traag et al. (2019)](https://www.nature.com/articles/s41598-019-41695-z)
- [CS224W: Machine Learning with Graphs | Louvain Algorithm; Jure Leskovec (2021)](https://www.youtube.com/watch?v=0zuiLBOIcsw)
- [The University of Florida Sparse Matrix Collection; Timothy A. Davis et al. (2011)](https://doi.org/10.1145/2049662.2049663)

<br>
<br>


[![](https://i.imgur.com/xmxAIC5.jpg)](https://www.youtube.com/watch?v=R8gdUldwD0I)<br>
[![ORG](https://img.shields.io/badge/org-puzzlef-green?logo=Org)](https://puzzlef.github.io)
