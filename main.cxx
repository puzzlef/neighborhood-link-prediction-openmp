#include <cstdint>
#include <cstdio>
#include <utility>
#include <random>
#include <tuple>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include "inc/main.hxx"

using namespace std;




#pragma region CONFIGURATION
#ifndef TYPE
/** Type of edge weights. */
#define TYPE float
#endif
#ifndef MAX_THREADS
/** Maximum number of threads to use. */
#define MAX_THREADS 64
#endif
#ifndef REPEAT_BATCH
/** Number of times to repeat each batch. */
#define REPEAT_BATCH 5
#endif
#ifndef REPEAT_METHOD
/** Number of times to repeat each method. */
#define REPEAT_METHOD 1
#endif
#pragma endregion




#pragma region METHODS
#pragma region HELPERS
/**
 * Obtain the modularity of community structure on a graph.
 * @param x original graph
 * @param a rak result
 * @param M sum of edge weights
 * @returns modularity
 */
template <class G, class R>
inline double getModularity(const G& x, const R& a, double M) {
  auto fc = [&](auto u) { return a.membership[u]; };
  return modularityBy(x, fc, M, 1.0);
}
#pragma endregion




#pragma region PERFORM EXPERIMENT
/**
 * Perform the experiment.
 * @param x original graph
 */
template <class G>
void runExperiment(const G& x) {
  using K = typename G::key_type;
  using V = typename G::edge_value_type;
  random_device dev;
  default_random_engine rnd(dev());
  int repeat  = REPEAT_METHOD;
  int retries = 5;
  vector<K> *init = nullptr;
  double M = edgeWeightOmp(x)/2;
  // Follow a specific result logging format, which can be easily parsed later.
  auto glog = [&](const auto& ans, const auto& pre, const char *technique, int numThreads, const auto& y, auto M, auto predictf) {
    printf(
      "{+%.3e predictf, +%.3e predict, %03d threads} -> "
      "{%09.1fms, %09.1fms preproc, %09.1fms predict, %04d iters, %01.9f modularity} %s\n",
      double(predictf), double(pre.edges.size()), numThreads,
      ans.time, ans.preprocessingTime, pre.time, ans.iterations, getModularity(y, ans, M), technique
    );
  };
  // Get community memberships on original graph (static).
  auto p0 = PredictLinkResult<K, V>();
  auto a0 = louvainStaticOmp(x, init);
  glog(a0, p0, "louvainStaticOmpOriginal", MAX_THREADS, x, M, 0.0);
  // Predict links and obtain updated community memberships.
  vector<float> percents {0.00001, 0.00005, 0.0001};
  for (float percent : percents) {
    // Predict links using hub promoted score.
    auto p1 = predictLinksHubPromotedOmp(x, {repeat, V()});
    auto fl = [](const tuple<K, K, V>& x, const tuple<K, K, V>& y) { return get<2>(x) > get<2>(y); };
    vector<tuple<K, K>>      deletions;
    vector<tuple<K, K, V>>&  insertions = p1.edges;
    sort(insertions.begin(), insertions.end(), fl);
    insertions.resize(size_t(percent * x.size()));
    // Add predicted links to original graph.
    auto y = duplicate(x);
    for (const auto& [u, v, w] : insertions)
      y.addEdge(u, v, V(1));
    updateOmpU(y);
    double M = edgeWeightOmp(y)/2;
    // Update community memberships on updated graph.
    auto a1 = rakDynamicFrontierOmp(y, deletions, insertions, &a0.membership, {repeat});
    glog(a1, p1, "predictLinksHubPromotedOmp_louvainRakDynamicFrontierOmp", MAX_THREADS, y, M, percent);
  }
}


/**
 * Main function.
 * @param argc argument count
 * @param argv argument values
 * @returns zero on success, non-zero on failure
 */
int main(int argc, char **argv) {
  using K = uint32_t;
  using V = TYPE;
  install_sigsegv();
  char *file     = argv[1];
  bool symmetric = argc>2? stoi(argv[2]) : false;
  bool weighted  = argc>3? stoi(argv[3]) : false;
  omp_set_num_threads(MAX_THREADS);
  LOG("OMP_NUM_THREADS=%d\n", MAX_THREADS);
  LOG("Loading graph %s ...\n", file);
  DiGraph<K, None, V> x;
  readMtxOmpW(x, file, weighted); LOG(""); println(x);
  if (!symmetric) { x = symmetricizeOmp(x); LOG(""); print(x); printf(" (symmetricize)\n"); }
  runExperiment(x);
  printf("\n");
  return 0;
}
#pragma endregion
#pragma endregion
