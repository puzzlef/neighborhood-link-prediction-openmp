#include <cstdint>
#include <cstdio>
#include <utility>
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



#pragma region EXPERIMENTAL SETUP
/**
 * Run a function on each batch update using link prediction, with a specified range of batch sizes.
 * @param x original graph
 * @param fn function to run on each batch update
 */
template <class G, class F>
inline void runBatches(const G& x, F fn) {
  using  K = typename G::key_type;
  using  V = typename G::edge_value_type;
  double i = BATCH_INSERTIONS_BEGIN;
  for (int epoch=0;; ++epoch) {
    for (int r=0; r<REPEAT_BATCH; ++r) {
      auto y  = duplicate(x);
      for (int sequence=0; sequence<BATCH_LENGTH; ++sequence) {
        vector<tuple<K, K>> deletions;  // No edge deletions
        // Predict links using hub promoted score.
        auto pred = predictLinksHubPromotedOmp(x, {1, V(), size_t(i * x.size()/2)});
        const auto& insertions = pred.edges;
        // Add predicted links to original graph.
        for (const auto& [u, v, w] : insertions) {
          y.addEdge(u, v, V(1));
          y.addEdge(v, u, V(1));
        }
        updateOmpU(y);
        // Run function on updated graph.
        fn(y, 0.0, deletions, i, insertions, pred, sequence, epoch);
      }
    }
    if (i>=BATCH_INSERTIONS_END) break;
    i BATCH_INSERTIONS_STEP;
    i = min(i, double(BATCH_INSERTIONS_END));
  }
}


/**
 * Run a function on each number of threads, with a specified range of thread counts.
 * @param fn function to run on each number of threads
 */
template <class F>
inline void runThreads(F fn) {
  for (int t=NUM_THREADS_BEGIN; t<=NUM_THREADS_END; t NUM_THREADS_STEP) {
    omp_set_num_threads(t);
    fn(t);
    omp_set_num_threads(MAX_THREADS);
  }
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
  int repeat  = REPEAT_METHOD;
  int retries = 5;
  vector<K> *init = nullptr;
  double M = edgeWeightOmp(x)/2;
  // Follow a specific result logging format, which can be easily parsed later.
  auto glog = [&](const auto& ans, const auto& pred, const char *technique, int numThreads, const auto& y, auto M, auto insertionsf) {
    printf(
      "{+%.3e batchf, %03d threads} -> "
      "{%09.1fms, %09.1fms preproc, %09.1fms predict, %04d iters, %01.9f modularity} %s\n",
      double(insertionsf), numThreads,
      ans.time, ans.preprocessingTime, pred.time, ans.iterations, getModularity(y, ans, M), technique
    );
  };
  // Get community memberships on original graph (static).
  auto p0 = PredictLinkResult<K, V>();
  auto a0 = louvainStaticOmp(x, init);
  glog(a0, p0, "louvainStaticOmpOriginal", MAX_THREADS, x, M, 0.0);
  // Predict links and obtain updated community memberships.
  runBatches(x, [&](const auto& y, auto deletionsf, const auto& deletions, auto insertionsf, const auto& insertions, const auto& pred, int sequence, int epoch) {
    double M = edgeWeightOmp(y)/2;
    runThreads([&](int numThreads) {
      // Update community memberships on updated graph.
      auto a1 = louvainStaticOmp(y, init, {repeat});
      glog(a1, pred, "predictLinksHubPromotedOmp_louvainStaticOmp", numThreads, y, M, insertionsf);
      auto b1 = rakDynamicFrontierOmp(y, deletions, insertions, &a0.membership, {repeat});
      glog(b1, pred, "predictLinksHubPromotedOmp_louvainRakDynamicFrontierOmp", numThreads, y, M, insertionsf);
    });
  });
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
