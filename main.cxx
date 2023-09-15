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



#pragma region EXPERIMENTAL SETUP
/**
 * Get directed edge insertions.
 * @param edges undirected edge insertions
 * @param wd default edge weight to use (0 => use original)
 * @param UNDIRECTED are edges undirected?
 */
template <class K, class V>
inline vector<tuple<K, K, V>> directedInsertions(const vector<tuple<K, K, V>>& edges, V wd, bool UNDIRECTED=false) {
  vector<tuple<K, K, V>> a;
  for (const auto& [u, v, w] : edges) {
    a.push_back({u, v, wd? wd : w});
    if (UNDIRECTED) a.push_back({v, u, wd? wd : w});
  }
  return a;
}


/**
 * Get directed edge insertions.
 * @param edges undirected edges
 * @param w edge weight to use
 * @param UNDIRECTED are edges undirected?
 */
template <class K, class V>
inline vector<tuple<K, K, V>> directedInsertions(const vector<tuple<K, K>>& edges, V w, bool UNDIRECTED=false) {
  vector<tuple<K, K, V>> a;
  for (const auto& [u, v] : edges) {
    a.push_back({u, v, w});
    if (UNDIRECTED) a.push_back({v, u, w});
  }
  return a;
}


/**
 * Get common edges between two sets of edges.
 * @param edges1 first set of edges
 * @param edges2 second set of edges
 * @returns common edges
 */
template <class K, class V>
inline vector<tuple<K, K, V>> commonEdges(const vector<tuple<K, K, V>>& edges1, const vector<tuple<K, K, V>>& edges2) {
  vector<tuple<K, K, V>> a;
  set_intersection(edges1.begin(), edges1.end(), edges2.begin(), edges2.end(), back_inserter(a));
  return a;
}


/**
 * Obtain a graph with a specified edge insertions.
 * @param x original graph
 * @param insertions edge insertions
 * @returns graph with edge insertions
 */
template <class G, class K, class V>
inline auto graphWithInsertions(const G& x, const vector<tuple<K, K, V>>& insertions) {
  auto y = duplicate(x);
  for (const auto& [u, v, w] : insertions)
    y.addEdge(u, v, w);
  updateOmpU(y);
  return y;
}


/**
 * Run a function on each batch update using link prediction, with a specified range of batch sizes.
 * @param x original graph
 * @param fn function to run on each batch update
 */
template <class G, class R, class F>
inline void runBatches(const G& x, R& rnd, F fn) {
  using  E = typename G::edge_value_type;
  double d = BATCH_DELETIONS_BEGIN;
  double i = BATCH_INSERTIONS_BEGIN;
  for (int epoch=0;; ++epoch) {
    for (int r=0; r<REPEAT_BATCH; ++r) {
      auto y  = duplicate(x);
      for (int sequence=0; sequence<BATCH_LENGTH; ++sequence) {
        auto deletions  = removeRandomEdges(y, rnd, size_t(d * x.size()/2), 1, x.span()-1);
        auto insertions = addRandomEdges   (y, rnd, size_t(i * x.size()/2), 1, x.span()-1, E(1));
        fn(y, d, deletions, i, insertions, sequence, epoch);
      }
    }
    if (d>=BATCH_DELETIONS_END && i>=BATCH_INSERTIONS_END) break;
    d BATCH_DELETIONS_STEP;
    i BATCH_INSERTIONS_STEP;
    d = min(d, double(BATCH_DELETIONS_END));
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
  random_device dev;
  default_random_engine rnd(dev());
  int repeat  = REPEAT_METHOD;
  int retries = 5;
  vector<K> *init = nullptr;
  double M = edgeWeightOmp(x)/2;
  // Follow a specific result logging format, which can be easily parsed later.
  auto plog = [&](const auto& ans, const char *technique, int numThreads, auto deletionsf, const auto& deletions, const auto& pinsertions, const auto& pcommon) {
    double accuracy  = double(pcommon.size()) / (deletions.size() + pinsertions.size() - pcommon.size());
    double precision = double(pcommon.size()) / pinsertions.size();
    printf(
      "{-%.3e batchf, %03d threads} -> "
      "{%09.1fms, %.3e accuracy, %.3e precision} %s\n",
      double(deletionsf), numThreads,
      ans.time, accuracy, precision, technique
    );
  };
  auto clog = [&](const auto& ans, const char *technique, int numThreads, auto deletionsf, const auto& y, auto M) {
    printf(
      "{-%.3e batchf, %03d threads} -> "
      "{%09.1fms, %09.1fms preproc, %04d iters, %01.9f modularity} %s\n",
      double(deletionsf), numThreads,
      ans.time, ans.preprocessingTime, ans.iterations, getModularity(y, ans, M), technique
    );
  };
  // Get community memberships on original graph (static).
  auto p0 = PredictLinkResult<K, V>();
  auto a0 = louvainStaticOmp(x, init);
  // Predict links and obtain updated community memberships.
  runBatches(x, rnd, [&](const auto& y, auto deletionsf, const auto& deletions, auto insertionsf, const auto& insertions, int sequence, int epoch) {
    // Sort edge deletions from random batch update, for finding common edges.
    vector<tuple<K, K, V>>   bdeletions = directedInsertions(deletions, V(1));
    sort(bdeletions.begin(), bdeletions.end());
    runThreads([&](int numThreads) {
      auto a1 = louvainStaticOmp(y, init, {repeat});
      {
        // Predict links using Hub promoted score.
        auto p1 = predictLinksHubPromotedOmp<4>(y, {repeat, bdeletions.size()});
        vector<tuple<K, K>>    pdeletions;
        vector<tuple<K, K, V>> pinsertions = directedInsertions(p1.edges, V(1));
        // Sort edge insertions, and find common edges.
        sort(pinsertions.begin(), pinsertions.end());
        vector<tuple<K, K, V>> pcommon = commonEdges(bdeletions, pinsertions);
        // Log results.
        plog(p1, "predictLinksHubPromotedOmp4", numThreads, deletionsf, bdeletions, pinsertions, pcommon);
        // Obtain updated graph.
        auto   z = graphWithInsertions(y, pinsertions);
        double M = edgeWeightOmp(z)/2;
        // Update community memberships on updated graph.
        auto a2 = louvainStaticOmp(z, init, {repeat});
        clog(a2, "louvainStaticOmp", numThreads, deletionsf, z, M);
        auto b2 = rakDynamicFrontierOmp(z, pdeletions, pinsertions, &a1.membership, {repeat});
        clog(b2, "louvainRakDynamicFrontierOmp", numThreads, deletionsf, z, M);
      }
      {
        // Predict links using Hub promoted score.
        auto p1 = predictLinksHubPromotedOmp<8>(y, {repeat, bdeletions.size()});
        vector<tuple<K, K>>    pdeletions;
        vector<tuple<K, K, V>> pinsertions = directedInsertions(p1.edges, V(1));
        // Sort edge insertions, and find common edges.
        sort(pinsertions.begin(), pinsertions.end());
        vector<tuple<K, K, V>> pcommon = commonEdges(bdeletions, pinsertions);
        // Log results.
        plog(p1, "predictLinksHubPromotedOmp8", numThreads, deletionsf, bdeletions, pinsertions, pcommon);
        // Obtain updated graph.
        auto   z = graphWithInsertions(y, pinsertions);
        double M = edgeWeightOmp(z)/2;
        // Update community memberships on updated graph.
        auto a2 = louvainStaticOmp(z, init, {repeat});
        clog(a2, "louvainStaticOmp", numThreads, deletionsf, z, M);
        auto b2 = rakDynamicFrontierOmp(z, pdeletions, pinsertions, &a1.membership, {repeat});
        clog(b2, "louvainRakDynamicFrontierOmp", numThreads, deletionsf, z, M);
      }
      {
        // Predict links using Hub promoted score.
        auto p1 = predictLinksHubPromotedOmp<16>(y, {repeat, bdeletions.size()});
        vector<tuple<K, K>>    pdeletions;
        vector<tuple<K, K, V>> pinsertions = directedInsertions(p1.edges, V(1));
        // Sort edge insertions, and find common edges.
        sort(pinsertions.begin(), pinsertions.end());
        vector<tuple<K, K, V>> pcommon = commonEdges(bdeletions, pinsertions);
        // Log results.
        plog(p1, "predictLinksHubPromotedOmp16", numThreads, deletionsf, bdeletions, pinsertions, pcommon);
        // Obtain updated graph.
        auto   z = graphWithInsertions(y, pinsertions);
        double M = edgeWeightOmp(z)/2;
        // Update community memberships on updated graph.
        auto a2 = louvainStaticOmp(z, init, {repeat});
        clog(a2, "louvainStaticOmp", numThreads, deletionsf, z, M);
        auto b2 = rakDynamicFrontierOmp(z, pdeletions, pinsertions, &a1.membership, {repeat});
        clog(b2, "louvainRakDynamicFrontierOmp", numThreads, deletionsf, z, M);
      }
      {
        // Predict links using Hub promoted score.
        auto p1 = predictLinksHubPromotedOmp<32>(y, {repeat, bdeletions.size()});
        vector<tuple<K, K>>    pdeletions;
        vector<tuple<K, K, V>> pinsertions = directedInsertions(p1.edges, V(1));
        // Sort edge insertions, and find common edges.
        sort(pinsertions.begin(), pinsertions.end());
        vector<tuple<K, K, V>> pcommon = commonEdges(bdeletions, pinsertions);
        // Log results.
        plog(p1, "predictLinksHubPromotedOmp32", numThreads, deletionsf, bdeletions, pinsertions, pcommon);
        // Obtain updated graph.
        auto   z = graphWithInsertions(y, pinsertions);
        double M = edgeWeightOmp(z)/2;
        // Update community memberships on updated graph.
        auto a2 = louvainStaticOmp(z, init, {repeat});
        clog(a2, "louvainStaticOmp", numThreads, deletionsf, z, M);
        auto b2 = rakDynamicFrontierOmp(z, pdeletions, pinsertions, &a1.membership, {repeat});
        clog(b2, "louvainRakDynamicFrontierOmp", numThreads, deletionsf, z, M);
      }
      {
        // Predict links using Jaccard's coefficient.
        auto p1 = predictLinksJaccardCoefficientOmp<4>(y, {repeat, bdeletions.size()});
        vector<tuple<K, K>>    pdeletions;
        vector<tuple<K, K, V>> pinsertions = directedInsertions(p1.edges, V(1));
        // Sort edge insertions, and find common edges.
        sort(pinsertions.begin(), pinsertions.end());
        vector<tuple<K, K, V>> pcommon = commonEdges(bdeletions, pinsertions);
        // Log results.
        plog(p1, "predictLinksJaccardCoefficientOmp4", numThreads, deletionsf, bdeletions, pinsertions, pcommon);
        // Obtain updated graph.
        auto   z = graphWithInsertions(y, pinsertions);
        double M = edgeWeightOmp(z)/2;
        // Update community memberships on updated graph.
        auto a2 = louvainStaticOmp(z, init, {repeat});
        clog(a2, "louvainStaticOmp", numThreads, deletionsf, z, M);
        auto b2 = rakDynamicFrontierOmp(z, pdeletions, pinsertions, &a1.membership, {repeat});
        clog(b2, "louvainRakDynamicFrontierOmp", numThreads, deletionsf, z, M);
      }
      {
        // Predict links using Jaccard's coefficient.
        auto p1 = predictLinksJaccardCoefficientOmp<8>(y, {repeat, bdeletions.size()});
        vector<tuple<K, K>>    pdeletions;
        vector<tuple<K, K, V>> pinsertions = directedInsertions(p1.edges, V(1));
        // Sort edge insertions, and find common edges.
        sort(pinsertions.begin(), pinsertions.end());
        vector<tuple<K, K, V>> pcommon = commonEdges(bdeletions, pinsertions);
        // Log results.
        plog(p1, "predictLinksJaccardCoefficientOmp8", numThreads, deletionsf, bdeletions, pinsertions, pcommon);
        // Obtain updated graph.
        auto   z = graphWithInsertions(y, pinsertions);
        double M = edgeWeightOmp(z)/2;
        // Update community memberships on updated graph.
        auto a2 = louvainStaticOmp(z, init, {repeat});
        clog(a2, "louvainStaticOmp", numThreads, deletionsf, z, M);
        auto b2 = rakDynamicFrontierOmp(z, pdeletions, pinsertions, &a1.membership, {repeat});
        clog(b2, "louvainRakDynamicFrontierOmp", numThreads, deletionsf, z, M);
      }
      {
        // Predict links using Jaccard's coefficient.
        auto p1 = predictLinksJaccardCoefficientOmp<16>(y, {repeat, bdeletions.size()});
        vector<tuple<K, K>>    pdeletions;
        vector<tuple<K, K, V>> pinsertions = directedInsertions(p1.edges, V(1));
        // Sort edge insertions, and find common edges.
        sort(pinsertions.begin(), pinsertions.end());
        vector<tuple<K, K, V>> pcommon = commonEdges(bdeletions, pinsertions);
        // Log results.
        plog(p1, "predictLinksJaccardCoefficientOmp16", numThreads, deletionsf, bdeletions, pinsertions, pcommon);
        // Obtain updated graph.
        auto   z = graphWithInsertions(y, pinsertions);
        double M = edgeWeightOmp(z)/2;
        // Update community memberships on updated graph.
        auto a2 = louvainStaticOmp(z, init, {repeat});
        clog(a2, "louvainStaticOmp", numThreads, deletionsf, z, M);
        auto b2 = rakDynamicFrontierOmp(z, pdeletions, pinsertions, &a1.membership, {repeat});
        clog(b2, "louvainRakDynamicFrontierOmp", numThreads, deletionsf, z, M);
      }
      {
        // Predict links using Jaccard's coefficient.
        auto p1 = predictLinksJaccardCoefficientOmp<32>(y, {repeat, bdeletions.size()});
        vector<tuple<K, K>>    pdeletions;
        vector<tuple<K, K, V>> pinsertions = directedInsertions(p1.edges, V(1));
        // Sort edge insertions, and find common edges.
        sort(pinsertions.begin(), pinsertions.end());
        vector<tuple<K, K, V>> pcommon = commonEdges(bdeletions, pinsertions);
        // Log results.
        plog(p1, "predictLinksJaccardCoefficientOmp32", numThreads, deletionsf, bdeletions, pinsertions, pcommon);
        // Obtain updated graph.
        auto   z = graphWithInsertions(y, pinsertions);
        double M = edgeWeightOmp(z)/2;
        // Update community memberships on updated graph.
        auto a2 = louvainStaticOmp(z, init, {repeat});
        clog(a2, "louvainStaticOmp", numThreads, deletionsf, z, M);
        auto b2 = rakDynamicFrontierOmp(z, pdeletions, pinsertions, &a1.membership, {repeat});
        clog(b2, "louvainRakDynamicFrontierOmp", numThreads, deletionsf, z, M);
      }
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
