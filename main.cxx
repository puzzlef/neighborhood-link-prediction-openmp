#include <cstdint>
#include <cstdio>
#include <utility>
#include <tuple>
#include <vector>
#include <string>
#include <random>
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
#define MAX_THREADS 32
#endif
#ifndef REPEAT_BATCH
/** Number of times to repeat each batch. */
#define REPEAT_BATCH 5
#endif
#ifndef REPEAT_METHOD
/** Number of times to repeat each method. */
#define REPEAT_METHOD 5
#endif
#pragma endregion




#pragma region MACROS
/**
 * Predict links using Modified approach.
 * @param fn prediction function
 * @param deg mindegree1 to use
 * @param insertionsf fraction of edges to insert
 * @param insertions0 original insertions/predictions
 */
#define PREDICT_LINKS(fn, deg, insertionsf, insertions0) \
  { \
    auto p1 = fn<deg>(x, {repeat, insertions0.size()}); \
    vector<tuple<K, K, V>> insertions1 = directedInsertions(p1.edges, V(1), true); \
    sort(insertions1.begin(), insertions1.end()); \
    unique(insertions1.begin(), insertions1.end()); \
    vector<tuple<K, K, V>> common1 = commonEdges(insertions0, insertions1); \
    glog(p1, #fn #deg, insertionsf, insertions0, insertions1, common1); \
  }


/**
 * Predict links with varying mindegree1 using Modified approach.
 * @param fn prediction function
 * @param insertionsf fraction of edges to insert
 * @param insertionsc number of edges to insert
 * @param insertions0 original insertions/predictions
 */
#define PREDICT_LINKS_ALL(fn, insertionsf, insertionsc, insertions0) \
  { \
    PREDICT_LINKS(fn, 0,    insertionsf, insertions0); \
    PREDICT_LINKS(fn, 2,    insertionsf, insertions0); \
    PREDICT_LINKS(fn, 4,    insertionsf, insertions0); \
    PREDICT_LINKS(fn, 8,    insertionsf, insertions0); \
    PREDICT_LINKS(fn, 16,   insertionsf, insertions0); \
    PREDICT_LINKS(fn, 32,   insertionsf, insertions0); \
    PREDICT_LINKS(fn, 64,   insertionsf, insertions0); \
    PREDICT_LINKS(fn, 128,  insertionsf, insertions0); \
    PREDICT_LINKS(fn, 256,  insertionsf, insertions0); \
    PREDICT_LINKS(fn, 512,  insertionsf, insertions0); \
    PREDICT_LINKS(fn, 1024, insertionsf, insertions0); \
  }
#pragma endregion




#pragma region METHODS
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
        auto deletions  = generateEdgeDeletions (rnd, y, size_t(d * x.size()/2), 1, x.span()-1, true);
        auto insertions = generateEdgeInsertions(rnd, y, size_t(i * x.size()/2), 1, x.span()-1, true, None());
        tidyBatchUpdateU(deletions, insertions, y);
        applyBatchUpdateOmpU(y, deletions, insertions);
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
  int repeat     = REPEAT_METHOD;
  int numThreads = MAX_THREADS;
  // Follow a specific result logging format, which can be easily parsed later.
  auto glog = [&](const auto& ans, const char *technique, double insertionsf, const auto& insertions0, const auto& insertions1, const auto& common1) {
    double accuracy  = double(common1.size()) / (insertions0.size() + insertions1.size() - common1.size());
    double precision = double(common1.size()) / insertions1.size();
    printf(
      "{-%.3e/+%.3e batchf, %03d threads} -> {%09.1fms, %.3e accuracy, %.3e precision} %s\n",
      0.0, insertionsf, numThreads, ans.time, accuracy, precision, technique
    );
  };
  // Get predicted links from Original Jaccard coefficient.
  runBatches(x, rnd, [&](const auto& y, auto deletionsf, const auto& deletions, auto insertionsf, const auto& insertions, int sequence, int epoch) {
    size_t deletionsc = deletionsf * x.size();
    if (deletionsc < 1) return;
    vector<tuple<K, K, V>>   deletions0 = directedInsertions(deletions, V(1));
    sort(deletions0.begin(), deletions0.end());
    PREDICT_LINKS_ALL(predictLinksJaccardCoefficientOmp,      deletionsf, deletionsc, deletions0);
    PREDICT_LINKS_ALL(predictLinksSorensenIndexOmp,           deletionsf, deletionsc, deletions0);
    PREDICT_LINKS_ALL(predictLinksSaltonCosineSimilarityOmp,  deletionsf, deletionsc, deletions0);
    PREDICT_LINKS_ALL(predictLinksHubPromotedOmp,             deletionsf, deletionsc, deletions0);
    PREDICT_LINKS_ALL(predictLinksHubDepressedOmp,            deletionsf, deletionsc, deletions0);
    PREDICT_LINKS_ALL(predictLinksLeichtHolmeNermanScoreOmp,  deletionsf, deletionsc, deletions0);
    PREDICT_LINKS_ALL(predictLinksAdamicAdarCoefficientOmp,   deletionsf, deletionsc, deletions0);
    PREDICT_LINKS_ALL(predictLinksResourceAllocationScoreOmp, deletionsf, deletionsc, deletions0);
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
  if (!symmetric) { x = symmetrizeOmp(x); LOG(""); print(x); printf(" (symmetrize)\n"); }
  runExperiment(x);
  printf("\n");
  return 0;
}
#pragma endregion
#pragma endregion
