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
  int repeat     = REPEAT_METHOD;
  int numThreads = MAX_THREADS;
  // Follow a specific result logging format, which can be easily parsed later.
  auto glog = [&](const auto& ans, const char *technique, int numThreads, double insertionsf, const auto& insertions0, const auto& insertions1, const auto& common1) {
    double accuracy  = double(common1.size()) / (insertions0.size() + insertions1.size() - common1.size());
    double precision = double(common1.size()) / insertions1.size();
    printf(
      "{-%.3e/+%.3e batchf, %03d threads} -> {%09.1fms, %.3e accuracy, %.3e precision} %s\n",
      0.0, insertionsf, numThreads, ans.time, accuracy, precision, technique
    );
  };
  // Get predicted links from Original Jaccard coefficient.
  for (float insertionsf=1e-7; insertionsf<=0.1; insertionsf*=10) {
    size_t insertionsc = insertionsf * x.size();
    auto p0 = predictLinksJaccardCoefficientOmp<0>(x, {repeat, insertionsc});
    vector<tuple<K, K, V>>    insertions0 = directedInsertions(p0.edges, V(1));
    sort(insertions0.begin(), insertions0.end());
    glog(p0, "predictLinksJaccardCoefficientOmp", numThreads, insertionsf, insertions0, insertions0, insertions0);
    {
      // Predict links using Modified Jaccard's coefficient.
      auto p1 = predictLinksJaccardCoefficientOmp<2>(x, {repeat, insertions0.size()});
      vector<tuple<K, K, V>> insertions1 = directedInsertions(p1.edges, V(1));
      sort(insertions1.begin(), insertions1.end());
      vector<tuple<K, K, V>> common1 = commonEdges(insertions0, insertions1);
      glog(p1, "predictLinksJaccardCoefficientOmp2", numThreads, insertionsf, insertions0, insertions1, common1);
    }
    {
      // Predict links using Modified Jaccard's coefficient.
      auto p1 = predictLinksJaccardCoefficientOmp<4>(x, {repeat, insertions0.size()});
      vector<tuple<K, K, V>> insertions1 = directedInsertions(p1.edges, V(1));
      sort(insertions1.begin(), insertions1.end());
      vector<tuple<K, K, V>> common1 = commonEdges(insertions0, insertions1);
      glog(p1, "predictLinksJaccardCoefficientOmp4", numThreads, insertionsf, insertions0, insertions1, common1);
    }
    {
      // Predict links using Modified Jaccard's coefficient.
      auto p1 = predictLinksJaccardCoefficientOmp<8>(x, {repeat, insertions0.size()});
      vector<tuple<K, K, V>> insertions1 = directedInsertions(p1.edges, V(1));
      sort(insertions1.begin(), insertions1.end());
      vector<tuple<K, K, V>> common1 = commonEdges(insertions0, insertions1);
      glog(p1, "predictLinksJaccardCoefficientOmp8", numThreads, insertionsf, insertions0, insertions1, common1);
    }
    {
      // Predict links using Modified Jaccard's coefficient.
      auto p1 = predictLinksJaccardCoefficientOmp<16>(x, {repeat, insertions0.size()});
      vector<tuple<K, K, V>> insertions1 = directedInsertions(p1.edges, V(1));
      sort(insertions1.begin(), insertions1.end());
      vector<tuple<K, K, V>> common1 = commonEdges(insertions0, insertions1);
      glog(p1, "predictLinksJaccardCoefficientOmp16", numThreads, insertionsf, insertions0, insertions1, common1);
    }
    {
      // Predict links using Modified Jaccard's coefficient.
      auto p1 = predictLinksJaccardCoefficientOmp<32>(x, {repeat, insertions0.size()});
      vector<tuple<K, K, V>> insertions1 = directedInsertions(p1.edges, V(1));
      sort(insertions1.begin(), insertions1.end());
      vector<tuple<K, K, V>> common1 = commonEdges(insertions0, insertions1);
      glog(p1, "predictLinksJaccardCoefficientOmp32", numThreads, insertionsf, insertions0, insertions1, common1);
    }
    {
      // Predict links using Modified Jaccard's coefficient.
      auto p1 = predictLinksJaccardCoefficientOmp<64>(x, {repeat, insertions0.size()});
      vector<tuple<K, K, V>> insertions1 = directedInsertions(p1.edges, V(1));
      sort(insertions1.begin(), insertions1.end());
      vector<tuple<K, K, V>> common1 = commonEdges(insertions0, insertions1);
      glog(p1, "predictLinksJaccardCoefficientOmp64", numThreads, insertionsf, insertions0, insertions1, common1);
    }
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
  if (!symmetric) { x = symmetrizeOmp(x); LOG(""); print(x); printf(" (symmetrize)\n"); }
  runExperiment(x);
  printf("\n");
  return 0;
}
#pragma endregion
#pragma endregion
