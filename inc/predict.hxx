#pragma once
#include <utility>
#include <tuple>
#include <vector>
#include <algorithm>
#include "_main.hxx"
#ifdef OPENMP
#include <omp.h>
#endif

using std::tuple;
using std::vector;
using std::move;
using std::min;
using std::sort;
using std::make_heap;




#pragma region TYPES
/**
 * Options for Link Prediction algorithm.
 * @tparam V edge weight/score type
 */
template <class V>
struct PredictLinkOptions {
  #pragma region DATA
  /** Number of times to repeat the algorithm [1]. */
  int repeat;
  /** Minimum score to consider a link [0]. */
  V   minScore;
  /** Maximum number of edges to predict [-1]. */
  size_t maxEdges;
  #pragma endregion


  #pragma region CONSTRUCTORS
  /**
   * Define options for Link Prediction algorithm.
   * @param repeat number of times to repeat the algorithm [1]
   * @param minScore minimum score to consider a link [0]
   * @param maxEdges maximum number of edges to predict [-1]
   */
  PredictLinkOptions(int repeat=1, V minScore=V(), size_t maxEdges=size_t(-1)) :
  repeat(repeat), minScore(minScore), maxEdges(maxEdges) {}
  #pragma endregion
};




/**
 * Result of Link Prediction algorithm.
 * @tparam K key type (vertex-id)
 * @tparam V edge weight/score type
 */
template <class K, class V>
struct PredictLinkResult {
  #pragma region DATA
  /** Predicted links (undirected). */
  vector<tuple<K, K, V>> edges;
  /** Time spent in milliseconds. */
  float time;
  #pragma endregion


  #pragma region CONSTRUCTORS
  /**
   * Empty Result of Link Prediction algorithm.
   */
  PredictLinkResult() :
  edges(), time() {}

  /**
   * Result of Link Prediction algorithm.
   * @param edges predicted links (undirected)
   * @param time time spent in milliseconds
   */
  PredictLinkResult(vector<tuple<K, K, V>>&& edges, float time=0) :
  edges(edges), time(time) {}


  /**
   * Result of Link Prediction algorithm.
   * @param edges predicted links (undirected)
   * @param time time spent in milliseconds
   */
  PredictLinkResult(vector<tuple<K, K, V>>& edges, float time=0) :
  edges(move(edges)), time(time) {}
  #pragma endregion
};
#pragma endregion




#pragma region METHODS
#pragma region HASHTABLES
/**
 * Allocate a number of hashtables.
 * @param vedgs edges linked to vertex u (output)
 * @param veout times edge u is linked to vertex u (output)
 * @param S size of each hashtable
 */
template <class K>
inline void predictAllocateHashtablesW(vector<vector<K>*>& vedgs, vector<vector<K>*>& veout, size_t S) {
  size_t N = vedgs.size();
  for (size_t i=0; i<N; ++i) {
    vedgs[i] = new vector<K>();
    veout[i] = new vector<K>(S);
  }
}


/**
 * Free a number of hashtables.
 * @param vedgs edges linked to vertex u (output)
 * @param veout times edge u is linked to vertex u (output)
 */
template <class K>
inline void predictFreeHashtablesW(vector<vector<K>*>& vedgs, vector<vector<K>*>& veout) {
  size_t N = vedgs.size();
  for (size_t i=0; i<N; ++i) {
    delete vedgs[i];
    delete veout[i];
  }
}
#pragma endregion




#pragma region SCAN EDGES
/**
 * Scan edges connected to a vertex.
 * @param vedgs edges linked to vertex u (updated)
 * @param veout times edge u is linked to vertex u (updated)
 * @param x original graph
 * @param u given vertex
 * @param ft include edge?
 */
template <class G, class K, class FT>
inline void predictScanEdgesU(vector<K>& vedgs, vector<K>& veout, const G& x, K u, FT ft) {
  x.forEachEdgeKey(u, [&](auto v) {
    if (!ft(v)) return;
    if (!veout[v]) vedgs.push_back(v);
    ++veout[v];
  });
}


/**
 * Clear edge scan data.
 * @param vedgs edges linked to vertex u (output)
 * @param veout times edge u is linked to vertex u (output)
 */
template <class K>
inline void predictClearScanW(vector<K>& vedgs, vector<K>& veout) {
  for (K v : vedgs)
    veout[v] = K();
  vedgs.clear();
}
#pragma endregion




#pragma region PREDICT LINKS HUB PROMOTED
/**
 * Main loop for predicting links using hub promoted score.
 * @param a predicted links (undirected, updated)
 * @param vedgs edges linked to vertex u (output)
 * @param veout times edge u is linked to vertex u (output)
 * @param x original graph
 * @param SMIN minimum score to consider a link
 * @param NMAX maximum number of edges to predict
 */
template <class G, class K, class V>
inline void predictLinksHubPromotedLoopU(vector<tuple<K, K, V>>& a, vector<K>& vedgs, vector<K>& veout, const G& x, V SMIN, size_t NMAX) {
  x.forEachVertexKey([&](auto u) {
    // Get second order edges, with link count.
    auto ft = [&](auto v) { return v>u; };
    predictClearScanW(vedgs, veout);
    x.forEachEdgeKey(u, [&](auto v) { predictScanEdgesU(vedgs, veout, x, v, ft); });
    // Get hub promoted score, and add to prediction list.
    for (K v : vedgs) {
      V   score = V(veout[v]) / min(x.degree(u), x.degree(v));
      if (score < SMIN) continue;  // Skip low scores
      // Add to prediction list.
      size_t A = a.size();
      auto  fl = [](const auto& x, const auto& y) { return get<2>(x) > get<2>(y); };
      if (A<NMAX) {
        // We have not reached the maximum number of edges to predict, simply add.
        a.push_back({u, v, score});
        // Convert to max-heap, if prediction list is full.
        if (A+1==NMAX) make_heap(a.begin(), a.end(), fl);
      }
      else {
        // Use min-heap to store top NMAX edges only.
        if (score < get<2>(a[0])) continue;
        pop_heap(a.begin(), a.end(), fl);
        a.push_back({u, v, score});
        push_heap(a.begin(), a.end(), fl);
      }
    }
  });
}


#ifdef OPENMP
/**
 * Main loop for predicting links using hub promoted score.
 * @param as per-thread predicted links (undirected, updated)
 * @param vedgs edges linked to vertex u (output)
 * @param veout times edge u is linked to vertex u (output)
 * @param x original graph
 * @param SMIN minimum score to consider a link
 * @param NMAX maximum number of edges to predict
 */
template <class G, class K, class V>
inline void predictLinksHubPromotedLoopOmpU(vector<vector<tuple<K, K, V>>*>& as, vector<vector<K>*>& vedgs, vector<vector<K>*>& veout, const G& x, V SMIN, size_t NMAX) {
  size_t S = x.span();
  #pragma omp parallel for schedule(dynamic, 2048)
  for (K u=0; u<S; ++u) {
    if (!x.hasVertex(u)) continue;
    int t = omp_get_thread_num();
    // Get second order edges, with link count.
    auto ft = [&](auto v) { return v>u; };
    predictClearScanW(*vedgs[t], *veout[t]);
    x.forEachEdgeKey(u, [&](auto v) { predictScanEdgesU(*vedgs[t], *veout[t], x, v, ft); });
    // Get hub promoted score, and add to prediction list.
    for (K v : *vedgs[t]) {
      V   score = V((*veout[t])[v]) / min(x.degree(u), x.degree(v));
      if (score < SMIN) continue;  // Skip low scores
      // Add to prediction list.
      size_t A = (*as[t]).size();
      auto  fl = [](const auto& x, const auto& y) { return get<2>(x) > get<2>(y); };
      if (A<NMAX) {
        // We have not reached the maximum number of edges to predict, simply add.
        (*as[t]).push_back({u, v, score});
        // Convert to max-heap, if prediction list is full.
        if (A+1==NMAX) make_heap((*as[t]).begin(), (*as[t]).end(), fl);
      }
      else {
        // Use min-heap to store top NMAX edges only.
        if (score < get<2>((*as[t])[0])) continue;
        pop_heap((*as[t]).begin(), (*as[t]).end(), fl);
        (*as[t]).push_back({u, v, score});
        push_heap((*as[t]).begin(), (*as[t]).end(), fl);
      }
    }
  }
}
#endif




/**
 * Predict links using hub promoted score.
 * @param x original graph
 * @param o predict link options
 * @returns [{u, v, score}] undirected predicted links, ordered by score (descending)
 */
template <class G, class V=float>
inline auto predictLinksHubPromoted(const G& x, const PredictLinkOptions<V>& o={}) {
  using  K = typename G::key_type;
  size_t S = x.span();
  vector<tuple<K, K, V>> a;
  vector<K> vedgs, veout(S);
  float ta = measureDuration([&]() {
    a.clear();
    predictLinksHubPromotedLoopU(a, vedgs, veout, x, o.minScore, o.maxEdges);
  }, o.repeat);
  auto fl = [](const auto& x, const auto& y) { return get<2>(x) > get<2>(y); };
  sort(a.begin(), a.end(), fl);
  return PredictLinkResult<K, V>(a, ta);
}


#ifdef OPENMP
/**
 * Predict links using hub promoted score.
 * @param x original graph
 * @param o predict link options
 * @returns [{u, v, score}] undirected predicted links, ordered by score (descending)
 */
template <class G, class V=float>
inline auto predictLinksHubPromotedOmp(const G& x, const PredictLinkOptions<V>& o={}) {
  using  K = typename G::key_type;
  size_t S = x.span();
  int    T = omp_get_max_threads();
  // Setup per-thread prediction lists.
  vector<tuple<K, K, V>>  a;
  vector<vector<tuple<K, K, V>>*> as(T);
  for (int t=0; t<T; ++t)
    as[t] = new vector<tuple<K, K, V>>();
  // Setup per-thread hashtables.
  vector<vector<K>*> vedgs(T);
  vector<vector<K>*> veout(T);
  predictAllocateHashtablesW(vedgs, veout, S);
  // Predict links in parallel.
  float ta = measureDuration([&]() {
    for (int t=0; t<T; ++t)
      (*as[t]).clear();
    predictLinksHubPromotedLoopOmpU(as, vedgs, veout, x, o.minScore, o.maxEdges);
  }, o.repeat);
  // Merge per-thread prediction lists.
  for (int t=0; t<T; ++t)
    a.insert(a.end(), (*as[t]).begin(), (*as[t]).end());
  auto fl = [](const auto& x, const auto& y) { return get<2>(x) > get<2>(y); };
  sort(a.begin(), a.end(), fl);
  // Truncate to maximum number of edges.
  if (a.size() > o.maxEdges) a.resize(o.maxEdges);
  // Free per-thread prediction lists.
  for (int t=0; t<T; ++t)
    delete as[t];
  // Free per-thread hashtables.
  predictFreeHashtablesW(vedgs, veout);
  return PredictLinkResult<K, V>(a, ta);
}
#endif
#pragma endregion
#pragma endregion
