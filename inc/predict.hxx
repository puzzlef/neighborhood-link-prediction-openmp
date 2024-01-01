#pragma once
#include <cmath>
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
using std::sqrt;
using std::log;
using std::move;
using std::min;
using std::max;
using std::sort;
using std::make_heap;




#pragma region TYPES
/**
 * Options for Link Prediction algorithm.
 * @tparam W edge weight/score type
 */
template <class W>
struct PredictLinkOptions {
  #pragma region DATA
  /** Number of times to repeat the algorithm [1]. */
  int repeat;
  /** Maximum number of edges to predict [-1]. */
  size_t maxEdges;
  /** Minimum score to consider a link [0]. */
  W   minScore;
  #pragma endregion


  #pragma region CONSTRUCTORS
  /**
   * Define options for Link Prediction algorithm.
   * @param repeat number of times to repeat the algorithm [1]
   * @param maxEdges maximum number of edges to predict [-1]
   * @param minScore minimum score to consider a link [0]
   */
  PredictLinkOptions(int repeat=1, size_t maxEdges=size_t(-1), W minScore=W()) :
  repeat(repeat), maxEdges(maxEdges), minScore(minScore) {}
  #pragma endregion
};




/**
 * Result of Link Prediction algorithm.
 * @tparam K key type (vertex-id)
 * @tparam W edge weight/score type
 */
template <class K, class W>
struct PredictLinkResult {
  #pragma region DATA
  /** Predicted links (undirected). */
  vector<tuple<K, K, W>> edges;
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
  PredictLinkResult(vector<tuple<K, K, W>>&& edges, float time=0) :
  edges(edges), time(time) {}


  /**
   * Result of Link Prediction algorithm.
   * @param edges predicted links (undirected)
   * @param time time spent in milliseconds
   */
  PredictLinkResult(vector<tuple<K, K, W>>& edges, float time=0) :
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
template <class K, class V>
inline void predictAllocateHashtablesW(vector<vector<K>*>& vedgs, vector<vector<V>*>& veout, size_t S) {
  size_t N = vedgs.size();
  for (size_t i=0; i<N; ++i) {
    vedgs[i] = new vector<K>();
    veout[i] = new vector<V>(S);
  }
}


/**
 * Free a number of hashtables.
 * @param vedgs edges linked to vertex u (output)
 * @param veout times edge u is linked to vertex u (output)
 */
template <class K, class V>
inline void predictFreeHashtablesW(vector<vector<K>*>& vedgs, vector<vector<V>*>& veout) {
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
template <class G, class K, class V, class FT>
inline void predictScanEdgesU(vector<K>& vedgs, vector<V>& veout, const G& x, K u, FT ft) {
  x.forEachEdgeKey(u, [&](auto v) {
    if (!ft(v)) return;
    if (!veout[v]) vedgs.push_back(v);
    ++veout[v];
  });
}


/**
 * Scan edges connected to a vertex.
 * @param vedgs edges linked to vertex u (updated)
 * @param veout times edge u is linked to vertex u (updated)
 * @param x original graph
 * @param u given vertex
 * @param ft include edge?
 * @param fu edge value update function (entry, v)
 */
template <class G, class K, class V, class FT, class FV>
inline void predictScanEdgesU(vector<K>& vedgs, vector<V>& veout, const G& x, K u, FT ft, FV fu) {
  x.forEachEdgeKey(u, [&](auto v) {
    if (!ft(v)) return;
    if (!veout[v]) vedgs.push_back(v);
    fu(veout[v], v);
  });
}


/**
 * Clear edge scan data.
 * @param vedgs edges linked to vertex u (output)
 * @param veout times edge u is linked to vertex u (output)
 */
template <class K, class V>
inline void predictClearScanW(vector<K>& vedgs, vector<V>& veout) {
  for (K v : vedgs)
    veout[v] = V();
  vedgs.clear();
}
#pragma endregion




#pragma region PREDICT LINKS WITH INTERSECTION
/**
 * Main loop for predicting links with intersection-based score.
 * @tparam MINDEGREE1 degree of high degree first order neighbors to skip (if set)
 * @tparam MAXFACTOR2 maximum degree factor between source and second order neighbor to allow (if set)
 * @tparam FORCEHEAP always use heap to store top edges
 * @tparam CUSTOMVALUE use custom edge value update function
 * @param a predicted links (undirected, updated)
 * @param vedgs edges linked to vertex u (output)
 * @param veout times edge u is linked to vertex u (output)
 * @param x original graph
 * @param SMIN minimum score to consider a link
 * @param NMAX maximum number of edges to predict
 * @param fs score function (u, v, |N(u) ∩ N(v)|) => score, where N(u) is the set of neighbors of u
 * @param fu edge value update function (entry, v)
 */
template <int MINDEGREE1=4, int MAXFACTOR2=0, bool FORCEHEAP=false, bool CUSTOMVALUE=false, class G, class K, class V, class W, class FS, class FU>
inline void predictLinksWithIntersectionLoopU(vector<tuple<K, K, W>>& a, vector<K>& vedgs, vector<V>& veout, const G& x, W SMIN, size_t NMAX, FS fs, FU fu) {
  x.forEachVertexKey([&](auto u) {
    // Get second order edges, with link count.
    auto ft = [&](auto v) {
      // Skip link prediction between low degree and high degree vertices (if MAXFACTOR2 is set).
      // Such links are unlikely to be formed, based on Jaccard's coefficient or Hub promoted score.
      return MAXFACTOR2? v>u && x.degree(u)<=MAXFACTOR2*x.degree(u) && x.degree(v)<=MAXFACTOR2*x.degree(u) : v>u;
    };
    predictClearScanW(vedgs, veout);
    x.forEachEdgeKey(u, [&](auto v) {
      // Skip high degree first order neighbors (if MINDEGREE1 is set).
      // This can significantly improve performance!
      if (MINDEGREE1 && x.degree(v) > MINDEGREE1) return;
      if (CUSTOMVALUE) predictScanEdgesU(vedgs, veout, x, v, ft, fu);
      else             predictScanEdgesU(vedgs, veout, x, v, ft);
    });
    // Get hub promoted score, and add to prediction list.
    for (K v : vedgs) {
      W   score = fs(u, v, veout[v]);
      if (score < SMIN) continue;  // Skip low scores
      // Add to prediction list.
      size_t A = a.size();
      auto  fl = [](const auto& x, const auto& y) { return get<2>(x) > get<2>(y); };
      if (FORCEHEAP) {
        if (A>=NMAX && score < get<2>(a[0])) continue;
        if (A>=NMAX) pop_heap(a.begin(), a.end(), fl);
        a.push_back({u, v, score});
        push_heap(a.begin(), a.end(), fl);
      }
      else if (A<NMAX) {
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
 * Main loop for predicting links with intersection-based score.
 * @tparam MINDEGREE1 degree of high degree first order neighbors to skip (if set)
 * @tparam MAXFACTOR2 maximum degree factor between source and second order neighbor to allow (if set)
 * @tparam FORCEHEAP always use heap to store top edges
 * @tparam CUSTOMVALUE use custom edge value update function
 * @param as per-thread predicted links (undirected, updated)
 * @param vedgs edges linked to vertex u (output)
 * @param veout times edge u is linked to vertex u (output)
 * @param x original graph
 * @param SMIN minimum score to consider a link
 * @param NMAX maximum number of edges to predict
 * @param fs score function (u, v, |N(u) ∩ N(v)|) => score, where N(u) is the set of neighbors of u
 * @param fu edge value update function (entry, v)
 */
template <int MINDEGREE1=4, int MAXFACTOR2=0, bool FORCEHEAP=false, bool CUSTOMVALUE=false, class G, class K, class W, class FS, class FU>
inline void predictLinksWithIntersectionLoopOmpU(vector<vector<tuple<K, K, W>>*>& as, vector<vector<K>*>& vedgs, vector<vector<K>*>& veout, const G& x, W SMIN, size_t NMAX, FS fs, FU fu) {
  size_t S = x.span();
  #pragma omp parallel for schedule(dynamic, 2048)
  for (K u=0; u<S; ++u) {
    if (!x.hasVertex(u)) continue;
    int t = omp_get_thread_num();
    // Get second order edges, with link count.
    auto ft = [&](auto v) {
      // Skip link prediction between low degree and high degree vertices (if MAXFACTOR2 is set).
      // Such links are unlikely to be formed, based on Jaccard's coefficient or Hub promoted score.
      return MAXFACTOR2? v>u && x.degree(u)<=MAXFACTOR2*x.degree(u) && x.degree(v)<=MAXFACTOR2*x.degree(u) : v>u;
    };
    predictClearScanW(*vedgs[t], *veout[t]);
    x.forEachEdgeKey(u, [&](auto v) {
      // Skip high degree first order neighbors (if MINDEGREE1 is set).
      // This can significantly improve performance!
      if (MINDEGREE1 && x.degree(v) > MINDEGREE1) return;
      if (CUSTOMVALUE) predictScanEdgesU(*vedgs[t], *veout[t], x, v, ft, fu);
      else             predictScanEdgesU(*vedgs[t], *veout[t], x, v, ft);
    });
    // Get hub promoted score, and add to prediction list.
    for (K v : *vedgs[t]) {
      W   score = fs(u, v, (*veout[t])[v]);
      if (score < SMIN) continue;  // Skip low scores
      // Add to prediction list.
      size_t A = (*as[t]).size();
      auto  fl = [](const auto& x, const auto& y) { return get<2>(x) > get<2>(y); };
      if (FORCEHEAP) {
        if (A>=NMAX && score < get<2>((*as[t])[0])) continue;
        if (A>=NMAX) pop_heap((*as[t]).begin(), (*as[t]).end(), fl);
        (*as[t]).push_back({u, v, score});
        push_heap((*as[t]).begin(), (*as[t]).end(), fl);
      }
      else if (A<NMAX) {
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
 * Predict links with intersection-based score.
 * @tparam MINDEGREE1 degree of high degree first order neighbors to skip (if set)
 * @tparam MAXFACTOR2 maximum degree factor between source and second order neighbor to allow (if set)
 * @tparam FORCEHEAP always use heap to store top edges
 * @tparam CUSTOMVALUE use custom edge value update function
 * @param x original graph
 * @param o predict link options
 * @param VT hashtable value type
 * @param fs score function (u, v, |N(u) ∩ N(v)|) => score, where N(u) is the set of neighbors of u
 * @param fu edge value update function (entry, v)
 * @returns [{u, v, score}] undirected predicted links, ordered by score (descending)
 */
template <int MINDEGREE1=4, int MAXFACTOR2=0, bool FORCEHEAP=false, bool CUSTOMVALUE=false, class G, class V, class W, class FS, class FU>
inline auto predictLinksWithIntersection(const G& x, const PredictLinkOptions<W>& o, V VT, FS fs, FU fu) {
  using  K = typename G::key_type;
  size_t S = x.span();
  vector<tuple<K, K, W>> a;
  vector<K> vedgs;
  vector<V> veout(S);
  float ta = measureDuration([&]() {
    a.clear();
    if (o.maxEdges > 0) predictLinksWithIntersectionLoopU<MINDEGREE1, MAXFACTOR2, FORCEHEAP, CUSTOMVALUE>(a, vedgs, veout, x, o.minScore, o.maxEdges, fs, fu);
  }, o.repeat);
  auto fl = [](const auto& x, const auto& y) { return get<2>(x) > get<2>(y); };
  sort(a.begin(), a.end(), fl);
  return PredictLinkResult<K, W>(a, ta);
}


/**
 * Predict links with intersection-based score.
 * @tparam MINDEGREE1 degree of high degree first order neighbors to skip (if set)
 * @tparam MAXFACTOR2 maximum degree factor between source and second order neighbor to allow (if set)
 * @tparam FORCEHEAP always use heap to store top edges
 * @param x original graph
 * @param o predict link options
 * @param fs score function (u, v, |N(u) ∩ N(v)|) => score, where N(u) is the set of neighbors of u
 * @returns [{u, v, score}] undirected predicted links, ordered by score (descending)
 */
template <int MINDEGREE1=4, int MAXFACTOR2=0, bool FORCEHEAP=false, class G, class W, class FS>
inline auto predictLinksWithIntersection(const G& x, const PredictLinkOptions<W>& o, FS fs) {
  using K = typename G::key_type;
  auto fu = [](auto& entry, auto v) { ++entry; };
  return predictLinksWithIntersection<MINDEGREE1, MAXFACTOR2, FORCEHEAP, false>(x, o, K(), fs, fu);
}


#ifdef OPENMP
/**
 * Predict links using hub promoted score.
 * @tparam MINDEGREE1 degree of high degree first order neighbors to skip (if set)
 * @tparam MAXFACTOR2 maximum degree factor between source and second order neighbor to allow (if set)
 * @tparam FORCEHEAP always use heap to store top edges
 * @tparam CUSTOMVALUE use custom edge value update function
 * @param x original graph
 * @param o predict link options
 * @param VT hashtable value type
 * @param fs score function (u, v, |N(u) ∩ N(v)|) => score, where N(u) is the set of neighbors of u
 * @param fu edge value update function (entry, v)
 * @returns [{u, v, score}] undirected predicted links, ordered by score (descending)
 */
template <int MINDEGREE1=4, int MAXFACTOR2=0, bool FORCEHEAP=false, bool CUSTOMVALUE=false, class G, class V, class W, class FS, class FU>
inline auto predictLinksWithIntersectionCustomOmp(const G& x, const PredictLinkOptions<W>& o, V VT, FS fs, FU fu) {
  using  K = typename G::key_type;
  size_t S = x.span();
  int    T = omp_get_max_threads();
  // Setup per-thread prediction lists.
  vector<tuple<K, K, W>>  a;
  vector<vector<tuple<K, K, W>>*> as(T);
  for (int t=0; t<T; ++t)
    as[t] = new vector<tuple<K, K, W>>();
  // Setup per-thread hashtables.
  vector<vector<K>*> vedgs(T);
  vector<vector<V>*> veout(T);
  predictAllocateHashtablesW(vedgs, veout, S);
  // Predict links in parallel.
  float ta = measureDuration([&]() {
    for (int t=0; t<T; ++t)
      (*as[t]).clear();
    if (o.maxEdges > 0) predictLinksWithIntersectionLoopOmpU<MINDEGREE1, MAXFACTOR2, FORCEHEAP, CUSTOMVALUE>(as, vedgs, veout, x, o.minScore, o.maxEdges, fs, fu);
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
  return PredictLinkResult<K, W>(a, ta);
}


/**
 * Predict links using hub promoted score.
 * @tparam MINDEGREE1 degree of high degree first order neighbors to skip (if set)
 * @tparam MAXFACTOR2 maximum degree factor between source and second order neighbor to allow (if set)
 * @tparam FORCEHEAP always use heap to store top edges
 * @param x original graph
 * @param o predict link options
 * @param fs score function (u, v, |N(u) ∩ N(v)|) => score, where N(u) is the set of neighbors of u
 * @returns [{u, v, score}] undirected predicted links, ordered by score (descending)
 */
template <int MINDEGREE1=4, int MAXFACTOR2=0, bool FORCEHEAP=false, class G, class W, class FS>
inline auto predictLinksWithIntersectionOmp(const G& x, const PredictLinkOptions<W>& o, FS fs) {
  using K = typename G::key_type;
  auto fu = [](auto& entry, auto v) { ++entry; };
  return predictLinksWithIntersectionCustomOmp<MINDEGREE1, MAXFACTOR2, FORCEHEAP, false>(x, o, K(), fs, fu);
}
#endif
#pragma endregion




#pragma region PREDICT LINKS WITH JACCARD COEFFICIENT
/**
 * Predict links using Jaccard's coefficient.
 * @tparam MINDEGREE1 degree of high degree first order neighbors to skip (if set)
 * @tparam MAXFACTOR2 maximum degree factor between source and second order neighbor to allow (if set)
 * @tparam FORCEHEAP always use heap to store top edges
 * @param x original graph
 * @param o predict link options
 * @returns [{u, v, score}] undirected predicted links, ordered by score (descending)
 */
template <int MINDEGREE1=4, int MAXFACTOR2=0, bool FORCEHEAP=false, class G, class W=float>
inline auto predictLinksJaccardCoefficient(const G& x, const PredictLinkOptions<W>& o={}) {
  auto fs = [&](auto u, auto v, auto Nuv) { return W(Nuv) / (x.degree(u) + x.degree(v) - Nuv); };
  return predictLinksWithIntersection<MINDEGREE1, MAXFACTOR2, FORCEHEAP>(x, o, fs);
}


#ifdef OPENMP
/**
 * Predict links using Jaccard's coefficient.
 * @tparam MINDEGREE1 degree of high degree first order neighbors to skip (if set)
 * @tparam MAXFACTOR2 maximum degree factor between source and second order neighbor to allow (if set)
 * @tparam FORCEHEAP always use heap to store top edges
 * @param x original graph
 * @param o predict link options
 * @returns [{u, v, score}] undirected predicted links, ordered by score (descending)
 */
template <int MINDEGREE1=4, int MAXFACTOR2=0, bool FORCEHEAP=false, class G, class W=float>
inline auto predictLinksJaccardCoefficientOmp(const G& x, const PredictLinkOptions<W>& o={}) {
  auto fs = [&](auto u, auto v, auto Nuv) { return W(Nuv) / (x.degree(u) + x.degree(v) - Nuv); };
  return predictLinksWithIntersectionOmp<MINDEGREE1, MAXFACTOR2, FORCEHEAP>(x, o, fs);
}
#endif
#pragma endregion




#pragma region PREDICT LINKS WITH SORENSEN INDEX
/**
 * Predict links using Sorensen Index.
 * @tparam MINDEGREE1 degree of high degree first order neighbors to skip (if set)
 * @tparam MAXFACTOR2 maximum degree factor between source and second order neighbor to allow (if set)
 * @tparam FORCEHEAP always use heap to store top edges
 * @param x original graph
 * @param o predict link options
 * @returns [{u, v, score}] undirected predicted links, ordered by score (descending)
 */
template <int MINDEGREE1=4, int MAXFACTOR2=0, bool FORCEHEAP=false, class G, class W=float>
inline auto predictLinksSorensenIndex(const G& x, const PredictLinkOptions<W>& o={}) {
  auto fs = [&](auto u, auto v, auto Nuv) { return W(Nuv) / (x.degree(u) + x.degree(v)); };
  return predictLinksWithIntersection<MINDEGREE1, MAXFACTOR2, FORCEHEAP>(x, o, fs);
}


#ifdef OPENMP
/**
 * Predict links using Sorensen Index.
 * @tparam MINDEGREE1 degree of high degree first order neighbors to skip (if set)
 * @tparam MAXFACTOR2 maximum degree factor between source and second order neighbor to allow (if set)
 * @tparam FORCEHEAP always use heap to store top edges
 * @param x original graph
 * @param o predict link options
 * @returns [{u, v, score}] undirected predicted links, ordered by score (descending)
 */
template <int MINDEGREE1=4, int MAXFACTOR2=0, bool FORCEHEAP=false, class G, class W=float>
inline auto predictLinksSorensenIndexOmp(const G& x, const PredictLinkOptions<W>& o={}) {
  auto fs = [&](auto u, auto v, auto Nuv) { return W(Nuv) / (x.degree(u) + x.degree(v)); };
  return predictLinksWithIntersectionOmp<MINDEGREE1, MAXFACTOR2, FORCEHEAP>(x, o, fs);
}
#endif
#pragma endregion




#pragma region PREDICT LINKS WITH SALTON COSINE SIMILARITY
/**
 * Predict links using Salton Cosine Similarity.
 * @tparam MINDEGREE1 degree of high degree first order neighbors to skip (if set)
 * @tparam MAXFACTOR2 maximum degree factor between source and second order neighbor to allow (if set)
 * @tparam FORCEHEAP always use heap to store top edges
 * @param x original graph
 * @param o predict link options
 * @returns [{u, v, score}] undirected predicted links, ordered by score (descending)
 */
template <int MINDEGREE1=4, int MAXFACTOR2=0, bool FORCEHEAP=false, class G, class W=float>
inline auto predictLinksSaltonCosineSimilarity(const G& x, const PredictLinkOptions<W>& o={}) {
  auto fs = [&](auto u, auto v, auto Nuv) { return W(Nuv) / sqrt(x.degree(u) * x.degree(v)); };
  return predictLinksWithIntersection<MINDEGREE1, MAXFACTOR2, FORCEHEAP>(x, o, fs);
}


#ifdef OPENMP
/**
 * Predict links using Salton Cosine Similarity.
 * @tparam MINDEGREE1 degree of high degree first order neighbors to skip (if set)
 * @tparam MAXFACTOR2 maximum degree factor between source and second order neighbor to allow (if set)
 * @tparam FORCEHEAP always use heap to store top edges
 * @param x original graph
 * @param o predict link options
 * @returns [{u, v, score}] undirected predicted links, ordered by score (descending)
 */
template <int MINDEGREE1=4, int MAXFACTOR2=0, bool FORCEHEAP=false, class G, class W=float>
inline auto predictLinksSaltonCosineSimilarityOmp(const G& x, const PredictLinkOptions<W>& o={}) {
  auto fs = [&](auto u, auto v, auto Nuv) { return W(Nuv) / sqrt(x.degree(u) * x.degree(v)); };
  return predictLinksWithIntersectionOmp<MINDEGREE1, MAXFACTOR2, FORCEHEAP>(x, o, fs);
}
#endif
#pragma endregion




#pragma region PREDICT LINKS WITH HUB PROMOTED SCORE
/**
 * Predict links using Hub promoted score.
 * @tparam MINDEGREE1 degree of high degree first order neighbors to skip (if set)
 * @tparam MAXFACTOR2 maximum degree factor between source and second order neighbor to allow (if set)
 * @tparam FORCEHEAP always use heap to store top edges
 * @param x original graph
 * @param o predict link options
 * @returns [{u, v, score}] undirected predicted links, ordered by score (descending)
 */
template <int MINDEGREE1=4, int MAXFACTOR2=0, bool FORCEHEAP=false, class G, class W=float>
inline auto predictLinksHubPromoted(const G& x, const PredictLinkOptions<W>& o={}) {
  auto fs = [&](auto u, auto v, auto Nuv) { return W(Nuv) / min(x.degree(u), x.degree(v)); };
  return predictLinksWithIntersection<MINDEGREE1, MAXFACTOR2, FORCEHEAP>(x, o, fs);
}


#ifdef OPENMP
/**
 * Predict links using Hub promoted score.
 * @tparam MINDEGREE1 degree of high degree first order neighbors to skip (if set)
 * @tparam MAXFACTOR2 maximum degree factor between source and second order neighbor to allow (if set)
 * @tparam FORCEHEAP always use heap to store top edges
 * @param x original graph
 * @param o predict link options
 * @returns [{u, v, score}] undirected predicted links, ordered by score (descending)
 */
template <int MINDEGREE1=4, int MAXFACTOR2=0, bool FORCEHEAP=false, class G, class W=float>
inline auto predictLinksHubPromotedOmp(const G& x, const PredictLinkOptions<W>& o={}) {
  auto fs = [&](auto u, auto v, auto Nuv) { return W(Nuv) / min(x.degree(u), x.degree(v)); };
  return predictLinksWithIntersectionOmp<MINDEGREE1, MAXFACTOR2, FORCEHEAP>(x, o, fs);
}
#endif
#pragma endregion




#pragma region PREDICT LINKS WITH HUB DEPRESSED SCORE
/**
 * Predict links using Hub depressed score.
 * @tparam MINDEGREE1 degree of high degree first order neighbors to skip (if set)
 * @tparam MAXFACTOR2 maximum degree factor between source and second order neighbor to allow (if set)
 * @tparam FORCEHEAP always use heap to store top edges
 * @param x original graph
 * @param o predict link options
 * @returns [{u, v, score}] undirected predicted links, ordered by score (descending)
 */
template <int MINDEGREE1=4, int MAXFACTOR2=0, bool FORCEHEAP=false, class G, class W=float>
inline auto predictLinksHubDepressed(const G& x, const PredictLinkOptions<W>& o={}) {
  auto fs = [&](auto u, auto v, auto Nuv) { return W(Nuv) / max(x.degree(u), x.degree(v)); };
  return predictLinksWithIntersection<MINDEGREE1, MAXFACTOR2, FORCEHEAP>(x, o, fs);
}


#ifdef OPENMP
/**
 * Predict links using Hub depressed score.
 * @tparam MINDEGREE1 degree of high degree first order neighbors to skip (if set)
 * @tparam MAXFACTOR2 maximum degree factor between source and second order neighbor to allow (if set)
 * @tparam FORCEHEAP always use heap to store top edges
 * @param x original graph
 * @param o predict link options
 * @returns [{u, v, score}] undirected predicted links, ordered by score (descending)
 */
template <int MINDEGREE1=4, int MAXFACTOR2=0, bool FORCEHEAP=false, class G, class W=float>
inline auto predictLinksHubDepressedOmp(const G& x, const PredictLinkOptions<W>& o={}) {
  auto fs = [&](auto u, auto v, auto Nuv) { return W(Nuv) / max(x.degree(u), x.degree(v)); };
  return predictLinksWithIntersectionOmp<MINDEGREE1, MAXFACTOR2, FORCEHEAP>(x, o, fs);
}
#endif
#pragma endregion




#pragma region PREDICT LINKS WITH LEICHT-HOLME-NERMAN SCORE
/**
 * Predict links using Leicht-Holme-Nerman score.
 * @tparam MINDEGREE1 degree of high degree first order neighbors to skip (if set)
 * @tparam MAXFACTOR2 maximum degree factor between source and second order neighbor to allow (if set)
 * @tparam FORCEHEAP always use heap to store top edges
 * @param x original graph
 * @param o predict link options
 * @returns [{u, v, score}] undirected predicted links, ordered by score (descending)
 */
template <int MINDEGREE1=4, int MAXFACTOR2=0, bool FORCEHEAP=false, class G, class W=float>
inline auto predictLinksLeichtHolmeNermanScore(const G& x, const PredictLinkOptions<W>& o={}) {
  auto fs = [&](auto u, auto v, auto Nuv) { return W(Nuv) / (x.degree(u) * x.degree(v)); };
  return predictLinksWithIntersection<MINDEGREE1, MAXFACTOR2, FORCEHEAP>(x, o, fs);
}


#ifdef OPENMP
/**
 * Predict links using Leicht-Holme-Nerman score.
 * @tparam MINDEGREE1 degree of high degree first order neighbors to skip (if set)
 * @tparam MAXFACTOR2 maximum degree factor between source and second order neighbor to allow (if set)
 * @tparam FORCEHEAP always use heap to store top edges
 * @param x original graph
 * @param o predict link options
 * @returns [{u, v, score}] undirected predicted links, ordered by score (descending)
 */
template <int MINDEGREE1=4, int MAXFACTOR2=0, bool FORCEHEAP=false, class G, class W=float>
inline auto predictLinksLeichtHolmeNermanScoreOmp(const G& x, const PredictLinkOptions<W>& o={}) {
  auto fs = [&](auto u, auto v, auto Nuv) { return W(Nuv) / (x.degree(u) * x.degree(v)); };
  return predictLinksWithIntersectionOmp<MINDEGREE1, MAXFACTOR2, FORCEHEAP>(x, o, fs);
}
#endif
#pragma endregion




#pragma region PREDICT LINKS WITH ADAMIC-ADAR COEFFICIENT
/**
 * Predict links using Adamic-Adar Coefficient.
 * @tparam MINDEGREE1 degree of high degree first order neighbors to skip (if set)
 * @tparam MAXFACTOR2 maximum degree factor between source and second order neighbor to allow (if set)
 * @tparam FORCEHEAP always use heap to store top edges
 * @param x original graph
 * @param o predict link options
 * @returns [{u, v, score}] undirected predicted links, ordered by score (descending)
 */
template <int MINDEGREE1=4, int MAXFACTOR2=0, bool FORCEHEAP=false, class G, class W=float>
inline auto predictLinksAdamicAdarCoefficient(const G& x, const PredictLinkOptions<W>& o={}) {
  auto fu = [&](auto& entry, auto v) { entry += 1.0 / log(x.degree(v)); };  // What if degree(v) == 1?
  auto fs = [&](auto u, auto v, auto Nuv) { return W(Nuv); };
  return predictLinksWithIntersectionCustom<MINDEGREE1, MAXFACTOR2, FORCEHEAP, true>(x, o, fs, fu);
}


#ifdef OPENMP
/**
 * Predict links using Adamic-Adar Coefficient.
 * @tparam MINDEGREE1 degree of high degree first order neighbors to skip (if set)
 * @tparam MAXFACTOR2 maximum degree factor between source and second order neighbor to allow (if set)
 * @tparam FORCEHEAP always use heap to store top edges
 * @param x original graph
 * @param o predict link options
 * @returns [{u, v, score}] undirected predicted links, ordered by score (descending)
 */
template <int MINDEGREE1=4, int MAXFACTOR2=0, bool FORCEHEAP=false, class G, class W=float>
inline auto predictLinksAdamicAdarCoefficientOmp(const G& x, const PredictLinkOptions<W>& o={}) {
  auto fu = [&](auto& entry, auto v) { entry += 1.0 / log(x.degree(v)); };  // What if degree(v) == 1?
  auto fs = [&](auto u, auto v, auto Nuv) { return W(Nuv); };
  return predictLinksWithIntersectionCustomOmp<MINDEGREE1, MAXFACTOR2, FORCEHEAP, true>(x, o, fs, fu);
}
#endif
#pragma endregion




#pragma region PREDICT LINKS WITH RESOURCE ALLOCATION SCORE
/**
 * Predict links using Resource Allocation score.
 * @tparam MINDEGREE1 degree of high degree first order neighbors to skip (if set)
 * @tparam MAXFACTOR2 maximum degree factor between source and second order neighbor to allow (if set)
 * @tparam FORCEHEAP always use heap to store top edges
 * @param x original graph
 * @param o predict link options
 * @returns [{u, v, score}] undirected predicted links, ordered by score (descending)
 */
template <int MINDEGREE1=4, int MAXFACTOR2=0, bool FORCEHEAP=false, class G, class W=float>
inline auto predictLinksResourceAllocationScore(const G& x, const PredictLinkOptions<W>& o={}) {
  auto fu = [&](auto& entry, auto v) { entry += 1.0 / x.degree(v); };
  auto fs = [&](auto u, auto v, auto Nuv) { return W(Nuv); };
  return predictLinksWithIntersectionCustom<MINDEGREE1, MAXFACTOR2, FORCEHEAP, true>(x, o, fs, fu);
}


#ifdef OPENMP
/**
 * Predict links using Resource Allocation score.
 * @tparam MINDEGREE1 degree of high degree first order neighbors to skip (if set)
 * @tparam MAXFACTOR2 maximum degree factor between source and second order neighbor to allow (if set)
 * @tparam FORCEHEAP always use heap to store top edges
 * @param x original graph
 * @param o predict link options
 * @returns [{u, v, score}] undirected predicted links, ordered by score (descending)
 */
template <int MINDEGREE1=4, int MAXFACTOR2=0, bool FORCEHEAP=false, class G, class W=float>
inline auto predictLinksResourceAllocationScoreOmp(const G& x, const PredictLinkOptions<W>& o={}) {
  auto fu = [&](auto& entry, auto v) { entry += 1.0 / x.degree(v); };
  auto fs = [&](auto u, auto v, auto Nuv) { return W(Nuv); };
  return predictLinksWithIntersectionCustomOmp<MINDEGREE1, MAXFACTOR2, FORCEHEAP, true>(x, o, fs, fu);
}
#endif
#pragma endregion
#pragma endregion
