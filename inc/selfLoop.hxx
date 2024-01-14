#pragma once
#include "update.hxx"




#pragma region METHODS
#pragma region COUNT SELF-LOOPS
/**
 * Count the number of self-loops in a graph.
 * @param x input graph
 * @returns number of self-loops
 */
template <class G>
inline size_t countSelfLoops(const G& x) {
  size_t a = 0;
  x.forEachVertexKey([&](auto u) { if (x.hasEdge(u, u)) ++a; });
  return a;
}
#pragma endregion




#pragma region ADD SELF-LOOPS
/**
 * Add self-loops to a graph.
 * @param a graph to add self-loops to (updated)
 * @param w edge weight of self-loops
 * @param ft test function to determine if self-loop should be added (vertex)
 */
template <class G, class E, class FT>
inline void addSelfLoopsU(G& a, E w, FT ft) {
  a.forEachVertexKey([&](auto u) { if (ft(u)) a.addEdge(u, u, w); });
  a.update();
}

/**
 * Add self-loops to a graph.
 * @param a input graph
 * @param w edge weight of self-loops
 * @param ft test function to determine if self-loop should be added (vertex)
 * @returns graph with self-loops added
 */
template <class G, class E, class FT>
inline G addSelfLoops(const G& x, E w, FT ft) {
  G a = x; addSelfLoopsU(a, w, ft);
  return a;
}


#ifdef OPENMP
/**
 * Add self-loops to a graph in parallel.
 * @param a graph to add self-loops to (updated)
 * @param w edge weight of self-loops
 * @param ft test function to determine if self-loop should be added (vertex)
 */
template <class G, class E, class FT>
inline void addSelfLoopsOmpU(G& a, E w, FT ft) {
  #pragma omp parallel
  {
    a.forEachVertexKey([&](auto u) { if (ft(u)) addEdgeOmpU(a, u, u, w); });
  }
  updateOmpU(a);
}

/**
 * Add self-loops to a graph in parallel.
 * @param a input graph
 * @param w edge weight of self-loops
 * @param ft test function to determine if self-loop should be added (vertex)
 * @returns graph with self-loops added
 */
template <class G, class E, class FT>
inline G addSelfLoopsOmp(const G& x, E w, FT ft) {
  G a = x; addSelfLoopsOmpU(a, w, ft);
  return a;
}
#endif
#pragma endregion




#pragma region REMOVE SELF-LOOPS
/**
 * Remove self-loops from a graph.
 * @param a graph to remove self-loops from (updated)
 * @param ft test function to determine if self-loop should be removed (vertex)
 */
template <class G, class FT>
inline void removeSelfLoopsU(G& a, FT ft) {
  a.forEachVertexKey([&](auto u) { if (ft(u)) a.removeEdge(u, u); });
  a.update();
}

/**
 * Remove self-loops from a graph.
 * @param a input graph
 * @param ft test function to determine if self-loop should be removed (vertex)
 * @returns graph with self-loops removed
 */
template <class G, class FT>
inline G removeSelfLoops(const G& x, FT ft) {
  G a = x; removeSelfLoopsU(a, ft);
  return a;
}


#ifdef OPENMP
/**
 * Remove self-loops from a graph in parallel.
 * @param a graph to remove self-loops from (updated)
 * @param ft test function to determine if self-loop should be removed (vertex)
 */
template <class G, class FT>
inline void removeSelfLoopsOmpU(G& a, FT ft) {
  #pragma omp parallel
  {
    a.forEachVertexKey([&](auto u) { if (ft(u)) removeEdgeOmpU(a, u, u); });
  }
  updateOmpU(a);
}

/**
 * Remove self-loops from a graph in parallel.
 * @param a input graph
 * @param ft test function to determine if self-loop should be removed (vertex)
 * @returns graph with self-loops removed
 */
template <class G, class E, class FT>
inline G removeSelfLoopsOmp(const G& x, FT ft) {
  G a = x; removeSelfLoopsOmpU(a, ft);
  return a;
}
#endif
#pragma endregion
#pragma endregion
