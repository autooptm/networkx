from __future__ import annotations

import random

try:
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.csgraph as csgraph
    _HAVE_SCIPY = True
except Exception:                                     # pragma: no cover
    _HAVE_SCIPY = False

import networkx as nx

# Chunk of pivots processed per sparse product. The dense working set is
# chunk x n float64; 64 x 2000 is ~1 MB, which keeps this comfortably small on a
# shared host while still amortizing the sparse product over many sources.
DEFAULT_CHUNK = 64


class FastGraph:
    """The graph ao_bench.py's analysis actually needs, without the dict-of-dicts.

    networkx's Graph keeps an adjacency dict-of-dicts: for G(2000, 10000) that is
    2000 outer dicts holding 20000 inner one-entry dicts, all built one
    `add_edge` at a time. Every consumer in this benchmark wants one of only two
    things from it -- a CSR matrix, or each node's neighbour list -- so this
    holds exactly those two and nothing else.

    It deliberately implements the small slice of the nx.Graph interface the
    functions below already use (`len`, iteration over nodes, `nodes()`,
    `number_of_nodes()`, `G[v]`), so they work on it unchanged.
    """

    __slots__ = ("n", "adj", "A")

    def __init__(self, n, adj, A):
        self.n = n
        self.adj = adj
        self.A = A

    def __len__(self):
        return self.n

    def __iter__(self):
        return iter(range(self.n))

    def __getitem__(self, v):
        return self.adj[v]

    def nodes(self):
        return range(self.n)

    def number_of_nodes(self):
        return self.n


def gnm_random_fast(n, m, seed=None):
    # n <= 256 would put k=200 betweenness pivots at or above the node count,
    # where betweenness_top hands the graph back to networkx -- which cannot read
    # a FastGraph. Such graphs are small enough that the stock path is free.
    if not _HAVE_SCIPY or n <= 256 or m >= n * (n - 1) / 2.0:
        return nx.gnm_random_graph(n, m, seed=seed)
    rng = random.Random(seed)
    nlist = list(range(n))
    adj = [[] for _ in range(n)]
    seen = set()
    us = np.empty(m, dtype=np.int32)
    vs = np.empty(m, dtype=np.int32)
    count = 0
    while count < m:
        u = rng.choice(nlist)
        v = rng.choice(nlist)
        if u == v:
            continue
        key = (u, v) if u < v else (v, u)
        if key in seen:
            continue
        seen.add(key)
        adj[u].append(v)
        adj[v].append(u)
        us[count] = u
        vs[count] = v
        count += 1
    rows = np.concatenate((us, vs))
    cols = np.concatenate((vs, us))
    A = sp.csr_array((np.ones(2 * m, dtype=np.float64), (rows, cols)), shape=(n, n))
    return FastGraph(n, adj, A)


def _linear_nodes(G):
    """True when G's nodes are exactly 0..n-1 in order, so a node IS its CSR index.
    gnm_random_graph always produces that; anything else takes the stock path."""
    n = G.number_of_nodes()
    if n == 0:
        return False
    it = iter(G)
    for expect in range(n):
        if next(it, None) != expect:
            return False
    return True


def csr_of(G):
    """CSR adjacency (float64, unweighted) with node i at index i."""
    return nx.to_scipy_sparse_array(G, nodelist=range(G.number_of_nodes()),
                                    dtype=np.float64, weight=None, format="csr")


def betweenness_top(G, k, seed, A=None, chunk=DEFAULT_CHUNK, dtype=np.float64 if _HAVE_SCIPY else None):
    """The node id `max(nx.betweenness_centrality(G, k=k, seed=seed), key=...)`.

    Reproduces networkx exactly:
      * pivots            random.Random(seed).sample(list(G.nodes()), k)  -- the
                          same call networkx makes through @py_random_state;
      * per pivot         Brandes: BFS shortest-path counts, then dependency
                          accumulation in reverse level order;
      * rescaling         the k-sampled, endpoints=False branch of _rescale:
                          1/((k-1)(n-2)) for a pivot, 1/(k(n-2)) for anything else;
      * tie-break         np.argmax returns the FIRST maximal index, which is what
                          max(dict, key=dict.get) does over 0..n-1 insertion order.
    """
    n = G.number_of_nodes()
    if not _HAVE_SCIPY or k is None or k >= n or n < 3 or not _linear_nodes(G):
        bc = nx.betweenness_centrality(G, k=k, seed=seed)
        return max(bc, key=bc.get)

    sources = random.Random(seed).sample(list(G.nodes()), k)
    if A is None:
        A = csr_of(G)
    if dtype is not None and A.dtype != dtype:
        A = A.astype(dtype)
    src = np.asarray(sources, dtype=np.int64)

    bc = np.zeros(n, dtype=np.float64)
    for start in range(0, k, chunk):
        blk = src[start:start + chunk]
        b = blk.size
        rows = np.arange(b)

        dist = np.full((b, n), -1, dtype=np.int32)
        sigma = np.zeros((b, n), dtype=dtype)
        dist[rows, blk] = 0
        sigma[rows, blk] = 1.0

        cur = np.zeros((b, n), dtype=bool)
        cur[rows, blk] = True
        masks = [cur]
        level = 0
        while True:
            # paths arriving one level out: for each node w, the sum of sigma[v]
            # over its neighbours v on the current level -- which IS sigma[w] when
            # w is being discovered now.
            nxt = (A @ np.where(cur, sigma, 0.0).T).T
            new = (dist < 0) & (nxt > 0.0)
            if not new.any():
                break
            level += 1
            dist[new] = level
            sigma[new] = nxt[new]
            masks.append(new)
            cur = new

        # Dependency accumulation, level by level from the deepest inwards. Every
        # node of level L has its final delta before level L-1 is touched, which is
        # exactly the invariant networkx's reverse-BFS stack pop relies on.
        delta = np.zeros((b, n), dtype=dtype)
        safe_sigma = np.where(sigma > 0.0, sigma, 1.0)
        for L in range(level, 0, -1):
            coeff = np.where(masks[L], (1.0 + delta) / safe_sigma, 0.0)
            contrib = (A @ coeff.T).T
            delta += np.where(masks[L - 1], sigma * contrib, 0.0)

        # networkx adds delta[w] for every reached w EXCEPT the source itself.
        bc += delta.sum(axis=0)
        bc[blk] -= delta[rows, blk]

    scale_source = 1.0 / ((k - 1) * (n - 2))
    scale_nonsource = 1.0 / (k * (n - 2))
    scales = np.full(n, scale_nonsource, dtype=np.float64)
    scales[src] = scale_source
    bc *= scales
    return int(np.argmax(bc))


def pagerank_top(G, alpha=0.85, A=None):
    """`max(nx.pagerank(G, alpha=alpha), key=...)`.

    nx.pagerank already dispatches to scipy; the only thing saved here is the
    second construction of the adjacency matrix, which the caller has built
    already for the betweenness pass.
    """
    if not _HAVE_SCIPY or A is None or not _linear_nodes(G):
        pr = nx.pagerank(G, alpha=alpha)
        return max(pr, key=pr.get)
    n = G.number_of_nodes()
    # Same formulation as networkx's _pagerank_scipy: row-normalized adjacency,
    # uniform personalization and dangling distribution, tol 1e-6, max_iter 100.
    S = np.asarray(A.sum(axis=1)).ravel()
    S_inv = np.where(S != 0, 1.0 / np.where(S != 0, S, 1.0), 0.0)
    Q = sp.csr_array(sp.dia_array((S_inv, 0), shape=(n, n)))
    M = (Q @ A).T
    x = np.full(n, 1.0 / n)
    p = np.full(n, 1.0 / n)
    dangling_weights = p
    is_dangling = np.where(S == 0)[0]
    for _ in range(100):
        xlast = x
        x = alpha * (M @ x + sum(x[is_dangling]) * dangling_weights) + (1 - alpha) * p
        if np.abs(x - xlast).sum() < n * 1.0e-6:
            break
    else:
        pr = nx.pagerank(G, alpha=alpha)
        return max(pr, key=pr.get)
    return int(np.argmax(x))


def components_stats(G, A=None):
    """(number of connected components, size of the largest) -- the two things
    ao_bench.py takes from `list(nx.connected_components(G))`."""
    if not _HAVE_SCIPY or A is None or not _linear_nodes(G):
        comps = list(nx.connected_components(G))
        return len(comps), max(len(c) for c in comps)
    ncomp, labels = csgraph.connected_components(A, directed=False)
    return int(ncomp), int(np.bincount(labels).max())


def sssp_stats(G, source=0, A=None):
    """(reachable count, eccentricity of `source`) -- what ao_bench.py takes from
    `nx.single_source_shortest_path_length(G, 0)` via len() and max(values())."""
    if not _HAVE_SCIPY or A is None or not _linear_nodes(G):
        spl = nx.single_source_shortest_path_length(G, source)
        return len(spl), max(spl.values())
    dist = csgraph.dijkstra(A, directed=False, unweighted=True, indices=source)
    finite = np.isfinite(dist)
    return int(finite.sum()), int(dist[finite].max())


def max_degree(G, A=None):
    """`len(nx.degree_histogram(G)) - 1` -- the largest degree present."""
    if not _HAVE_SCIPY or A is None or not _linear_nodes(G):
        return len(nx.degree_histogram(G)) - 1
    return int(np.asarray(A.sum(axis=1)).ravel().max())


def average_clustering(G, trials, seed):
    """`nx.approximation.average_clustering(G, trials=trials, seed=seed)`.

    THE DRAWS MUST MATCH EXACTLY, so this keeps the stock structure: the same
    `random.Random(seed)`, the same list comprehension that pulls all `trials`
    floats FIRST, then one `sample(nbrs, 2)` per trial that has >= 2 neighbours.
    The only change is that neighbour lists come from a prebuilt list-of-lists in
    the graph's own adjacency-insertion order -- the order `list(G[v])` yields --
    and the membership test is a set lookup instead of a dict lookup.
    """
    if not _linear_nodes(G):
        return nx.approximation.average_clustering(G, trials=trials, seed=seed)
    rng = random.Random(seed)
    n = len(G)
    nodes = list(G)
    adj = [list(G[v]) for v in nodes]
    adjset = [set(a) for a in adj]
    triangles = 0
    for i in [int(rng.random() * n) for _ in range(trials)]:
        nbrs = adj[i]
        if len(nbrs) < 2:
            continue
        u, v = rng.sample(nbrs, 2)
        if u in adjset[v]:
            triangles += 1
    return triangles / trials


def average_clustering_lists(G, trials, seed):
    """Same estimator as `average_clustering` above, without the per-unit set build.

    `clusfast` lost 9% by constructing n neighbour SETS (2000 of them for
    G(2000, 10000)) to serve only `trials` membership tests. At mean degree 10 a
    list scan `u in adj[v]` is cheaper than building the set that would replace
    it, so this keeps the lists and pays the scan. Draws are unchanged.
    """
    if not _linear_nodes(G):
        return nx.approximation.average_clustering(G, trials=trials, seed=seed)
    rng = random.Random(seed)
    n = len(G)
    adj = G.adj if isinstance(G, FastGraph) else [list(G[v]) for v in range(n)]
    triangles = 0
    for i in [int(rng.random() * n) for _ in range(trials)]:
        nbrs = adj[i]
        if len(nbrs) < 2:
            continue
        u, v = rng.sample(nbrs, 2)
        if u in adj[v]:
            triangles += 1
    return triangles / trials


def gnm_random_fast2(n, m, seed=None):
    if not _HAVE_SCIPY or n <= 256 or m >= n * (n - 1) / 2.0:
        return nx.gnm_random_graph(n, m, seed=seed)
    rng = random.Random(seed)
    try:
        below = rng._randbelow
        below(n)
        rng = random.Random(seed)
        below = rng._randbelow
    except Exception:
        return gnm_random_fast(n, m, seed=seed)

    adj = [[] for _ in range(n)]
    seen = set()
    us = np.empty(m, dtype=np.int32)
    vs = np.empty(m, dtype=np.int32)
    count = 0
    while count < m:
        u = below(n)
        v = below(n)
        if u == v:
            continue
        key = (u, v) if u < v else (v, u)
        if key in seen:
            continue
        seen.add(key)
        adj[u].append(v)
        adj[v].append(u)
        us[count] = u
        vs[count] = v
        count += 1
    rows = np.concatenate((us, vs))
    cols = np.concatenate((vs, us))
    A = sp.csr_array((np.ones(2 * m, dtype=np.float64), (rows, cols)), shape=(n, n))
    return FastGraph(n, adj, A)


def gnm_random_fast3(n, m, seed=None):
    if not _HAVE_SCIPY or n <= 256 or m >= n * (n - 1) / 2.0:
        return nx.gnm_random_graph(n, m, seed=seed)
    rng = random.Random(seed)
    nlist = list(range(n))
    choice = rng.choice
    adj = [[] for _ in range(n)]
    seen = set()
    seen_add = seen.add
    us = []
    vs = []
    ua = us.append
    va = vs.append
    count = 0
    while count < m:
        u = choice(nlist)
        v = choice(nlist)
        if u == v:
            continue
        key = u * n + v if u < v else v * n + u
        if key in seen:
            continue
        seen_add(key)
        adj[u].append(v)
        adj[v].append(u)
        ua(u)
        va(v)
        count += 1
    us = np.fromiter(us, dtype=np.int32, count=m)
    vs = np.fromiter(vs, dtype=np.int32, count=m)
    rows = np.concatenate((us, vs))
    cols = np.concatenate((vs, us))
    A = sp.csr_array((np.ones(2 * m, dtype=np.float64), (rows, cols)), shape=(n, n))
    return FastGraph(n, adj, A)


def analyse(G, seed, dtype=None):
    if dtype is None:
        dtype = np.float64
    A = getattr(G, "A", None)
    if A is None:
        A = csr_of(G)
    top_bc = betweenness_top(G, k=200, seed=seed, A=A, dtype=dtype)
    top_pr = pagerank_top(G, alpha=0.85, A=A)
    reachable, ecc = sssp_stats(G, 0, A=A)
    ncomp, largest = components_stats(G, A=A)
    clus = average_clustering_lists(G, trials=2000, seed=seed)
    return {"top_betweenness": top_bc, "top_pagerank": top_pr,
            "reachable": reachable, "eccentricity_0": ecc,
            "components": ncomp, "largest": largest,
            "clustering": round(clus, 5), "max_degree": max_degree(G, A=A)}
