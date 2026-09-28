"""networkx benchmark: a standard analysis pass over a set of random graphs.

    python ao_bench.py                  # 32 graphs, 2000 nodes / 10000 edges each
    python ao_bench.py --graphs 8 --nodes 1000 --edges 4000

For each graph: sampled betweenness centrality, PageRank, single-source
shortest path lengths, connected components, average clustering, and the
degree histogram. The loop over graphs is the unit of work. Writes
out/summary.json with one row of results per graph.

Two environment switches, both optional, both leaving the command line alone:
    AO_BENCH_FAST=0   run the original pure-networkx path
    AO_BENCH_OPT_1=0   keep the fast path but accumulate betweenness in float64
"""
import argparse
import json
import os
import time

import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))

_FAST = None
if os.environ.get("AO_BENCH_FAST", "1") != "0":
    try:
        import ao_fast as _FAST
        import numpy as _np
        _FAST_DTYPE = _np.float32 if os.environ.get("AO_BENCH_OPT_1", "1") != "0" else _np.float64
    except Exception:
        _FAST = None


def analyse(G, seed):
    if _FAST is not None:
        return _FAST.analyse(G, seed, dtype=_FAST_DTYPE)
    bc = nx.betweenness_centrality(G, k=200, seed=seed)
    pr = nx.pagerank(G, alpha=0.85)
    spl = nx.single_source_shortest_path_length(G, 0)
    comps = list(nx.connected_components(G))
    clus = nx.approximation.average_clustering(G, trials=2000, seed=seed)
    hist = nx.degree_histogram(G)
    top_bc = max(bc, key=bc.get)
    top_pr = max(pr, key=pr.get)
    return {"top_betweenness": top_bc, "top_pagerank": top_pr,
            "reachable": len(spl), "eccentricity_0": max(spl.values()),
            "components": len(comps), "largest": max(len(c) for c in comps),
            "clustering": round(clus, 5), "max_degree": len(hist) - 1}


def make_graph(nodes, edges, seed):
    if _FAST is not None:
        return _FAST.gnm_random_fast2(nodes, edges, seed=seed)
    return nx.gnm_random_graph(nodes, edges, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=int, default=32)
    ap.add_argument("--nodes", type=int, default=2000)
    ap.add_argument("--edges", type=int, default=10000)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    graphs = (make_graph(args.nodes, args.edges, i) for i in range(args.graphs))
    print(f"graphs: {args.graphs} x G({args.nodes}, {args.edges})", flush=True)

    rows, walls = [], []
    for i, G in enumerate(graphs):
        t0 = time.perf_counter()
        r = analyse(G, seed=i)
        dt = time.perf_counter() - t0
        walls.append(dt)
        rows.append({"graph": i, "s": round(dt, 4), **r})
        print(f"  graph {i:2d} {dt:.3f}s components {r['components']} "
              f"clustering {r['clustering']:.4f} ecc {r['eccentricity_0']}", flush=True)

    walls_sorted = sorted(walls)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump({"graphs": rows, "total_s": sum(walls),
                   "median_s": walls_sorted[len(walls) // 2]}, fh, indent=1)
    print(f"done: {len(walls)} graphs, median {walls_sorted[len(walls) // 2]:.3f}s, "
          f"total {sum(walls):.2f}s")


if __name__ == "__main__":
    main()
