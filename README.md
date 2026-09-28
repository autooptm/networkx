<div align="center">
  <a href="https://autooptm.com"><img src=".autooptm/logo.png" width="96" alt="AutoOptm"></a>

  <h1>networkx · optimized by <a href="https://autooptm.com">AutoOptm</a></h1>

  <p><b>14.65x faster end to end</b> on the command below, output verified against the stock program.</p>

  <p>
    <a href="https://autooptm.com"><img alt="speedup" src="https://img.shields.io/badge/end--to--end-14.65x-2ea44f"></a>
    <a href="https://github.com/networkx/networkx/commit/4e74880b0da01977da79915167c64e5c2af38b47"><img alt="base" src="https://img.shields.io/badge/upstream-4e74880b0da0-blue"></a>
    <img alt="card" src="https://img.shields.io/badge/measured%20on-CPU%20only%20no%20card-lightgrey">
  </p>
</div>

> This is a fork of [networkx/networkx](https://github.com/networkx/networkx) at commit
> [`4e74880b0da0`](https://github.com/networkx/networkx/commit/4e74880b0da01977da79915167c64e5c2af38b47) with a benchmark driver (`ao_bench.py`) and the AutoOptm patch applied on top.
> The optimisation was found, measured and verified automatically by [AutoOptm](https://autooptm.com);
> the patch is kept under [`.autooptm/`](.autooptm/).

Every optimisation is on by default and the command runs unchanged — same file, same flags, same outputs. Every change is behind a switch that defaults on; see `.autooptm/autooptm.patch`.

## The result — `python ao_bench.py`

| | |
|---|---|
| **Command** | `python ao_bench.py` |
| **Entry point** | `ao_bench.py` |
| **Unit measured** | one random graph through ao_bench.py's six analyses (betweenness, pagerank, clustering, …); CPU only |
| **Before (stock)** | 1166 ms per unit |
| **After (this tree, all switches default ON)** | 78.4 ms per unit |
| **Speedup** | **14.65x** end to end on CPU only (no card), host noise floor 0.6% |
| **Output** | default tree: every reported field identical across 10 checked graphs (one declared numeric deviation, in the path-count accumulation); one switch gives the bit-exact path at 11.15x |

### What changed

| File | Where | Gain (alone) |
|---|---|---|
| `ao_bench.py` | analyse() | 1.0x |
| `ao_fast.py` | new module | 8.68x |
| `ao_bench.py` | make_graph() | 1.16x |
| `ao_fast.py` | analyse() | 1.04x |
| `ao_fast.py` | betweenness_top(dtype) | 1.2x |
| `ao_bench.py` | main() | 1.0x |


## Reproduce

```bash
git clone https://github.com/autooptm/networkx-ao.git
cd networkx-ao
# set up exactly as upstream documents, then:
python ao_bench.py
```

Everything AutoOptm added is the single commit on top of upstream: the benchmark driver `ao_bench.py` and the module `ao_fast.py`, both added by this fork, and the optimisation; `git diff 4e74880b0da0` is the same change as the patch file under `.autooptm/`.

---

<div align="center"><sub>Optimized by <a href="https://autooptm.com">AutoOptm</a> — point it at a repository, get back a verified speedup and the patch.</sub></div>

---

The upstream README is [`README.rst`](README.rst), unchanged.

