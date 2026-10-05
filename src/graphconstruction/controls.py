"""
Control graphs for testing whether a model actually uses the spatial structure.

- ``identity_graph``: every node connected only to itself. In ``HHH4Module`` the
  self-loops are removed, so the neighbourhood branch sees no neighbours and
  becomes a learned constant.
- ``rewired_graph``: degree-preserving rewiring (double-edge swaps). Every node
  keeps the same number of neighbours, but which regions are neighbours is
  scrambled. This keeps the capacity of the neighbourhood branch and removes the
  geography, so it is the stronger control.
- ``complete_graph``: everyone connected to everyone, i.e. a national average.

Compare the real graph against several rewired draws: the rank of the real graph
among them gives a permutation p-value, p = (1 + #{rewired >= real}) / (B + 1)
for a score where higher is better (for WIS, count rewired <= real).
"""
from __future__ import annotations

import numpy as np
import torch

from .graphobjects.structure import GraphStructure


def identity_graph(num_nodes: int) -> GraphStructure:
    """Self-loops only."""
    idx = torch.arange(num_nodes)
    return GraphStructure(torch.stack([idx, idx]), torch.ones(num_nodes), num_nodes)


def complete_graph(num_nodes: int) -> GraphStructure:
    """All pairs (i != j), unit weights."""
    i, j = np.where(~np.eye(num_nodes, dtype=bool))
    ei = torch.tensor(np.stack([i, j]), dtype=torch.long)
    return GraphStructure(ei, torch.ones(ei.shape[1]), num_nodes)


def _is_symmetric(graph: GraphStructure) -> bool:
    edges = set(map(tuple, graph.edge_index.t().tolist()))
    return all((b, a) in edges for a, b in edges)


def rewired_graph(graph: GraphStructure,
                  seed: int = 0,
                  swaps_per_edge: int = 10) -> GraphStructure:
    """
    Degree-preserving rewiring by repeated double-edge swaps.

    For an undirected (symmetric) graph, two edges (a-b, c-d) become (a-d, c-b)
    when that creates no self-loop and no duplicate edge. Each edge keeps its
    weight. Self-loops in the input are kept as they are. Directed graphs are
    rewired the same way on directed edges (out- and in-degrees preserved).

    Parameters
    ----------
    graph : GraphStructure
        The real graph.
    seed : int
        Random seed; use several seeds to get several control graphs.
    swaps_per_edge : int
        Swap attempts per edge. 10 is plenty for the graph to forget its geography.
    """
    rng = np.random.default_rng(seed)
    ei  = graph.edge_index.cpu().numpy()
    ew  = graph.edge_weight.cpu().numpy()

    loops = ei[0] == ei[1]
    symmetric = _is_symmetric(graph)

    if symmetric:
        keep = (ei[0] < ei[1]) & ~loops                  # one entry per undirected edge
    else:
        keep = ~loops

    edges   = [tuple(e) for e in ei[:, keep].T]
    weights = list(ew[keep])
    present = set(edges)
    if symmetric:
        present |= {(b, a) for a, b in edges}

    m = len(edges)
    if m >= 2:
        for _ in range(swaps_per_edge * m):
            i, j = rng.integers(0, m, size=2)
            if i == j:
                continue
            a, b = edges[i]
            c, d = edges[j]
            if symmetric and rng.random() < 0.5:         # undirected: pick an orientation
                c, d = d, c
            if len({a, b, c, d}) < 4:
                continue
            new1, new2 = (a, d), (c, b)
            if new1 in present or new2 in present:
                continue
            if symmetric and ((d, a) in present or (b, c) in present):
                continue

            for e in [(a, b), (c, d)] + ([(b, a), (d, c)] if symmetric else []):
                present.discard(e)
            for e in [new1, new2] + ([(d, a), (b, c)] if symmetric else []):
                present.add(e)
            edges[i], edges[j] = new1, new2

    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    w   = list(weights)
    if symmetric:
        src, dst, w = src + dst, dst + src, w + w

    loop_idx = np.where(loops)[0]
    src += list(ei[0, loop_idx]); dst += list(ei[1, loop_idx]); w += list(ew[loop_idx])

    return GraphStructure(torch.tensor([src, dst], dtype=torch.long),
                          torch.tensor(w, dtype=torch.float32),
                          graph.num_nodes)


def degree_sequence(graph: GraphStructure) -> np.ndarray:
    """Out-degree per node, excluding self-loops."""
    ei = graph.edge_index.cpu().numpy()
    ei = ei[:, ei[0] != ei[1]]
    return np.bincount(ei[0], minlength=graph.num_nodes)
