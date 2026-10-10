"""Coarse-to-fine pyramid block selection."""

from __future__ import annotations

import numpy as np

from pisa_attn.pyramid import build_pyramid, lse_score


class BlockSelection(list):
    """Selected leaf-block indices per query.

    With one KV head this is a list of length T. Each item is a 1d int array
    of leaf block indices, sorted ascending. With several KV heads this is a
    list of length Hkv of those per-query lists. Forced blocks are included
    and do not displace blocks already chosen by the top-k ranking.

    ``score_counts`` is the number of candidates scored with LogSumExp, per
    query (shape ``(T,)`` or ``(Hkv, T)``).
    """

    def __init__(
        self,
        indices,
        *,
        block_size: int,
        group_factor: int,
        n_levels: int,
        score_counts,
        n_kv: int,
        seq_len: int,
        n_queries: int,
    ):
        super().__init__(indices)
        self.block_size = int(block_size)
        self.group_factor = int(group_factor)
        self.n_levels = int(n_levels)
        self.score_counts = np.asarray(score_counts, dtype=np.int32)
        self.n_score_evaluations = self.score_counts
        self.n_kv = int(n_kv)
        self.seq_len = int(seq_len)
        self.n_queries = int(n_queries)

    def __repr__(self) -> str:
        return (
            f"BlockSelection(n={len(self)}, block_size={self.block_size}, "
            f"n_levels={self.n_levels}, n_kv={self.n_kv})"
        )


def _normalize_q(queries):
    queries = np.asarray(queries)
    if queries.ndim == 2:
        return queries[None, :, :], True
    if queries.ndim == 3:
        return queries, False
    raise ValueError("queries must have shape (T, d) or (Hq, T, d)")


def _forced_leaf_indices(query_pos: int, n_leaves: int, block_size: int, causal: bool):
    """First, previous, and current leaf blocks, if they exist and are eligible."""
    if n_leaves <= 0 or query_pos < 0:
        return []
    current = query_pos // block_size
    wanted = [0]
    if current - 1 >= 0:
        wanted.append(current - 1)
    if current >= 0:
        wanted.append(current)
    out = []
    for block in wanted:
        if block < 0 or block >= n_leaves or block in out:
            continue
        start = block * block_size
        if causal and start > query_pos:
            continue
        out.append(block)
    return out


def _stable_topk(candidates, scores, top_k: int):
    """Highest score first. Exact ties keep the lower block index."""
    order = sorted(
        range(len(candidates)),
        key=lambda i: (-float(scores[i]), int(candidates[i])),
    )
    return [candidates[i] for i in order[:top_k]]


def _select_one(pyramid, query_vec, query_pos: int, top_k: int, causal: bool, forced: bool, kv_head: int):
    n_leaves = pyramid.n_blocks[0]
    if top_k <= 0:
        selected = []
        if forced:
            selected = _forced_leaf_indices(query_pos, n_leaves, pyramid.block_size, causal)
        return np.asarray(sorted(set(selected)), dtype=np.int32), 0

    level = pyramid.n_levels - 1
    candidates = [0]
    score_count = 0
    while True:
        if causal:
            candidates = [
                c
                for c in candidates
                if int(pyramid.starts[level][c]) <= query_pos
            ]
        # Keep every candidate without scoring when the shortlist already fits.
        if len(candidates) > top_k:
            scores = []
            for cand in candidates:
                child_keys = pyramid.child_keys(level, cand, kv_head=kv_head)
                scores.append(lse_score(query_vec, child_keys))
                score_count += 1
            candidates = _stable_topk(candidates, scores, top_k)
        if level == 0:
            selected = list(candidates)
            break
        expanded = []
        for cand in candidates:
            expanded.extend(pyramid.children(level, cand))
        candidates = expanded
        level -= 1

    if forced:
        # Union, so a forced block never displaces a ranked block.
        for block in _forced_leaf_indices(query_pos, n_leaves, pyramid.block_size, causal):
            if block not in selected:
                selected.append(block)
    selected = sorted(set(int(b) for b in selected))
    return np.asarray(selected, dtype=np.int32), score_count


def select_blocks(
    queries,
    keys,
    block_size,
    top_k,
    group_factor=2,
    causal=True,
    forced=True,
    debug=False,
    return_stats=False,
):
    """Select leaf blocks per query with coarse-to-fine LogSumExp ranking.

    ``queries`` is ``(T, d)`` or ``(Hq, T, d)``. ``keys`` is ``(S, d)`` or
    ``(Hkv, S, d)``. Query heads that share a KV head have their per-head
    LogSumExp scores summed before SelectK. Pass ``debug=True`` (or
    ``return_stats=True``) to also return per-query score-evaluation counts.
    """
    block_size = int(block_size)
    top_k = int(top_k)
    group_factor = int(group_factor)
    q, q_squeezed = _normalize_q(queries)
    pyramid = build_pyramid(keys, block_size, group_factor)
    n_heads, n_queries, head_dim = q.shape
    if head_dim != pyramid.head_dim:
        raise ValueError("query and key head dimensions differ")
    if pyramid.n_kv == 1:
        groups = [list(range(n_heads))]
    else:
        if n_heads % pyramid.n_kv != 0:
            raise ValueError("Hq must be divisible by Hkv for GQA")
        group_size = n_heads // pyramid.n_kv
        groups = [
            list(range(g * group_size, (g + 1) * group_size))
            for g in range(pyramid.n_kv)
        ]

    per_kv = []
    counts = np.zeros((pyramid.n_kv, n_queries), dtype=np.int32)
    for kv_head, heads in enumerate(groups):
        chosen = []
        q_group = q[heads]  # (G, T, d)
        for t in range(n_queries):
            query_vec = q_group[:, t, :]
            if query_vec.shape[0] == 1 and q_squeezed:
                query_vec = query_vec[0]
            indices, n_scored = _select_one(
                pyramid,
                query_vec,
                t,
                top_k,
                bool(causal),
                bool(forced),
                kv_head,
            )
            chosen.append(indices)
            counts[kv_head, t] = n_scored
        per_kv.append(chosen)

    if pyramid.n_kv == 1:
        selection = BlockSelection(
            per_kv[0],
            block_size=block_size,
            group_factor=group_factor,
            n_levels=pyramid.n_levels,
            score_counts=counts[0],
            n_kv=1,
            seq_len=pyramid.seq_len,
            n_queries=n_queries,
        )
    else:
        inner = [
            BlockSelection(
                per_kv[h],
                block_size=block_size,
                group_factor=group_factor,
                n_levels=pyramid.n_levels,
                score_counts=counts[h],
                n_kv=1,
                seq_len=pyramid.seq_len,
                n_queries=n_queries,
            )
            for h in range(pyramid.n_kv)
        ]
        selection = BlockSelection(
            inner,
            block_size=block_size,
            group_factor=group_factor,
            n_levels=pyramid.n_levels,
            score_counts=counts,
            n_kv=pyramid.n_kv,
            seq_len=pyramid.seq_len,
            n_queries=n_queries,
        )
    if debug or return_stats:
        return selection, selection.score_counts
    return selection
