"""Mean-pooled key pyramid used by PISA block selection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.nn as jnn
import jax.numpy as jnp


def n_pyramid_levels(seq_len: int, block_size: int, group_factor: int = 2) -> int:
    """Number of pyramid levels, including the leaf level and the root."""
    if seq_len <= 0:
        return 0
    n = math.ceil(seq_len / block_size)
    levels = 1
    while n > 1:
        n = math.ceil(n / group_factor)
        levels += 1
    return levels


def lse_score(query, child_keys):
    """LogSumExp attention score of a block from its children.

    ``s = logsumexp(q · k_child / sqrt(d))``. For several query heads that
    share one KV head, scores are summed across heads before SelectK.
    ``query`` is ``(d,)`` or ``(H, d)``. ``child_keys`` is ``(C, d)``.
    """
    query = jnp.asarray(query)
    child_keys = jnp.asarray(child_keys)
    if child_keys.ndim != 2:
        raise ValueError("child_keys must have shape (n_children, d)")
    if child_keys.shape[0] == 0:
        return jnp.asarray(-jnp.inf, dtype=child_keys.dtype)
    d = child_keys.shape[-1]
    scale = jnp.sqrt(jnp.asarray(d, dtype=child_keys.dtype))
    if query.ndim == 1:
        logits = (child_keys @ query) / scale
        return jnn.logsumexp(logits)
    if query.ndim != 2:
        raise ValueError("query must have shape (d,) or (H, d)")
    if query.shape[-1] != d:
        raise ValueError("query and child_keys head dimensions differ")
    logits = jnp.einsum("hd,cd->hc", query, child_keys) / scale
    return jnp.sum(jnn.logsumexp(logits, axis=-1))


@dataclass
class Pyramid:
    """Hierarchy of mean-pooled key summaries.

    Level 0 is the leaf blocks. The last level is a single root.
    Each summary equals the mean of the original keys in its span. Padding
    does not enter the pooling denominator: a partial leaf divides by the
    number of valid tokens, and a partial parent divides by the total valid
    token count of its real children (zero-weight pad slots are ignored).
    """

    block_size: int
    group_factor: int
    seq_len: int
    head_dim: int
    n_kv: int
    squeezed: bool
    keys: jnp.ndarray
    n_blocks: list
    starts: list
    ends: list
    summaries: list
    token_counts: list

    @property
    def n_levels(self) -> int:
        return len(self.n_blocks)

    def span(self, level: int, index: int) -> tuple[int, int]:
        return int(self.starts[level][index]), int(self.ends[level][index])

    def summary(self, level: int, index: int):
        vec = self.summaries[level][:, index, :]
        if self.squeezed:
            return vec[0]
        return vec

    def children(self, level: int, index: int) -> list[int]:
        if level <= 0:
            raise ValueError("leaf blocks have token children, not block children")
        n_finer = self.n_blocks[level - 1]
        start = index * self.group_factor
        end = min(start + self.group_factor, n_finer)
        return list(range(start, end))

    def child_keys(self, level: int, index: int, kv_head: int = 0):
        """Keys used to score this block: tokens at a leaf, child summaries above."""
        if level == 0:
            start, end = self.span(0, index)
            return self.keys[kv_head, start:end, :]
        ids = jnp.asarray(self.children(level, index), dtype=jnp.int32)
        return self.summaries[level - 1][kv_head, ids, :]


def _as_kv_keys(keys):
    keys = jnp.asarray(keys)
    if keys.ndim == 2:
        return keys[None, :, :], True
    if keys.ndim == 3:
        return keys, False
    raise ValueError("keys must have shape (S, d) or (Hkv, S, d)")


def build_pyramid(keys, block_size: int, group_factor: int = 2) -> Pyramid:
    """Build a mean-pooled pyramid until a single root remains."""
    block_size = int(block_size)
    group_factor = int(group_factor)
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if group_factor < 2:
        raise ValueError("group_factor must be at least 2")
    kv_keys, squeezed = _as_kv_keys(keys)
    n_kv, seq_len, head_dim = kv_keys.shape
    if seq_len < 1:
        raise ValueError("keys must be non-empty")

    n_leaves = math.ceil(seq_len / block_size)
    leaf_counts = []
    leaf_starts = []
    leaf_ends = []
    leaf_summaries = []
    for index in range(n_leaves):
        start = index * block_size
        end = min(start + block_size, seq_len)
        count = end - start
        # Sum ignores nothing here: the slice already drops pad tokens.
        # Denominator is the valid token count, not block_size.
        summary = jnp.sum(kv_keys[:, start:end, :], axis=1) / count
        leaf_counts.append(count)
        leaf_starts.append(start)
        leaf_ends.append(end)
        leaf_summaries.append(summary)

    n_blocks = [n_leaves]
    starts = [leaf_starts]
    ends = [leaf_ends]
    token_counts = [leaf_counts]
    summaries = [jnp.stack(leaf_summaries, axis=1)]

    while n_blocks[-1] > 1:
        n_child = n_blocks[-1]
        n_parent = math.ceil(n_child / group_factor)
        parent_counts = []
        parent_starts = []
        parent_ends = []
        parent_summaries = []
        child_summaries = summaries[-1]
        child_counts = token_counts[-1]
        for index in range(n_parent):
            c0 = index * group_factor
            c1 = min(c0 + group_factor, n_child)
            # Weight by valid tokens so the summary is the mean of original
            # keys. Pad slots that were not created contribute zero weight,
            # and a partial child does not get a full-block denominator.
            counts = jnp.asarray(child_counts[c0:c1], dtype=child_summaries.dtype)
            total = jnp.sum(counts)
            weighted = child_summaries[:, c0:c1, :] * counts[None, :, None]
            summary = jnp.sum(weighted, axis=1) / total
            parent_counts.append(int(total))
            parent_starts.append(starts[-1][c0])
            parent_ends.append(ends[-1][c1 - 1])
            parent_summaries.append(summary)
        n_blocks.append(n_parent)
        starts.append(parent_starts)
        ends.append(parent_ends)
        token_counts.append(parent_counts)
        summaries.append(jnp.stack(parent_summaries, axis=1))

    return Pyramid(
        block_size=block_size,
        group_factor=group_factor,
        seq_len=seq_len,
        head_dim=head_dim,
        n_kv=n_kv,
        squeezed=squeezed,
        keys=kv_keys,
        n_blocks=n_blocks,
        starts=starts,
        ends=ends,
        summaries=summaries,
        token_counts=token_counts,
    )
