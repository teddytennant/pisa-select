"""Sparse causal attention over selected leaf blocks."""

from __future__ import annotations

import math

import jax.nn as jnn
import jax.numpy as jnp
import numpy as np


def _as_heads(array, name):
    array = jnp.asarray(array)
    if array.ndim == 2:
        return array[None, :, :], True
    if array.ndim == 3:
        return array, False
    raise ValueError(f"{name} must have shape (L, d) or (H, L, d)")


def _is_per_query(selected) -> bool:
    n_kv = getattr(selected, "n_kv", None)
    if n_kv == 1:
        return True
    if n_kv not in (None, 1):
        return False
    if len(selected) == 0:
        return True
    first = selected[0]
    if isinstance(first, (list, tuple)):
        return False
    ndim = getattr(first, "ndim", None)
    return ndim == 1


def _rows_for_queries(selected_one, n_queries: int):
    if isinstance(selected_one, np.ndarray) and selected_one.ndim == 2:
        rows = []
        for t in range(n_queries):
            row = np.asarray(selected_one[t])
            rows.append(row[row >= 0])
        return rows
    if len(selected_one) != n_queries:
        raise ValueError(
            f"selected length {len(selected_one)} does not match queries {n_queries}"
        )
    return selected_one


def _block_allow_mask(n_queries, seq_len, selected_one, block_size):
    rows = _rows_for_queries(selected_one, n_queries)
    mask = np.zeros((n_queries, seq_len), dtype=bool)
    for t in range(n_queries):
        blocks = np.asarray(rows[t], dtype=np.int64).reshape(-1)
        for block in blocks:
            if block < 0:
                continue
            start = int(block) * block_size
            end = min(start + block_size, seq_len)
            if 0 <= start < seq_len:
                mask[t, start:end] = True
    return jnp.asarray(mask)


def sparse_attention(queries, keys, values, selected, block_size=None, causal=True):
    """Causal attention restricted to tokens inside ``selected`` leaf blocks.

    Tokens after the query are masked even when their block was selected.
    When every causally eligible block is selected, the result matches dense
    causal attention. ``selected`` is the object returned by ``select_blocks``
    or a list of per-query block-index arrays. ``block_size`` is read from
    ``selected`` when omitted.
    """
    if block_size is None:
        block_size = getattr(selected, "block_size", None)
    if block_size is None:
        raise ValueError("block_size is required when selected does not carry it")
    block_size = int(block_size)

    q, q_squeezed = _as_heads(queries, "queries")
    k, k_squeezed = _as_heads(keys, "keys")
    v, v_squeezed = _as_heads(values, "values")
    n_q, n_queries, d = q.shape
    n_kv, seq_len, dk = k.shape
    if d != dk:
        raise ValueError("query and key head dimensions differ")
    if v.shape[0] != n_kv or v.shape[1] != seq_len:
        raise ValueError("values must match keys on KV heads and sequence length")
    if n_kv == 1:
        group = n_q
    else:
        if n_q % n_kv != 0:
            raise ValueError("Hq must be divisible by Hkv")
        group = n_q // n_kv

    if _is_per_query(selected):
        if n_kv != 1 and getattr(selected, "n_kv", 1) == 1:
            per_kv = [selected] * n_kv
        elif n_kv == 1:
            per_kv = [selected]
        else:
            raise ValueError("multi-KV keys need a per-KV selection")
    else:
        if len(selected) != n_kv:
            raise ValueError("selected KV heads do not match keys")
        per_kv = list(selected)

    scale = 1.0 / math.sqrt(d)
    q_pos = jnp.arange(n_queries)[:, None]
    k_pos = jnp.arange(seq_len)[None, :]
    causal_mask = k_pos <= q_pos if causal else jnp.ones((n_queries, seq_len), dtype=bool)

    outs = []
    for head in range(n_q):
        kv_head = 0 if n_kv == 1 else head // group
        logits = (q[head] @ jnp.swapaxes(k[kv_head], -1, -2)) * scale
        allow = _block_allow_mask(n_queries, seq_len, per_kv[kv_head], block_size)
        mask = causal_mask & allow
        logits = jnp.where(mask, logits, -jnp.inf)
        weights = jnn.softmax(logits, axis=-1)
        weights = jnp.where(jnp.isfinite(weights), weights, 0.0)
        outs.append(weights @ v[kv_head])
    out = jnp.stack(outs, axis=0)
    if q_squeezed and k_squeezed and v_squeezed:
        return out[0]
    if q_squeezed and n_kv == 1:
        return out[0]
    return out
