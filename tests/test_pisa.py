"""Tests for PISA pyramid selection and sparse causal attention."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from pisa_attn import build_pyramid, lse_score, select_blocks, sparse_attention
from pisa_attn.pyramid import n_pyramid_levels


def _dense_causal(queries, keys, values):
    q = jnp.asarray(queries)
    k = jnp.asarray(keys)
    v = jnp.asarray(values)
    squeeze_q = q.ndim == 2
    squeeze_k = k.ndim == 2
    if q.ndim == 2:
        q = q[None, :, :]
    if k.ndim == 2:
        k = k[None, :, :]
    if v.ndim == 2:
        v = v[None, :, :]
    n_q, n_queries, d = q.shape
    n_kv, seq_len, _ = k.shape
    group = n_q if n_kv == 1 else n_q // n_kv
    scale = 1.0 / math.sqrt(d)
    q_pos = jnp.arange(n_queries)[:, None]
    k_pos = jnp.arange(seq_len)[None, :]
    causal = k_pos <= q_pos
    outs = []
    for head in range(n_q):
        kv_head = 0 if n_kv == 1 else head // group
        logits = (q[head] @ jnp.swapaxes(k[kv_head], -1, -2)) * scale
        logits = jnp.where(causal, logits, -jnp.inf)
        weights = jax.nn.softmax(logits, axis=-1)
        outs.append(weights @ v[kv_head])
    out = jnp.stack(outs, axis=0)
    if squeeze_q and squeeze_k:
        return out[0]
    return out


def test_lse_score_matches_numpy_on_hand_built_pyramid():
    keys = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, -0.5],
            [1.0, 1.0, 0.25],
            [1.0, -1.0, 0.0],
            [0.25, 0.5, 1.0],
            [-1.0, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    block_size = 3
    group_factor = 2
    pyramid = build_pyramid(keys, block_size, group_factor)
    assert pyramid.n_levels == 2
    q = np.array([0.5, -0.25, 0.125], dtype=np.float64)
    children = np.asarray(pyramid.child_keys(1, 0))
    hand = np.stack(
        [keys[0:3].mean(axis=0), keys[3:6].mean(axis=0)],
        axis=0,
    )
    np.testing.assert_allclose(children, hand, atol=1e-12)
    score = np.asarray(lse_score(q, children), dtype=np.float64)
    logits = hand @ q / math.sqrt(q.shape[0])
    ref = np.logaddexp.reduce(logits)
    np.testing.assert_allclose(score, ref, atol=1e-12)

    tokens = np.asarray(pyramid.child_keys(0, 1))
    leaf_score = np.asarray(lse_score(q, tokens), dtype=np.float64)
    leaf_ref = np.logaddexp.reduce(tokens @ q / math.sqrt(q.shape[0]))
    np.testing.assert_allclose(leaf_score, leaf_ref, atol=1e-12)


def test_mean_pool_equals_mean_of_original_keys():
    rng = np.random.default_rng(1)
    keys = rng.normal(size=(13, 5)).astype(np.float64)
    pyramid = build_pyramid(keys, block_size=4, group_factor=2)
    assert pyramid.n_blocks[0] == 4
    assert pyramid.n_blocks[-1] == 1
    for level in range(pyramid.n_levels):
        for index in range(pyramid.n_blocks[level]):
            start, end = pyramid.span(level, index)
            assert 0 <= start < end <= keys.shape[0]
            # Partial last leaf must not divide by the padded width.
            if level == 0 and index == pyramid.n_blocks[0] - 1:
                assert end - start == keys.shape[0] - start
                assert end - start < 4
            ref = keys[start:end].mean(axis=0)
            np.testing.assert_allclose(np.asarray(pyramid.summary(level, index)), ref, atol=1e-12)

    # A zero pad inside the last block would pull this mean toward zero.
    partial = np.array([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]], dtype=np.float64)
    partial_pyramid = build_pyramid(partial, block_size=2, group_factor=2)
    np.testing.assert_allclose(partial_pyramid.summary(0, 1), np.array([5.0, 4.0]))
    np.testing.assert_allclose(partial_pyramid.summary(1, 0), partial.mean(axis=0))
    naive_unweighted = 0.5 * (partial[0:2].mean(axis=0) + partial[2:3].mean(axis=0))
    assert not np.allclose(naive_unweighted, partial.mean(axis=0))


def test_forced_blocks_always_present_when_eligible():
    rng = np.random.default_rng(2)
    seq_len, dim, block_size = 32, 8, 4
    queries = rng.normal(size=(seq_len, dim))
    keys = rng.normal(size=(seq_len, dim))
    selected = select_blocks(
        queries, keys, block_size=block_size, top_k=1, group_factor=2, causal=True, forced=True
    )
    n_leaves = math.ceil(seq_len / block_size)
    for t in range(seq_len):
        blocks = set(int(b) for b in selected[t])
        current = t // block_size
        for block in (0, current - 1, current):
            if 0 <= block < n_leaves and block * block_size <= t:
                assert block in blocks


def test_forced_blocks_do_not_displace_ranked_blocks():
    # Block 3 dominates the ranking. top_k=1 keeps it, and forced blocks are added.
    seq_len, dim, block_size = 16, 4, 4
    keys = np.zeros((seq_len, dim), dtype=np.float64)
    keys[12:16, 0] = 5.0
    queries = np.zeros((seq_len, dim), dtype=np.float64)
    queries[:, 0] = 1.0
    selected = select_blocks(
        queries, keys, block_size=block_size, top_k=1, causal=True, forced=True
    )
    blocks = set(int(b) for b in selected[15])
    assert 3 in blocks
    assert {0, 2, 3} <= blocks


def test_no_selected_block_starts_after_query():
    rng = np.random.default_rng(3)
    seq_len, dim, block_size = 37, 6, 5
    queries = rng.normal(size=(seq_len, dim))
    keys = rng.normal(size=(seq_len, dim))
    selected = select_blocks(
        queries, keys, block_size=block_size, top_k=3, group_factor=3, causal=True, forced=True
    )
    for t in range(seq_len):
        for block in selected[t]:
            assert int(block) * block_size <= t


def test_sparse_matches_dense_when_topk_keeps_every_eligible_block():
    rng = np.random.default_rng(4)
    seq_len, dim, block_size = 13, 8, 4
    queries = rng.normal(size=(seq_len, dim)).astype(np.float32)
    keys = rng.normal(size=(seq_len, dim)).astype(np.float32)
    values = rng.normal(size=(seq_len, dim)).astype(np.float32)
    n_leaves = math.ceil(seq_len / block_size)
    selected = select_blocks(
        queries,
        keys,
        block_size=block_size,
        top_k=n_leaves,
        group_factor=2,
        causal=True,
        forced=True,
    )
    for t in range(seq_len):
        eligible = [b for b in range(n_leaves) if b * block_size <= t]
        assert set(int(b) for b in selected[t]) == set(eligible)
    out = sparse_attention(queries, keys, values, selected)
    ref = _dense_causal(queries, keys, values)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), atol=1e-5)

    # Future tokens inside the current block must stay masked.
    queries64 = queries.astype(np.float64)
    keys64 = keys.astype(np.float64)
    values64 = values.astype(np.float64)
    selected64 = select_blocks(
        queries64, keys64, block_size=block_size, top_k=n_leaves, causal=True, forced=True
    )
    base = np.asarray(sparse_attention(queries64, keys64, values64, selected64))
    leaked = keys64.copy()
    leaked[2, :] += 10.0
    changed = np.asarray(sparse_attention(queries64, leaked, values64, selected64))
    np.testing.assert_allclose(base[0], changed[0], atol=1e-12)
    np.testing.assert_allclose(base[1], changed[1], atol=1e-12)
    assert not np.allclose(base[2], changed[2])


def test_score_evaluations_within_bound():
    rng = np.random.default_rng(5)
    cases = [
        (64, 4, 2, 2),
        (60, 5, 3, 1),
        (17, 4, 2, 3),
        (8, 2, 2, 1),
        (4, 4, 2, 1),
    ]
    for seq_len, block_size, group_factor, top_k in cases:
        dim = 8
        queries = rng.normal(size=(seq_len, dim))
        keys = rng.normal(size=(seq_len, dim))
        selected, counts = select_blocks(
            queries,
            keys,
            block_size=block_size,
            top_k=top_k,
            group_factor=group_factor,
            causal=True,
            forced=True,
            debug=True,
        )
        n_levels = n_pyramid_levels(seq_len, block_size, group_factor)
        assert selected.n_levels == n_levels
        bound = group_factor * top_k * n_levels
        counts = np.asarray(counts)
        assert counts.shape == (seq_len,)
        assert np.all(counts <= bound)
        assert np.array_equal(counts, selected.score_counts)

    queries = rng.normal(size=(32, 4))
    keys = rng.normal(size=(32, 4))
    selected, counts = select_blocks(
        queries, keys, block_size=4, top_k=100, debug=True, forced=False
    )
    assert np.all(np.asarray(counts) == 0)


def test_gqa_score_is_sum_of_per_head_lse():
    rng = np.random.default_rng(6)
    n_heads, n_children, dim = 4, 5, 8
    query = rng.normal(size=(n_heads, dim)).astype(np.float64)
    children = rng.normal(size=(n_children, dim)).astype(np.float64)
    score = np.asarray(lse_score(query, children), dtype=np.float64)
    scale = math.sqrt(dim)
    ref = 0.0
    for head in range(n_heads):
        ref += np.logaddexp.reduce(children @ query[head] / scale)
    np.testing.assert_allclose(score, ref, atol=1e-12)

    # Selection uses that sum: two heads disagree, the sum picks block 1.
    seq_len, block_size = 8, 4
    keys = np.zeros((seq_len, dim), dtype=np.float64)
    keys[0:4, 0] = 1.0
    keys[4:8, 1] = 1.0
    queries = np.zeros((2, seq_len, dim), dtype=np.float64)
    queries[0, :, 0] = 3.0
    queries[1, :, 1] = 1.0
    # Head 0 prefers block 0. Head 1 prefers block 1, but more weakly.
    # Sum still prefers block 0, so flip head 1 to dominate.
    queries[1, :, 1] = 8.0
    head0 = np.logaddexp.reduce(keys[0:4] @ queries[0, -1] / math.sqrt(dim))
    head0_b1 = np.logaddexp.reduce(keys[4:8] @ queries[0, -1] / math.sqrt(dim))
    head1_b0 = np.logaddexp.reduce(keys[0:4] @ queries[1, -1] / math.sqrt(dim))
    head1_b1 = np.logaddexp.reduce(keys[4:8] @ queries[1, -1] / math.sqrt(dim))
    assert head0 + head1_b0 < head0_b1 + head1_b1
    selected = select_blocks(
        queries, keys, block_size=block_size, top_k=1, causal=True, forced=False
    )
    assert list(selected[seq_len - 1]) == [1]
    single = lse_score(queries[:, -1, :], keys[4:8])
    manual = lse_score(queries[0, -1], keys[4:8]) + lse_score(queries[1, -1], keys[4:8])
    np.testing.assert_allclose(np.asarray(single), np.asarray(manual), atol=1e-12)


def test_gqa_sparse_matches_dense():
    rng = np.random.default_rng(7)
    n_q, n_kv, seq_len, dim, block_size = 4, 2, 12, 8, 4
    queries = rng.normal(size=(n_q, seq_len, dim)).astype(np.float64)
    keys = rng.normal(size=(n_kv, seq_len, dim)).astype(np.float64)
    values = rng.normal(size=(n_kv, seq_len, dim)).astype(np.float64)
    n_leaves = math.ceil(seq_len / block_size)
    selected = select_blocks(
        queries, keys, block_size=block_size, top_k=n_leaves, causal=True, forced=True
    )
    assert selected.n_kv == n_kv
    assert len(selected) == n_kv
    out = sparse_attention(queries, keys, values, selected)
    ref = _dense_causal(queries, keys, values)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), atol=1e-5)
    assert out.shape == (n_q, seq_len, dim)


def test_deterministic():
    rng = np.random.default_rng(8)
    seq_len, dim, block_size = 24, 8, 4
    queries = rng.normal(size=(seq_len, dim))
    keys = rng.normal(size=(seq_len, dim))
    values = rng.normal(size=(seq_len, dim))
    first = select_blocks(queries, keys, block_size=block_size, top_k=2, forced=True)
    second = select_blocks(queries, keys, block_size=block_size, top_k=2, forced=True)
    assert len(first) == len(second)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(first.score_counts, second.score_counts)
    out_a = sparse_attention(queries, keys, values, first)
    out_b = sparse_attention(queries, keys, values, second)
    np.testing.assert_array_equal(np.asarray(out_a), np.asarray(out_b))


def test_ties_keep_lower_index():
    seq_len, dim, block_size = 16, 4, 4
    keys = np.ones((seq_len, dim), dtype=np.float64)
    queries = np.ones((seq_len, dim), dtype=np.float64)
    selected = select_blocks(
        queries, keys, block_size=block_size, top_k=1, causal=True, forced=False
    )
    for t in range(seq_len):
        assert list(selected[t]) == [0]


def test_shared_kv_gqa_selection_length_matches_queries():
    rng = np.random.default_rng(9)
    queries = rng.normal(size=(3, 10, 4))
    keys = rng.normal(size=(10, 4))
    selected = select_blocks(queries, keys, block_size=2, top_k=2, forced=True)
    assert selected.n_kv == 1
    assert len(selected) == 10
    out = sparse_attention(queries, keys, rng.normal(size=(10, 4)), selected)
    assert out.shape == (3, 10, 4)
