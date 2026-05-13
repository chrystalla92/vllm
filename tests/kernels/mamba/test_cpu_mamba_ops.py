# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for the pure-PyTorch CPU Mamba-2 fallback ops.

All tests run on CPU and require no GPU.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.config.mamba import MambaBackendEnum, MambaConfig
from vllm.model_executor.layers.mamba.ops.cpu_mamba_ops import (
    causal_conv1d_fn as cpu_causal_conv1d_fn,
    causal_conv1d_update as cpu_causal_conv1d_update,
    mamba_chunk_scan_combined_varlen as cpu_mamba_chunk_scan_combined_varlen,
)
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    get_mamba_ssu_backend,
    initialize_mamba_ssu_backend,
)
from vllm.v1.attention.backends.mamba2_attn import compute_varlen_chunk_metadata

pytestmark = pytest.mark.cpu_test


# ---------------------------------------------------------------------------
# test_causal_conv1d_fn_cpu
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("has_bias", [False, True])
def test_causal_conv1d_fn_cpu(activation, has_bias):
    """CPU causal_conv1d_fn matches F.conv1d reference with causal padding."""
    torch.manual_seed(0)

    batch = 2
    channels = 16
    seq_len = 8
    width = 4  # kernel width
    state_len = width - 1  # 3

    # Packed input: (channels, batch * seq_len)
    x = torch.randn(channels, batch * seq_len, dtype=torch.float32)
    weight = torch.randn(channels, width, dtype=torch.float32)
    bias = torch.randn(channels, dtype=torch.float32) if has_bias else None

    # conv_states — slot 0 is reserved as null block, slots 1..batch hold data.
    # Zero-initialised so no initial state prefix is added.
    conv_states = torch.zeros(batch + 1, channels, state_len, dtype=torch.float32)

    # query_start_loc: (batch+1,)
    query_start_loc = torch.tensor(
        [i * seq_len for i in range(batch + 1)], dtype=torch.int32
    )
    # cache_indices: slot 0 is NULL_BLOCK_ID; use 1-based indices.
    cache_indices = torch.arange(1, batch + 1, dtype=torch.int32)

    # ---------- CPU implementation ----------
    out = cpu_causal_conv1d_fn(
        x=x,
        weight=weight,
        bias=bias,
        conv_states=conv_states,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        activation=activation,
    )

    # ---------- F.conv1d reference ----------
    # For each sequence with zero initial state, apply causal padding (width-1
    # zeros on the left) then F.conv1d with groups=channels.
    w_dw = weight.unsqueeze(1)  # (channels, 1, width)
    ref = torch.zeros_like(x)

    for i in range(batch):
        s = i * seq_len
        e = s + seq_len
        x_seq = x[:, s:e].unsqueeze(0)  # (1, channels, seq_len)
        # zero initial state → pad left with state_len zeros
        x_padded = F.pad(x_seq, (state_len, 0))
        conv_out = F.conv1d(x_padded, w_dw, bias=bias, groups=channels)
        if activation == "silu":
            conv_out = F.silu(conv_out)
        ref[:, s:e] = conv_out.squeeze(0)

    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# test_causal_conv1d_update_cpu
# ---------------------------------------------------------------------------


def test_causal_conv1d_update_cpu():
    """CPU causal_conv1d_update matches a manual sliding-window reference."""
    torch.manual_seed(0)

    batch = 3
    dim = 8
    width = 4
    state_len = width - 1  # 3

    x = torch.randn(batch, dim, dtype=torch.float32)
    # conv_state: (batch+1, dim, state_len) — slot 0 is the null block,
    # slots 1..batch hold pre-filled random state data.
    conv_state = torch.zeros(batch + 1, dim, state_len, dtype=torch.float32)
    # Fill only the active slots (1..batch) with random data.
    conv_state[1:] = torch.randn(batch, dim, state_len, dtype=torch.float32)
    conv_state_ref = conv_state[1:].clone()  # reference copy of active slots
    weight = torch.randn(dim, width, dtype=torch.float32)

    # conv_state_indices: 1-based to skip the null-block slot at index 0.
    conv_state_indices = torch.arange(1, batch + 1, dtype=torch.int32)

    # ---------- CPU implementation ----------
    out = cpu_causal_conv1d_update(
        x=x,
        conv_state=conv_state,
        weight=weight,
        bias=None,
        activation=None,
        conv_state_indices=conv_state_indices,
    )

    # ---------- Manual reference ----------
    # For each sequence: concat [state | x_t], slide window, dot with weight.
    w_dw = weight.unsqueeze(1)  # (dim, 1, width)
    for i in range(batch):
        old_state = conv_state_ref[i]  # (dim, state_len)
        x_i = x[i].unsqueeze(-1)  # (dim, 1)
        full_in = torch.cat([old_state, x_i], dim=-1)  # (dim, width)
        expected_out = F.conv1d(
            full_in.unsqueeze(0), w_dw, bias=None, groups=dim
        ).squeeze()  # (dim,)
        torch.testing.assert_close(out[i], expected_out, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# test_mamba_chunk_scan_combined_cpu
# ---------------------------------------------------------------------------


def test_mamba_chunk_scan_combined_cpu():
    """CPU mamba_chunk_scan_combined_varlen: shape, dtype, and finiteness."""
    torch.manual_seed(0)

    batch = 2
    total_seq = 32  # 16 tokens per sequence
    seq_per_batch = total_seq // batch  # 16
    nheads = 4
    headdim = 16  # nheads * headdim = 64 = "d_model"
    dstate = 16
    ngroups = 1
    chunk_size = 8

    device = torch.device("cpu")
    dtype = torch.float32

    x = torch.randn(total_seq, nheads, headdim, dtype=dtype, device=device)
    dt = torch.randn(total_seq, nheads, dtype=dtype, device=device)
    A = -torch.rand(nheads, dtype=dtype, device=device)
    B = torch.randn(total_seq, ngroups, dstate, dtype=dtype, device=device)
    C = torch.randn(total_seq, ngroups, dstate, dtype=dtype, device=device)
    D = torch.ones(nheads, dtype=dtype, device=device)
    dt_bias = torch.zeros(nheads, dtype=dtype, device=device)

    # Two equal-length sequences
    cu_seqlens = torch.tensor(
        [0, seq_per_batch, total_seq], dtype=torch.int32, device=device
    )

    cu_chunk_seqlens, last_chunk_indices, seq_idx_chunks = (
        compute_varlen_chunk_metadata(cu_seqlens, chunk_size)
    )

    out = torch.zeros(total_seq, nheads, headdim, dtype=dtype, device=device)

    final_states = cpu_mamba_chunk_scan_combined_varlen(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        cu_chunk_seqlens=cu_chunk_seqlens,
        last_chunk_indices=last_chunk_indices,
        seq_idx=seq_idx_chunks,
        out=out,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
    )

    # Shape checks
    assert out.shape == (total_seq, nheads, headdim), (
        f"out shape mismatch: {out.shape}"
    )
    assert final_states.shape == (batch, nheads, headdim, dstate), (
        f"final_states shape mismatch: {final_states.shape}"
    )

    # Dtype checks
    assert out.dtype == dtype
    assert final_states.dtype == dtype

    # Finiteness
    assert torch.isfinite(out).all(), "out contains non-finite values"
    assert torch.isfinite(final_states).all(), (
        "final_states contains non-finite values"
    )


# ---------------------------------------------------------------------------
# test_cpu_ssu_backend
# ---------------------------------------------------------------------------


def test_cpu_ssu_backend():
    """CPUSSUBackend.selective_state_update: correct output shape and dtype."""
    torch.manual_seed(0)

    # Initialize the CPU backend (no kv_cache_config needed)
    initialize_mamba_ssu_backend(MambaConfig(backend=MambaBackendEnum.CPU))
    backend = get_mamba_ssu_backend()
    assert backend.name == "cpu"

    batch = 2
    nheads = 2
    dim = 8   # headdim per head
    dstate = 4
    ngroups = 1

    device = torch.device("cpu")
    dtype = torch.float32

    # state: (batch, nheads, dim, dstate)
    state = torch.randn(batch, nheads, dim, dstate, dtype=dtype, device=device)
    x = torch.randn(batch, nheads, dim, dtype=dtype, device=device)
    dt = torch.randn(batch, nheads, dim, dtype=dtype, device=device)
    A = -torch.rand(nheads, dim, dstate, dtype=dtype, device=device)
    B = torch.randn(batch, ngroups, dstate, dtype=dtype, device=device)
    C = torch.randn(batch, ngroups, dstate, dtype=dtype, device=device)
    D = torch.randn(nheads, dim, dtype=dtype, device=device)
    dt_bias = torch.randn(nheads, dim, dtype=dtype, device=device)
    out = torch.zeros(batch, nheads, dim, dtype=dtype, device=device)

    backend(
        state=state,
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        out=out,
    )

    # Shape and dtype
    assert out.shape == (batch, nheads, dim), f"out shape mismatch: {out.shape}"
    assert out.dtype == dtype

    # Finiteness
    assert torch.isfinite(out).all(), "SSU output contains non-finite values"
