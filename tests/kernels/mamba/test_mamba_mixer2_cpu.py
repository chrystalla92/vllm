# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Integration tests for MambaMixer2 CPU inference.

Applies the same monkey-patches as CPUModelRunner._postprocess_triton() and
exercises both the prefill and decode forward paths using the pure-PyTorch CPU
fallback ops.  No GPU is required.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.config.mamba import MambaBackendEnum, MambaConfig
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    get_mamba_ssu_backend,
    initialize_mamba_ssu_backend,
)
from vllm.v1.attention.backends.mamba2_attn import compute_varlen_chunk_metadata

pytestmark = pytest.mark.cpu_test

# ---------------------------------------------------------------------------
# Model dimensions used throughout
# ---------------------------------------------------------------------------
D_MODEL = 128        # total model / intermediate size
D_STATE = 16         # SSM state dimension
D_CONV = 4           # causal conv kernel width
NHEADS = 2           # number of SSM heads
HEAD_DIM = D_MODEL // NHEADS   # 64
NGROUPS = 1          # B/C groups
CHUNK_SIZE = 8


# ---------------------------------------------------------------------------
# Fixture: apply CPU monkey-patches (replicate _postprocess_triton)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def apply_cpu_patches():
    """
    Mirror the three monkey-patches applied by CPUModelRunner._postprocess_triton().

    Both the ops-module namespace and mamba_mixer2's own module-level bindings
    are patched, exactly as the runner does it.
    """
    import vllm.model_executor.layers.mamba.mamba_mixer2 as _mamba_mixer2
    import vllm.model_executor.layers.mamba.ops.causal_conv1d as _causal_conv1d
    import vllm.model_executor.layers.mamba.ops.ssd_combined as _ssd_combined
    from vllm.model_executor.layers.mamba.ops.cpu_mamba_ops import (
        causal_conv1d_fn as cpu_causal_conv1d_fn,
        causal_conv1d_update as cpu_causal_conv1d_update,
        mamba_chunk_scan_combined_varlen as cpu_mamba_chunk_scan_combined_varlen,
    )

    # Save originals for teardown
    orig_cc_fn = _causal_conv1d.causal_conv1d_fn
    orig_cc_up = _causal_conv1d.causal_conv1d_update
    orig_scan = _ssd_combined.mamba_chunk_scan_combined_varlen
    orig_m2_fn = _mamba_mixer2.causal_conv1d_fn
    orig_m2_up = _mamba_mixer2.causal_conv1d_update
    orig_m2_scan = _mamba_mixer2.mamba_chunk_scan_combined_varlen

    # Apply patches
    _causal_conv1d.causal_conv1d_fn = cpu_causal_conv1d_fn
    _causal_conv1d.causal_conv1d_update = cpu_causal_conv1d_update
    _ssd_combined.mamba_chunk_scan_combined_varlen = cpu_mamba_chunk_scan_combined_varlen
    _mamba_mixer2.causal_conv1d_fn = cpu_causal_conv1d_fn
    _mamba_mixer2.causal_conv1d_update = cpu_causal_conv1d_update
    _mamba_mixer2.mamba_chunk_scan_combined_varlen = cpu_mamba_chunk_scan_combined_varlen

    # Register CPU SSU backend
    initialize_mamba_ssu_backend(MambaConfig(backend=MambaBackendEnum.CPU))

    yield

    # Restore originals
    _causal_conv1d.causal_conv1d_fn = orig_cc_fn
    _causal_conv1d.causal_conv1d_update = orig_cc_up
    _ssd_combined.mamba_chunk_scan_combined_varlen = orig_scan
    _mamba_mixer2.causal_conv1d_fn = orig_m2_fn
    _mamba_mixer2.causal_conv1d_update = orig_m2_up
    _mamba_mixer2.mamba_chunk_scan_combined_varlen = orig_m2_scan


# ---------------------------------------------------------------------------
# Helper: build prefill inputs
# ---------------------------------------------------------------------------


def _make_prefill_inputs(batch: int, seq_len: int):
    """
    Return (x_packed, weight, bias, conv_states, ssm_state,
            query_start_loc, cu_seqlens, cu_chunk_seqlens,
            last_chunk_indices, seq_idx, dt, A, B, C, D, dt_bias).
    """
    torch.manual_seed(42)
    total_tokens = batch * seq_len
    state_len = D_CONV - 1  # 3

    device = torch.device("cpu")
    dtype = torch.float32

    # Conv inputs
    x_packed = torch.randn(D_MODEL, total_tokens, dtype=dtype, device=device)
    weight = torch.randn(D_MODEL, D_CONV, dtype=dtype, device=device)
    bias = torch.randn(D_MODEL, dtype=dtype, device=device)
    # Slot 0 is NULL_BLOCK_ID; allocate batch+1 slots so sequences use 1-based indices.
    conv_states = torch.zeros(batch + 1, D_MODEL, state_len, dtype=dtype, device=device)

    query_start_loc = torch.tensor(
        [i * seq_len for i in range(batch + 1)], dtype=torch.int32, device=device
    )
    cache_indices = torch.arange(1, batch + 1, dtype=torch.int32, device=device)

    # SSM inputs
    ssm_state = torch.zeros(
        batch, NHEADS, HEAD_DIM, D_STATE, dtype=dtype, device=device
    )
    dt = F.softplus(
        torch.randn(total_tokens, NHEADS, dtype=dtype, device=device) - 4.0
    )
    A = -torch.rand(NHEADS, dtype=dtype, device=device)
    B = torch.randn(total_tokens, NGROUPS, D_STATE, dtype=dtype, device=device)
    C = torch.randn(total_tokens, NGROUPS, D_STATE, dtype=dtype, device=device)
    D_param = torch.ones(NHEADS, dtype=dtype, device=device)
    dt_bias = torch.zeros(NHEADS, dtype=dtype, device=device)

    cu_seqlens = query_start_loc
    cu_chunk_seqlens, last_chunk_indices, seq_idx = compute_varlen_chunk_metadata(
        cu_seqlens, CHUNK_SIZE
    )

    return dict(
        x_packed=x_packed,
        weight=weight,
        bias=bias,
        conv_states=conv_states,
        ssm_state=ssm_state,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        cu_seqlens=cu_seqlens,
        cu_chunk_seqlens=cu_chunk_seqlens,
        last_chunk_indices=last_chunk_indices,
        seq_idx=seq_idx,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D_param=D_param,
        dt_bias=dt_bias,
        total_tokens=total_tokens,
        dtype=dtype,
        device=device,
    )


# ---------------------------------------------------------------------------
# Helper: build decode inputs
# ---------------------------------------------------------------------------


def _make_decode_inputs(batch: int):
    """
    Return inputs for a single decode step (seq_len=1 per sequence).
    """
    torch.manual_seed(7)
    state_len = D_CONV - 1  # 3

    device = torch.device("cpu")
    dtype = torch.float32

    # Conv state from a "previous" prefill step (non-zero).
    # Slot 0 is the null block; use batch+1 slots with 1-based indices.
    conv_state = torch.zeros(batch + 1, D_MODEL, state_len, dtype=dtype, device=device)
    conv_state[1:] = torch.randn(batch, D_MODEL, state_len, dtype=dtype, device=device)
    x_dec = torch.randn(batch, D_MODEL, dtype=dtype, device=device)
    weight = torch.randn(D_MODEL, D_CONV, dtype=dtype, device=device)
    bias = torch.randn(D_MODEL, dtype=dtype, device=device)
    conv_state_indices = torch.arange(1, batch + 1, dtype=torch.int32, device=device)

    # SSM state
    ssm_state = torch.randn(
        batch, NHEADS, HEAD_DIM, D_STATE, dtype=dtype, device=device
    )
    x_ssm = torch.randn(batch, NHEADS, HEAD_DIM, dtype=dtype, device=device)
    dt = torch.randn(batch, NHEADS, HEAD_DIM, dtype=dtype, device=device)
    A = -torch.rand(NHEADS, HEAD_DIM, D_STATE, dtype=dtype, device=device)
    B = torch.randn(batch, NGROUPS, D_STATE, dtype=dtype, device=device)
    C = torch.randn(batch, NGROUPS, D_STATE, dtype=dtype, device=device)
    D_param = torch.ones(NHEADS, HEAD_DIM, dtype=dtype, device=device)
    dt_bias = torch.zeros(NHEADS, HEAD_DIM, dtype=dtype, device=device)
    out = torch.zeros(batch, NHEADS, HEAD_DIM, dtype=dtype, device=device)

    return dict(
        conv_state=conv_state,
        x_dec=x_dec,
        weight=weight,
        bias=bias,
        conv_state_indices=conv_state_indices,
        ssm_state=ssm_state,
        x_ssm=x_ssm,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D_param=D_param,
        dt_bias=dt_bias,
        out=out,
        dtype=dtype,
        device=device,
    )


# ---------------------------------------------------------------------------
# Prefill test
# ---------------------------------------------------------------------------


def test_mamba_mixer2_prefill_cpu(apply_cpu_patches):
    """
    Prefill path: causal_conv1d_fn + mamba_chunk_scan_combined_varlen on CPU.

    Verifies that outputs are:
    - on CPU
    - have the expected shape and dtype
    - contain no NaN or Inf
    """
    batch = 2
    seq_len = 8
    inp = _make_prefill_inputs(batch, seq_len)

    # Import the patched functions from their respective module namespaces
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_fn,
    )
    from vllm.model_executor.layers.mamba.ops.ssd_combined import (
        mamba_chunk_scan_combined_varlen,
    )

    # --- Causal conv (prefill) ---
    conv_out = causal_conv1d_fn(
        x=inp["x_packed"],
        weight=inp["weight"],
        bias=inp["bias"],
        conv_states=inp["conv_states"],
        query_start_loc=inp["query_start_loc"],
        cache_indices=inp["cache_indices"],
        activation="silu",
    )

    assert conv_out.device.type == "cpu", "conv_out not on CPU"
    assert conv_out.shape == (D_MODEL, inp["total_tokens"]), (
        f"conv_out shape mismatch: {conv_out.shape}"
    )
    assert conv_out.dtype == inp["dtype"]
    assert torch.isfinite(conv_out).all(), "conv_out contains non-finite values"

    # --- SSM scan (prefill) ---
    # Reshape conv_out from (D_MODEL, total_tokens) → (total_tokens, NHEADS, HEAD_DIM)
    ssm_in = conv_out.t().view(inp["total_tokens"], NHEADS, HEAD_DIM)
    ssm_out = torch.zeros_like(ssm_in)

    final_states = mamba_chunk_scan_combined_varlen(
        x=ssm_in,
        dt=inp["dt"],
        A=inp["A"],
        B=inp["B"],
        C=inp["C"],
        chunk_size=CHUNK_SIZE,
        cu_seqlens=inp["cu_seqlens"],
        cu_chunk_seqlens=inp["cu_chunk_seqlens"],
        last_chunk_indices=inp["last_chunk_indices"],
        seq_idx=inp["seq_idx"],
        out=ssm_out,
        D=inp["D_param"],
        dt_bias=inp["dt_bias"],
        dt_softplus=True,
    )

    assert ssm_out.device.type == "cpu", "ssm_out not on CPU"
    assert ssm_out.shape == (inp["total_tokens"], NHEADS, HEAD_DIM), (
        f"ssm_out shape mismatch: {ssm_out.shape}"
    )
    assert ssm_out.dtype == inp["dtype"]
    assert torch.isfinite(ssm_out).all(), "ssm_out contains non-finite values"

    assert final_states.device.type == "cpu", "final_states not on CPU"
    assert final_states.shape == (batch, NHEADS, HEAD_DIM, D_STATE), (
        f"final_states shape mismatch: {final_states.shape}"
    )
    assert torch.isfinite(final_states).all(), (
        "final_states contains non-finite values"
    )


# ---------------------------------------------------------------------------
# Decode test
# ---------------------------------------------------------------------------


def test_mamba_mixer2_decode_cpu(apply_cpu_patches):
    """
    Decode path: causal_conv1d_update + CPUSSUBackend on CPU.

    Verifies that outputs are:
    - on CPU
    - have the expected shape and dtype
    - contain no NaN or Inf
    """
    batch = 2
    inp = _make_decode_inputs(batch)

    # Import the patched function from the ops-module namespace
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )

    # --- Causal conv update (decode single step) ---
    conv_out = causal_conv1d_update(
        x=inp["x_dec"],
        conv_state=inp["conv_state"],
        weight=inp["weight"],
        bias=inp["bias"],
        activation="silu",
        conv_state_indices=inp["conv_state_indices"],
    )

    assert conv_out.device.type == "cpu", "conv_out not on CPU"
    assert conv_out.shape == (batch, D_MODEL), (
        f"conv_out shape mismatch: {conv_out.shape}"
    )
    assert conv_out.dtype == inp["dtype"]
    assert torch.isfinite(conv_out).all(), "conv_out contains non-finite values"

    # --- SSU (decode) ---
    backend = get_mamba_ssu_backend()
    assert backend.name == "cpu"

    backend(
        state=inp["ssm_state"],
        x=inp["x_ssm"],
        dt=inp["dt"],
        A=inp["A"],
        B=inp["B"],
        C=inp["C"],
        D=inp["D_param"],
        dt_bias=inp["dt_bias"],
        dt_softplus=True,
        out=inp["out"],
    )
    ssu_out = inp["out"]

    assert ssu_out.device.type == "cpu", "SSU output not on CPU"
    assert ssu_out.shape == (batch, NHEADS, HEAD_DIM), (
        f"SSU output shape mismatch: {ssu_out.shape}"
    )
    assert ssu_out.dtype == inp["dtype"]
    assert torch.isfinite(ssu_out).all(), "SSU output contains non-finite values"
