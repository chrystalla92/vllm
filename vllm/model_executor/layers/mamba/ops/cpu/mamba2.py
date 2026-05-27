# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Pure-PyTorch CPU implementations of the Mamba2 SSM kernels.

These replace the Triton/CUDA kernels in
vllm/model_executor/layers/mamba/ops/{ssd_combined,mamba_ssm}.py
when running on CPU.  They are intentionally simple sequential
implementations — correctness over throughput.

Shapes & conventions used throughout
─────────────────────────────────────
  nheads   = num_heads // tp_size
  headdim  = head_dim
  ngroups  = n_groups // tp_size
  dstate   = ssm_state_size

  A   : (nheads,)          — stored as −exp(A_log); already negative
  D   : (nheads,)          — skip-connection scalar per head
  dt_bias : (nheads,)      — per-head bias for time-step
  B/C : (tokens, ngroups, dstate)
  x   : (tokens, nheads, headdim)
  dt  : (tokens, nheads)   — before bias / softplus

In the decode path the caller expands (A, D, dt_bias, dt) by `headdim`
via `expand()` (stride-0 views).  The CPU functions extract the scalar
at index 0 along that dimension.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mamba2_ssm_single_seq_cpu(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
    dt_bias: torch.Tensor,
    out: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    dt_softplus: bool = True,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
) -> torch.Tensor:
    """Sequential Mamba2 SSM scan for a *single* contiguous sequence.

    Recurrence (per token t):
        delta_t  = softplus(dt_t + dt_bias)
        dA_t     = exp(A * delta_t)              # (nheads,)
        state    = state * dA_t + delta_t * outer(x_t, B_t)
        y_t      = inner(C_t, state) + D * x_t

    Args:
        x:    (seqlen, nheads, headdim)
        dt:   (seqlen, nheads)
        A:    (nheads,)  — negative
        B:    (seqlen, ngroups, dstate)
        C:    (seqlen, ngroups, dstate)
        D:    (nheads,) or None
        dt_bias: (nheads,)
        out:  (seqlen, nheads, headdim) — written in-place
        initial_state: (nheads, headdim, dstate) or None
        dt_softplus, dt_limit: passed through

    Returns:
        state: (nheads, headdim, dstate) — the final SSM state
    """
    seqlen, nheads, headdim = x.shape
    ngroups = B.shape[1]
    dstate = B.shape[2]
    nheads_per_group = nheads // ngroups

    # Compute δ = softplus(dt + dt_bias), optionally clamp
    dt_f = dt.float() + dt_bias.float().unsqueeze(0)   # (seqlen, nheads)
    if dt_softplus:
        dt_f = F.softplus(dt_f)
    dt_min, dt_max = dt_limit
    if dt_min > 0.0 or dt_max < float("inf"):
        dt_f = dt_f.clamp(min=dt_min, max=dt_max)

    # Initialise SSM state
    if initial_state is not None:
        state = initial_state.float().clone()
    else:
        state = x.new_zeros(nheads, headdim, dstate, dtype=torch.float32)

    A_f = A.float()
    x_f = x.float()
    B_f = B.float()
    C_f = C.float()
    D_f = D.float() if D is not None else None

    for t in range(seqlen):
        dt_t = dt_f[t]                                     # (nheads,)
        x_t  = x_f[t]                                      # (nheads, headdim)
        # Expand ngroups → nheads
        B_t = B_f[t].repeat_interleave(nheads_per_group, dim=0)   # (nheads, dstate)
        C_t = C_f[t].repeat_interleave(nheads_per_group, dim=0)   # (nheads, dstate)

        dA = torch.exp(A_f * dt_t)                         # (nheads,)

        # outer product: (nheads, headdim, dstate)
        dBx = (dt_t[:, None] * x_t)[:, :, None] * B_t[:, None, :]

        state = state * dA[:, None, None] + dBx

        y_t = (state * C_t[:, None, :]).sum(dim=-1)        # (nheads, headdim)
        if D_f is not None:
            y_t = y_t + D_f[:, None] * x_t
        out[t] = y_t.to(out.dtype)

    return state  # (nheads, headdim, dstate)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def mamba2_ssm_prefill_varlen_cpu(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
    dt_bias: torch.Tensor,
    out: torch.Tensor,
    query_start_loc: torch.Tensor,
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    initial_states: torch.Tensor | None = None,
    dt_softplus: bool = True,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
) -> None:
    """Varlen Mamba2 SSM prefill — iterates over sequences.

    Args:
        x:    (total_prefill_tokens, nheads, headdim)
        dt:   (total_prefill_tokens, nheads)
        A:    (nheads,)
        B:    (total_prefill_tokens, ngroups, dstate)
        C:    (total_prefill_tokens, ngroups, dstate)
        D:    (nheads,) or None
        dt_bias: (nheads,)
        out:  (total_prefill_tokens, nheads, headdim) — written in-place
        query_start_loc: (num_prefills+1,) token offsets
        ssm_state: (num_slots, nheads, headdim, dstate) — KV cache
        state_indices: (num_prefills,) — destination slot per sequence
        initial_states: (num_prefills, nheads, headdim, dstate) or None
        dt_softplus, dt_limit: passed through to the inner scan
    """
    num_seqs = query_start_loc.shape[0] - 1

    for s in range(num_seqs):
        bos = int(query_start_loc[s].item())
        eos = int(query_start_loc[s + 1].item())
        if eos <= bos:
            continue

        init = initial_states[s] if initial_states is not None else None
        final_state = _mamba2_ssm_single_seq_cpu(
            x=x[bos:eos],
            dt=dt[bos:eos],
            A=A,
            B=B[bos:eos],
            C=C[bos:eos],
            D=D,
            dt_bias=dt_bias,
            out=out[bos:eos],
            initial_state=init,
            dt_softplus=dt_softplus,
            dt_limit=dt_limit,
        )

        slot = int(state_indices[s].item())
        ssm_state[slot] = final_state.to(ssm_state.dtype)


def mamba2_ssm_decode_cpu(
    ssm_state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    out: torch.Tensor,
    state_batch_indices: torch.Tensor | None = None,
    dst_state_batch_indices: torch.Tensor | None = None,
    dt_softplus: bool = True,
    null_block_id: int = 2**31 - 1,
) -> None:
    """Pure-PyTorch CPU Mamba2 SSM decode (one token per request).

    Args:
        ssm_state: (num_slots, nheads, headdim, dstate) — KV cache (in-place)
        x:    (num_decode_tokens, nheads, headdim)
        dt:   (num_decode_tokens, nheads, headdim) — stride-0 in headdim
        A:    (nheads, headdim, dstate)            — stride-0 in headdim, dstate
        B:    (num_decode_tokens, ngroups, dstate)
        C:    (num_decode_tokens, ngroups, dstate)
        D:    (nheads, headdim)                    — stride-0 in headdim
        dt_bias: (nheads, headdim)                 — stride-0 in headdim
        out:  (num_decode_tokens, nheads, headdim) — written in-place
        state_batch_indices:     (num_tokens, 1) src slot indices or None
        dst_state_batch_indices: (num_tokens, 1) dst slot indices or None
        dt_softplus: apply softplus to dt
        null_block_id: sentinel for padding entries to skip
    """
    num_tokens, nheads, headdim = x.shape
    ngroups = B.shape[1]
    nheads_per_group = nheads // ngroups

    # Extract unique scalars from stride-0 expanded dims
    A_h     = A[:, 0, 0].float()         # (nheads,)
    D_h     = D[:, 0].float()            # (nheads,)
    dt_b_h  = dt_bias[:, 0].float()      # (nheads,)

    for i in range(num_tokens):
        # Resolve source / destination state slots
        if state_batch_indices is not None:
            src_slot = int(state_batch_indices[i, 0].item())
        else:
            src_slot = i
        if src_slot == null_block_id:
            continue
        if dst_state_batch_indices is not None:
            dst_slot = int(dst_state_batch_indices[i, 0].item())
        else:
            dst_slot = src_slot

        state = ssm_state[src_slot].float()              # (nheads, headdim, dstate)

        dt_i = dt[i, :, 0].float() + dt_b_h             # (nheads,)
        if dt_softplus:
            dt_i = F.softplus(dt_i)

        x_i = x[i].float()                               # (nheads, headdim)
        B_i = B[i].float().repeat_interleave(nheads_per_group, dim=0)  # (nheads, dstate)
        C_i = C[i].float().repeat_interleave(nheads_per_group, dim=0)  # (nheads, dstate)

        dA  = torch.exp(A_h * dt_i)                      # (nheads,)
        dBx = (dt_i[:, None] * x_i)[:, :, None] * B_i[:, None, :]  # (nheads, hd, ds)
        state = state * dA[:, None, None] + dBx

        y = (state * C_i[:, None, :]).sum(dim=-1) + D_h[:, None] * x_i
        out[i] = y.to(out.dtype)

        if dst_slot != null_block_id:
            ssm_state[dst_slot] = state.to(ssm_state.dtype)


def causal_conv1d_decode_cpu(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    conv_state_indices: torch.Tensor,
) -> torch.Tensor:
    """CPU causal conv1d update for decode (continuous batching).

    Args:
        x: (num_decode_tokens, conv_dim) — input
        conv_state: (num_slots, conv_dim, state_len) — modified in-place
        weight: (conv_dim, width)
        bias: (conv_dim,) or None
        activation: "silu", "swish", or None
        conv_state_indices: (num_decode_tokens,) slot indices

    Returns:
        out: (num_decode_tokens, conv_dim)
    """
    from vllm.model_executor.layers.mamba.ops.cpu.causal_conv1d import (
        causal_conv1d_update_torch,
    )

    assert activation in {None, "silu", "swish"}
    num_tokens = x.shape[0]
    out = torch.empty_like(x)

    for i in range(num_tokens):
        slot = int(conv_state_indices[i].item())
        # causal_conv1d_update_torch expects (1, conv_dim, seq_len)
        # state_i is a view of conv_state[slot]; the in-place update inside
        # causal_conv1d_update_torch propagates back to conv_state[slot].
        x_i     = x[i].unsqueeze(0).unsqueeze(-1)   # (1, conv_dim, 1)
        state_i = conv_state[slot].unsqueeze(0)      # (1, conv_dim, state_len) view
        out_i = causal_conv1d_update_torch(
            x=x_i,
            conv_state=state_i,
            weight=weight,
            bias=bias,
            activation=activation,
        )
        out[i] = out_i.squeeze(0).squeeze(-1)

    return out
