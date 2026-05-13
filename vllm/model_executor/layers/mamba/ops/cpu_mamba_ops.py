# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Pure-PyTorch CPU-fallback implementations of the three Triton/GPU kernel
entry-points consumed by MambaMixer2:

  * causal_conv1d_fn   – prefill depthwise causal convolution
  * causal_conv1d_update – single-step (decode) conv state update
  * mamba_chunk_scan_combined_varlen – full SSM scan for prefill

Signatures match those in
  vllm/model_executor/layers/mamba/ops/causal_conv1d.py  and
  vllm/model_executor/layers/mamba/ops/ssd_combined.py
exactly, so these functions can be monkey-patched transparently.

No GPU, Triton, causal-conv1d, or mamba-ssm package dependencies.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_activation(x: torch.Tensor, activation: str | bool | None) -> torch.Tensor:
    """Element-wise activation dispatch (silu / swish / gelu / identity)."""
    if activation is None:
        return x
    if activation is True or activation in ("silu", "swish"):
        return F.silu(x)
    if activation == "gelu":
        return F.gelu(x)
    return x


def _get_slot_1d_or_2d(
    indices: torch.Tensor,
    seq_i: int,
    block_j: int,
) -> int:
    """Read a cache-slot index from a 1-D or 2-D indices tensor."""
    if indices.dim() == 1:
        return int(indices[seq_i].item())
    return int(indices[seq_i, block_j].item())


# ---------------------------------------------------------------------------
# causal_conv1d_fn  – prefill path
# ---------------------------------------------------------------------------


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    null_block_id: int = NULL_BLOCK_ID,
    block_idx_first_scheduled_token: torch.Tensor | None = None,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    num_computed_tokens: torch.Tensor | None = None,
    block_size_to_align: int = 0,
    metadata=None,
    validate_data: bool = False,
) -> torch.Tensor:
    """
    Depthwise causal 1-D convolution over a packed (varlen) token batch.

    x              : (dim, cu_seqlen)   – channel-first packed sequences.
    weight         : (dim, kernel_width)
    conv_states    : (num_cache_lines, dim, kernel_width - 1)  – updated in-place.
    query_start_loc: (batch + 1,) int32  – cumulative sequence lengths.
    cache_indices  : (batch,) or (batch, n_blocks) int32  – cache-slot mapping.
    has_initial_state: (batch,) bool  – use stored conv state as initial prefix.
    Returns        : (dim, cu_seqlen)  – same shape/dtype as *x*.
    """
    if isinstance(activation, bool) and activation:
        activation = "silu"

    original_dtype = x.dtype
    # Work in float32 for numerical stability; cast back at the end.
    x_f32 = x.float()
    w_f32 = weight.float()
    b_f32 = bias.float() if bias is not None else None
    # Depthwise-conv weight shape required by F.conv1d: (dim, 1, width)
    w_depthwise = w_f32.unsqueeze(1)

    dim, _cu = x_f32.shape
    _, width = w_f32.shape
    state_len = width - 1  # length of the stored conv state per cache line
    batch = query_start_loc.size(0) - 1

    out_f32 = torch.zeros_like(x_f32)  # pre-filled with 0 for skipped seqs

    for i in range(batch):
        seq_start = int(query_start_loc[i].item())
        seq_end = int(query_start_loc[i + 1].item())
        seqlen = seq_end - seq_start
        if seqlen == 0:
            continue

        # ---- determine read/write cache slots ---------------------------------
        if cache_indices is not None:
            if cache_indices.dim() == 2:
                # APC (Automatic Prefix Caching) mode – 2-D index tensor
                init_blk = int(initial_state_idx[i].item()) if initial_state_idx is not None else 0
                read_slot = int(cache_indices[i, init_blk].item())
                last_blk = (
                    int(block_idx_last_scheduled_token[i].item())
                    if block_idx_last_scheduled_token is not None
                    else init_blk
                )
                write_slot = int(cache_indices[i, last_blk].item())
            else:
                # Simple mode – 1-D index tensor
                read_slot = int(cache_indices[i].item())
                write_slot = read_slot
                init_blk = 0
        else:
            read_slot = i
            write_slot = i
            init_blk = 0

        # Skip padded / null slots
        if read_slot == null_block_id:
            continue

        # ---- build initial state (left padding for causal conv) ---------------
        if state_len > 0:
            if has_initial_state is not None and bool(has_initial_state[i].item()):
                # Load stored conv state: (dim, state_len)
                init_state = conv_states[read_slot].float()
            else:
                init_state = torch.zeros(
                    dim, state_len, dtype=torch.float32, device=x_f32.device
                )
        else:
            init_state = torch.empty(dim, 0, dtype=torch.float32, device=x_f32.device)

        # ---- causal convolution -----------------------------------------------
        x_seq = x_f32[:, seq_start:seq_end]  # (dim, seqlen)

        # Prepend the stored initial state so that position 0 of the output
        # has access to kernel_width tokens (state_len prior + current).
        if state_len > 0:
            full_in = torch.cat([init_state, x_seq], dim=1)  # (dim, state_len+seqlen)
        else:
            full_in = x_seq  # (dim, seqlen)

        # F.conv1d cross-correlation matches the Triton kernel's weight ordering:
        #   output[d,t] = sum_k  weight[d,k] * full_in[d, t+k]
        # because full_in[t] to full_in[t+width-1] spans exactly the causal window.
        conv_out = F.conv1d(
            full_in.unsqueeze(0),  # (1, dim, state_len+seqlen)
            w_depthwise,
            bias=b_f32,
            groups=dim,
        ).squeeze(0)  # (dim, seqlen)

        # Apply activation element-wise to this sequence's output
        conv_out = _apply_activation(conv_out, activation)
        out_f32[:, seq_start:seq_end] = conv_out

        # ---- update conv state in the cache -----------------------------------
        if state_len > 0 and write_slot != null_block_id and write_slot != pad_slot_id:
            # The new state is the last *state_len* positions of the causal window.
            # full_in has length state_len + seqlen; slicing from seqlen onwards
            # gives exactly the last state_len positions.
            new_state = full_in[:, seqlen:]  # (dim, state_len)
            conv_states[write_slot].copy_(new_state.to(conv_states.dtype))

    return out_f32.to(original_dtype)


# ---------------------------------------------------------------------------
# causal_conv1d_update  – decode (single-step) path
# ---------------------------------------------------------------------------


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    null_block_id: int = NULL_BLOCK_ID,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    validate_data: bool = False,
) -> torch.Tensor:
    """
    Single decode-step depthwise causal conv1d with in-place state update.

    Supports three input layouts:
      * (batch, dim)            – one token per sequence (unsqueezed internally)
      * (batch, dim, seqlen)    – multiple tokens per sequence
      * (num_tokens, dim)       – varlen (when *query_start_loc* is provided)

    conv_state   : (num_cache_lines, dim, state_len)  – mutated in-place.
    weight       : (dim, kernel_width)
    Returns      : same shape/dtype as *x* (in-place modification).
    """
    if isinstance(activation, bool):
        activation = "silu" if activation else None

    original_dtype = x.dtype
    x = x.to(conv_state.dtype)  # work in conv_state's dtype

    _, width = weight.shape
    state_len = width - 1

    w_f32 = weight.float()
    b_f32 = bias.float() if bias is not None else None
    w_depthwise = w_f32.unsqueeze(1)  # (dim, 1, width) for F.conv1d groups=dim

    # ---- varlen decode mode (query_start_loc provided) -----------------------
    if query_start_loc is not None:
        _num_tokens, dim = x.shape
        batch = (
            conv_state_indices.size(0)
            if conv_state_indices is not None
            else (query_start_loc.size(0) - 1)
        )

        for i in range(batch):
            seq_start = int(query_start_loc[i].item())
            seq_end = int(query_start_loc[i + 1].item())
            actual_len = seq_end - seq_start
            if actual_len == 0:
                continue

            # cache slot lookup
            read_slot, write_slot = _decode_slots(
                conv_state_indices, i, initial_state_idx, block_idx_last_scheduled_token
            )
            if read_slot == null_block_id:
                continue

            cur_state = conv_state[read_slot].float()  # (dim, state_len)
            x_seq = x[seq_start:seq_end].float().t()  # (dim, actual_len)

            full_in = torch.cat([cur_state, x_seq], dim=1)  # (dim, state_len+actual_len)

            # Update stored state: last *state_len* positions of the window
            if state_len > 0 and write_slot != null_block_id:
                conv_state[write_slot].copy_(
                    full_in[:, actual_len:].to(conv_state.dtype)
                )

            # Compute conv output and apply activation
            out_seq = F.conv1d(
                full_in.unsqueeze(0), w_depthwise, bias=b_f32, groups=dim
            ).squeeze(0)  # (dim, actual_len)
            out_seq = _apply_activation(out_seq, activation)

            x[seq_start:seq_end].copy_(out_seq.t().to(x.dtype))

        return x.to(original_dtype)

    # ---- batched (non-varlen) mode -------------------------------------------
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)  # (batch, dim, 1)

    batch, dim, seqlen = x.shape

    for i in range(batch):
        read_slot, write_slot = _decode_slots(
            conv_state_indices, i, initial_state_idx, block_idx_last_scheduled_token
        )
        if read_slot == null_block_id:
            continue

        cur_state = conv_state[read_slot].float()  # (dim, state_len)
        x_seq = x[i].float()  # (dim, seqlen)

        full_in = torch.cat([cur_state, x_seq], dim=1)  # (dim, state_len+seqlen)

        if state_len > 0 and write_slot != null_block_id:
            conv_state[write_slot].copy_(
                full_in[:, seqlen:].to(conv_state.dtype)
            )

        out_seq = F.conv1d(
            full_in.unsqueeze(0), w_depthwise, bias=b_f32, groups=dim
        ).squeeze(0)  # (dim, seqlen)
        out_seq = _apply_activation(out_seq, activation)

        x[i].copy_(out_seq.to(x.dtype))

    if unsqueeze:
        x = x.squeeze(-1)

    return x.to(original_dtype)


def _decode_slots(
    conv_state_indices: torch.Tensor | None,
    seq_i: int,
    initial_state_idx: torch.Tensor | None,
    block_idx_last_scheduled_token: torch.Tensor | None,
) -> tuple[int, int]:
    """Return (read_slot, write_slot) for a single decode sequence."""
    if conv_state_indices is None:
        return seq_i, seq_i

    if conv_state_indices.dim() == 2:
        init_blk = int(initial_state_idx[seq_i].item()) if initial_state_idx is not None else 0
        read_slot = int(conv_state_indices[seq_i, init_blk].item())
        last_blk = (
            int(block_idx_last_scheduled_token[seq_i].item())
            if block_idx_last_scheduled_token is not None
            else init_blk
        )
        write_slot = int(conv_state_indices[seq_i, last_blk].item())
    else:
        read_slot = int(conv_state_indices[seq_i].item())
        write_slot = read_slot

    return read_slot, write_slot


# ---------------------------------------------------------------------------
# mamba_chunk_scan_combined_varlen  – SSM prefill scan
# ---------------------------------------------------------------------------


def mamba_chunk_scan_combined_varlen(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.Tensor,
    cu_chunk_seqlens: torch.Tensor,
    last_chunk_indices: torch.Tensor,
    seq_idx: torch.Tensor,
    out: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    dt_softplus: bool = False,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
    return_intermediate_states: bool = False,
    state_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Sequential SSM scan (Mamba-2 / SSD) for a packed varlen batch.

    Tensor shapes
    -------------
    x              : (seqlen, nheads, headdim)
    dt             : (seqlen, nheads)
    A              : (nheads,)
    B              : (seqlen, ngroups, dstate)
    C              : (seqlen, ngroups, dstate)
    D              : (nheads,) or (nheads, headdim) or None
    z              : (seqlen, nheads, headdim) or None
    dt_bias        : (nheads,) or None
    cu_seqlens     : (batch + 1,)   cumulative sequence-token counts
    cu_chunk_seqlens: (nchunks + 1,) cumulative chunk-token counts
    last_chunk_indices: (batch,)    index of the last chunk per sequence
    seq_idx        : (nchunks,)     sequence index for each chunk
    out            : (seqlen, nheads, headdim)  preallocated; updated in-place
    initial_states : (batch, nheads, headdim, dstate) or None
    state_dtype    : desired dtype for the returned state tensor

    Returns
    -------
    If return_intermediate_states is True:
        (nchunks, nheads, headdim, dstate)  – state at end of every chunk
    Else:
        (batch, nheads, headdim, dstate)    – final state per sequence
    """
    _seqlen, nheads, headdim = x.shape
    _, ngroups, dstate = B.shape
    heads_per_group = nheads // ngroups
    batch = int(cu_seqlens.size(0)) - 1
    nchunks = int(cu_chunk_seqlens.size(0)) - 1

    s_dtype = state_dtype if state_dtype is not None else C.dtype
    device = x.device

    # ------------------------------------------------------------------
    # 1. Discretise Δt  →  dt_disc : (seqlen, nheads)
    # ------------------------------------------------------------------
    dt_f = dt.float()
    if dt_bias is not None:
        dt_f = dt_f + dt_bias.float().unsqueeze(0)  # broadcast over seqlen
    if dt_softplus:
        dt_f = F.softplus(dt_f)
    lo, hi = dt_limit
    dt_f = dt_f.clamp(lo, hi)  # (seqlen, nheads)

    # ------------------------------------------------------------------
    # 2. Pre-compute dA = exp(dt_disc * A) : (seqlen, nheads)
    # ------------------------------------------------------------------
    dA = torch.exp(dt_f * A.float().unsqueeze(0))  # (seqlen, nheads)

    # ------------------------------------------------------------------
    # 3. Precompute head-to-group index mapping  (nheads,)
    # ------------------------------------------------------------------
    if heads_per_group == 1:
        # each head corresponds directly to one group
        group_for_head = torch.arange(ngroups, device=device)
    else:
        group_for_head = torch.arange(nheads, device=device) // heads_per_group

    # ------------------------------------------------------------------
    # 4. Initialise per-sequence hidden states  h: (nheads, headdim, dstate)
    # ------------------------------------------------------------------
    h_states: list[torch.Tensor] = []
    for s in range(batch):
        if initial_states is not None:
            h = initial_states[s].float().clone()
        else:
            h = torch.zeros(nheads, headdim, dstate, dtype=torch.float32, device=device)
        h_states.append(h)

    # ------------------------------------------------------------------
    # 5. Sequential scan over chunks
    # ------------------------------------------------------------------
    chunk_end_states: list[torch.Tensor] = []  # filled when return_intermediate

    x_f = x.float()
    B_f = B.float()
    C_f = C.float()
    z_f = z.float() if z is not None else None
    D_f = D.float() if D is not None else None

    for c in range(nchunks):
        t_start = int(cu_chunk_seqlens[c].item())
        t_end = int(cu_chunk_seqlens[c + 1].item())
        s = int(seq_idx[c].item())

        h = h_states[s]  # (nheads, headdim, dstate)

        for t in range(t_start, t_end):
            # --- Expand B and C from group to head dimension ---------------
            # B_t: (nheads, dstate),  C_t: (nheads, dstate)
            B_t = B_f[t][group_for_head]  # (nheads, dstate)
            C_t = C_f[t][group_for_head]  # (nheads, dstate)

            # dA_t: (nheads,) → (nheads, 1, 1) for broadcasting with h
            dA_t = dA[t].view(nheads, 1, 1)

            # dB_t = dt_disc_t · B_t : (nheads, dstate) → (nheads, 1, dstate)
            dB_t = (dt_f[t].unsqueeze(-1) * B_t).unsqueeze(1)

            # x_t: (nheads, headdim) → (nheads, headdim, 1)
            x_t = x_f[t].unsqueeze(-1)

            # SSM state update: h = dA * h + dB * x
            h = dA_t * h + dB_t * x_t  # (nheads, headdim, dstate)

            # Output: y_t = (h * C_t).sum(-1) + D * x_t
            # C_t_exp: (nheads, 1, dstate)
            y_t = (h * C_t.unsqueeze(1)).sum(-1)  # (nheads, headdim)

            if D_f is not None:
                if D_f.dim() == 1:
                    y_t = y_t + D_f.unsqueeze(-1) * x_f[t]
                else:
                    y_t = y_t + D_f * x_f[t]

            # Optional z-gate
            if z_f is not None:
                y_t = F.silu(z_f[t]) * y_t

            out[t] = y_t.to(out.dtype)

        h_states[s] = h

        if return_intermediate_states:
            chunk_end_states.append(h.to(s_dtype).clone())

    # ------------------------------------------------------------------
    # 6. Assemble return tensor
    # ------------------------------------------------------------------
    if return_intermediate_states:
        if len(chunk_end_states) == 0:
            return torch.zeros(0, nheads, headdim, dstate, dtype=s_dtype, device=device)
        return torch.stack(chunk_end_states, dim=0)  # (nchunks, nheads, headdim, dstate)

    # Non-intermediate: one final state per sequence  (batch, nheads, headdim, dstate)
    if batch == 0:
        return torch.zeros(0, nheads, headdim, dstate, dtype=s_dtype, device=device)
    final = torch.stack([h_states[s].to(s_dtype) for s in range(batch)], dim=0)
    return final
