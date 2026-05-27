# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch
import torch.nn.functional as F


# Dtypes not natively supported by F.conv1d on CPU — will be temporarily
# upcast to float32 before the convolution and then cast back.
_CONV1D_REQUIRES_UPCAST = frozenset({torch.bfloat16, torch.float16})


# for prefill
def causal_conv1d_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    activation: str | None = "silu",
) -> torch.Tensor:
    out = torch.empty_like(x)
    state_len = weight.shape[1] - 1
    assert activation in {None, "silu", "swish"}

    # F.conv1d on CPU does not always support BF16/FP16; upcast if needed.
    input_dtype = x.dtype
    compute_dtype = torch.float32 if input_dtype in _CONV1D_REQUIRES_UPCAST else input_dtype
    if compute_dtype != input_dtype:
        x = x.float()
        weight = weight.float()
        if bias is not None:
            bias = bias.float()

    seq_begin_end_idx = [
        (int(query_start_loc[idx].item()), int(query_start_loc[idx + 1].item()))
        for idx in range(query_start_loc.shape[0] - 1)
    ]
    weight = weight.unsqueeze(1)
    for seq_idx, (bos, eos) in enumerate(seq_begin_end_idx):
        slot = int(cache_indices[seq_idx].item())

        seq_x = x[:, bos:eos].unsqueeze(0)
        if bool(has_initial_state[seq_idx].item()):
            initial_state = conv_states[slot, :, :state_len].float().unsqueeze(0)
        else:
            initial_state = torch.zeros(
                1,
                weight.shape[0],
                state_len,
                device=seq_x.device,
                dtype=compute_dtype,
            )

        conv_input = torch.cat([initial_state, seq_x], dim=-1)
        seq_out = F.conv1d(
            conv_input,
            weight,
            bias,
            padding=0,
            groups=weight.shape[0],
        )
        seq_out = seq_out[..., -seq_x.shape[-1] :].to(dtype=input_dtype)
        if activation in ("silu", "swish"):
            seq_out = F.silu(seq_out)

        out[:, bos:eos] = seq_out.squeeze(0)
        # Write state back in original dtype
        conv_states[slot, :, :state_len].copy_(
            conv_input[..., -state_len:].squeeze(0).to(conv_states.dtype)
        )

    return out


# for decode
def causal_conv1d_update_torch(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    assert activation in {None, "silu", "swish"}

    _, dim, seq_len = x.shape
    state_len = conv_state.shape[-1]

    # F.conv1d on CPU does not always support BF16/FP16; upcast if needed.
    input_dtype = x.dtype
    compute_dtype = torch.float32 if input_dtype in _CONV1D_REQUIRES_UPCAST else input_dtype
    if compute_dtype != input_dtype:
        x = x.float()
        weight = weight.float()
        if bias is not None:
            bias = bias.float()

    x_new = torch.cat([conv_state.to(compute_dtype), x], dim=-1)
    # Update conv_state in-place (keep original dtype)
    conv_state.copy_(x_new[:, :, -state_len:].to(conv_state.dtype))

    out = F.conv1d(
        x_new,
        weight.unsqueeze(1),
        bias,
        padding=0,
        groups=dim,
    )[:, :, -seq_len:]
    out = out.to(input_dtype)
    if activation in ("silu", "swish"):
        out = F.silu(out)
    return out
