# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm import _custom_ops as ops

from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.layers.mamba.ops.cpu.causal_conv1d import (
    causal_conv1d_torch,
    causal_conv1d_update_torch,
)
from vllm.model_executor.layers.mamba.ops.cpu.recurrent_gated_delta_rule import (
    chunk_gated_delta_rule,
    gdn_gating,
    recurrent_gated_delta_rule,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

_CPU_GDN_ATTENTION_OPS_REGISTERED = False


def _get_cpu_gdn_packed_conv_weight(layer, conv_weights: torch.Tensor) -> torch.Tensor:
    packed_weight = getattr(layer, "_cpu_gdn_conv1d_packed_weight", None)
    if packed_weight is None:
        packed_weight = ops.causal_conv1d_weight_pack(conv_weights.contiguous())
        layer._cpu_gdn_conv1d_packed_weight = packed_weight
    return packed_weight


def _can_use_cpu_gdn_decode_conv_update(
    layer,
    mixed_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices: torch.Tensor,
) -> bool:
    return (
        layer.activation in ("silu", "swish")
        and layer.conv_kernel_size == 4
        and mixed_qkv.dim() == 2
        and mixed_qkv.is_contiguous()
        and conv_state.dim() == 3
        and conv_state.is_contiguous()
        and conv_state.dtype == mixed_qkv.dtype
        and conv_weights.dtype == mixed_qkv.dtype
        and conv_state.shape[1] == mixed_qkv.shape[1]
        and conv_state.shape[2] == layer.conv_kernel_size - 1
        and conv_weights.shape == (mixed_qkv.shape[1], layer.conv_kernel_size)
        and state_indices.dtype == torch.int32
        and not is_conv_state_dim_first()
    )


def _can_use_cpu_gdn_prefill_chunk_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> bool:
    return (
        query.dim() == 4
        and key.shape == query.shape
        and value.dim() == 4
        and g.dim() == 3
        and beta.dim() == 3
        and initial_state.dim() == 4
        and cu_seqlens.dim() == 1
        and query.dtype == torch.bfloat16
        and key.dtype == torch.bfloat16
        and value.dtype == torch.bfloat16
        and beta.dtype == torch.bfloat16
        and g.dtype == torch.float32
        and initial_state.dtype == torch.float32
        and cu_seqlens.dtype == torch.int32
        and query.shape[0] == 1
        and g.shape[:2] == query.shape[:2]
        and beta.shape == g.shape
        and initial_state.shape[0] + 1 == cu_seqlens.shape[0]
        and value.shape[2] == g.shape[2]
        and initial_state.shape[1] == value.shape[2]
        and initial_state.shape[2] == query.shape[3]
        and initial_state.shape[3] == value.shape[3]
        and query.shape[3] % 32 == 0
        and value.shape[3] % 32 == 0
    )


def cpu_gdn_attention_core(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """CPU custom op for the core GDN attention computation."""
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]

    attn_metadata = forward_context.attn_metadata

    if attn_metadata is None:
        return

    assert isinstance(attn_metadata, dict)
    attn_metadata_i = attn_metadata[layer.prefix]
    assert isinstance(attn_metadata_i, GDNAttentionMetadata)

    if attn_metadata_i.num_actual_tokens == 0:
        return

    assert (
        attn_metadata_i.spec_sequence_masks is None
        and attn_metadata_i.num_accepted_tokens is None
    ), "speculative decode not supported in CPU GDN attention."

    state_indices_tensor = attn_metadata_i.non_spec_state_indices_tensor
    query_start_loc = attn_metadata_i.non_spec_query_start_loc
    assert state_indices_tensor is not None
    assert query_start_loc is not None

    # [num_allocated_slots, conv_dim, kernel - 1]
    conv_state = layer.kv_cache[0]
    if not is_conv_state_dim_first():
        conv_state = conv_state.transpose(-1, -2)

    # [num_allocated_slots, num_v_heads / tp_size, v_dim, k_dim]
    ssm_state = layer.kv_cache[1]

    num_decodes = attn_metadata_i.num_decodes
    num_decode_tokens = attn_metadata_i.num_decode_tokens
    num_prefills = attn_metadata_i.num_prefills
    num_prefill_tokens = attn_metadata_i.num_prefill_tokens

    conv_weights = layer.conv1d.weight.view(
        layer.conv1d.weight.size(0), layer.conv1d.weight.size(2)
    )

    # all decode requests (batched)
    if num_decodes > 0:
        decode_mixed_qkv = mixed_qkv[:num_decode_tokens]
        decode_b = b[:num_decode_tokens]
        decode_a = a[:num_decode_tokens]
        decode_state_indices = state_indices_tensor[:num_decodes]

        if _can_use_cpu_gdn_decode_conv_update(
            layer,
            decode_mixed_qkv,
            conv_state,
            conv_weights,
            decode_state_indices,
        ):
            decode_mixed_qkv = ops.causal_conv1d_update_cpu(
                decode_mixed_qkv,
                conv_state,
                _get_cpu_gdn_packed_conv_weight(layer, conv_weights),
                layer.conv1d.bias,
                True,
                None,
                decode_state_indices,
                -1,
                True,
            )
        else:
            decode_conv_state = conv_state[decode_state_indices].contiguous()
            decode_mixed_qkv = causal_conv1d_update_torch(
                # [B, dim] -> [B, dim, 1]
                x=decode_mixed_qkv.unsqueeze(-1),
                conv_state=decode_conv_state,
                weight=conv_weights,
                bias=layer.conv1d.bias,
                activation=layer.activation,
            ).squeeze(-1)
            conv_state[decode_state_indices] = decode_conv_state

        query, key, value = layer.rearrange_mixed_qkv(decode_mixed_qkv)

        # [1, L, H, D] -> [B, 1, H, D] for batched decode
        query = query.transpose(0, 1).contiguous()
        key = key.transpose(0, 1).contiguous()
        value = value.transpose(0, 1).contiguous()

        g, beta_output = gdn_gating(
            A_log=layer.A_log,
            a=decode_a,
            b=decode_b,
            dt_bias=layer.dt_bias,
        )
        if g.ndim == 2:
            g = g.unsqueeze(1)
            beta_output = beta_output.unsqueeze(1)

        initial_state = ssm_state[decode_state_indices].contiguous()
        attn_out, last_recurrent_state = recurrent_gated_delta_rule(
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta_output,
            initial_state=initial_state,
            scale=None,
            use_qk_l2norm_in_kernel=True,
        )
        ssm_state[decode_state_indices] = last_recurrent_state.to(
            ssm_state.dtype
        ).contiguous()
        core_attn_out[:num_decode_tokens] = attn_out.squeeze(1)

    # all prefill requests: (varlen) currently naively loops over sequences
    if num_prefills > 0:
        has_initial_state = attn_metadata_i.has_initial_state
        assert has_initial_state is not None

        prefill_token_start = num_decode_tokens
        prefill_token_end = prefill_token_start + num_prefill_tokens
        prefill_mixed_qkv = mixed_qkv[prefill_token_start:prefill_token_end]
        prefill_b = b[prefill_token_start:prefill_token_end]
        prefill_a = a[prefill_token_start:prefill_token_end]
        prefill_state_indices = state_indices_tensor[
            num_decodes : num_decodes + num_prefills
        ]
        prefill_query_start_loc = (
            query_start_loc[num_decodes : num_decodes + num_prefills + 1]
            - num_decode_tokens
        )
        prefill_has_initial_state = has_initial_state[
            num_decodes : num_decodes + num_prefills
        ]

        prefill_mixed_qkv = causal_conv1d_torch(
            x=prefill_mixed_qkv.transpose(0, 1),
            weight=conv_weights,
            bias=layer.conv1d.bias,
            conv_states=conv_state,
            query_start_loc=prefill_query_start_loc,
            cache_indices=prefill_state_indices,
            has_initial_state=prefill_has_initial_state,
            activation=layer.activation,
        ).transpose(0, 1)

        query, key, value = layer.rearrange_mixed_qkv(prefill_mixed_qkv)
        g, beta = gdn_gating(layer.A_log, prefill_a, prefill_b, layer.dt_bias)
        if g.ndim == 2:
            g = g.unsqueeze(0)
            beta = beta.unsqueeze(0)

        initial_state = ssm_state[prefill_state_indices].contiguous()
        initial_state[~prefill_has_initial_state, ...] = 0
        initial_state = initial_state.to(torch.float32)
        if _can_use_cpu_gdn_prefill_chunk_kernel(
            query,
            key,
            value,
            g,
            beta,
            initial_state,
            prefill_query_start_loc,
        ):
            attn_out, last_recurrent_state = ops.chunk_gated_delta_rule_cpu(
                query,
                key,
                value,
                g,
                beta,
                initial_state,
                True,
                prefill_query_start_loc,
                False,
                True,
            )
        else:
            attn_out, last_recurrent_state = chunk_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                scale=None,
                initial_state=initial_state,
                cu_seqlens=prefill_query_start_loc,
                use_qk_l2norm_in_kernel=True,
            )
        ssm_state[prefill_state_indices] = last_recurrent_state.to(ssm_state.dtype)
        core_attn_out[prefill_token_start:prefill_token_end] = attn_out.squeeze(0)


def cpu_gdn_attention_core_fake(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Fake implementation for torch.compile."""
    return


def register_cpu_gdn_attention_ops() -> None:
    global _CPU_GDN_ATTENTION_OPS_REGISTERED
    if _CPU_GDN_ATTENTION_OPS_REGISTERED:
        return

    direct_register_custom_op(
        op_name="cpu_gdn_attention_core",
        op_func=cpu_gdn_attention_core,
        mutates_args=["core_attn_out"],
        fake_impl=cpu_gdn_attention_core_fake,
    )
    _CPU_GDN_ATTENTION_OPS_REGISTERED = True
