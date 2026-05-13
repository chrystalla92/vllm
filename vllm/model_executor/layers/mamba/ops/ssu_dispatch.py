# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Dispatch module for Mamba selective state update (SSU) backends.

Provides a unified `selective_state_update` function that dispatches to
either the Triton or FlashInfer backend based on the configured
`MambaBackendEnum`. Follows SGLang's dispatch pattern adapted for vLLM.
"""

from abc import ABC, abstractmethod

import torch

from vllm.config.mamba import MambaBackendEnum, MambaConfig
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec

logger = init_logger(__name__)


class MambaSSUBackend(ABC):
    """Abstract base class for Mamba SSU backends."""

    def __init__(self, mamba_config: MambaConfig):
        self._mamba_config = mamba_config

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        z: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        dst_state_batch_indices: torch.Tensor | None = None,
        null_block_id: int = NULL_BLOCK_ID,
        out: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        is_blackwell: bool = False,
    ) -> None: ...


class TritonSSUBackend(MambaSSUBackend):
    """Triton-based SSU backend (vLLM's default)."""

    def __init__(self, mamba_config: MambaConfig):
        super().__init__(mamba_config)
        from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
            selective_state_update as _triton_selective_state_update,
        )

        self._kernel = _triton_selective_state_update

    @property
    def name(self) -> str:
        return "triton"

    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        z: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        dst_state_batch_indices: torch.Tensor | None = None,
        null_block_id: int = NULL_BLOCK_ID,
        out: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        is_blackwell: bool = False,
    ) -> None:
        self._kernel(
            state,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            state_batch_indices=state_batch_indices,
            dst_state_batch_indices=dst_state_batch_indices,
            null_block_id=null_block_id,
            out=out,
            num_accepted_tokens=num_accepted_tokens,
            cu_seqlens=cu_seqlens,
            is_blackwell=is_blackwell,
            enable_stochastic_rounding=self._mamba_config.enable_stochastic_rounding,
            cache_philox_rounds=self._mamba_config.stochastic_rounding_philox_rounds,
        )


class FlashInferSSUBackend(MambaSSUBackend):
    """FlashInfer-based SSU backend."""

    def __init__(self, mamba_config: MambaConfig):
        super().__init__(mamba_config)
        try:
            from flashinfer.mamba import selective_state_update as _fi_ssu
        except ImportError as e:
            raise ImportError(
                "FlashInfer is required for the flashinfer Mamba SSU backend. "
                "Please install flashinfer (>= 0.6.4): "
                "pip install flashinfer-python"
            ) from e
        self._kernel = _fi_ssu

    @property
    def name(self) -> str:
        return "flashinfer"

    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        z: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        dst_state_batch_indices: torch.Tensor | None = None,
        null_block_id: int = NULL_BLOCK_ID,
        out: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        is_blackwell: bool = False,
    ) -> None:
        rand_seed = (
            torch.randint(0, 2**32, (1,), device=state.device)
            if self._mamba_config.enable_stochastic_rounding
            else None
        )

        self._kernel(
            state,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            state_batch_indices=state_batch_indices,
            dst_state_batch_indices=dst_state_batch_indices,
            cu_seqlens=cu_seqlens,
            num_accepted_tokens=num_accepted_tokens,
            cache_steps=state_batch_indices.size(-1)
            if cu_seqlens is not None and state_batch_indices is not None
            else 0,
            pad_slot_id=null_block_id,
            out=out,
            rand_seed=rand_seed,
            philox_rounds=self._mamba_config.stochastic_rounding_philox_rounds or 10,
        )


class CPUSSUBackend(MambaSSUBackend):
    """Pure-PyTorch SSU backend for CPU platforms.

    Implements the SSM selective state update (decode step) without any
    GPU-specific or Triton dependencies.

    The computation follows:
        dt  = softplus(dt + dt_bias)   [if dt_softplus]
        dA  = exp(dt * A)
        dB  = dt * B
        new_state = dA * state + dB * x
        y   = (new_state * C).sum(-1) + D * x
    """

    @property
    def name(self) -> str:
        return "cpu"

    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        z: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        dst_state_batch_indices: torch.Tensor | None = None,
        null_block_id: int = NULL_BLOCK_ID,
        out: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        is_blackwell: bool = False,
    ) -> None:
        import torch.nn.functional as F

        # Normalise all inputs to 4-D / per-head form, matching the Triton
        # wrapper's convention:
        #   state   : (total_slots_or_batch, nheads, dim, dstate)
        #   x/dt    : (batch, nheads, dim)
        #   A       : (nheads, dim, dstate)
        #   B/C     : (batch, ngroups, dstate)
        #   D       : (nheads, dim)
        #   dt_bias : (nheads, dim)
        #   out     : (batch, nheads, dim)
        if state.dim() == 3:
            state = state.unsqueeze(1)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if dt.dim() == 2:
            dt = dt.unsqueeze(1)
        if A.dim() == 2:
            A = A.unsqueeze(0)
        if B.dim() == 2:
            B = B.unsqueeze(1)
        if C.dim() == 2:
            C = C.unsqueeze(1)
        if D.dim() == 1:
            D = D.unsqueeze(0)
        if dt_bias.dim() == 1:
            dt_bias = dt_bias.unsqueeze(0)
        if out is not None and out.dim() == 2:
            out = out.unsqueeze(1)
        if state_batch_indices is not None and state_batch_indices.dim() == 1:
            state_batch_indices = state_batch_indices.unsqueeze(1)
        if dst_state_batch_indices is not None and dst_state_batch_indices.dim() == 1:
            dst_state_batch_indices = dst_state_batch_indices.unsqueeze(1)
        if dst_state_batch_indices is None:
            dst_state_batch_indices = state_batch_indices

        batch = x.shape[0]
        _, nheads, dim, dstate = state.shape
        ngroups = B.shape[1]
        heads_per_group = nheads // ngroups

        # Cast to float32 for numerically stable computation.
        compute_dtype = torch.float32
        x_f = x.to(compute_dtype)
        dt_f = dt.to(compute_dtype)
        A_f = A.to(compute_dtype)
        B_f = B.to(compute_dtype)
        C_f = C.to(compute_dtype)
        D_f = D.to(compute_dtype)
        dt_bias_f = dt_bias.to(compute_dtype)

        # 1. Discretise dt.
        dt_f = dt_f + dt_bias_f.unsqueeze(0)  # (batch, nheads, dim)
        if dt_softplus:
            dt_f = F.softplus(dt_f)

        def _ssm_step(
            state_chunk: torch.Tensor,  # (n, nheads, dim, dstate)
            x_chunk: torch.Tensor,      # (n, nheads, dim)
            dt_chunk: torch.Tensor,     # (n, nheads, dim)
            B_chunk: torch.Tensor,      # (n, ngroups, dstate)
            C_chunk: torch.Tensor,      # (n, ngroups, dstate)
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Return (new_state, y) for a subset of the batch."""
            # Expand B/C from group-level to head-level.
            # (n, ngroups, dstate) -> (n, nheads, dstate)
            B_h = B_chunk.repeat_interleave(heads_per_group, dim=1)
            C_h = C_chunk.repeat_interleave(heads_per_group, dim=1)

            # 2. dA = exp(dt * A)  ->  (n, nheads, dim, dstate)
            dA = torch.exp(
                dt_chunk.unsqueeze(-1) * A_f.unsqueeze(0)
            )

            # 3. dB = dt * B  ->  (n, nheads, dim, dstate)
            #    B is shared across `dim`, so unsqueeze the dim axis.
            dB = dt_chunk.unsqueeze(-1) * B_h.unsqueeze(2)

            # 4. new_state = dA * state + dB * x
            new_s = dA * state_chunk + dB * x_chunk.unsqueeze(-1)

            # 5. y = (new_state * C).sum(-1) + D * x
            y = (new_s * C_h.unsqueeze(2)).sum(-1) + D_f.unsqueeze(0) * x_chunk

            return new_s, y

        if state_batch_indices is not None:
            # Paged KV-cache path: state is (total_slots, nheads, dim, dstate).
            # Use the first column of state_batch_indices for decode (seqlen=1).
            src_idx = state_batch_indices[:batch, 0]  # (batch,)
            dst_idx = dst_state_batch_indices[:batch, 0]  # (batch,)

            valid_mask = src_idx != null_block_id

            if valid_mask.any():
                v_src = src_idx[valid_mask]
                v_dst = dst_idx[valid_mask]

                gathered = state[v_src].to(compute_dtype)

                new_s, y = _ssm_step(
                    gathered,
                    x_f[valid_mask],
                    dt_f[valid_mask],
                    B_f[valid_mask],
                    C_f[valid_mask],
                )

                # Write updated state back into the paged cache.
                state[v_dst] = new_s.to(state.dtype)

                if out is not None:
                    out[valid_mask] = y.to(out.dtype)
        else:
            # Non-paged path: state is (batch, nheads, dim, dstate).
            state_f = state.to(compute_dtype)
            new_s, y = _ssm_step(state_f, x_f, dt_f, B_f, C_f)

            state.copy_(new_s.to(state.dtype))
            if out is not None:
                out.copy_(y.to(out.dtype))


_BACKEND_REGISTRY: dict[MambaBackendEnum, type[MambaSSUBackend]] = {
    MambaBackendEnum.TRITON: TritonSSUBackend,
    MambaBackendEnum.FLASHINFER: FlashInferSSUBackend,
    MambaBackendEnum.CPU: CPUSSUBackend,
}

_mamba_ssu_backend: MambaSSUBackend | None = None


def initialize_mamba_ssu_backend(
    mamba_config: MambaConfig,
    kv_cache_config: KVCacheConfig,
) -> None:
    """Initialize the global Mamba SSU backend.

    No-op if `kv_cache_config` contains no specs that call
    selective_state_update.
    """
    if not any(
        isinstance(g.kv_cache_spec, MambaSpec)
        and g.kv_cache_spec.mamba_type in ("mamba1", "mamba2")
        for g in kv_cache_config.kv_cache_groups
    ):
        return

    global _mamba_ssu_backend

    backend = mamba_config.backend
    if backend not in _BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown Mamba SSU backend: {backend}. "
            f"Valid options: {list(_BACKEND_REGISTRY.keys())}"
        )

    backend_cls = _BACKEND_REGISTRY[backend]
    if isinstance(_mamba_ssu_backend, backend_cls):
        return

    _mamba_ssu_backend = backend_cls(mamba_config)
    logger.info("Using %s Mamba SSU backend.", _mamba_ssu_backend.name)


def get_mamba_ssu_backend() -> MambaSSUBackend:
    """Get the current Mamba SSU backend. Raises if not initialized."""
    if _mamba_ssu_backend is None:
        raise RuntimeError(
            "Mamba SSU backend has not been initialized. "
            "Call initialize_mamba_ssu_backend() first."
        )
    return _mamba_ssu_backend


def selective_state_update(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    z: torch.Tensor | None = None,
    dt_softplus: bool = False,
    state_batch_indices: torch.Tensor | None = None,
    dst_state_batch_indices: torch.Tensor | None = None,
    null_block_id: int = NULL_BLOCK_ID,
    out: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    is_blackwell: bool = False,
) -> None:
    """Unified dispatch for Mamba selective state update.

    Delegates to the initialized backend (Triton or FlashInfer).
    """
    get_mamba_ssu_backend()(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        dt_bias,
        z=z,
        dt_softplus=dt_softplus,
        state_batch_indices=state_batch_indices,
        dst_state_batch_indices=dst_state_batch_indices,
        null_block_id=null_block_id,
        out=out,
        num_accepted_tokens=num_accepted_tokens,
        cu_seqlens=cu_seqlens,
        is_blackwell=is_blackwell,
    )
