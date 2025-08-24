from __future__ import annotations
import math
from typing import Optional, Tuple

import torch

# Triton is optional; fall back gracefully
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


# ---- Triton kernel: fused double GEMM ----
# Computes:
#   OUT_S = K @ SIN
#   OUT_C = K @ COS
# where K: [N, N], SIN: [N, R], COS: [N, R]
# We accumulate both results in the same tile loop to reuse K tiles.
if _TRITON_AVAILABLE:
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4),
            triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
            triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4),
        ],
        key=['N', 'R'],
    )
    @triton.jit
    def _k_mul_two_kernel(
        K_ptr, SIN_ptr, COS_ptr, OUTS_ptr, OUTC_ptr,
        N: tl.constexpr, R: tl.constexpr,
        stride_km, stride_kn,
        stride_vm, stride_vn,
        stride_om, stride_on,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)  # rows of K / OUT
        pid_n = tl.program_id(axis=1)  # cols of SIN/COS / OUT

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Accumulators in FP32
        accS = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        accC = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # k-loop
        for k0 in range(0, N, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)

            # Load K tile [BLOCK_M, BLOCK_K]
            k_mask = (offs_m[:, None] < N) & (offs_k[None, :] < N)
            K_tile = tl.load(
                K_ptr + offs_m[:, None] * stride_km + offs_k[None, :] * stride_kn,
                mask=k_mask, other=0.0
            )

            # Load SIN/COS tiles [BLOCK_K, BLOCK_N]
            v_mask = (offs_k[:, None] < N) & (offs_n[None, :] < R)
            SIN_tile = tl.load(
                SIN_ptr + offs_k[:, None] * stride_vm + offs_n[None, :] * stride_vn,
                mask=v_mask, other=0.0
            )
            COS_tile = tl.load(
                COS_ptr + offs_k[:, None] * stride_vm + offs_n[None, :] * stride_vn,
                mask=v_mask, other=0.0
            )

            # Fused dot products: accS += K @ SIN; accC += K @ COS
            accS += tl.dot(K_tile, SIN_tile)
            accC += tl.dot(K_tile, COS_tile)

        # Store results
        o_mask = (offs_m[:, None] < N) & (offs_n[None, :] < R)
        tl.store(
            OUTS_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            accS, mask=o_mask
        )
        tl.store(
            OUTC_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            accC, mask=o_mask
        )


def fused_k_mul_two(
    K: torch.Tensor,  # [N, N], float32/float16/bfloat16, CUDA
    sin_in: torch.Tensor,  # [N, R], same dtype/device
    cos_in: torch.Tensor,  # [N, R], same dtype/device
    force_triton: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute (K @ sin_in, K @ cos_in) with a single fused pass over K.

    Returns:
        (S, C): both [N, R] tensors on same device/dtype as inputs.
    """
    if (not _TRITON_AVAILABLE) and (force_triton):
        raise RuntimeError("Triton not available but force_triton=True.")

    # Preconditions
    assert K.is_cuda and sin_in.is_cuda and cos_in.is_cuda, "Inputs must be CUDA tensors"
    assert K.shape[0] == K.shape[1], "K must be square [N,N]"
    N = K.shape[0]
    assert sin_in.shape == (N, sin_in.shape[1])
    assert cos_in.shape == (N, cos_in.shape[1])
    R = sin_in.shape[1]
    assert cos_in.shape[1] == R

    # Ensure contiguous row-major buffers
    Kc = K.contiguous()
    Sc = sin_in.contiguous()
    Cc = cos_in.contiguous()

    # Dtype: support fp32/fp16/bf16; accumulators are fp32 inside kernel
    dtype = Kc.dtype
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        Kc = Kc.float(); dtype = torch.float32
    if Sc.dtype != dtype: Sc = Sc.to(dtype)
    if Cc.dtype != dtype: Cc = Cc.to(dtype)

    # Allocate outputs
    OutS = torch.empty((N, R), device=K.device, dtype=torch.float32)
    OutC = torch.empty((N, R), device=K.device, dtype=torch.float32)

    if not _TRITON_AVAILABLE:
        # Fallback: two matmuls (still faster than Python loops)
        OutS.copy_(Kc @ Sc)
        OutC.copy_(Kc @ Cc)
        return OutS, OutC

    # Strides (row-major)
    stride_km, stride_kn = Kc.stride()
    stride_vm, stride_vn = Sc.stride()
    stride_om, stride_on = OutS.stride()

    grid = lambda META: (triton.cdiv(N, META['BLOCK_M']),
                         triton.cdiv(R, META['BLOCK_N']))

    _k_mul_two_kernel[grid](
        Kc, Sc, Cc, OutS, OutC,
        N, R,
        stride_km, stride_kn,
        stride_vm, stride_vn,
        stride_om, stride_on,
    )
    return OutS, OutC
