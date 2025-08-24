"""
GPU-accelerated kernels for Phase-Ising optimization
"""
from .triton_ops import fused_k_mul_two, _TRITON_AVAILABLE

__all__ = ["fused_k_mul_two", "_TRITON_AVAILABLE"]
