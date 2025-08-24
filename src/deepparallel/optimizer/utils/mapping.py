from __future__ import annotations
from typing import Tuple
import numpy as np

def symmetrize(Q: np.ndarray) -> np.ndarray:
    Q = np.asarray(Q, dtype=np.float64)
    return 0.5 * (Q + Q.T)

def qubo_to_ising(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    Qs = symmetrize(Q)
    n = Qs.shape[0]
    # Standard transform with s in {-1,1}, x=(s+1)/2
    J = 0.25 * Qs
    h = 0.5 * (Qs @ np.ones(n))
    const = 0.25 * float(np.ones(n).T @ Qs @ np.ones(n))
    return J, h, const

def add_cardinality_penalty(Q: np.ndarray, k: int, lam: float) -> np.ndarray:
    Q = np.array(Q, dtype=np.float64, copy=True)
    n = Q.shape[0]
    # λ*(sum_i x_i - k)^2 = λ*(sum_i x_i) + 2λ*sum_{i<j} x_i x_j - 2λk*sum_i x_i + λk^2
    # Diagonal: (1 - 2k)*λ; Off-diagonal: +2λ
    Q += 2.0 * lam * (np.ones((n, n)) - np.eye(n))
    Q[np.diag_indices(n)] += (1.0 - 2.0 * k) * lam
    return Q

def add_capacity_penalty(Q: np.ndarray, w: np.ndarray, B: float, lam: float) -> np.ndarray:
    Q = np.array(Q, dtype=np.float64, copy=True)
    w = np.asarray(w, dtype=np.float64)
    # λ*(w^T x - B)^2 = λ*(x^T ww^T x) - 2λB*w^T x + λB^2
    Q += lam * np.outer(w, w)
    Q[np.diag_indices(Q.shape[0])] += lam * (w**2 - 2.0 * B * w)
    return Q
