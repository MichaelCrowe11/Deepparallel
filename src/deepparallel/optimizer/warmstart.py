from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from collections import OrderedDict
import numpy as np
import hashlib

def _block_stats(block: np.ndarray, eps: float) -> np.ndarray:
    """Return a compact vector of block statistics, robust to scale."""
    if block.size == 0:
        return np.zeros(6, dtype=np.float32)
    a = block
    abs_a = np.abs(a)
    nz_frac = float(np.mean(abs_a > eps))          # density
    mean = float(np.mean(a))
    mean_abs = float(np.mean(abs_a))
    std_abs = float(np.std(abs_a))
    pos_frac = float(np.mean(a > 0))
    # simple spectral proxy: 1-step power iter norm (cheap)
    v = np.ones((a.shape[1],), dtype=np.float64) / max(1, a.shape[1])
    spec = float(np.linalg.norm(a @ v, ord=2))
    return np.array([nz_frac, mean, mean_abs, std_abs, pos_frac, spec], dtype=np.float32)

def structure_fingerprint(
    Q: np.ndarray,
    block_size: int = 64,
    eps: float = 1e-8
) -> Tuple[str, np.ndarray]:
    """
    Produce a structure hash key and an embedding vector for nearest-neighbor warm starts.
    - Divide Q into (ceil(n/bs))^2 blocks; compute robust stats per block.
    - Normalize features; hash quantized features.
    """
    Q = np.asarray(Q, dtype=np.float64)
    n = Q.shape[0]
    bs = max(8, int(block_size))
    B = (int(np.ceil(n / bs)), int(np.ceil(n / bs)))
    feats: List[float] = []
    for bi in range(B[0]):
        i0, i1 = bi * bs, min(n, (bi + 1) * bs)
        for bj in range(B[1]):
            j0, j1 = bj * bs, min(n, (bj + 1) * bs)
            feats.extend(_block_stats(Q[i0:i1, j0:j1], eps).tolist())
    feats = np.asarray(feats, dtype=np.float32)
    # global normalization (z-score)
    mu = feats.mean() if feats.size else 0.0
    sd = feats.std() if feats.size else 1.0
    if sd < 1e-12: sd = 1.0
    norm_feats = (feats - mu) / sd
    # quantize for stable hashing
    q = np.clip(np.round(norm_feats * 128), -32768, 32767).astype(np.int16).tobytes()
    h = hashlib.blake2b(q, digest_size=16).hexdigest()
    key = f"n{n}-bs{bs}-b{B[0]}x{B[1]}-{h}"
    # L2-normalize embedding for cosine similarity
    emb = norm_feats.astype(np.float32)
    denom = float(np.linalg.norm(emb) + 1e-12)
    emb /= denom
    return key, emb

@dataclass
class CacheItem:
    key: str
    emb: np.ndarray
    solution: np.ndarray    # binary 0/1 vector
    energy: float
    n: int

class WarmStartCache:
    """
    LRU cache with exact-key lookup and cosine-NN fallback.
    Stores {structure_key -> (embedding, solution, energy)}.
    """
    def __init__(self, max_items: int = 1024, nn_threshold: float = 0.92):
        self.max_items = int(max_items)
        self.nn_threshold = float(nn_threshold)
        self._lru: "OrderedDict[str, CacheItem]" = OrderedDict()

    def put(self, key: str, emb: np.ndarray, solution: np.ndarray, energy: float):
        n = int(solution.shape[0])
        item = CacheItem(key, emb.astype(np.float32), solution.astype(np.int32), float(energy), n)
        if key in self._lru:
            self._lru.pop(key)
        self._lru[key] = item
        # enforce capacity
        while len(self._lru) > self.max_items:
            self._lru.popitem(last=False)

    def get_exact(self, key: str) -> Optional[CacheItem]:
        it = self._lru.get(key)
        if it is not None:
            # refresh LRU
            self._lru.move_to_end(key, last=True)
        return it

    def get_nearest(self, emb: np.ndarray, n: int) -> Optional[CacheItem]:
        """Return most similar item by cosine similarity, if above threshold and same n."""
        if not self._lru:
            return None
        e = emb.astype(np.float32)
        best_sim, best_key = -1.0, None
        for k, it in self._lru.items():
            if it.n != n:  # require same dimensionality for direct reuse
                continue
            sim = float(np.dot(e, it.emb))
            if sim > best_sim:
                best_sim, best_key = sim, k
        if best_key is not None and best_sim >= self.nn_threshold:
            self._lru.move_to_end(best_key, last=True)
            return self._lru[best_key]
        return None
