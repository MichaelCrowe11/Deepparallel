from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Any

def qubo_for_paths(props: Dict[str, Dict[str, float]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """props[name] = {'accuracy': a, 'cost': c, 'gpu_mb': w}"""
    names = list(props.keys()); n = len(names)
    Q = np.zeros((n, n))
    for i, nm in enumerate(names):
        a, c = props[nm]['accuracy'], max(1e-6, props[nm]['cost'])
        Q[i, i] = -(a / c)
    # redundancy (can be learned)
    for i in range(n):
        for j in range(i+1, n):
            Q[i, j] = Q[j, i] = 0.2
    capacity = {
        "weights": [props[nm]['gpu_mb'] for nm in names],
        "budget": None,  # set at call time
        "lambda": 0.0    # auto-scale in solver
    }
    return Q, {"names": names, "capacity": capacity}
