from __future__ import annotations
import httpx, json, time, os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

@dataclass
class OptimizerClientCfg:
    base_url: str = os.getenv("DP_OPTIMIZER_URL", "http://localhost:8000")
    timeout_s: float = 3.0
    retries: int = 2
    backoff: float = 0.25

class OptimizerClient:
    def __init__(self, cfg: OptimizerClientCfg = OptimizerClientCfg()):
        self.cfg = cfg

    def solve(self, Q: List[List[float]], k_eq: Optional[int]=None,
              capacity: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {"cost_matrix": Q, "k_eq": k_eq, "k_penalty": 5.0}
        if capacity:
            payload.update({
                "capacity_weights": capacity["weights"],
                "capacity_budget": capacity["budget"],
                "capacity_lambda": capacity.get("lambda")
            })
        last_exc = None
        for attempt in range(self.cfg.retries + 1):
            try:
                with httpx.Client(timeout=self.cfg.timeout_s) as c:
                    r = c.post(f"{self.cfg.base_url}/optimize", json=payload)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_exc = e
                time.sleep(self.cfg.backoff * (2**attempt))
        raise RuntimeError(f"Optimizer unavailable after retries: {last_exc}")
