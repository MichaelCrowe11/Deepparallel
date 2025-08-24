from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import json, hashlib, time

@dataclass
class DecisionRecord:
    trace_id: str
    epoch_ms: int
    qubo_hash: str
    inputs: Dict[str, Any]      # props, constraints
    solution: Dict[str, Any]    # {solution, energy, steps, R2, seed_used}
    policy_version: str = "v1"

    @staticmethod
    def hash_qubo(Q: List[List[float]]) -> str:
        b = json.dumps(Q, separators=(',',':')).encode()
        return hashlib.blake2b(b, digest_size=16).hexdigest()

    @classmethod
    def from_run(cls, trace_id: str, Q: List[List[float]], inputs: Dict[str, Any], solution: Dict[str, Any]):
        return cls(trace_id=trace_id,
                   epoch_ms=int(time.time()*1000),
                   qubo_hash=cls.hash_qubo(Q),
                   inputs=inputs,
                   solution=solution)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(',',':'))
