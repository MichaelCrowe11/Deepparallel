from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
    _YAML_OK = True
except Exception:
    _YAML_OK = False

from ..engine import PhaseIsingConfig

def load_yaml_config(path: str) -> Dict[str, Any]:
    if not _YAML_OK:
        raise RuntimeError("PyYAML not installed. pip install pyyaml")
    with open(path, 'r') as f:
        data = yaml.safe_load(f) or {}
    return data

def config_from_yaml(data: Dict[str, Any]) -> PhaseIsingConfig:
    osc = data.get('oscillator', {})
    # map YAML fields to PhaseIsingConfig
    mapping = dict(
        use_gpu=osc.get('use_gpu'),
        dt=osc.get('dt'),
        max_steps=osc.get('max_steps'),
        noise_schedule=tuple(osc.get('noise_anneal', (0.15, 0.0))),
        alpha_schedule=(osc.get('alpha_min', 0.0), osc.get('alpha_max', 1.0)),
        n_restarts=osc.get('restarts'),
        polish_steps=osc.get('postprocess_bitflips'),
        warmstart_enabled=osc.get('warmstart_enable'),
    )
    # prune Nones
    clean = {k: v for k, v in mapping.items() if v is not None}
    cfg = PhaseIsingConfig()
    for k, v in clean.items():
        setattr(cfg, k, v)
    return cfg
