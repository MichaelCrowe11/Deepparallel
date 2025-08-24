from __future__ import annotations
import psutil, torch

def snapshot_resources() -> dict:
    gpu_mb = 0
    if torch.cuda.is_available() and torch.cuda.device_count()>0:
        dev = torch.cuda.current_device()
        gpu_mb = torch.cuda.get_device_properties(dev).total_memory // 1048576
    return {"gpu_memory_mb": int(gpu_mb), "cpu_cores": int(psutil.cpu_count(logical=True) or 1)}
