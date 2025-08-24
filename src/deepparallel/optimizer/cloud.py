"""
Distributed optimization using Ray for large-scale problems
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np
import time

try:
    import ray
    _RAY_AVAILABLE = True
except ImportError:
    # Provide a minimal stub so the module can be imported without Ray
    from types import SimpleNamespace
    def _noop_remote_decorator(obj):
        return obj
    ray = SimpleNamespace(remote=_noop_remote_decorator, is_initialized=lambda: False, init=lambda: None, kill=lambda actor: None, get=lambda x: x)
    _RAY_AVAILABLE = False

from .engine import PhaseIsingNetwork, PhaseIsingConfig
from .warmstart import WarmStartCache

@ray.remote
class DistributedSolver:
    """Ray actor for distributed Phase-Ising solving."""
    
    def __init__(self, config: PhaseIsingConfig):
        self.solver = PhaseIsingNetwork(config)
        self.cache = WarmStartCache(max_items=512, nn_threshold=0.92)
        self.solver.attach_warmstart_cache(self.cache)
    
    def solve_batch(self, Q_list: List[np.ndarray], worker_id: int) -> List[Dict[str, Any]]:
        """Solve a batch of problems."""
        results = []
        for i, Q in enumerate(Q_list):
            result = self.solver.solve(Q)
            result['worker_id'] = worker_id
            result['batch_index'] = i
            results.append(result)
        return results
    
    def solve_single(self, Q: np.ndarray, worker_id: int) -> Dict[str, Any]:
        """Solve a single problem."""
        result = self.solver.solve(Q)
        result['worker_id'] = worker_id
        return result

def optimize_distributed(
    Q: np.ndarray,
    n_workers: int = 4,
    restarts_per_worker: int = 4,
    max_steps: int = 2000,
    use_gpu: bool = True,
    return_all: bool = False
) -> Dict[str, Any]:
    """
    Solve QUBO problem using distributed Phase-Ising optimization.
    
    Args:
        Q: QUBO cost matrix
        n_workers: Number of Ray workers
        restarts_per_worker: Restarts per worker (total = n_workers * restarts_per_worker)
        max_steps: Maximum steps per restart
        use_gpu: Enable GPU acceleration
        return_all: Return all solutions or just the best
    
    Returns:
        Best solution and metadata from all workers
    """
    if not _RAY_AVAILABLE:
        raise RuntimeError("Ray not available. Install with: pip install ray")
    
    # Initialize Ray if not already running
    if not ray.is_initialized():
        ray.init()
    
    # Create worker configuration
    config = PhaseIsingConfig(
        n_restarts=restarts_per_worker,
        max_steps=max_steps,
        use_gpu=use_gpu,
        warmstart_enabled=True
    )
    
    # Launch workers
    workers = [DistributedSolver.remote(config) for _ in range(n_workers)]
    
    print(f"Launched {n_workers} workers with {restarts_per_worker} restarts each")
    print(f"Total optimization budget: {n_workers * restarts_per_worker} restarts")
    
    # Submit tasks
    start_time = time.time()
    futures = [worker.solve_single.remote(Q, i) for i, worker in enumerate(workers)]
    
    # Collect results
    all_results = ray.get(futures)
    total_time = time.time() - start_time
    
    # Find best solution
    best_result = min(all_results, key=lambda x: x['energy'])
    
    # Aggregate statistics
    energies = [r['energy'] for r in all_results]
    steps = [r['steps'] for r in all_results]
    r2_values = [r['R2'] for r in all_results]
    
    # Cleanup workers
    for worker in workers:
        ray.kill(worker)
    
    result = {
        'solution': best_result['solution'],
        'energy': best_result['energy'],
        'best_worker': best_result['worker_id'],
        'steps': best_result['steps'],
        'R2': best_result['R2'],
        'device': best_result['device'],
        'distributed_stats': {
            'n_workers': n_workers,
            'total_restarts': n_workers * restarts_per_worker,
            'total_time_s': total_time,
            'parallel_efficiency': (sum(r.get('solve_time_s', 0) for r in all_results) / total_time) if total_time > 0 else 0,
            'energy_stats': {
                'min': float(np.min(energies)),
                'max': float(np.max(energies)),
                'mean': float(np.mean(energies)),
                'std': float(np.std(energies))
            },
            'steps_stats': {
                'min': int(np.min(steps)),
                'max': int(np.max(steps)),
                'mean': float(np.mean(steps))
            },
            'r2_stats': {
                'min': float(np.min(r2_values)),
                'max': float(np.max(r2_values)),
                'mean': float(np.mean(r2_values))
            }
        }
    }
    
    if return_all:
        result['all_results'] = all_results
    
    return result

def distributed_benchmark(
    problems: List[tuple],  # [(name, Q), ...]
    n_workers: int = 4,
    restarts_per_worker: int = 4
) -> Dict[str, Any]:
    """
    Run distributed benchmark on multiple problems.
    
    Args:
        problems: List of (name, Q) tuples
        n_workers: Number of Ray workers
        restarts_per_worker: Restarts per worker per problem
    
    Returns:
        Benchmark results with timing and quality metrics
    """
    if not _RAY_AVAILABLE:
        raise RuntimeError("Ray not available. Install with: pip install ray")
    
    if not ray.is_initialized():
        ray.init()
    
    results = []
    total_start = time.time()
    
    for name, Q in problems:
        print(f"Solving {name} (n={Q.shape[0]}) with {n_workers} workers...")
        
        problem_start = time.time()
        result = optimize_distributed(
            Q, n_workers=n_workers, 
            restarts_per_worker=restarts_per_worker,
            return_all=False
        )
        problem_time = time.time() - problem_start
        
        results.append({
            'problem': name,
            'n': Q.shape[0],
            'energy': result['energy'],
            'best_worker': result['best_worker'],
            'time_s': problem_time,
            'distributed_stats': result['distributed_stats']
        })
        
        print(f"  Best energy: {result['energy']:.6f} "
              f"(worker {result['best_worker']}, {problem_time:.2f}s)")
    
    total_time = time.time() - total_start
    
    return {
        'problems': results,
        'total_time_s': total_time,
        'n_problems': len(problems),
        'n_workers': n_workers,
        'restarts_per_worker': restarts_per_worker
    }
