#!/usr/bin/env python3
"""
Command-line interface for Deep Parallel Phase-Ising Optimizer
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from typing import Dict, Any
import numpy as np

from .engine import PhaseIsingNetwork, PhaseIsingConfig
from .warmstart import WarmStartCache

def create_random_qubo(n: int, density: float = 0.5, seed: int = 42) -> np.ndarray:
    """Generate a random QUBO matrix for testing."""
    np.random.seed(seed)
    Q = np.zeros((n, n))
    
    # Add random entries
    for i in range(n):
        for j in range(i, n):
            if np.random.random() < density:
                val = np.random.normal(0, 1)
                Q[i, j] = val
                if i != j:
                    Q[j, i] = val
    
    return Q

def create_max_cut_qubo(n: int, edge_prob: float = 0.3, seed: int = 42) -> np.ndarray:
    """Generate a Max-Cut QUBO instance."""
    np.random.seed(seed)
    Q = np.zeros((n, n))
    
    # Random graph
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.random() < edge_prob:
                # Max-Cut QUBO: Q[i,j] = -w_ij, Q[i,i] += w_ij
                weight = np.random.uniform(0.5, 2.0)
                Q[i, j] = -weight
                Q[j, i] = -weight
                Q[i, i] += weight
                Q[j, j] += weight
    
    return Q

def benchmark_solver(solver: PhaseIsingNetwork, problems: list, verbose: bool = True) -> Dict[str, Any]:
    """Benchmark solver on a set of problems."""
    results = []
    total_time = 0
    
    for i, (name, Q) in enumerate(problems):
        if verbose:
            print(f"Solving {name} (n={Q.shape[0]})...")
        
        start_time = time.time()
        result = solver.solve(Q)
        solve_time = time.time() - start_time
        total_time += solve_time
        
        results.append({
            "problem": name,
            "n": Q.shape[0],
            "energy": result["energy"],
            "steps": result["steps"],
            "R2": result["R2"],
            "time_s": solve_time,
            "device": result["device"],
            "warmstart_used": result.get("warmstart_used", False)
        })
        
        if verbose:
            print(f"  Energy: {result['energy']:.6f}, Steps: {result['steps']}, "
                  f"Time: {solve_time:.3f}s, R²: {result['R2']:.3f}")
    
    return {
        "results": results,
        "total_time_s": total_time,
        "avg_time_s": total_time / len(problems) if problems else 0,
        "problems_solved": len(problems)
    }

def main():
    parser = argparse.ArgumentParser(description="Deep Parallel Phase-Ising Optimizer CLI")
    parser.add_argument("--n", type=int, default=100, help="Problem size")
    parser.add_argument("--problem", choices=["random", "maxcut"], default="random", 
                       help="Problem type")
    parser.add_argument("--density", type=float, default=0.5, help="QUBO density")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Solver configuration
    parser.add_argument("--restarts", type=int, default=16, help="Number of restarts")
    parser.add_argument("--max-steps", type=int, default=2000, help="Max steps per restart")
    parser.add_argument("--use-gpu", type=bool, default=True, help="Use GPU if available")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], 
                       default="float32", help="Floating point precision")
    parser.add_argument("--warmstart", action="store_true", help="Enable warm start cache")
    
    # Execution mode
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark suite")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    
    args = parser.parse_args()
    
    # Configure solver
    config = PhaseIsingConfig(
        n_restarts=args.restarts,
        max_steps=args.max_steps,
        use_gpu=args.use_gpu,
        dtype=args.dtype,
        warmstart_enabled=args.warmstart
    )
    
    solver = PhaseIsingNetwork(config)
    
    # Set up warm start cache
    if args.warmstart:
        cache = WarmStartCache(max_items=1024, nn_threshold=0.92)
        solver.attach_warmstart_cache(cache)
    
    if args.benchmark:
        # Benchmark suite
        problems = []
        sizes = [50, 100, 200, 500] if args.n <= 200 else [args.n]
        
        for size in sizes:
            if args.problem == "random":
                Q = create_random_qubo(size, args.density, args.seed)
                problems.append((f"random_{size}", Q))
            elif args.problem == "maxcut":
                Q = create_max_cut_qubo(size, 0.3, args.seed)
                problems.append((f"maxcut_{size}", Q))
        
        print(f"Running benchmark on {len(problems)} problems...")
        results = benchmark_solver(solver, problems, args.verbose)
        
        if args.verbose:
            print(f"\nBenchmark Summary:")
            print(f"  Problems solved: {results['problems_solved']}")
            print(f"  Total time: {results['total_time_s']:.3f}s")
            print(f"  Average time: {results['avg_time_s']:.3f}s")
            
            avg_r2 = np.mean([r["R2"] for r in results["results"]])
            print(f"  Average R²: {avg_r2:.3f}")
    
    else:
        # Single problem
        if args.problem == "random":
            Q = create_random_qubo(args.n, args.density, args.seed)
            problem_name = f"random_{args.n}"
        elif args.problem == "maxcut":
            Q = create_max_cut_qubo(args.n, 0.3, args.seed)
            problem_name = f"maxcut_{args.n}"
        
        if args.verbose:
            print(f"Solving {problem_name}...")
            print(f"  Size: {Q.shape[0]} x {Q.shape[1]}")
            print(f"  Density: {np.mean(np.abs(Q) > 1e-12):.3f}")
            print(f"  Config: {args.restarts} restarts, {args.max_steps} steps, "
                  f"GPU={args.use_gpu}, dtype={args.dtype}")
        
        start_time = time.time()
        result = solver.solve(Q)
        solve_time = time.time() - start_time
        
        results = {
            "problem": problem_name,
            "config": {
                "n": args.n,
                "restarts": args.restarts,
                "max_steps": args.max_steps,
                "use_gpu": args.use_gpu,
                "dtype": args.dtype,
                "warmstart": args.warmstart
            },
            "result": result,
            "solve_time_s": solve_time
        }
        
        if args.verbose:
            print(f"\nResults:")
            print(f"  Solution: {result['solution'][:10]}{'...' if len(result['solution']) > 10 else ''}")
            print(f"  Energy: {result['energy']:.6f}")
            print(f"  Steps: {result['steps']}")
            print(f"  R²: {result['R2']:.3f}")
            print(f"  Time: {solve_time:.3f}s")
            print(f"  Device: {result['device']}")
            print(f"  Warm start used: {result.get('warmstart_used', False)}")
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        if args.verbose:
            print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
