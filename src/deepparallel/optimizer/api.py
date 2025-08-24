from __future__ import annotations
from typing import Dict, List, Optional, Any
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .engine import PhaseIsingNetwork, PhaseIsingConfig
from .warmstart import WarmStartCache

# Global optimizer instance with warm-start cache
_global_cache = WarmStartCache(max_items=2048, nn_threshold=0.92)
_global_optimizer = None

def get_optimizer() -> PhaseIsingNetwork:
    global _global_optimizer
    if _global_optimizer is None:
        config = PhaseIsingConfig(
            n_restarts=16,
            max_steps=2000,
            use_gpu=True,
            warmstart_enabled=True
        )
        _global_optimizer = PhaseIsingNetwork(config)
        _global_optimizer.attach_warmstart_cache(_global_cache)
    return _global_optimizer

# API Models
class OptimizeRequest(BaseModel):
    cost_matrix: List[List[float]] = Field(..., description="QUBO cost matrix Q")
    k_eq: Optional[int] = Field(None, description="Number of variables to set to 1 (equality constraint)")
    k_penalty: float = Field(5.0, description="Penalty weight for equality constraint")
    capacity_weights: Optional[List[float]] = Field(None, description="Resource weights for capacity constraint")
    capacity_budget: Optional[float] = Field(None, description="Resource budget limit")
    capacity_lambda: Optional[float] = Field(0.0, description="Lagrange multiplier for capacity constraint")
    
    # Solver configuration overrides
    n_restarts: Optional[int] = Field(None, description="Number of restarts")
    max_steps: Optional[int] = Field(None, description="Maximum steps per restart")
    use_warmstart: Optional[bool] = Field(None, description="Enable warm start")

class OptimizeResponse(BaseModel):
    solution: List[int] = Field(..., description="Binary solution vector")
    energy: float = Field(..., description="QUBO objective value")
    steps: int = Field(..., description="Number of optimization steps")
    R2: float = Field(..., description="Convergence quality (R²)")
    restarts: int = Field(..., description="Number of restarts used")
    device: str = Field(..., description="Compute device used")
    warmstart_used: bool = Field(..., description="Whether warm start was used")
    metadata: Dict[str, Any] = Field(..., description="Additional solver metadata")

class HealthResponse(BaseModel):
    status: str
    version: str = "0.3.0"
    gpu_available: bool
    cache_size: int

# FastAPI app
app = FastAPI(
    title="Deep Parallel Phase-Ising Optimizer",
    description="GPU-accelerated quantum-inspired optimization for QUBO/Ising problems",
    version="0.3.0"
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    import torch
    return HealthResponse(
        status="healthy",
        gpu_available=torch.cuda.is_available(),
        cache_size=len(_global_cache._lru) if _global_cache else 0
    )

@app.post("/optimize", response_model=OptimizeResponse)
async def optimize(request: OptimizeRequest):
    """
    Solve QUBO optimization problem using Phase-Ising network.
    
    The cost matrix Q defines the objective: minimize x^T Q x subject to x ∈ {0,1}^n
    
    Optional constraints:
    - Equality: exactly k variables set to 1
    - Capacity: weighted sum ≤ budget
    """
    try:
        # Validate input
        Q = np.array(request.cost_matrix, dtype=np.float64)
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise HTTPException(status_code=400, detail="Cost matrix must be square")
        
        n = Q.shape[0]
        if n < 2 or n > 10000:
            raise HTTPException(status_code=400, detail="Problem size must be between 2 and 10000")
        
        # Apply constraints by modifying Q
        Q_constrained = Q.copy()
        
        # Equality constraint: k variables must be 1
        if request.k_eq is not None:
            if not (0 <= request.k_eq <= n):
                raise HTTPException(status_code=400, detail="k_eq must be between 0 and n")
            
            # Add penalty terms to enforce sum(x) = k_eq
            penalty = request.k_penalty
            # Penalty: λ * (sum(x) - k_eq)²
            # Expanded: λ * (sum(x²) + sum_i≠j(x_i*x_j) - 2*k_eq*sum(x) + k_eq²)
            # For binary x: x² = x, so this becomes:
            # λ * (sum(x) + sum_i≠j(x_i*x_j) - 2*k_eq*sum(x) + k_eq²)
            # = λ * ((1-2*k_eq)*sum(x) + sum_i≠j(x_i*x_j) + k_eq²)
            
            # Diagonal terms: (1-2*k_eq) coefficient
            for i in range(n):
                Q_constrained[i, i] += penalty * (1 - 2 * request.k_eq)
            
            # Off-diagonal terms: cross products
            for i in range(n):
                for j in range(i + 1, n):
                    Q_constrained[i, j] += penalty
                    Q_constrained[j, i] += penalty
        
        # Capacity constraint: weighted sum ≤ budget
        if (request.capacity_weights is not None and 
            request.capacity_budget is not None and 
            request.capacity_lambda is not None):
            
            weights = np.array(request.capacity_weights, dtype=np.float64)
            if len(weights) != n:
                raise HTTPException(status_code=400, detail="Capacity weights length must match problem size")
            
            budget = float(request.capacity_budget)
            lam = float(request.capacity_lambda)
            
            if lam > 0:
                # Penalty: λ * max(0, weighted_sum - budget)²
                # Approximated as: λ * (weighted_sum - budget)² for weighted_sum > budget
                # This is a quadratic penalty, so we add λ*w_i*w_j terms
                for i in range(n):
                    for j in range(n):
                        Q_constrained[i, j] += lam * weights[i] * weights[j]
                    # Linear term: -2*λ*budget*w_i
                    Q_constrained[i, i] -= 2 * lam * budget * weights[i]
        
        # Get optimizer with optional config overrides
        optimizer = get_optimizer()
        if any(x is not None for x in [request.n_restarts, request.max_steps, request.use_warmstart]):
            # Create temporary optimizer with custom config
            config = PhaseIsingConfig()
            if request.n_restarts is not None:
                config.n_restarts = request.n_restarts
            if request.max_steps is not None:
                config.max_steps = request.max_steps
            if request.use_warmstart is not None:
                config.warmstart_enabled = request.use_warmstart
            
            optimizer = PhaseIsingNetwork(config)
            if config.warmstart_enabled:
                optimizer.attach_warmstart_cache(_global_cache)
        
        # Solve
        result = optimizer.solve(Q_constrained)
        
        return OptimizeResponse(**result)
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@app.get("/cache/stats")
async def cache_stats():
    """Get warm-start cache statistics."""
    if _global_cache is None:
        return {"cache_enabled": False}
    
    return {
        "cache_enabled": True,
        "size": len(_global_cache._lru),
        "max_size": _global_cache.max_items,
        "nn_threshold": _global_cache.nn_threshold
    }

@app.post("/cache/clear")
async def clear_cache():
    """Clear the warm-start cache."""
    if _global_cache is not None:
        _global_cache._lru.clear()
    return {"status": "cache cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
