from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
from .kernels.triton_ops import fused_k_mul_two, _TRITON_AVAILABLE
from .warmstart import WarmStartCache, structure_fingerprint

@dataclass
class PhaseIsingConfig:
    """Configuration for Phase-Ising network solver."""
    n_restarts: int = 16
    max_steps: int = 2000
    dt: float = 0.01
    K_schedule: Tuple[float, float] = (0.1, 2.0)  # (start, end)
    alpha_schedule: Tuple[float, float] = (0.0, 1.0)  # (start, end)
    noise_schedule: Tuple[float, float] = (0.1, 0.001)  # (start, end)
    
    # Early stopping
    plateau_patience: int = 200
    plateau_threshold: float = 1e-6
    
    # Elite tracking
    elite_fraction: float = 0.25
    
    # Bit-flip polishing
    polish_steps: int = 50
    
    # GPU settings
    use_gpu: bool = True
    dtype: str = "float32"  # float32, float16, bfloat16
    
    # Warm start
    warmstart_enabled: bool = True
    warmstart_noise: float = 0.05

class PhaseIsingNetwork:
    """
    GPU-accelerated Phase-Ising network for QUBO/Ising optimization.
    
    Uses parametric bistability with oscillator phases φ_i and coupling matrix K.
    Energy function: E = -0.5 * Σ_ij Q_ij * sin²(φ_i - φ_j + π/4)
    
    Dynamics: dφ_i/dt = -α*sin(2φ_i) + K * Σ_j Q_ij * sin(φ_j - φ_i) + h_i + noise
    """
    
    def __init__(self, config: PhaseIsingConfig = PhaseIsingConfig()):
        self.cfg = config
        self.device = torch.device("cuda" if config.use_gpu and torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, config.dtype)
        self.warmstart_cache: Optional[WarmStartCache] = None
        
    def attach_warmstart_cache(self, cache: WarmStartCache):
        """Attach a warm-start cache for solution reuse."""
        self.warmstart_cache = cache
        
    def solve(self, Q: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Solve QUBO problem: minimize x^T Q x subject to x ∈ {0,1}^n
        
        Returns:
            Dictionary with solution, energy, steps, and metadata
        """
        Q = np.asarray(Q, dtype=np.float64)
        n = Q.shape[0]
        
        # Convert to Ising format: H = -0.5 * Σ_ij J_ij * s_i * s_j - Σ_i h_i * s_i
        # where s_i ∈ {-1, +1} and x_i = (s_i + 1)/2
        J = np.zeros((n, n), dtype=np.float64)
        h = np.zeros(n, dtype=np.float64)
        
        # QUBO to Ising transformation
        for i in range(n):
            h[i] = Q[i, i] + sum(Q[i, j] + Q[j, i] for j in range(n) if j != i) / 2
            for j in range(i + 1, n):
                J[i, j] = (Q[i, j] + Q[j, i]) / 4
                J[j, i] = J[i, j]
        
        # Check warm start cache
        warmstart_phases = None
        if self.cfg.warmstart_enabled and self.warmstart_cache is not None:
            key, emb = structure_fingerprint(Q)
            
            # Try exact match first
            cached = self.warmstart_cache.get_exact(key)
            if cached is None:
                # Try nearest neighbor
                cached = self.warmstart_cache.get_nearest(emb, n)
            
            if cached is not None:
                # Convert binary solution to phases
                s_cached = 2 * cached.solution.astype(np.float64) - 1  # {0,1} -> {-1,1}
                warmstart_phases = np.arcsin(s_cached) + np.random.normal(0, self.cfg.warmstart_noise, n)
        
        # Move to GPU
        J_gpu = torch.tensor(J, device=self.device, dtype=self.dtype)
        h_gpu = torch.tensor(h, device=self.device, dtype=self.dtype)
        
        best_energy = float('inf')
        best_solution = None
        best_metadata = {}
        
        # Multi-restart optimization
        for restart in range(self.cfg.n_restarts):
            result = self._single_run(J_gpu, h_gpu, n, warmstart_phases if restart == 0 else None)
            
            if result['energy'] < best_energy:
                best_energy = result['energy']
                best_solution = result['solution']
                best_metadata = result['metadata']
                best_metadata['best_restart'] = restart
        
        # Convert Ising solution back to QUBO
        s_best = 2 * best_solution - 1  # {0,1} -> {-1,1}
        x_best = (s_best + 1) / 2  # {-1,1} -> {0,1}
        E_qubo = float(x_best @ Q @ x_best)
        
        # Update warm start cache
        if self.cfg.warmstart_enabled and self.warmstart_cache is not None:
            key, emb = structure_fingerprint(Q)
            self.warmstart_cache.put(key, emb, best_solution, E_qubo)
        
        return {
            'solution': best_solution.astype(int).tolist(),
            'energy': E_qubo,
            'steps': best_metadata['steps'],
            'R2': best_metadata.get('R2', 0.0),
            'restarts': self.cfg.n_restarts,
            'device': str(self.device),
            'warmstart_used': warmstart_phases is not None,
            'metadata': best_metadata
        }
    
    def _single_run(self, J: torch.Tensor, h: torch.Tensor, n: int, 
                   warmstart_phases: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Single optimization run with annealing schedule."""
        
        # Initialize phases
        if warmstart_phases is not None:
            phi = torch.tensor(warmstart_phases, device=self.device, dtype=self.dtype)
        else:
            phi = torch.rand(n, device=self.device, dtype=self.dtype) * 2 * math.pi
        
        # Annealing schedules
        steps = self.cfg.max_steps
        t_vals = torch.linspace(0, 1, steps, device=self.device)
        
        K_start, K_end = self.cfg.K_schedule
        alpha_start, alpha_end = self.cfg.alpha_schedule
        noise_start, noise_end = self.cfg.noise_schedule
        
        K_t = K_start + (K_end - K_start) * t_vals
        alpha_t = alpha_start + (alpha_end - alpha_start) * t_vals
        noise_t = noise_start + (noise_end - noise_start) * t_vals
        
        # Energy tracking for early stopping
        energy_history = []
        best_energy = float('inf')
        plateau_count = 0
        
        # Main dynamics loop
        for step in range(steps):
            # Current parameters
            K = K_t[step].item()
            alpha = alpha_t[step].item()
            noise_std = noise_t[step].item()
            
            # Phase dynamics: dφ/dt = -α*sin(2φ) + K*(J @ sin(φ_diff)) + h + noise
            sin_phi = torch.sin(phi)
            cos_phi = torch.cos(phi)
            
            # Compute coupling terms efficiently
            if _TRITON_AVAILABLE and J.is_cuda and n >= 64:
                # Use fused Triton kernel for large problems
                sin_phi_expanded = sin_phi.unsqueeze(1).expand(n, 1)
                cos_phi_expanded = cos_phi.unsqueeze(1).expand(n, 1)
                J_sin, J_cos = fused_k_mul_two(J, sin_phi_expanded, cos_phi_expanded)
                J_sin = J_sin.squeeze(1)
                J_cos = J_cos.squeeze(1)
            else:
                # Standard PyTorch implementation
                J_sin = J @ sin_phi
                J_cos = J @ cos_phi
            
            # Phase differences and coupling force
            coupling_force = J_sin * cos_phi - J_cos * sin_phi
            
            # Local field (bistability)
            local_field = -alpha * torch.sin(2 * phi)
            
            # External field
            external_field = h
            
            # Noise
            noise = torch.randn_like(phi) * noise_std
            
            # Update phases
            dphi_dt = local_field + K * coupling_force + external_field + noise
            phi += self.cfg.dt * dphi_dt
            
            # Wrap phases to [0, 2π)
            phi = torch.fmod(phi + 2 * math.pi, 2 * math.pi)
            
            # Compute current energy every 10 steps
            if step % 10 == 0:
                current_energy = self._compute_energy(phi, J, h)
                energy_history.append(current_energy)
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    plateau_count = 0
                else:
                    plateau_count += 1
                
                # Early stopping on plateau
                if (plateau_count >= self.cfg.plateau_patience and 
                    len(energy_history) > 20):
                    recent_energies = energy_history[-20:]
                    if max(recent_energies) - min(recent_energies) < self.cfg.plateau_threshold:
                        break
        
        # Extract binary solution
        # Map phases to spins: s_i = sign(sin(φ_i))
        spins = torch.sign(torch.sin(phi))
        spins = torch.where(spins == 0, torch.ones_like(spins), spins)  # Handle exactly zero
        
        # Convert to {0,1}
        solution = ((spins + 1) / 2).cpu().numpy()
        
        # Bit-flip polishing
        if self.cfg.polish_steps > 0:
            solution = self._bit_flip_polish(solution, J.cpu().numpy(), h.cpu().numpy())
        
        final_energy = self._compute_energy_numpy(solution, J.cpu().numpy(), h.cpu().numpy())
        
        # Compute R² fit quality
        R2 = self._compute_r2(energy_history) if len(energy_history) > 1 else 0.0
        
        return {
            'solution': solution,
            'energy': final_energy,
            'metadata': {
                'steps': step + 1,
                'energy_history': [float(e) for e in energy_history[-50:]],  # Last 50 values
                'R2': R2,
                'plateau_count': plateau_count,
                'final_K': K,
                'final_alpha': alpha
            }
        }
    
    def _compute_energy(self, phi: torch.Tensor, J: torch.Tensor, h: torch.Tensor) -> float:
        """Compute Ising energy from phases."""
        spins = torch.sign(torch.sin(phi))
        spins = torch.where(spins == 0, torch.ones_like(spins), spins)
        
        # E = -0.5 * s^T J s - h^T s
        energy = -0.5 * torch.sum(spins.unsqueeze(0) * (J @ spins.unsqueeze(1)).squeeze()) - torch.sum(h * spins)
        return energy.item()
    
    def _compute_energy_numpy(self, solution: np.ndarray, J: np.ndarray, h: np.ndarray) -> float:
        """Compute Ising energy from binary solution."""
        spins = 2 * solution - 1  # {0,1} -> {-1,1}
        energy = -0.5 * np.sum(spins * (J @ spins)) - np.sum(h * spins)
        return float(energy)
    
    def _bit_flip_polish(self, solution: np.ndarray, J: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Local search with bit flips."""
        solution = solution.copy()
        n = len(solution)
        
        current_energy = self._compute_energy_numpy(solution, J, h)
        
        for _ in range(self.cfg.polish_steps):
            improved = False
            for i in range(n):
                # Try flipping bit i
                solution[i] = 1 - solution[i]
                new_energy = self._compute_energy_numpy(solution, J, h)
                
                if new_energy < current_energy:
                    current_energy = new_energy
                    improved = True
                else:
                    # Revert flip
                    solution[i] = 1 - solution[i]
            
            if not improved:
                break
        
        return solution
    
    def _compute_r2(self, energy_history: List[float]) -> float:
        """Compute R² for energy convergence quality."""
        if len(energy_history) < 2:
            return 0.0
        
        y = np.array(energy_history)
        x = np.arange(len(y))
        
        # Linear regression
        A = np.vstack([x, np.ones(len(x))]).T
        try:
            slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
            y_pred = slope * x + intercept
            
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            
            if ss_tot == 0:
                return 1.0
            
            r2 = 1 - (ss_res / ss_tot)
            return max(0.0, min(1.0, r2))  # Clamp to [0,1]
        except:
            return 0.0
