"""
Deep Parallel Genesis Configuration System
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from omegaconf import DictConfig
import os


@dataclass
class ModelConfig:
    """Configuration for base models and cloud storage"""
    
    # Base models to download and use
    base_models: Dict[str, str] = field(default_factory=lambda: {
        "llama_7b": "meta-llama/Llama-2-7b-hf",
        "llama_13b": "meta-llama/Llama-2-13b-hf", 
        "llama_70b": "meta-llama/Llama-2-70b-hf",
        "llama3_8b": "meta-llama/Meta-Llama-3-8B",
        "llama3_70b": "meta-llama/Meta-Llama-3-70B",
        "llama32_1b": "meta-llama/Llama-3.2-1B",
        "llama32_3b": "meta-llama/Llama-3.2-3B",
        "gemma_2b": "google/gemma-2b",
        "gemma_7b": "google/gemma-7b",
        "gemma2_9b": "google/gemma-2-9b",
        "gemma2_27b": "google/gemma-2-27b",
        "qwen_7b": "Qwen/Qwen2-7B",
        "qwen_72b": "Qwen/Qwen2-72B",
        "qwen_math": "Qwen/Qwen2-Math-7B-Instruct",
        "deepseek_math": "deepseek-ai/deepseek-math-7b-base",
        "minerva": "google/minerva-62b",
        "galactica": "facebook/galactica-6.7b"
    })
    
    # Cloud storage configuration
    cloud_storage: Dict[str, Any] = field(default_factory=lambda: {
        "provider": "huggingface",  # huggingface, aws, gcp, azure
        "bucket_name": "deepparallel-models",
        "cache_dir": "./model_cache",
        "use_auth_token": True,
        "download_mode": "reuse_dataset_if_exists"
    })
    
    # Model loading preferences
    load_config: Dict[str, Any] = field(default_factory=lambda: {
        "torch_dtype": "auto",
        "device_map": "auto", 
        "trust_remote_code": True,
        "use_flash_attention": True,
        "load_in_8bit": False,
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "float16"
    })


@dataclass 
class ComputeConfig:
    """Configuration for compute resources"""
    
    # Hardware configuration
    use_gpu: bool = True
    use_multiple_gpus: bool = True
    gpu_memory_limit: Optional[int] = None
    cpu_workers: int = 8
    
    # Distributed computing
    use_ray: bool = True
    ray_cluster_address: Optional[str] = None
    use_mpi: bool = False
    
    # Quantum computing
    quantum_backend: str = "aer_simulator"
    quantum_shots: int = 1024
    enable_quantum_acceleration: bool = True


@dataclass
class SimulationConfig:
    """Configuration for simulation engines"""
    
    # Physics simulation
    physics_scales: List[str] = field(default_factory=lambda: [
        "quantum", "atomic", "molecular", "mesoscopic", "macroscopic"
    ])
    
    # Molecular dynamics
    md_timestep: float = 1e-15  # femtoseconds
    md_max_steps: int = 1000000
    md_temperature: float = 300.0  # Kelvin
    
    # Quantum chemistry
    qc_basis_set: str = "cc-pVDZ"
    qc_method: str = "MP2"
    qc_convergence: float = 1e-8
    
    # Virtual experiments
    max_virtual_experiments: int = 10000
    experiment_timeout: int = 3600  # seconds
    success_threshold: float = 0.95


@dataclass
class TrainingConfig:
    """Configuration for training the Genesis Engine"""
    
    # Training parameters
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    
    # Curriculum learning
    difficulty_levels: List[str] = field(default_factory=lambda: [
        "undergraduate", "graduate", "research", "frontier"
    ])
    
    # Data generation
    data_scale_per_domain: int = 10**9  # 1B examples per domain
    simulation_data_points: int = 10**6
    
    # Training domains
    training_domains: List[str] = field(default_factory=lambda: [
        "physics", "chemistry", "biology", "materials", "mathematics"
    ])


@dataclass
class EvaluationConfig:
    """Configuration for evaluation and benchmarking"""
    
    # Benchmarks to evaluate on
    benchmarks: List[str] = field(default_factory=lambda: [
        "frontier_math", "science_qa", "arc_challenge", "mmlu_stem", 
        "math_level5", "physics_gre", "chemistry_olympiad"
    ])
    
    # FrontierMath specific
    frontier_math_tiers: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    max_compute_per_problem: int = 3600  # 1 hour
    
    # Performance targets
    target_accuracy: Dict[str, float] = field(default_factory=lambda: {
        "frontier_math_overall": 0.65,
        "science_qa": 0.995,
        "arc_challenge": 0.99,
        "mmlu_stem": 0.995,
        "math_level5": 0.98,
        "physics_gre": 0.99
    })


@dataclass
class APIConfig:
    """Configuration for the Genesis API"""
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    
    # Redis cache
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    cache_ttl: int = 86400  # 24 hours
    
    # Rate limiting
    rate_limit: str = "100/minute"
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    
    # API keys and authentication
    require_api_key: bool = False
    admin_api_key: Optional[str] = None


@dataclass
class GenesisConfig:
    """Main configuration for the Genesis Engine"""
    
    # Sub-configurations
    models: ModelConfig = field(default_factory=ModelConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig) 
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    api: APIConfig = field(default_factory=APIConfig)
    
    # Global settings
    debug: bool = False
    log_level: str = "INFO"
    seed: int = 42
    output_dir: str = "./outputs"
    
    # Performance targets
    performance_targets: Dict[str, float] = field(default_factory=lambda: {
        "virtual_experiment_success_rate": 0.995,
        "simulation_accuracy": 0.999,
        "confidence_calibration": 0.99,
        "uncertainty_quantification": 0.95,
        "physical_law_consistency": 0.9999,
        "conservation_law_adherence": 1.0
    })


def load_config(config_path: Optional[str] = None) -> GenesisConfig:
    """Load configuration from file or return default"""
    if config_path and os.path.exists(config_path):
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(config_path)
        return GenesisConfig(**cfg)
    return GenesisConfig()
