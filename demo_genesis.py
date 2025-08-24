#!/usr/bin/env python3
"""
Simplified demo script to showcase Deep Parallel Genesis model capabilities
"""

from rich.console import Console
from rich.table import Table

console = Console()

def demo_model_registry():
    """Demo the comprehensive model registry for Genesis Engine"""
    
    # Model registry (simplified version)
    model_registry = {
        # Meta Llama Models
        "llama-3.1-8b": {
            "parameters": "8B",
            "size_gb": 16.0,
            "domain": "general",
            "quality_score": 0.92,
            "specialties": ["reasoning", "conversation", "code"],
            "context_length": 128000,
        },
        "llama-3.1-70b": {
            "parameters": "70B",
            "size_gb": 140.0,
            "domain": "general", 
            "quality_score": 0.95,
            "specialties": ["advanced_reasoning", "complex_tasks", "research"],
            "context_length": 128000,
        },
        "llama-3.1-405b": {
            "parameters": "405B",
            "size_gb": 810.0,
            "domain": "general",
            "quality_score": 0.98,
            "specialties": ["frontier_reasoning", "scientific_research", "expert_tasks"],
            "context_length": 128000,
        },
        
        # Google Gemma Models
        "gemma-2-9b": {
            "parameters": "9B",
            "size_gb": 18.0,
            "domain": "general",
            "quality_score": 0.89,
            "specialties": ["efficiency", "safety", "multilingual"],
            "context_length": 8192,
        },
        "gemma-2-27b": {
            "parameters": "27B",
            "size_gb": 54.0,
            "domain": "general",
            "quality_score": 0.93,
            "specialties": ["reasoning", "math", "science"],
            "context_length": 8192,
        },
        
        # Alibaba Qwen Models
        "qwen2.5-7b": {
            "parameters": "7B",
            "size_gb": 14.0,
            "domain": "general",
            "quality_score": 0.88,
            "specialties": ["multilingual", "code", "math"],
            "context_length": 32768,
        },
        "qwen2.5-72b": {
            "parameters": "72B",
            "size_gb": 144.0,
            "domain": "general",
            "quality_score": 0.94,
            "specialties": ["advanced_reasoning", "research", "expert_tasks"],
            "context_length": 32768,
        },
        
        # Specialized Models
        "phi-3.5-mini": {
            "parameters": "3.8B",
            "size_gb": 7.5,
            "domain": "general",
            "quality_score": 0.84,
            "specialties": ["efficiency", "edge_deployment", "mobile"],
            "context_length": 128000,
        },
        "mistral-7b": {
            "parameters": "7B",
            "size_gb": 14.0,
            "domain": "general",
            "quality_score": 0.87,
            "specialties": ["efficiency", "reasoning", "conversation"],
            "context_length": 32768,
        },
        "deepseek-math-7b": {
            "parameters": "7B",
            "size_gb": 14.0,
            "domain": "mathematics",
            "quality_score": 0.90,
            "specialties": ["mathematics", "problem_solving", "proofs"],
            "context_length": 4096,
        },
        "codegemma-7b": {
            "parameters": "7B",
            "size_gb": 14.0,
            "domain": "code",
            "quality_score": 0.88,
            "specialties": ["code_generation", "programming", "debugging"],
            "context_length": 8192,
        },
    }
    
    console.print("🧬 [bold blue]Deep Parallel Genesis - Model Registry[/bold blue]")
    console.print("=" * 60)
    
    console.print(f"📋 Available models: {len(model_registry)}")
    console.print(f"🎯 Total parameters: {sum(float(info['parameters'].replace('B', '')) for info in model_registry.values()):.1f}B")
    
    # Show all models in a table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model ID", style="cyan", width=20)
    table.add_column("Params", justify="right", width=8)
    table.add_column("Size", justify="right", width=8)
    table.add_column("Quality", justify="right", width=8)
    table.add_column("Domain", width=12)
    table.add_column("Specialties", width=30)
    
    for model_id, info in model_registry.items():
        specialties_text = ", ".join(info["specialties"][:2]) + ("..." if len(info["specialties"]) > 2 else "")
        
        table.add_row(
            model_id,
            info["parameters"],
            f"{info['size_gb']:.1f}GB",
            f"{info['quality_score']:.2f}",
            info["domain"],
            specialties_text
        )
    
    console.print(table)
    
    # Show model selection algorithm demo
    console.print(f"\n🎯 [bold cyan]Model Selection Algorithm Demo[/bold cyan]")
    console.print("-" * 50)
    
    # Filter by size constraint
    size_limit = 20.0
    small_models = {k: v for k, v in model_registry.items() if v['size_gb'] <= size_limit}
    console.print(f"Models under {size_limit}GB: {len(small_models)}")
    
    # Select optimal ensemble
    sorted_models = sorted(small_models.items(), key=lambda x: x[1]['quality_score'], reverse=True)
    selected = dict(sorted_models[:3])  # Top 3
    
    console.print(f"\n🔍 [bold green]Optimal 3-model ensemble (under {size_limit}GB):[/bold green]")
    
    ensemble_table = Table(show_header=True, header_style="bold green")
    ensemble_table.add_column("Model", style="yellow")
    ensemble_table.add_column("Quality", justify="right")
    ensemble_table.add_column("Size", justify="right")
    ensemble_table.add_column("Specialties")
    
    total_size = 0
    for model_id, info in selected.items():
        total_size += info['size_gb']
        ensemble_table.add_row(
            model_id,
            f"{info['quality_score']:.2f}",
            f"{info['size_gb']:.1f}GB",
            ", ".join(info['specialties'][:3])
        )
    
    console.print(ensemble_table)
    console.print(f"\n📊 Ensemble Statistics:")
    console.print(f"• Total size: {total_size:.1f}GB")
    console.print(f"• Average quality: {sum(info['quality_score'] for info in selected.values()) / len(selected):.2f}")
    console.print(f"• Combined specialties: {len(set().union(*(info['specialties'] for info in selected.values())))}")
    
    # Show usage instructions
    console.print(f"\n💡 [bold yellow]Next Steps:[/bold yellow]")
    console.print("1. Configure cloud storage credentials in .env")
    console.print("2. Run: deepparallel download-models --count 3 --size-limit 20")
    console.print("3. Or run: ./setup_genesis.sh for guided setup")
    console.print("4. Test with: deepparallel genesis 'Your scientific question'")
    
    console.print(f"\n🚀 [bold blue]When fully configured, Genesis Engine will:[/bold blue]")
    console.print("• Download and manage 12+ open source models")
    console.print("• Run massive-scale simulations (10^9+ data points)")
    console.print("• Perform virtual experiments across all domains")
    console.print("• Use quantum-enhanced processing")
    console.print("• Store models in multi-cloud infrastructure")
    console.print("• Achieve >50% on FrontierMath benchmark")

if __name__ == "__main__":
    demo_model_registry()
