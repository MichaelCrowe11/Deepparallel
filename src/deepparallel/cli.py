import typer
from typing import Optional, List
import asyncio
import json
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .reasoning import DeepParallelReasoning, DeepParallelEnsemble
from .config import GenesisConfig
from .model_downloader import ModelDownloadManager

app = typer.Typer(help="Deep Parallel: Path to 100% Benchmark Domination")
console = Console()


@app.command()
def reason(
    question: str = typer.Argument(..., help="Scientific question to analyze"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file"),
):
    """Run deep parallel reasoning on a scientific question"""
    console.print(f"[bold blue]Deep Parallel Reasoning[/bold blue]")
    console.print(f"Question: {question}")
    
    # Use the existing DeepParallelReasoning class
    reasoner = DeepParallelReasoning()
    result = reasoner.parallel_reason(question)
    
    console.print(f"\n[bold green]Result:[/bold green] {result}")
    
    if output:
        with open(output, 'w') as f:
            json.dump({
                "question": question,
                "result": result,
            }, f, indent=2)
        console.print(f"Results saved to {output}")


@app.command()
def ensemble(
    question: str = typer.Argument(..., help="Scientific question to analyze"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file"),
):
    """Run ensemble reasoning across multiple models"""
    console.print(f"[bold blue]Ensemble Reasoning[/bold blue]")
    console.print(f"Question: {question}")
    
    # Use the existing DeepParallelEnsemble class
    ensemble = DeepParallelEnsemble()
    answer, confidence = ensemble.answer(question)
    
    console.print(f"\n[bold green]Answer:[/bold green] {answer}")
    console.print(f"[bold yellow]Confidence:[/bold yellow] {confidence:.2%}")
    
    if output:
        with open(output, 'w') as f:
            json.dump({
                "question": question,
                "answer": answer,
                "confidence": confidence
            }, f, indent=2)
        console.print(f"Results saved to {output}")


@app.command()
def genesis(
    question: str = typer.Argument(..., help="Scientific question for Genesis Engine"),
    simulation_scale: str = typer.Option("massive", "--scale", "-s", help="Simulation scale: minimal, moderate, massive"),
    enable_quantum: bool = typer.Option(True, "--quantum/--no-quantum", help="Enable quantum processing"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file"),
):
    """Run the Genesis Engine for ultimate scientific reasoning"""
    console.print(f"[bold blue]🧬 Deep Parallel Genesis Engine[/bold blue]")
    console.print(f"Question: {question}")
    console.print(f"Scale: {simulation_scale}, Quantum: {enable_quantum}")
    
    async def run_genesis():
        try:
            from .genesis import GenesisEngine
            
            config = GenesisConfig()
            config.simulation.scale = simulation_scale
            config.compute.enable_quantum = enable_quantum
            
            genesis = GenesisEngine(config)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Initializing Genesis Engine...", total=None)
                
                result = await genesis.analyze_with_full_simulation(question)
                
                progress.update(task, description="Analysis complete!")
            
            # Display results
            console.print(f"\n[bold green]Genesis Analysis:[/bold green]")
            console.print(f"Confidence: {result.get('confidence', 0):.2f}")
            console.print(f"Simulation Points: {result.get('simulation_results', {}).get('data_points', 0):,}")
            
            if output:
                with open(output, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                console.print(f"Results saved to {output}")
                
        except ImportError as e:
            console.print(f"[yellow]Genesis Engine not fully configured: {e}[/yellow]")
            console.print(f"Using basic reasoning instead...")
            
            # Fallback to basic reasoning
            reasoner = DeepParallelReasoning()
            result = reasoner.parallel_reason(question)
            console.print(f"\n[bold green]Result:[/bold green] {result}")
    
    asyncio.run(run_genesis())


@app.command()
def download_models(
    target_count: int = typer.Option(12, "--count", "-c", help="Target number of models"),
    size_limit: Optional[float] = typer.Option(None, "--size-limit", "-s", help="Size limit in GB"),
    exclude_licenses: Optional[List[str]] = typer.Option(None, "--exclude", "-e", help="Licenses to exclude"),
    cloud_upload: bool = typer.Option(True, "--cloud/--no-cloud", help="Upload to cloud storage"),
):
    """Download open source models for Genesis Engine"""
    console.print(f"[bold blue]📥 Downloading Genesis Model Ensemble[/bold blue]")
    console.print(f"Target: {target_count} models")
    if size_limit:
        console.print(f"Size limit: {size_limit} GB per model")
    
    async def download_models_async():
        try:
            config = GenesisConfig()
            downloader = ModelDownloadManager(config)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Downloading models...", total=None)
                
                results = await downloader.download_all_genesis_models(
                    target_ensemble_size=target_count,
                    size_constraint_gb=size_limit,
                    exclude_licenses=exclude_licenses or []
                )
                
                progress.update(task, description="Downloads complete!")
            
            # Display results
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Model", style="dim")
            table.add_column("Status")
            table.add_column("Size (GB)", justify="right")
            table.add_column("Cloud URLs")
            
            total_size = 0.0
            success_count = 0
            
            for model_id, result in results.items():
                status = result.get("status", "unknown")
                size_gb = result.get("size_gb", 0)
                cloud_urls = result.get("cloud_urls", {})
                
                if status == "success":
                    status_color = "[green]✓ Success[/green]"
                    success_count += 1
                elif status == "exists":
                    status_color = "[yellow]✓ Exists[/yellow]"
                    success_count += 1
                else:
                    status_color = "[red]✗ Failed[/red]"
                
                total_size += size_gb
                
                cloud_text = ", ".join(cloud_urls.keys()) if cloud_urls else "Local only"
                
                table.add_row(
                    model_id,
                    status_color,
                    f"{size_gb:.1f}",
                    cloud_text
                )
            
            console.print(table)
            console.print(f"\n[bold green]Downloaded {success_count}/{len(results)} models")
            console.print(f"Total size: {total_size:.1f} GB[/bold green]")
            
        except Exception as e:
            console.print(f"[red]Download failed: {e}[/red]")
            console.print(f"[yellow]This may be due to missing dependencies or network issues[/yellow]")
    
    asyncio.run(download_models_async())


@app.command()
def model_status():
    """Check status of downloaded models"""
    console.print(f"[bold blue]📊 Model Status[/bold blue]")
    
    async def check_status():
        try:
            config = GenesisConfig()
            downloader = ModelDownloadManager(config)
            
            status = await downloader.get_model_status()
            
            console.print(f"Cache directory: {status['local_cache']}")
            console.print(f"Total models: {status['total_models']}")
            console.print(f"Total size: {status['total_size_gb']:.1f} GB")
            
            if status.get('models'):
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Model ID")
                table.add_column("Status")
                table.add_column("Size (GB)", justify="right")
                table.add_column("Cloud Storage")
                
                for model_id, model_info in status['models'].items():
                    cloud_providers = list(model_info.get('cloud_urls', {}).keys())
                    cloud_text = ", ".join(cloud_providers) if cloud_providers else "None"
                    
                    table.add_row(
                        model_id,
                        model_info.get('status', 'unknown'),
                        f"{model_info.get('size_gb', 0):.1f}",
                        cloud_text
                    )
                
                console.print(table)
            else:
                console.print("[yellow]No models downloaded yet[/yellow]")
                
        except Exception as e:
            console.print(f"[red]Error checking status: {e}[/red]")
    
    asyncio.run(check_status())


@app.command()
def cleanup_cache(
    keep_recent: int = typer.Option(5, "--keep", "-k", help="Number of recent models to keep"),
):
    """Cleanup local model cache"""
    console.print(f"[bold blue]🧹 Cleaning Model Cache[/bold blue]")
    
    async def cleanup():
        try:
            config = GenesisConfig()
            downloader = ModelDownloadManager(config)
            
            result = await downloader.cleanup_local_cache(keep_recent=keep_recent)
            
            if result['status'] == 'success':
                console.print(f"[green]✓ Cleaned {result['cleaned']} models")
                console.print(f"Freed {result['freed_gb']:.1f} GB")
                console.print(f"Remaining: {result['remaining']} models[/green]")
            else:
                console.print(f"[yellow]No cleanup needed: {result['status']}[/yellow]")
                
        except Exception as e:
            console.print(f"[red]Cleanup failed: {e}[/red]")
    
    asyncio.run(cleanup())
