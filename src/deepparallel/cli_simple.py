import typer
from typing import Optional
import json
from rich.console import Console
from rich.table import Table

from .reasoning import DeepParallelReasoning, DeepParallelEnsemble

app = typer.Typer(help="Deep Parallel: Path to 100% Benchmark Domination")
console = Console()


@app.command()
def reason(
    question: str = typer.Argument(..., help="Scientific question to analyze"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file"),
):
    """Run deep parallel reasoning on a scientific question"""
    console.print(f"[bold blue]🧬 Deep Parallel Reasoning[/bold blue]")
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
    """Run ensemble reasoning across multiple approaches"""
    console.print(f"[bold blue]🔬 Ensemble Reasoning[/bold blue]")
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
def demo():
    """Run a demo of Deep Parallel capabilities"""
    console.print(f"[bold blue]🚀 Deep Parallel Demo[/bold blue]")
    console.print()
    
    test_questions = [
        "What is the optimal temperature for protein folding?",
        "How does quantum entanglement work?",
        "What causes superconductivity?",
        "How do neural networks learn?",
        "What is the mechanism of photosynthesis?"
    ]
    
    console.print("[bold cyan]Available test questions:[/bold cyan]")
    for i, q in enumerate(test_questions, 1):
        console.print(f"{i}. {q}")
    
    console.print()
    console.print("[bold green]Running ensemble reasoning on first question...[/bold green]")
    
    # Demo ensemble reasoning
    ensemble = DeepParallelEnsemble()
    answer, confidence = ensemble.answer(test_questions[0])
    
    console.print(f"\n[bold yellow]Question:[/bold yellow] {test_questions[0]}")
    console.print(f"[bold green]Answer:[/bold green] {answer}")
    console.print(f"[bold blue]Confidence:[/bold blue] {confidence:.2%}")
    
    console.print()
    console.print("[bold cyan]To test other questions:[/bold cyan]")
    console.print("deepparallel reason 'Your question here'")
    console.print("deepparallel ensemble 'Your question here'")


@app.command()
def info():
    """Show information about Deep Parallel Genesis"""
    console.print(f"[bold blue]🧬 Deep Parallel Genesis Information[/bold blue]")
    console.print()
    
    console.print("[bold cyan]Current Implementation:[/bold cyan]")
    console.print("• ✅ Multi-path parallel reasoning")
    console.print("• ✅ Ensemble decision making")
    console.print("• ✅ Scientific domain expertise")
    console.print("• ✅ Confidence estimation")
    console.print()
    
    console.print("[bold yellow]Genesis Engine Features (Planned):[/bold yellow]")
    console.print("• 🔬 Massive-scale data simulation")
    console.print("• 🧪 Virtual experiment engine")
    console.print("• ⚛️  Quantum-enhanced processing")
    console.print("• 🤖 Multi-model ensemble (12+ models)")
    console.print("• ☁️  Cloud-distributed computation")
    console.print("• 📊 Reality-scale benchmarking")
    console.print()
    
    console.print("[bold green]Usage Examples:[/bold green]")
    console.print("deepparallel demo                    # Run interactive demo")
    console.print("deepparallel reason 'question'       # Single question analysis")
    console.print("deepparallel ensemble 'question'     # Multi-approach analysis")
    console.print()
    
    console.print("[bold magenta]Future Commands (Genesis Engine):[/bold magenta]")
    console.print("deepparallel genesis 'question'      # Full Genesis analysis")
    console.print("deepparallel download-models         # Download model ensemble")
    console.print("deepparallel model-status            # Check model status")


@app.command()
def benchmark():
    """Run benchmark tests on standard questions"""
    console.print(f"[bold blue]📊 Deep Parallel Benchmark[/bold blue]")
    console.print()
    
    benchmark_questions = [
        "What is the relationship between temperature and enzyme activity?",
        "Explain the mechanism of DNA replication",
        "How does photosynthesis convert light to chemical energy?",
        "What causes the greenhouse effect?",
        "How do vaccines work to prevent disease?"
    ]
    
    results = []
    
    for i, question in enumerate(benchmark_questions, 1):
        console.print(f"[bold cyan]Question {i}:[/bold cyan] {question[:60]}...")
        
        # Test both reasoning approaches
        reasoner = DeepParallelReasoning()
        ensemble = DeepParallelEnsemble()
        
        reasoning_result = reasoner.parallel_reason(question)
        ensemble_answer, ensemble_confidence = ensemble.answer(question)
        
        results.append({
            "question": question,
            "reasoning_result": reasoning_result,
            "ensemble_answer": ensemble_answer,
            "ensemble_confidence": ensemble_confidence
        })
        
        console.print(f"  ✓ Parallel reasoning: {len(reasoning_result)} chars")
        console.print(f"  ✓ Ensemble confidence: {ensemble_confidence:.1%}")
        console.print()
    
    # Summary table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Question", style="dim", width=40)
    table.add_column("Confidence", justify="right")
    table.add_column("Status")
    
    avg_confidence = 0
    for i, result in enumerate(results):
        confidence = result["ensemble_confidence"]
        avg_confidence += confidence
        
        status = "✅ Good" if confidence > 0.7 else "⚠️  Fair" if confidence > 0.5 else "❌ Low"
        
        table.add_row(
            result["question"][:37] + "..." if len(result["question"]) > 40 else result["question"],
            f"{confidence:.1%}",
            status
        )
    
    console.print(table)
    console.print()
    console.print(f"[bold green]Average Confidence: {avg_confidence/len(results):.1%}[/bold green]")
    console.print(f"[bold blue]Benchmark Complete: {len(results)} questions processed[/bold blue]")


if __name__ == "__main__":
    app()
