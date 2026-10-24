"""
Command-Line Interface (CLI) for Verity:
Supports running benchmarks, debiased evaluations, annotator audits, and DPO dataset export.
"""

import json
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from verity.exporters.dpo_exporter import DPOExporter
from verity.judges.debiaser import DebiasingJudge
from verity.judges.pairwise import PairwiseJudge
from verity.models.factory import get_chat_model
from verity.rubrics.definitions import ERROR_TAXONOMY_DESCRIPTIONS, RUBRIC_CRITERIA
from verity.runner import EvaluationRunner
from verity.schemas.models import (
    DebiasedEvaluation,
    PreferenceChoice,
    PromptSample,
    RatingDimension,
)
from verity.stats.agreement import compute_benchmark_summary

app = typer.Typer(
    name="verity",
    help="Enterprise RLHF & AI Training Evaluation Engine with LLM Judge Debiasing.",
    add_completion=False,
)
console = Console()


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host address to bind"),
    port: int = typer.Option(8000, "--port", "-p", help="Port number"),
):
    """Launch the interactive Verity Studio Web UI."""
    console.print(Panel.fit(
        f"[bold cyan]Launching Verity Studio[/bold cyan]\n"
        f"Server running at: [bold green]http://{host}:{port}[/bold green]\n"
        f"Press [yellow]Ctrl+C[/yellow] to stop.",
        title="Verity Web Studio",
    ))
    from verity.web.app import start_server
    start_server(host=host, port=port)


@app.command()
def rubrics():
    """Display the RLHF multi-dimensional rubrics and error taxonomy."""
    console.print(Panel.fit("[bold cyan]Verity RLHF Evaluation Rubrics[/bold cyan]"))

    rubric_table = Table(title="Evaluation Dimensions & Scoring Criteria", show_header=True)
    rubric_table.add_column("Dimension", style="bold yellow", width=22)
    rubric_table.add_column("Focus", style="cyan", width=35)
    rubric_table.add_column("Score 1 (Fail)", style="red", width=25)
    rubric_table.add_column("Score 5 (Exemplary)", style="green", width=25)

    for dim, criteria in RUBRIC_CRITERIA.items():
        rubric_table.add_row(
            dim.value,
            criteria["description"],
            criteria["score_1"][:80] + "...",
            criteria["score_5"][:80] + "...",
        )
    console.print(rubric_table)

    err_table = Table(title="Error Taxonomy & Failure Modes", show_header=True)
    err_table.add_column("Error Tag", style="bold magenta", width=30)
    err_table.add_column("Definition & Violation Criteria", style="white")

    for err, desc in ERROR_TAXONOMY_DESCRIPTIONS.items():
        err_table.add_row(err.value, desc)
    console.print(err_table)


@app.command()
def audit(
    prompt: str = typer.Option(..., "--prompt", "-p", help="Benchmark user prompt"),
    candidate_a: str = typer.Option(..., "--model-a", "-a", help="Response from Candidate A"),
    candidate_b: str = typer.Option(..., "--model-b", "-b", help="Response from Candidate B"),
    constraint: Optional[str] = typer.Option(None, "--constraint", "-c", help="Negative or formatting constraint"),
    judge_provider: str = typer.Option("mock", "--provider", help="Judge provider (gemini, openai, ollama, mock)"),
    judge_model: str = typer.Option("mock-judge", "--model", help="Judge model name"),
):
    """Run an instant debiased audit comparing two completions."""
    console.print(f"[bold green]Running debiased audit using {judge_provider}:{judge_model}...[/bold green]")

    sample = PromptSample(
        id="audit-single-01",
        prompt=prompt,
        constraints=[constraint] if constraint else [],
    )

    chat_model = get_chat_model(judge_provider, judge_model)
    pairwise_judge = PairwiseJudge(chat_model)
    debiaser = DebiasingJudge(pairwise_judge)

    evaluation = debiaser.evaluate_with_debiasing(
        sample=sample,
        response_a=candidate_a,
        response_b=candidate_b,
        model_a_name="Candidate A",
        model_b_name="Candidate B",
    )

    _display_single_evaluation(evaluation)


@app.command()
def benchmark(
    dataset_path: str = typer.Option(..., "--dataset", "-d", help="Path to JSON benchmark dataset file"),
    judge_provider: str = typer.Option("mock", "--provider", help="Judge provider (gemini, openai, ollama, mock)"),
    judge_model: str = typer.Option("mock-judge", "--model", help="Judge model name"),
    export_dpo: Optional[str] = typer.Option(None, "--export-dpo", help="Output path for clean DPO JSONL export"),
    export_summary: Optional[str] = typer.Option(None, "--export-summary", help="Output path for benchmark summary JSON"),
):
    """Execute a comprehensive benchmark evaluation over a dataset of samples."""
    path = Path(dataset_path)
    if not path.exists():
        console.print(f"[bold red]Dataset file not found: {dataset_path}[/bold red]")
        raise typer.Exit(code=1)

    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    samples: list[PromptSample] = []
    responses_a: dict[str, str] = {}
    responses_b: dict[str, str] = {}

    for item in raw_data:
        s = PromptSample.model_validate(item["sample"])
        samples.append(s)
        responses_a[s.id] = item.get("response_a", "")
        responses_b[s.id] = item.get("response_b", "")

    console.print(f"[bold cyan]Loaded {len(samples)} benchmark samples from {path.name}[/bold cyan]")

    chat_model = get_chat_model(judge_provider, judge_model)
    pairwise_judge = PairwiseJudge(chat_model)
    debiaser = DebiasingJudge(pairwise_judge)

    runner = EvaluationRunner(
        debiasing_judge=debiaser,
        model_a_name=raw_data[0].get("model_a_name", "Model A") if raw_data else "Model A",
        model_b_name=raw_data[0].get("model_b_name", "Model B") if raw_data else "Model B",
    )

    evaluations = runner.run_benchmark(
        samples=samples,
        precomputed_responses_a=responses_a,
        precomputed_responses_b=responses_b,
    )

    # Compute Statistical Benchmark Summary
    samples_map = {s.id: s for s in samples}
    summary = compute_benchmark_summary(evaluations, samples_map)

    # Render Summary Tables
    _display_benchmark_summary(summary, evaluations)

    # Export DPO dataset if requested
    if export_dpo:
        exporter = DPOExporter(min_confidence=0.80)
        triplets, filter_stats = exporter.create_triplets(
            evaluations=evaluations,
            samples_by_id=samples_map,
            responses_a_by_id=responses_a,
            responses_b_by_id=responses_b,
        )
        out_p = exporter.export_jsonl(triplets, export_dpo)
        console.print(f"[bold green]Exported {len(triplets)} curated DPO triplets to: {out_p}[/bold green]")
        console.print(f"Filter statistics: {filter_stats}")

    # Export JSON Summary if requested
    if export_summary:
        out_sum = Path(export_summary)
        out_sum.parent.mkdir(parents=True, exist_ok=True)
        with open(out_sum, "w", encoding="utf-8") as f:
            f.write(summary.model_dump_json(indent=2))
        console.print(f"[bold green]Saved benchmark summary metrics to: {out_sum}[/bold green]")


def _display_single_evaluation(ev: DebiasedEvaluation):
    status_color = "green" if ev.final_consensus_winner != PreferenceChoice.INDETERMINATE else "red"
    console.print(Panel(
        f"[bold {status_color}]Consensus Winner: {ev.final_consensus_winner.value.upper()}[/bold {status_color}]\n"
        f"Confidence: {ev.confidence_score * 100:.0f}%\n"
        f"Positional Bias Detected: {'YES (Order changed verdict)' if ev.positional_bias_detected else 'NO (Order-invariant)'}\n"
        f"Verbosity Bias Flag: {'YES' if ev.verbosity_bias_flag else 'NO'}\n\n"
        f"[bold]Summary Critique:[/bold]\n{ev.summary_critique}",
        title=f"Evaluation Verdict for Sample: {ev.sample_id}",
    ))


def _display_benchmark_summary(summary, evaluations):
    table = Table(title="[bold yellow]Verity Benchmark Summary[/bold yellow]", show_header=True)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="bold white")

    table.add_row("Total Samples Evaluated", str(summary.total_samples))
    table.add_row("Model A Win Rate", f"{summary.model_a_wins / summary.total_samples * 100:.1f}% ({summary.model_a_wins})")
    table.add_row("Model B Win Rate", f"{summary.model_b_wins / summary.total_samples * 100:.1f}% ({summary.model_b_wins})")
    table.add_row("Tie Rate", f"{summary.ties / summary.total_samples * 100:.1f}% ({summary.ties})")
    table.add_row("Indeterminate (Positional Bias)", f"{summary.indeterminate / summary.total_samples * 100:.1f}% ({summary.indeterminate})")
    table.add_row("Positional Inconsistency Rate", f"{summary.positional_bias_rate * 100:.1f}%")
    table.add_row("Verbosity Bias Rate", f"{summary.verbosity_bias_rate * 100:.1f}%")

    if summary.cohens_kappa_vs_human is not None:
        table.add_row(
            "Cohen's Kappa (LLM vs Human)",
            f"{summary.cohens_kappa_vs_human:.3f} (Inter-Annotator Agreement)"
        )

    console.print(table)


def main():
    app()


if __name__ == "__main__":
    main()

@app.command()
def export(
    dataset: Path = typer.Option(..., "--dataset", "-d", help="Path to evaluated JSON"),
    output: Path = typer.Option(..., "--output", "-o", help="Target output file"),
):
    """Export evaluated triplets to downstream fine-tuning formats (WIP)."""
    # TODO: Wire with DPOExporter once HuggingFace integration is complete
    # TODO: Add interactive prompt mode using rich prompts / prompt_toolkit
    # TODO: Add CSV input ingestion support in CLI audit command
    typer.echo("Error: 'verity export' is currently under construction. Use 'verity benchmark --export-dpo' instead.")
    raise typer.Exit(code=1)

