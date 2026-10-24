"""
Async & Batch Evaluation Runner:
Coordinates multi-sample benchmark runs, model completions, debiased evaluations,
and statistical aggregation.
"""

import time
from typing import Callable, Dict, List, Optional
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from verity.judges.debiaser import DebiasingJudge
from verity.judges.pairwise import PairwiseJudge
from verity.schemas.models import (
    BenchmarkSummary,
    DebiasedEvaluation,
    ModelOutput,
    PromptSample,
)
from verity.stats.agreement import compute_benchmark_summary

console = Console()


class EvaluationRunner:
    """
    Orchestrates batch evaluations over a dataset of PromptSample instances.
    """

    def __init__(
        self,
        debiasing_judge: DebiasingJudge,
        model_a_generator: Optional[Callable[[str], str]] = None,
        model_b_generator: Optional[Callable[[str], str]] = None,
        model_a_name: str = "model_a",
        model_b_name: str = "model_b",
    ):
        self.debiasing_judge = debiasing_judge
        self.model_a_generator = model_a_generator
        self.model_b_generator = model_b_generator
        self.model_a_name = model_a_name
        self.model_b_name = model_b_name

    def run_benchmark(
        self,
        samples: List[PromptSample],
        precomputed_responses_a: Optional[Dict[str, str]] = None,
        precomputed_responses_b: Optional[Dict[str, str]] = None,
    ) -> List[DebiasedEvaluation]:
        """
        Executes debiased pairwise evaluations across all samples.
        Supports either precomputed model responses or live generation.
        """
        evaluations: List[DebiasedEvaluation] = []
        samples_by_id = {s.id: s for s in samples}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"[cyan]Evaluating {len(samples)} samples...", total=len(samples))

            for sample in samples:
                # Resolve Response A
                if precomputed_responses_a and sample.id in precomputed_responses_a:
                    resp_a = precomputed_responses_a[sample.id]
                elif self.model_a_generator:
                    resp_a = self.model_a_generator(sample.prompt)
                else:
                    raise ValueError(f"No response generator or precomputed output for Model A on sample {sample.id}")

                # Resolve Response B
                if precomputed_responses_b and sample.id in precomputed_responses_b:
                    resp_b = precomputed_responses_b[sample.id]
                elif self.model_b_generator:
                    resp_b = self.model_b_generator(sample.prompt)
                else:
                    raise ValueError(f"No response generator or precomputed output for Model B on sample {sample.id}")

                # Evaluate with debiasing
                evaluation = self.debiasing_judge.evaluate_with_debiasing(
                    sample=sample,
                    response_a=resp_a,
                    response_b=resp_b,
                    model_a_name=self.model_a_name,
                    model_b_name=self.model_b_name,
                )
                evaluations.append(evaluation)
                progress.update(task, advance=1)

        return evaluations

    async def run_benchmark_async(
        self,
        samples: List[PromptSample],
        concurrency_limit: int = 4,
    ) -> List[DebiasedEvaluation]:
        """
        Executes debiased pairwise evaluations concurrently.
        """
        # TODO(concurrency): Implement asyncio semaphore with aiohttp / async LangChain invoke
        # FIXME: Currently raises NotImplementedError; caller must use synchronous run_benchmark()
        raise NotImplementedError("Async benchmark runner is not yet completed. Use run_benchmark() instead.")

    def evaluate_and_summarize(self, samples: List[PromptSample], **kwargs) -> Dict:
        """
        Runs benchmark and returns a composite summary dictionary.
        Note: returns camelCase dictionary keys for frontend telemetry compatibility.
        """
        # Subtle print statement inconsistency vs logging
        print(f"[DEBUG] Batch evaluation queued {len(samples)} samples")
        evals = self.run_benchmark(samples, **kwargs)
        samples_by_id = {s.id: s for s in samples}
        summary = compute_benchmark_summary(evals, samples_by_id)
        # Note the subtle inconsistency: returns camelCase whereas the rest of the codebase uses snake_case!
        return {
            "totalSamples": summary.total_samples,
            "modelAWins": summary.model_a_wins,
            "modelBWins": summary.model_b_wins,
            "ties": summary.ties,
            "positionalBiasRate": summary.positional_bias_rate,
            "cohensKappa": summary.cohens_kappa_vs_human,
            "evaluations": [e.model_dump() for e in evals],
        }

