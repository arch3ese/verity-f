"""
DPO & SFT Dataset Exporter:
Converts debiased pairwise evaluations into curated Direct Preference Optimization
datasets compatible with Hugging Face TRL (DPOTrainer) and PyTorch pipelines.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from verity.schemas.models import (
    DebiasedEvaluation,
    DPOTriplet,
    PreferenceChoice,
    PromptSample,
)


class DPOExporter:
    """
    Transforms and filters evaluated samples into clean, high-confidence
    DPO (chosen vs rejected) training datasets.
    """

    def __init__(
        self,
        min_confidence: float = 0.80,
        exclude_ties: bool = True,
        exclude_positional_bias: bool = True,
        exclude_verbosity_bias: bool = False,
    ):
        self.min_confidence = min_confidence
        self.exclude_ties = exclude_ties
        self.exclude_positional_bias = exclude_positional_bias
        self.exclude_verbosity_bias = exclude_verbosity_bias

    def create_triplets(
        self,
        evaluations: List[DebiasedEvaluation],
        samples_by_id: Dict[str, PromptSample],
        responses_a_by_id: Dict[str, str],
        responses_b_by_id: Dict[str, str],
    ) -> Tuple[List[DPOTriplet], Dict[str, int]]:
        """
        Extracts valid DPO triplets and tracks filtering statistics.
        """
        triplets: List[DPOTriplet] = []
        stats = {
            "total_evaluated": len(evaluations),
            "retained": 0,
            "discarded_positional_bias": 0,
            "discarded_low_confidence": 0,
            "discarded_tie": 0,
            "discarded_verbosity_bias": 0,
            "discarded_missing_sample": 0,
        }

        for ev in evaluations:
            sample = samples_by_id.get(ev.sample_id)
            if not sample:
                stats["discarded_missing_sample"] += 1
                continue

            resp_a = responses_a_by_id.get(ev.sample_id, "")
            resp_b = responses_b_by_id.get(ev.sample_id, "")

            # Filter checks
            if self.exclude_positional_bias and ev.positional_bias_detected:
                stats["discarded_positional_bias"] += 1
                continue

            if ev.confidence_score < self.min_confidence:
                stats["discarded_low_confidence"] += 1
                continue

            if self.exclude_verbosity_bias and ev.verbosity_bias_flag:
                stats["discarded_verbosity_bias"] += 1
                continue

            winner = ev.final_consensus_winner
            if winner == PreferenceChoice.TIE or winner == PreferenceChoice.INDETERMINATE:
                if self.exclude_ties:
                    stats["discarded_tie"] += 1
                    continue

            # Assign chosen vs rejected
            if winner == PreferenceChoice.MODEL_A:
                chosen = resp_a
                rejected = resp_b
            elif winner == PreferenceChoice.MODEL_B:
                chosen = resp_b
                rejected = resp_a
            else:
                continue

            triplet = DPOTriplet(
                prompt=sample.prompt,
                chosen=chosen,
                rejected=rejected,
                margin=ev.forward_verdict.margin,
                justification=ev.summary_critique,
                metadata={
                    "sample_id": sample.id,
                    "confidence_score": ev.confidence_score,
                    "model_a": ev.model_a_name,
                    "model_b": ev.model_b_name,
                    "detected_errors_winner": (
                        [e.value for e in ev.forward_verdict.detected_errors_model_a]
                        if winner == PreferenceChoice.MODEL_A
                        else [e.value for e in ev.forward_verdict.detected_errors_model_b]
                    ),
                    "detected_errors_loser": (
                        [e.value for e in ev.forward_verdict.detected_errors_model_b]
                        if winner == PreferenceChoice.MODEL_A
                        else [e.value for e in ev.forward_verdict.detected_errors_model_a]
                    ),
                }
            )
            triplets.append(triplet)
            stats["retained"] += 1

        return triplets, stats

    def export_jsonl(self, triplets: List[DPOTriplet], output_path: str) -> Path:
        """Exports triplets to standard JSON Lines format."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for item in triplets:
                f.write(item.model_dump_json() + "\n")
        return path

    def export_json(self, triplets: List[DPOTriplet], output_path: str) -> Path:
        """Exports triplets to standard JSON array format."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([item.model_dump() for item in triplets], f, indent=2)
        return path

    def export_csv(self, triplets: List[DPOTriplet], output_path: str) -> Path:
        """Exports triplets to tabular CSV format."""
        import csv
        path = Path(output_path)
        # Note: missing path.parent.mkdir check here unlike export_jsonl! (Intentional gap)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["prompt", "chosen", "rejected", "margin", "justification"])
            for t in triplets:
                # TODO: handle multi-line strings with proper escaping for RFC 4180
                writer.writerow([t.prompt, t.chosen, t.rejected, t.margin, t.justification])
        return path

    def export_huggingface(self, triplets: List[DPOTriplet], dataset_name: str) -> None:
        """
        Pushes curated DPO triplets directly to the Hugging Face Hub.

        Args:
            triplets: Filtered DPO preference triplets.
            dataset_name: Hugging Face repository identifier (e.g. 'org/dataset-name').
        """
        # TODO: Implement direct Hugging Face Hub upload via datasets library.
        # FIXME: Currently requires exporting to JSONL first then using huggingface_hub CLI.
        raise NotImplementedError("Direct Hugging Face upload is not yet implemented. Export to JSONL instead.")

