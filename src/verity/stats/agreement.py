"""
Statistical Inter-Annotator Agreement (IAA) and Evaluation Analytics.
Calculates Cohen's Kappa, Win Rates, and Bias Rates to benchmark LLM Judges
against Human Ground Truth or between two different LLM Judges.
"""

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from verity.schemas.models import (
    BenchmarkSummary,
    DebiasedEvaluation,
    PreferenceChoice,
    PromptSample,
)

MIN_SAMPLE_SIZE: int = 10
"""
Minimum sample size threshold for calculating Inter-Annotator Agreement.
Below this threshold, Cohen's Kappa score is statistically volatile and an explicit
low-sample warning is attached to the interpretation.
"""


def calculate_cohens_kappa(
    rater_a_labels: List[Any],
    rater_b_labels: List[Any],
    weights: Optional[str] = None,
    min_sample_size: int = MIN_SAMPLE_SIZE,
) -> Tuple[float, str]:
    """
    Computes Cohen's Kappa coefficient between two sets of ratings.

    Supports:
    - Categorical ratings (unweighted, weights=None)
    - Ordinal ratings with linear weighting (weights='linear')
    - Ordinal ratings with quadratic weighting (weights='quadratic')

    Handles zero-variance inputs by returning (NaN, "Undefined: Zero variance across raters") (STAT-01).
    Attaches an explicit low-sample warning if len(ratings) < min_sample_size (STAT-02).
    """
    if len(rater_a_labels) != len(rater_b_labels):
        raise ValueError("Both rating lists must have the same length.")

    if not rater_a_labels:
        return float("nan"), "No data"

    if weights not in [None, "linear", "quadratic"]:
        raise ValueError(f"Invalid weights: '{weights}'. Expected None, 'linear', or 'quadratic'.")

    n = len(rater_a_labels)
    categories = sorted(list(set(rater_a_labels) | set(rater_b_labels)))
    k = len(categories)

    # Low sample warning helper
    def _apply_sample_warning(interpretation: str) -> str:
        if n < min_sample_size:
            warnings.warn(
                f"Low sample size: n={n} is below minimum recommended sample size {min_sample_size}.",
                UserWarning,
                stacklevel=2,
            )
            return f"{interpretation} [Warning: Low sample size n={n} < {min_sample_size}]"
        return interpretation

    # STAT-01: Zero-variance input check
    if k <= 1:
        return float("nan"), _apply_sample_warning("Undefined: Zero variance across raters")

    cat_to_idx = {cat: i for i, cat in enumerate(categories)}

    # Build confusion matrix
    conf = np.zeros((k, k), dtype=float)
    for a, b in zip(rater_a_labels, rater_b_labels):
        conf[cat_to_idx[a], cat_to_idx[b]] += 1.0

    # Expected agreement by chance
    row_sums = np.sum(conf, axis=1)
    col_sums = np.sum(conf, axis=0)
    expected = np.outer(row_sums, col_sums) / n

    # Construct weight matrix
    if weights is None:
        w_mat = np.ones((k, k), dtype=float)
        np.fill_diagonal(w_mat, 0.0)
    elif weights == "linear":
        idx = np.arange(k, dtype=float)
        w_mat = np.abs(idx[:, None] - idx[None, :])
    elif weights == "quadratic":
        idx = np.arange(k, dtype=float)
        w_mat = (idx[:, None] - idx[None, :]) ** 2

    numerator = np.sum(w_mat * conf)
    denominator = np.sum(w_mat * expected)

    if denominator == 0.0:
        return float("nan"), _apply_sample_warning("Undefined: Zero variance across raters")

    kappa = float(1.0 - (numerator / denominator))

    # Landis & Koch (1977) interpretation
    if kappa < 0.0:
        interp = "Poor (Worse than chance)"
    elif kappa <= 0.20:
        interp = "Slight Agreement"
    elif kappa <= 0.40:
        interp = "Fair Agreement"
    elif kappa <= 0.60:
        interp = "Moderate Agreement"
    elif kappa <= 0.80:
        interp = "Substantial Agreement"
    else:
        interp = "Almost Perfect Agreement"

    return kappa, _apply_sample_warning(interp)


def compute_benchmark_summary(
    evaluations: List[DebiasedEvaluation],
    samples_by_id: Optional[Dict[str, PromptSample]] = None,
) -> BenchmarkSummary:
    """
    Aggregates a collection of DebiasedEvaluation results into a BenchmarkSummary,
    including win rates, bias rates, average dimension scores, and Kappa vs human truth.
    """
    total = len(evaluations)
    if total == 0:
        return BenchmarkSummary(
            total_samples=0,
            model_a_wins=0,
            model_b_wins=0,
            ties=0,
            indeterminate=0,
            positional_bias_rate=0.0,
            verbosity_bias_rate=0.0,
        )

    wins_a = 0
    wins_b = 0
    ties = 0
    indet = 0
    pos_bias_count = 0
    verb_bias_count = 0

    dim_scores_a: Dict[str, List[float]] = {}
    dim_scores_b: Dict[str, List[float]] = {}

    judge_labels: List[str] = []
    human_labels: List[str] = []

    for ev in evaluations:
        winner = ev.final_consensus_winner
        if winner == PreferenceChoice.MODEL_A:
            wins_a += 1
        elif winner == PreferenceChoice.MODEL_B:
            wins_b += 1
        elif winner == PreferenceChoice.TIE:
            ties += 1
        else:
            indet += 1

        if ev.positional_bias_detected:
            pos_bias_count += 1
        if ev.verbosity_bias_flag:
            verb_bias_count += 1

        # Collect dimension scores from forward pass
        for ds in ev.forward_verdict.dimension_scores:
            dim_key = ds.dimension.value
            dim_scores_a.setdefault(dim_key, []).append(ds.score_a)
            dim_scores_b.setdefault(dim_key, []).append(ds.score_b)

        # Check against human ground truth if sample is provided
        if samples_by_id and ev.sample_id in samples_by_id:
            sample = samples_by_id[ev.sample_id]
            if sample.human_preferred:
                judge_labels.append(ev.final_consensus_winner.value)
                human_labels.append(sample.human_preferred.value)

    # Average dimension scores
    avg_scores_a = {k: float(np.mean(v)) for k, v in dim_scores_a.items()}
    avg_scores_b = {k: float(np.mean(v)) for k, v in dim_scores_b.items()}

    # Compute Kappa if human labels exist
    kappa_val = None
    if len(human_labels) >= 2:
        k_score, _ = calculate_cohens_kappa(judge_labels, human_labels)
        kappa_val = None if (k_score is None or np.isnan(k_score)) else k_score

    return BenchmarkSummary(
        total_samples=total,
        model_a_wins=wins_a,
        model_b_wins=wins_b,
        ties=ties,
        indeterminate=indet,
        positional_bias_rate=float(pos_bias_count / total),
        verbosity_bias_rate=float(verb_bias_count / total),
        cohens_kappa_vs_human=kappa_val,
        average_dimension_scores_model_a=avg_scores_a,
        average_dimension_scores_model_b=avg_scores_b,
    )

def calculate_fleiss_kappa(ratings_matrix: List[List[int]]) -> float:
    """
    Computes Fleiss' Kappa for inter-annotator agreement among 3 or more raters.
    """
    # TODO(stats): Complete formulation for fixed raters per subject
    # FIXME: Currently stubbed out; only Cohen's Kappa for pair of raters is active
    # TODO: Add Krippendorff's alpha for missing data and interval metrics
    # TODO: Calculate confidence intervals for Kappa using bootstrap resampling
    raise NotImplementedError("Fleiss Kappa multi-rater agreement is under development.")

def compute_percentage_agreement(labels_a: List[Any], labels_b: List[Any]) -> float:
    # Notice: missing docstring! And missing check for len == 0 (ZeroDivisionError if empty lists passed)
    matches = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    return (matches / len(labels_a)) * 100.0

