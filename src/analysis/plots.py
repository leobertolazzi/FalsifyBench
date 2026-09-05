"""Generate publication-ready comparison plots from evaluated model results.

Usage examples:
    python -m src.analysis.plots
    python -m src.analysis.plots --results-dir results --output-dir results/plots
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import matplotlib.pyplot as plt
from scipy import stats

from src.benchmark.evaluation import (
    discover_model_bias_falsification_correlation_rows,
    GameEvaluator,
    discover_model_aggregates,
    discover_model_guess_counts,
    discover_model_relation_distributions,
    discover_model_relation_distributions_by_outcome,
    discover_model_relation_metrics,
    discover_model_success_bias_correlation_rows,
    discover_model_target_property_guess_counts,
)


RELATION_LABELS: List[str] = [
    "identical",
    "disjoint",
    "partial_overlap",
    "hypothesis_included_in_target",
    "target_included_in_hypothesis",
]
EXPERIMENT_RELATIONS: List[str] = ["disjoint", "partial_overlap"]


METRICS: List[Tuple[str, str, str]] = [
    ("success_rate", "Success Rate", "proportion"),
    ("avg_turns_to_solution", "Avg Turns to Solution", "count"),
    (
        "avg_positive_testing_bias",
        "Avg Confirmation Bias",
        "ratio",
    ),
    ("avg_conclusive_falsification_rate", "Avg Conclusive Falsification Rate", "ratio"),
]

STACKED_RELATION_LABELS: List[str] = [
    "partial_overlap",
    "disjoint",
    "identical",
    "target_included_in_hypothesis",
    "hypothesis_included_in_target",
]
RELATION_DISTRIBUTION_LABELS: List[str] = [
    "target_included_in_hypothesis",
    "disjoint",
    "partial_overlap",
    "identical",
    "hypothesis_included_in_target",
]

MODEL_XTICK_LABEL_SIZE = 7
BAR_PLOT_MODEL_LABEL_SIZE = 7
BAR_PLOT_VALUE_LABEL_SIZE = 8
GROUPED_BAR_PLOT_VALUE_LABEL_SIZE = 4
RELATION_DISPLAY_LABELS: Dict[str, str] = {
    "identical": "Identical",
    "disjoint": "Disjoint",
    "partial_overlap": "Partial Overlap",
    "hypothesis_included_in_target": "Hypothesis in Target",
    "target_included_in_hypothesis": "Target in Hypothesis",
}
RELATION_COLORS: Dict[str, Tuple[float, float, float, float]] = {
    "identical": (0.84, 0.35, 0.29, 1.0),
    "disjoint": (0.89, 0.63, 0.27, 1.0),
    "partial_overlap": (0.35, 0.66, 0.39, 1.0),
    "hypothesis_included_in_target": (0.27, 0.56, 0.86, 1.0),
    "target_included_in_hypothesis": (0.60, 0.40, 0.80, 1.0),
}
RELATION_DISTRIBUTION_COLORS: Dict[str, Tuple[float, float, float, float]] = {
    "target_included_in_hypothesis": (0.15, 0.45, 0.24, 1.0),
    "disjoint": (0.28, 0.62, 0.33, 1.0),
    "partial_overlap": (0.55, 0.79, 0.52, 1.0),
    "identical": (0.55, 0.58, 0.60, 1.0),
    "hypothesis_included_in_target": (0.27, 0.56, 0.86, 1.0),
}
RELATION_DISTRIBUTION_DISPLAY_LABELS: Dict[str, str] = {
    "target_included_in_hypothesis": r"$R \subset H$",
    "disjoint": "Disjoint",
    "partial_overlap": "Partial Overlap",
    "identical": r"$H = R$",
    "hypothesis_included_in_target": r"$H \subset R$",
}
TARGET_PROPERTIES: List[str] = ["animal", "artifact", "body part", "food", "plant"]
TARGET_PROPERTY_DISPLAY_LABELS: Dict[str, str] = {
    "animal": "Animal",
    "artifact": "Artifact",
    "body part": "Body Part",
    "food": "Food",
    "plant": "Plant",
}
ERROR_ANALYSIS_RESULTS_FILENAMES: List[str] = [
    "error_analysis_results.json",
    "relation_experiments_results.json",
]

PERCENTAGE_METRICS = {
    "success_rate",
    "avg_positive_testing_bias",
    "avg_conclusive_falsification_rate",
}
RELATION_AWARE_PLOT_METRICS = {
    "avg_conclusive_falsification_rate",
}
FIXED_MAX_TURN_METRICS = {
    "avg_turns_to_solution",
}
FIXED_100_SCORE_METRICS: set[str] = set()
AUTO_UPPER_TURN_METRICS: set[str] = set()
DISPERSION_METRICS = {
    "avg_guess_count",
    "avg_turns_to_solution",
    "avg_positive_testing_bias",
    "avg_conclusive_falsification_rate",
}

OVERALL_BAR_WIDTH = 0.14
GROUPED_BAR_WIDTH = 0.08
MODEL_SPACING = 0.2

MODEL_COLOR_PAIRS: List[Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]] = [
    ((219 / 255.0, 87 / 255.0, 87 / 255.0, 1.0), (225 / 255.0, 141 / 255.0, 135 / 255.0, 1.0)),
    ((219 / 255.0, 96 / 255.0, 87 / 255.0, 1.0), (225 / 255.0, 141 / 255.0, 135 / 255.0, 1.0)),
    ((219 / 255.0, 145 / 255.0, 87 / 255.0, 1.0), (236 / 255.0, 201 / 255.0, 175 / 255.0, 1.0)),
    ((219 / 255.0, 194 / 255.0, 87 / 255.0, 1.0), (236 / 255.0, 224 / 255.0, 175 / 255.0, 1.0)),
    ((146 / 255.0, 219 / 255.0, 88 / 255.0, 1.0), (209 / 255.0, 231 / 255.0, 190 / 255.0, 1.0)),
    ((88 / 255.0, 219 / 255.0, 128 / 255.0, 1.0), (175 / 255.0, 228 / 255.0, 192 / 255.0, 1.0)),
    ((87 / 255.0, 212 / 255.0, 219 / 255.0, 1.0), (164 / 255.0, 224 / 255.0, 228 / 255.0, 1.0)),
    ((87 / 255.0, 163 / 255.0, 219 / 255.0, 1.0), (164 / 255.0, 199 / 255.0, 230 / 255.0, 1.0)),
    ((87 / 255.0, 113 / 255.0, 219 / 255.0, 1.0), (164 / 255.0, 177 / 255.0, 230 / 255.0, 1.0)),
    ((161 / 255.0, 87 / 255.0, 219 / 255.0, 1.0), (202 / 255.0, 170 / 255.0, 228 / 255.0, 1.0)),
    ((220 / 255.0, 88 / 255.0, 178 / 255.0, 1.0), (225 / 255.0, 142 / 255.0, 199 / 255.0, 1.0)),
    ((219 / 255.0, 87 / 255.0, 129 / 255.0, 1.0), (235 / 255.0, 170 / 255.0, 193 / 255.0, 1.0)),
]

PAPER_PAIR_PANEL_WIDTH = 6.5
PAPER_PAIR_PANEL_HEIGHT = 3.25

MODEL_DISPLAY_OVERRIDES = {
    "deepseekapi": "DeepSeek-V3.2",
    "Mistral-Small-24B-Instruct-2501": "Mistral-Small-24B",
    "meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8": "Llama-4-Maverick",
    "Llama-4-Maverick-17B-128E-Instruct-FP8": "Llama-4-Maverick",
    "openai_gpt-oss-120b": "GPT-OSS-120B",
    "gpt-oss-120b": "GPT-OSS-120B",
    "openai_gpt-oss-20b": "GPT-OSS-20B",
    "gpt-oss-20b": "GPT-OSS-20B",
    "gpt-5-mini": "GPT-5-Mini",
    "gpt-5-nano": "GPT-5-Nano",
    "gpt-5.2-chat": "GPT-5.2-Chat",
}
NON_REASONING_DISPLAY_NAMES = {
    "DeepSeek-V3.1",
    "MiniMax-M2.5",
    "Mistral-Small-24B",
    "Llama-4-Maverick",
}
OVERALL_BAR_EMPHASIZED_METRICS = {
    "success_rate",
    "avg_turns_to_solution",
    "avg_positive_testing_bias",
    "avg_guess_count",
}


def configure_plot_style() -> None:
    """Set Matplotlib defaults tuned for paper-friendly figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.labelsize": 5,
            "ytick.labelsize": 9,
        }
    )


def _safe_model_name(name: str) -> str:
    base_name = name.split("_")[1] if "_" in name else name
    display_name = MODEL_DISPLAY_OVERRIDES.get(name, MODEL_DISPLAY_OVERRIDES.get(base_name, base_name))
    if display_name.startswith("gpt"):
        return f"GPT{display_name[3:]}"
    return display_name


def _is_reasoning_display_name(display_name: str) -> bool:
    return display_name not in NON_REASONING_DISPLAY_NAMES


def _build_model_palette_map(
    rows: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Tuple[float, float, float, float]]]:
    """Assign deterministic full/light color pairs to each model."""
    model_names = _canonical_model_order(rows)
    palette_map: Dict[str, Dict[str, Tuple[float, float, float, float]]] = {}

    for index, model_name in enumerate(model_names):
        full_color, light_color = MODEL_COLOR_PAIRS[index % len(MODEL_COLOR_PAIRS)]
        palette_map[model_name] = {"full": full_color, "light": light_color}

    return palette_map


def _canonical_model_order(rows: List[Dict[str, Any]]) -> List[str]:
    """Stable model order for all plots: non-reasoning first, then reasoning."""
    models = []
    for row in rows:
        model_id = str(row.get("model"))
        if model_id and model_id not in models:
            models.append(model_id)

    non_reasoning = [
        model_id
        for model_id in models
        if not _is_reasoning_display_name(_safe_model_name(model_id))
    ]
    reasoning = [
        model_id
        for model_id in models
        if _is_reasoning_display_name(_safe_model_name(model_id))
    ]

    non_reasoning.sort(key=lambda model_id: _safe_model_name(model_id))
    reasoning.sort(key=lambda model_id: _safe_model_name(model_id))
    return non_reasoning + reasoning


def _separator_x_position(x_positions: List[float], reasoning_flags: List[bool]) -> Optional[float]:
    """Return x-position separating non-reasoning and reasoning groups, if both exist."""
    if not x_positions or not reasoning_flags:
        return None

    if all(reasoning_flags) or not any(reasoning_flags):
        return None

    first_reasoning_index = reasoning_flags.index(True)
    if first_reasoning_index <= 0:
        return None

    left = x_positions[first_reasoning_index - 1]
    right = x_positions[first_reasoning_index]
    return (left + right) / 2.0


def _order_entries_by_canonical_model_order(
    entries: List[Dict[str, Any]],
    canonical_order: List[str],
    model_key: str = "model_id",
) -> List[Dict[str, Any]]:
    """Order entries by a fixed canonical model order."""
    by_model = {str(entry.get(model_key)): entry for entry in entries}
    return [by_model[model_id] for model_id in canonical_order if model_id in by_model]


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _transform_metric_value(metric_key: str, value: float) -> float:
    """Convert ratio/proportion metrics to percentage for plotting."""
    if metric_key in PERCENTAGE_METRICS:
        return value * 100.0
    return value


def _output_metric_slug(metric_key: str) -> str:
    """Map internal metric keys to plot filename slugs."""
    if metric_key == "avg_positive_testing_bias":
        return "avg_confirmation_bias"
    return metric_key


def _rounded_percentage_label_value(value: float) -> int:
    """Round percentage labels using Python's standard rounding for display."""
    return int(round(value))


def _percentage_label_lower_bound(value: float) -> int:
    """Use the lower integer bound when deciding whether to annotate a segment."""
    return int(math.floor(value))


def _metric_value(metrics: Dict[str, Any], metric_key: str) -> Any:
    return metrics.get(metric_key)


def _dispersion_metric_key(metric_key: str) -> Optional[str]:
    if metric_key in DISPERSION_METRICS:
        return f"{metric_key}_std"
    return None


def _uncertainty_metric_key(metric_key: str) -> Optional[str]:
    if metric_key in DISPERSION_METRICS:
        return f"{metric_key}_ci95"
    return None


def _dispersion_scale_metric_key(metric_key: str) -> str:
    return metric_key


def _metric_dispersion_value(metrics: Dict[str, Any], metric_key: str) -> Optional[float]:
    uncertainty_key = _uncertainty_metric_key(metric_key)
    dispersion_key = _dispersion_metric_key(metric_key)
    if uncertainty_key is None or dispersion_key is None:
        return None

    value = metrics.get(uncertainty_key)
    if not isinstance(value, (int, float)):
        value = metrics.get(dispersion_key)
    if not isinstance(value, (int, float)):
        return None

    return _transform_metric_value(_dispersion_scale_metric_key(metric_key), float(value))


def _metric_ylabel_fontsize(metric_key: str) -> float:
    """Use a slightly smaller y-axis label for the longest metric label."""
    if metric_key == "avg_conclusive_falsification_rate":
        return 8.0
    return 9.0


def _compute_auto_upper_bound(values: List[float], padding_ratio: float = 0.1) -> float:
    """Compute an upper bound with headroom for auto-scaled metrics."""
    if not values:
        return 1.0

    value_max = max(values)
    if value_max <= 0:
        return 1.0

    padded = value_max * (1.0 + padding_ratio)
    return padded if padded > value_max else value_max + 1.0


def _spearman_correlation_with_p_value(x_values: List[float], y_values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Compute Spearman rank correlation and p-value."""
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None, None

    rho, p_value = stats.spearmanr(x_values, y_values)
    if math.isnan(rho) or math.isnan(p_value):
        return None, None

    return float(rho), float(p_value)


def _pearson_correlation_with_p_value(x_values: List[float], y_values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Compute Pearson correlation and p-value."""
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None, None

    correlation, p_value = stats.pearsonr(x_values, y_values)
    if math.isnan(correlation) or math.isnan(p_value):
        return None, None

    return float(correlation), float(p_value)


def _linear_regression_fit_with_confidence_interval(
    x_values: List[float],
    y_values: List[float],
    x_line: List[float],
    confidence_level: float = 0.95,
) -> Optional[Tuple[List[float], List[float], List[float]]]:
    """Return fitted y values and mean-response confidence band for a regression line."""
    if len(x_values) != len(y_values) or len(x_values) < 3 or not x_line:
        return None

    regression = stats.linregress(x_values, y_values)
    x_mean = sum(x_values) / len(x_values)
    sample_size = len(x_values)
    sum_squared_x = sum((x_value - x_mean) ** 2 for x_value in x_values)
    if sum_squared_x == 0.0:
        return None

    fitted_y = [regression.slope * x_value + regression.intercept for x_value in x_line]
    residual_sum_of_squares = sum(
        (y_value - (regression.slope * x_value + regression.intercept)) ** 2
        for x_value, y_value in zip(x_values, y_values)
    )

    degrees_of_freedom = sample_size - 2
    if degrees_of_freedom <= 0:
        return None

    residual_standard_error = math.sqrt(residual_sum_of_squares / degrees_of_freedom)
    alpha = 1.0 - confidence_level
    critical_t = stats.t.ppf(1.0 - alpha / 2.0, degrees_of_freedom)

    ci_delta: List[float] = []
    for x_value in x_line:
        standard_error = residual_standard_error * math.sqrt(
            (1.0 / sample_size) + (((x_value - x_mean) ** 2) / sum_squared_x)
        )
        ci_delta.append(float(critical_t * standard_error))

    lower_band = [y_value - delta for y_value, delta in zip(fitted_y, ci_delta)]
    upper_band = [y_value + delta for y_value, delta in zip(fitted_y, ci_delta)]
    return fitted_y, lower_band, upper_band


def _figure_size_for_bars(model_count: int, group_half_width: float) -> Tuple[float, float]:
    """Compute a compact figure size from bar geometry and model count."""
    if model_count <= 0:
        return (4.8, 3.2)

    span_units = (model_count - 1) * MODEL_SPACING + (2.0 * group_half_width)
    horizontal_margin_units = max(group_half_width, 0.08) * 2.0
    total_units = span_units + horizontal_margin_units

    width_inches = max(4.8, min(7.2, total_units * 2.8))
    return (width_inches, 3.2)


def _figure_size_for_overall_metric(metric_key: str, model_count: int) -> Tuple[float, float]:
    """Overall plots keep width but use a shorter height."""
    base_width, _ = _figure_size_for_bars(model_count, OVERALL_BAR_WIDTH / 2)

    if metric_key == "avg_conclusive_falsification_rate":
        return (PAPER_PAIR_PANEL_WIDTH, PAPER_PAIR_PANEL_HEIGHT)

    return (base_width, 2.4)


def _apply_metric_axis_limits(
    ax: plt.Axes,
    metric_key: str,
    max_turns: int,
    plotted_values: List[float],
    error_values: Optional[List[float]] = None,
) -> None:
    """Apply metric-specific y-axis limits."""
    upper_values = list(plotted_values)
    if error_values:
        upper_values = [
            value + max(0.0, error_values[index] if index < len(error_values) else 0.0)
            for index, value in enumerate(plotted_values)
        ]

    if metric_key in PERCENTAGE_METRICS:
        ax.set_ylim(0, max(100.0, _compute_auto_upper_bound(upper_values, padding_ratio=0.05)))
    elif metric_key in FIXED_100_SCORE_METRICS:
        ax.set_ylim(0, max(100.0, _compute_auto_upper_bound(upper_values, padding_ratio=0.05)))
    elif metric_key in FIXED_MAX_TURN_METRICS:
        ax.set_ylim(0, max(float(max_turns), _compute_auto_upper_bound(upper_values, padding_ratio=0.05)))
    elif metric_key in AUTO_UPPER_TURN_METRICS:
        ax.set_ylim(0, _compute_auto_upper_bound(upper_values))


def _format_bar_value(metric_key: str, value: float) -> str:
    """Format bar labels consistently with each metric's scale."""
    if metric_key in PERCENTAGE_METRICS:
        return f"{value:.1f}%"
    if metric_key in FIXED_100_SCORE_METRICS:
        return f"{value:.1f}"
    if metric_key in FIXED_MAX_TURN_METRICS or metric_key in AUTO_UPPER_TURN_METRICS:
        return f"{value:.2f}"
    return f"{value:.2f}"


def _format_value_with_dispersion(metric_key: str, value: float, dispersion: Optional[float]) -> str:
    if dispersion is None:
        return _format_bar_value(metric_key, value)
    return f"{_format_bar_value(metric_key, value)}\n±{_format_bar_value(metric_key, dispersion)}"


def _format_p_value_for_legend(p_value: Optional[float]) -> str:
    if p_value is None:
        return "n/a"
    if p_value < 0.001:
        return "< 0.001"
    return f"{p_value:.3g}"


def _contrast_text_color(color: Tuple[float, float, float, float]) -> str:
    """Choose black or white text based on perceived luminance."""
    red, green, blue, _ = color
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "black" if luminance >= 0.6 else "white"


def _lighten_color(
    color: Tuple[float, float, float, float],
    mix_with_white: float = 0.2,
) -> Tuple[float, float, float, float]:
    """Blend a color slightly toward white while preserving opacity."""
    red, green, blue, alpha = color
    mix = max(0.0, min(1.0, mix_with_white))
    return (
        red + ((1.0 - red) * mix),
        green + ((1.0 - green) * mix),
        blue + ((1.0 - blue) * mix),
        alpha,
    )


def _annotate_bar_values(
    ax: plt.Axes,
    bars: Any,
    metric_key: str,
    plotted_values: List[float],
    error_values: Optional[List[float]] = None,
    font_size: int = 6,
) -> None:
    """Add numeric labels above bars using a small offset from the axis range."""
    if not plotted_values:
        return

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    offset = y_range * 0.015 if y_range > 0 else 0.1

    for bar, value in zip(bars, plotted_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + offset,
            _format_bar_value(metric_key, value),
            ha="center",
            va="bottom",
            fontsize=font_size,
            rotation=0,
            zorder=6,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
        )


def discover_model_experiment_1_rows(results_dir: Path) -> List[Dict[str, Any]]:
    """Read experiment 1 outputs and extract mean gaps, effect sizes, and p-values."""
    rows: List[Dict[str, Any]] = []

    seen_models = set()
    for filename in ERROR_ANALYSIS_RESULTS_FILENAMES:
        for experiment_path in sorted(results_dir.glob(f"*/{filename}")):
            model_dir = experiment_path.parent
            if model_dir in seen_models:
                continue
            seen_models.add(model_dir)
            payload = _load_json(experiment_path)
            experiment_payload = payload.get("experiment_1_distributional_difference") or {}
            relations = experiment_payload.get("relations") or {}

            row: Dict[str, Any] = {"model": model_dir.name}
            found_value = False
            for relation_label in EXPERIMENT_RELATIONS:
                relation_payload = relations.get(relation_label) or {}
                mean_difference = relation_payload.get("mean_difference")
                cliffs_delta = relation_payload.get("cliffs_delta")
                mann_whitney_p_value = relation_payload.get("mann_whitney_p_value")
                mann_whitney_p_value_holm = relation_payload.get("mann_whitney_p_value_holm")
                if isinstance(mean_difference, (int, float)):
                    row[f"{relation_label}_mean_difference"] = float(mean_difference)
                    found_value = True
                if isinstance(cliffs_delta, (int, float)):
                    row[f"{relation_label}_cliffs_delta"] = float(cliffs_delta)
                    found_value = True
                if isinstance(mann_whitney_p_value, (int, float)):
                    row[f"{relation_label}_mann_whitney_p_value"] = float(mann_whitney_p_value)
                    found_value = True
                if isinstance(mann_whitney_p_value_holm, (int, float)):
                    row[f"{relation_label}_mann_whitney_p_value_holm"] = float(mann_whitney_p_value_holm)
                    found_value = True

            if not found_value:
                continue
            rows.append(row)

    return rows


def discover_model_linguistic_pattern_rows(results_dir: Path) -> List[Dict[str, Any]]:
    """Read heuristic linguistic-pattern summaries from error-analysis artifacts."""
    rows: List[Dict[str, Any]] = []

    seen_models = set()
    for filename in ERROR_ANALYSIS_RESULTS_FILENAMES:
        for experiment_path in sorted(results_dir.glob(f"*/{filename}")):
            model_dir = experiment_path.parent
            if model_dir in seen_models:
                continue
            seen_models.add(model_dir)

            payload = _load_json(experiment_path)
            experiment_payload = payload.get("experiment_3_linguistic_shift") or {}
            if not isinstance(experiment_payload, dict) or experiment_payload.get("status") != "ok":
                continue

            relations = experiment_payload.get("relations") or {}
            row: Dict[str, Any] = {
                "model": model_dir.name,
                "shift_rate_in_successful_games": experiment_payload.get("shift_rate_in_successful_games"),
                "shift_rate_in_unsuccessful_games": experiment_payload.get("shift_rate_in_unsuccessful_games"),
                "post_linguistic_no_recovery_rate_in_successful_games": experiment_payload.get(
                    "post_linguistic_no_recovery_rate_in_successful_games"
                ),
                "post_linguistic_no_recovery_rate_in_unsuccessful_games": experiment_payload.get(
                    "post_linguistic_no_recovery_rate_in_unsuccessful_games"
                ),
                "games_with_post_linguistic_recovery_opportunity": experiment_payload.get(
                    "games_with_post_linguistic_recovery_opportunity"
                ),
                "post_linguistic_no_recovery_fisher_exact": experiment_payload.get(
                    "post_linguistic_no_recovery_fisher_exact"
                )
                or {},
                "failure_fisher_exact": experiment_payload.get("failure_fisher_exact") or {},
                "matched_rule_counts": experiment_payload.get("matched_rule_counts") or {},
                "target_relations_combined_linguistic_share": (
                    (relations.get("target_relations_combined") or {}).get("linguistic_share")
                ),
            }

            rows.append(row)

    return rows


def discover_pooled_linguistic_pattern_payload(results_dir: Path) -> Dict[str, Any]:
    """Read the pooled heuristic linguistic-pattern summary from the results root."""
    summary_path = results_dir / "error_analysis_summary.json"
    if not summary_path.exists():
        return {}

    summary_payload = _load_json(summary_path)
    pooled_payload = summary_payload.get("pooled_experiment_3_linguistic_shift") or {}
    return pooled_payload if isinstance(pooled_payload, dict) else {}


def discover_model_target_property_metrics(results_dir: Path) -> List[Dict[str, Any]]:
    """Compute core evaluation metrics for target-property-specific game subsets."""
    rows: List[Dict[str, Any]] = []

    for game_results_path in sorted(results_dir.glob("*/game_results.json")):
        model_dir = game_results_path.parent
        evaluator = GameEvaluator()
        evaluator.load_results(str(game_results_path))

        by_target_property: Dict[str, Dict[str, Any]] = {}
        for target_property in TARGET_PROPERTIES:
            metrics = evaluator.compute_aggregate_metrics(target_property=target_property)
            if metrics:
                by_target_property[target_property] = metrics

        if by_target_property:
            rows.append(
                {
                    "model": model_dir.name,
                    "by_target_property": by_target_property,
                }
            )

    return rows


def discover_model_target_property_relation_metrics(results_dir: Path) -> List[Dict[str, Any]]:
    """Compute relation-aware metrics for target-property-specific game subsets."""
    rows: List[Dict[str, Any]] = []

    for game_results_path in sorted(results_dir.glob("*/game_results.json")):
        model_dir = game_results_path.parent
        annotation_path = model_dir / "turn_relation_annotations.json"
        if not annotation_path.exists():
            continue

        evaluator = GameEvaluator()
        evaluator.load_results(str(game_results_path))
        evaluator.load_relation_annotations(str(annotation_path))

        by_target_property: Dict[str, Dict[str, Any]] = {}
        for target_property in TARGET_PROPERTIES:
            metrics = evaluator.compute_aggregate_metrics(target_property=target_property)
            if metrics and isinstance(metrics.get("avg_conclusive_falsification_rate"), (int, float)):
                by_target_property[target_property] = {
                    "avg_conclusive_falsification_rate": metrics.get("avg_conclusive_falsification_rate"),
                    "avg_conclusive_falsification_rate_std": metrics.get("avg_conclusive_falsification_rate_std"),
                    "avg_conclusive_falsification_rate_ci95": metrics.get("avg_conclusive_falsification_rate_ci95"),
                    "total_games": metrics.get("total_games"),
                }

        if by_target_property:
            rows.append(
                {
                    "model": model_dir.name,
                    "by_target_property": by_target_property,
                }
            )

    return rows


def _plot_overall_metric(
    rows: List[Dict[str, Any]],
    metric_key: str,
    metric_label: str,
    output_path: Path,
    max_turns: int,
    model_palette_map: Dict[str, Dict[str, Tuple[float, float, float, float]]],
) -> None:
    entries = []
    canonical_order = _canonical_model_order(rows)

    for row in rows:
        model_id = str(row.get("model"))
        value = _metric_value(row["overall"], metric_key)
        if not isinstance(value, (int, float)):
            continue

        display_name = _safe_model_name(model_id)
        entry = {
            "model_id": model_id,
            "display_name": display_name,
            "value": _transform_metric_value(metric_key, float(value)),
            "dispersion": _metric_dispersion_value(row["overall"], metric_key),
            "color": model_palette_map.get(model_id, {}).get("full", (0.3, 0.45, 0.75, 1.0)),
        }

        entry["is_reasoning"] = _is_reasoning_display_name(display_name)
        entries.append(entry)

    ordered = _order_entries_by_canonical_model_order(entries, canonical_order)

    if not ordered:
        return

    models_filtered = [entry["display_name"] for entry in ordered]
    values_filtered = [entry["value"] for entry in ordered]
    dispersions_filtered = [entry["dispersion"] if isinstance(entry["dispersion"], float) else 0.0 for entry in ordered]
    colors_filtered = [entry["color"] for entry in ordered]
    reasoning_flags = [bool(entry["is_reasoning"]) for entry in ordered]

    x_positions = [index * MODEL_SPACING for index in range(len(models_filtered))]

    fig, ax = plt.subplots(figsize=_figure_size_for_overall_metric(metric_key, len(models_filtered)))
    bars = ax.bar(
        x_positions,
        values_filtered,
        color=colors_filtered,
        width=OVERALL_BAR_WIDTH,
        yerr=dispersions_filtered,
        ecolor="#333333",
        capsize=2,
        error_kw={"elinewidth": 0.8, "capthick": 0.8},
    )
    y_label = f"{metric_label} (%)" if metric_key in PERCENTAGE_METRICS else metric_label
    y_label_fontsize = 10 if metric_key == "avg_conclusive_falsification_rate" else _metric_ylabel_fontsize(metric_key)
    ax.set_ylabel(y_label, fontsize=y_label_fontsize)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models_filtered)
    if metric_key == "avg_conclusive_falsification_rate":
        ax.tick_params(axis="x", rotation=25, labelsize=6)
    elif metric_key in OVERALL_BAR_EMPHASIZED_METRICS:
        ax.tick_params(axis="x", rotation=25, labelsize=BAR_PLOT_MODEL_LABEL_SIZE)
    else:
        ax.tick_params(axis="x", rotation=25)
    if x_positions:
        x_margin = max(OVERALL_BAR_WIDTH * 0.8, 0.08)
        ax.set_xlim(min(x_positions) - x_margin, max(x_positions) + x_margin)
        separator = _separator_x_position(x_positions, reasoning_flags)
        if separator is not None:
            ax.axvline(separator, color="#000000", linewidth=1.0, alpha=1.0)
    _apply_metric_axis_limits(ax, metric_key, max_turns, list(values_filtered), dispersions_filtered)
    if metric_key == "avg_conclusive_falsification_rate":
        annotation_size = 8
    elif metric_key in OVERALL_BAR_EMPHASIZED_METRICS:
        annotation_size = BAR_PLOT_VALUE_LABEL_SIZE
    else:
        annotation_size = 6
    _annotate_bar_values(
        ax,
        bars,
        metric_key,
        list(values_filtered),
        error_values=dispersions_filtered,
        font_size=annotation_size,
    )

    fig.savefig(output_path, format="png")
    plt.close(fig)


def _plot_target_property_metric(
    rows: List[Dict[str, Any]],
    metric_key: str,
    metric_label: str,
    output_path: Path,
    max_turns: int,
) -> None:
    """Plot one heatmap per metric across models and target properties."""
    entries = []
    canonical_order = _canonical_model_order(rows)

    for row in rows:
        model_id = str(row.get("model"))
        property_metrics = row.get("by_target_property") or {}
        property_values: Dict[str, float] = {}
        property_dispersions: Dict[str, float] = {}

        for target_property in TARGET_PROPERTIES:
            metrics = property_metrics.get(target_property) or {}
            value = _metric_value(metrics, metric_key)
            if isinstance(value, (int, float)):
                property_values[target_property] = _transform_metric_value(metric_key, float(value))
                dispersion_value = _metric_dispersion_value(metrics, metric_key)
                if isinstance(dispersion_value, float):
                    property_dispersions[target_property] = dispersion_value

        if not property_values:
            continue

        display_name = _safe_model_name(model_id)
        entries.append(
            {
                "model_id": model_id,
                "display_name": display_name,
                "property_values": property_values,
                "property_dispersions": property_dispersions,
                "is_reasoning": _is_reasoning_display_name(display_name),
            }
        )

    ordered = _order_entries_by_canonical_model_order(entries, canonical_order)
    if not ordered:
        return

    models_filtered = [entry["display_name"] for entry in ordered]
    reasoning_flags = [bool(entry["is_reasoning"]) for entry in ordered]
    plotted_values: List[float] = []
    matrix: List[List[float]] = []
    dispersion_matrix: List[List[float]] = []
    for target_property in TARGET_PROPERTIES:
        row_values: List[float] = []
        row_dispersions: List[float] = []
        for entry in ordered:
            value = entry["property_values"].get(target_property)
            if isinstance(value, (int, float)):
                row_values.append(float(value))
                plotted_values.append(float(value))
                dispersion_value = entry["property_dispersions"].get(target_property)
                row_dispersions.append(float(dispersion_value) if isinstance(dispersion_value, (int, float)) else float("nan"))
            else:
                row_values.append(float("nan"))
                row_dispersions.append(float("nan"))
        matrix.append(row_values)
        dispersion_matrix.append(row_dispersions)

    if not plotted_values:
        return

    if metric_key in PERCENTAGE_METRICS or metric_key in FIXED_100_SCORE_METRICS:
        color_min, color_max = 0.0, 100.0
    elif metric_key in FIXED_MAX_TURN_METRICS:
        color_min, color_max = 0.0, float(max_turns)
    elif metric_key in AUTO_UPPER_TURN_METRICS:
        color_min, color_max = 0.0, _compute_auto_upper_bound(plotted_values)
    else:
        color_min, color_max = min(plotted_values), max(plotted_values)

    cell_size = 0.6
    fig_width = max(6.4, 2.4 + len(models_filtered) * cell_size)
    fig_height = max(4.0, 1.6 + len(TARGET_PROPERTIES) * cell_size)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    heatmap = ax.pcolormesh(
        matrix,
        cmap="viridis",
        vmin=color_min,
        vmax=color_max,
        edgecolors="#f5f5f5",
        linewidth=1.1,
        shading="flat",
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.grid(False)
    ax.xaxis.grid(False, which="both")
    ax.yaxis.grid(False, which="both")

    y_label = f"{metric_label} (%)" if metric_key in PERCENTAGE_METRICS else metric_label
    ax.set_ylabel("Target Rule")
    ax.set_xticks([index + 0.5 for index in range(len(models_filtered))])
    ax.set_xticklabels(models_filtered)
    ax.set_yticks([index + 0.5 for index in range(len(TARGET_PROPERTIES))])
    ax.set_yticklabels([TARGET_PROPERTY_DISPLAY_LABELS[target_property] for target_property in TARGET_PROPERTIES])
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", labelsize=9)

    if reasoning_flags and not all(reasoning_flags) and any(reasoning_flags):
        first_reasoning_index = reasoning_flags.index(True)
        if first_reasoning_index > 0:
            ax.axvline(first_reasoning_index, color="#000000", linewidth=1.0, alpha=1.0)

    midpoint = (color_min + color_max) / 2.0 if color_max > color_min else color_max
    annotation_font_size = 9.0 if len(models_filtered) <= 10 else 8.0
    for row_index, row_values in enumerate(matrix):
        for column_index, value in enumerate(row_values):
            if math.isnan(value):
                continue
            dispersion_value = dispersion_matrix[row_index][column_index]
            text_color = "white" if value < midpoint else "black"
            ax.text(
                column_index + 0.5,
                row_index + 0.5,
                _format_value_with_dispersion(
                    metric_key,
                    value,
                    None if math.isnan(dispersion_value) else dispersion_value,
                ),
                ha="center",
                va="center",
                fontsize=annotation_font_size,
                color=text_color,
            )

    colorbar = fig.colorbar(heatmap, ax=ax, fraction=0.035, pad=0.03, shrink=0.92, aspect=30)
    colorbar.set_label(y_label)

    fig.savefig(output_path, format="png")
    plt.close(fig)


def _plot_overall_guess_counts(
    rows: List[Dict[str, Any]],
    output_path: Path,
    model_palette_map: Dict[str, Dict[str, Tuple[float, float, float, float]]],
) -> None:
    """Plot overall average number of guesses per game."""
    entries = []
    canonical_order = _canonical_model_order(rows)

    for row in rows:
        model_id = str(row.get("model"))
        avg_guess_count = row.get("avg_guess_count")
        if not isinstance(avg_guess_count, (int, float)):
            continue

        display_name = _safe_model_name(model_id)
        entries.append(
            {
                "model_id": model_id,
                "display_name": display_name,
                "value": float(avg_guess_count),
                "dispersion": float(row.get("avg_guess_count_ci95", row.get("avg_guess_count_std", 0.0)) or 0.0),
                "color": model_palette_map.get(model_id, {}).get("full", (0.3, 0.45, 0.75, 1.0)),
                "is_reasoning": _is_reasoning_display_name(display_name),
            }
        )

    ordered = _order_entries_by_canonical_model_order(entries, canonical_order)
    if not ordered:
        return

    models_filtered = [entry["display_name"] for entry in ordered]
    values_filtered = [entry["value"] for entry in ordered]
    dispersions_filtered = [entry["dispersion"] for entry in ordered]
    colors_filtered = [entry["color"] for entry in ordered]
    reasoning_flags = [bool(entry["is_reasoning"]) for entry in ordered]

    x_positions = [index * MODEL_SPACING for index in range(len(models_filtered))]

    fig, ax = plt.subplots(figsize=_figure_size_for_overall_metric("avg_guess_count", len(models_filtered)))
    bars = ax.bar(
        x_positions,
        values_filtered,
        color=colors_filtered,
        width=OVERALL_BAR_WIDTH,
        yerr=dispersions_filtered,
        ecolor="#333333",
        capsize=2,
        error_kw={"elinewidth": 0.8, "capthick": 0.8},
    )

    ax.set_ylabel("Avg Guesses per Game")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models_filtered)
    ax.tick_params(axis="x", rotation=25, labelsize=BAR_PLOT_MODEL_LABEL_SIZE)
    if x_positions:
        x_margin = max(OVERALL_BAR_WIDTH * 0.8, 0.08)
        ax.set_xlim(min(x_positions) - x_margin, max(x_positions) + x_margin)
        separator = _separator_x_position(x_positions, reasoning_flags)
        if separator is not None:
            ax.axvline(separator, color="#000000", linewidth=1.0, alpha=1.0)

    ax.set_ylim(0, _compute_auto_upper_bound(values_filtered + [
        value + error for value, error in zip(values_filtered, dispersions_filtered)
    ]))
    _annotate_bar_values(
        ax,
        bars,
        "avg_guess_count",
        values_filtered,
        error_values=dispersions_filtered,
        font_size=BAR_PLOT_VALUE_LABEL_SIZE,
    )

    fig.savefig(output_path, format="png")
    plt.close(fig)


def _annotate_significance_marker(ax: plt.Axes, x_position: float, y_value: float, p_value: Optional[float]) -> None:
    """Mark coefficients with a star when their p-value is below 0.05."""
    if p_value is None or p_value >= 0.05:
        return

    y_min, y_max = ax.get_ylim()
    offset = (y_max - y_min) * 0.03 if y_max > y_min else 0.05
    ax.text(
        x_position,
        y_value + (offset if y_value >= 0 else -offset),
        "*",
        ha="center",
        va="bottom" if y_value >= 0 else "top",
        fontsize=12,
        fontweight="bold",
    )


def _plot_experiment_1_distributional_difference(
    rows: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot experiment 1 mean gaps with statistical-significance annotations."""
    filtered_rows = []
    canonical_order = _canonical_model_order(rows)

    for row in rows:
        model_id = str(row.get("model"))
        display_name = _safe_model_name(model_id)
        if not any(isinstance(row.get(f"{relation_label}_mean_difference"), (int, float)) for relation_label in EXPERIMENT_RELATIONS):
            continue

        filtered_row = {
            "model_id": model_id,
            "model": display_name,
            "is_reasoning": _is_reasoning_display_name(display_name),
        }
        for relation_label in EXPERIMENT_RELATIONS:
            if isinstance(row.get(f"{relation_label}_mean_difference"), (int, float)):
                filtered_row[f"{relation_label}_mean_difference"] = float(row[f"{relation_label}_mean_difference"])
            if isinstance(row.get(f"{relation_label}_mann_whitney_p_value"), (int, float)):
                filtered_row[f"{relation_label}_mann_whitney_p_value"] = float(
                    row[f"{relation_label}_mann_whitney_p_value"]
                )
            if isinstance(row.get(f"{relation_label}_mann_whitney_p_value_holm"), (int, float)):
                filtered_row[f"{relation_label}_mann_whitney_p_value_holm"] = float(
                    row[f"{relation_label}_mann_whitney_p_value_holm"]
                )
        filtered_rows.append(filtered_row)

    ordered_rows = _order_entries_by_canonical_model_order(filtered_rows, canonical_order)
    if not ordered_rows:
        return

    models_filtered = [row["model"] for row in ordered_rows]
    reasoning_flags = [bool(row["is_reasoning"]) for row in ordered_rows]
    x_positions = [index * MODEL_SPACING for index in range(len(models_filtered))]
    width = GROUPED_BAR_WIDTH
    offsets = [-width / 2.0, width / 2.0]

    fig, ax = plt.subplots(figsize=_figure_size_for_overall_metric("success_rate", len(models_filtered)))
    plotted_values: List[float] = []

    for relation_label, offset in zip(EXPERIMENT_RELATIONS, offsets):
        mean_values = [float(row.get(f"{relation_label}_mean_difference", 0.0)) * 100.0 for row in ordered_rows]
        plotted_values.extend(mean_values)
        bars = ax.bar(
            [position + offset for position in x_positions],
            mean_values,
            width=width,
            color=_lighten_color(RELATION_DISTRIBUTION_COLORS[relation_label], mix_with_white=0.18),
            label=RELATION_DISTRIBUTION_DISPLAY_LABELS[relation_label],
        )

        for bar, ordered_row, value in zip(bars, ordered_rows, mean_values):
            p_value = ordered_row.get(f"{relation_label}_mann_whitney_p_value_holm")
            if not isinstance(p_value, float):
                p_value = ordered_row.get(f"{relation_label}_mann_whitney_p_value")
            if isinstance(p_value, float) and p_value < 0.05:
                _annotate_significance_marker(ax, bar.get_x() + bar.get_width() / 2.0, value, p_value)

    if x_positions:
        x_margin = max(width * 1.4, 0.08)
        ax.set_xlim(min(x_positions) - x_margin, max(x_positions) + x_margin)
        separator = _separator_x_position(x_positions, reasoning_flags)
        if separator is not None:
            ax.axvline(separator, color="#000000", linewidth=1.0, alpha=1.0)
    if plotted_values:
        positive_values = [value for value in plotted_values if value >= 0]
        negative_values = [value for value in plotted_values if value < 0]
        upper_bound = _compute_auto_upper_bound(positive_values or [0.0])
        if negative_values:
            lower_bound = max(-10.0, min(negative_values) * 1.15)
        else:
            lower_bound = 0.0
        ax.set_ylim(lower_bound, upper_bound)
    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_ylabel("Mean Gap (pp)", fontsize=10)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models_filtered, rotation=25)
    ax.tick_params(axis="x", labelsize=BAR_PLOT_MODEL_LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.18), fontsize=9)

    fig.savefig(output_path, format="png")
    plt.close(fig)


def _failed_vs_successful_linguistic_shift_p_value(row: Dict[str, Any]) -> Optional[float]:
    """Return a one-sided Fisher exact p-value for failed > successful shift rates."""
    failure_fisher_exact = row.get("failure_fisher_exact") or {}
    table = failure_fisher_exact.get("table") or {}
    if not isinstance(table, dict):
        return None

    failed_shifted = table.get("shifted_failures")
    failed_unshifted = table.get("unshifted_failures")
    successful_shifted = table.get("shifted_successes")
    successful_unshifted = table.get("unshifted_successes")
    counts = [failed_shifted, failed_unshifted, successful_shifted, successful_unshifted]
    if not all(isinstance(count, (int, float)) for count in counts):
        return None

    _, p_value = stats.fisher_exact(
        [
            [int(failed_shifted), int(failed_unshifted)],
            [int(successful_shifted), int(successful_unshifted)],
        ],
        alternative="greater",
    )
    if math.isnan(p_value):
        return None
    return float(p_value)


def _plot_success_failure_linguistic_pattern_share(
    rows: List[Dict[str, Any]],
    output_path: Path,
    model_palette_map: Dict[str, Dict[str, Tuple[float, float, float, float]]],
) -> None:
    """Plot linguistic-pattern rates for successful versus failed games by model."""
    filtered_rows = []
    canonical_order = _canonical_model_order(rows)

    for row in rows:
        model_id = str(row.get("model"))
        successful_share = row.get("shift_rate_in_successful_games")
        failed_games_with_pattern_share = row.get("shift_rate_in_unsuccessful_games")
        color_pair = model_palette_map.get(model_id)
        if color_pair is None:
            continue
        if not isinstance(successful_share, (int, float)) or not isinstance(failed_games_with_pattern_share, (int, float)):
            continue

        display_name = _safe_model_name(model_id)
        filtered_rows.append(
            {
                "model_id": model_id,
                "model": display_name,
                "is_reasoning": _is_reasoning_display_name(display_name),
                "successful_value": float(successful_share) * 100.0,
                "failed_value": float(failed_games_with_pattern_share) * 100.0,
                "successful_color": color_pair["light"],
                "failed_color": color_pair["full"],
                "failed_higher_p_value": _failed_vs_successful_linguistic_shift_p_value(row),
            }
        )

    ordered_rows = _order_entries_by_canonical_model_order(filtered_rows, canonical_order)
    if not ordered_rows:
        return

    models_filtered = [row["model"] for row in ordered_rows]
    reasoning_flags = [bool(row["is_reasoning"]) for row in ordered_rows]
    x = [index * MODEL_SPACING for index in range(len(models_filtered))]
    width = GROUPED_BAR_WIDTH
    successful_values = [row["successful_value"] for row in ordered_rows]
    failed_values = [row["failed_value"] for row in ordered_rows]

    fig, ax = plt.subplots(figsize=_figure_size_for_overall_metric("success_rate", len(models_filtered)))
    successful_bars = ax.bar(
        [index - width / 2.0 for index in x],
        successful_values,
        width=width,
        color=[row["successful_color"] for row in ordered_rows],
        label="Successful",
    )
    failed_bars = ax.bar(
        [index + width / 2.0 for index in x],
        failed_values,
        width=width,
        color=[row["failed_color"] for row in ordered_rows],
        label="Failed",
    )

    ax.set_ylabel("Games With Surface-level Hyp (%)", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(models_filtered, rotation=25)
    ax.tick_params(axis="x", labelsize=BAR_PLOT_MODEL_LABEL_SIZE)
    if x:
        x_margin = max(width * 1.1, 0.08)
        ax.set_xlim(min(x) - x_margin, max(x) + x_margin)
        separator = _separator_x_position(x, reasoning_flags)
        if separator is not None:
            ax.axvline(separator, color="#000000", linewidth=1.0, alpha=1.0)
    ax.set_ylim(0, 100)
    legend_handles = [
        Patch(facecolor="#808080", edgecolor="none", label="Successful"),
        Patch(facecolor="#000000", edgecolor="none", label="Failed"),
    ]
    ax.legend(handles=legend_handles, loc="best")
    _annotate_bar_values(ax, successful_bars, "success_rate", successful_values, font_size=GROUPED_BAR_PLOT_VALUE_LABEL_SIZE)
    _annotate_bar_values(ax, failed_bars, "success_rate", failed_values, font_size=GROUPED_BAR_PLOT_VALUE_LABEL_SIZE)

    for x_position, ordered_row in zip(x, ordered_rows):
        failed_value = ordered_row["failed_value"]
        successful_value = ordered_row["successful_value"]
        p_value = ordered_row["failed_higher_p_value"]
        if failed_value <= successful_value:
            continue
        _annotate_significance_marker(ax, x_position, max(failed_value, successful_value), p_value)

    fig.savefig(output_path, format="png")
    plt.close(fig)


def _plot_pooled_linguistic_pattern_rule_counts(payload: Dict[str, Any], output_path: Path) -> None:
    """Plot the pooled cue frequencies used to classify linguistic/meta-linguistic hypotheses."""
    rule_counts = payload.get("matched_rule_counts") or {}
    if not isinstance(rule_counts, dict) or not rule_counts:
        return

    ordered_items = sorted(
        [(str(rule_name), int(count)) for rule_name, count in rule_counts.items() if isinstance(count, (int, float)) and count > 0],
        key=lambda item: (-item[1], item[0]),
    )
    if not ordered_items:
        return

    labels = [item[0].replace("_", " ").title() for item in ordered_items]
    values = [item[1] for item in ordered_items]
    y_positions = list(range(len(labels)))

    fig_height = max(2.4, len(labels) * 0.28)
    fig, ax = plt.subplots(figsize=(6.4, fig_height))
    bars = ax.barh(y_positions, values, color=(0.27, 0.56, 0.86, 1.0))
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Matched Surface-level Pattern Turns", fontsize=9)
    ax.set_xlim(0, max(values) * 1.15 if values else 1.0)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.02,
            bar.get_y() + bar.get_height() / 2.0,
            f"{value}",
            va="center",
            ha="left",
            fontsize=8,
        )

    fig.savefig(output_path, format="png")
    plt.close(fig)


def _plot_relation_distribution(
    rows: List[Dict[str, Any]],
    output_path: Path,
    title: Optional[str] = None,
) -> None:
    """Plot compact horizontal stacked relation distributions for each model."""
    canonical_order = _canonical_model_order(rows)
    ordered_rows = _order_entries_by_canonical_model_order(
        rows,
        canonical_order,
        model_key="model",
    )

    if not ordered_rows:
        return

    model_names = [_safe_model_name(str(row.get("model"))) for row in ordered_rows]
    reasoning_flags = [_is_reasoning_display_name(name) for name in model_names]
    relation_labels = RELATION_DISTRIBUTION_LABELS
    top_row_relation_labels = [
        "target_included_in_hypothesis",
        "disjoint",
        "partial_overlap",
    ]
    second_row_relation_labels = [
        "hypothesis_included_in_target",
        "identical",
    ]
    label_threshold = 5
    y_positions = list(range(len(model_names)))

    fig_width = 8.4
    fig_height = max(4.2, 1.8 + len(model_names) * 0.42)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    left_offsets = [0.0 for _ in ordered_rows]
    for relation_label in relation_labels:
        relation_color = _lighten_color(RELATION_DISTRIBUTION_COLORS[relation_label], mix_with_white=0.18)
        values = [
            _transform_metric_value(
                "success_rate",
                float((row.get("relation_distribution") or {}).get(relation_label, 0.0)),
            )
            for row in ordered_rows
        ]
        bars = ax.barh(
            y_positions,
            values,
            left=left_offsets,
            height=0.84,
            color=relation_color,
            label=RELATION_DISTRIBUTION_DISPLAY_LABELS[relation_label],
        )

        for bar, value, left in zip(bars, values, left_offsets):
            rounded_label_value = _rounded_percentage_label_value(value)
            if _percentage_label_lower_bound(value) < label_threshold:
                continue
            ax.text(
                left + (value / 2.0),
                bar.get_y() + bar.get_height() / 2.0,
                f"{rounded_label_value}%",
                ha="center",
                va="center",
                fontsize=8.0,
                fontweight="semibold",
                color="#111111",
            )

        left_offsets = [left + value for left, value in zip(left_offsets, values)]

    ax.set_ylabel("")
    ax.set_xlabel("Share (%) of TEST turns", fontsize=13, color="#444444")
    ax.set_xlim(0.0, 100.0)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", labelsize=9)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(model_names, fontsize=8.5)
    ax.invert_yaxis()
    ax.margins(y=0.02)

    if reasoning_flags and not all(reasoning_flags) and any(reasoning_flags):
        first_reasoning_index = reasoning_flags.index(True)
        if first_reasoning_index > 0:
            ax.axhline(first_reasoning_index - 0.5, color="#000000", linewidth=1.0, alpha=1.0)

    top_row_handles = [
        Patch(
            facecolor=_lighten_color(RELATION_DISTRIBUTION_COLORS[relation_label], mix_with_white=0.18),
            edgecolor="none",
            label=RELATION_DISTRIBUTION_DISPLAY_LABELS[relation_label],
        )
        for relation_label in top_row_relation_labels
    ]
    second_row_handles = [
        Patch(
            facecolor=_lighten_color(RELATION_DISTRIBUTION_COLORS[relation_label], mix_with_white=0.18),
            edgecolor="none",
            label=RELATION_DISTRIBUTION_DISPLAY_LABELS[relation_label],
        )
        for relation_label in second_row_relation_labels
    ]

    if title:
        fig.suptitle(title, y=0.98, fontsize=11.5)
        fig.subplots_adjust(top=0.80)
    else:
        fig.subplots_adjust(top=0.86)

    top_legend = ax.legend(handles=top_row_handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.16), fontsize=11)
    ax.add_artist(top_legend)
    second_legend = ax.legend(handles=second_row_handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.09), fontsize=11)

    fig.savefig(output_path, format="png", bbox_extra_artists=(top_legend, second_legend))
    plt.close(fig)


def _plot_positive_testing_bias_vs_success_rate_scatter(
    rows: List[Dict[str, Any]],
    output_path: Path,
    model_palette_map: Dict[str, Dict[str, Tuple[float, float, float, float]]],
) -> Optional[float]:
    """Plot confirmation bias against success rate with Spearman rho."""
    if len(rows) < 2:
        return None

    canonical_order = _canonical_model_order(rows)
    ordered_rows = _order_entries_by_canonical_model_order(rows, canonical_order, model_key="model")
    if len(ordered_rows) < 2:
        return None

    x_values = [100.0 * float(_metric_value(row, "avg_positive_testing_bias")) for row in ordered_rows]
    y_values = [100.0 * float(row["success_rate"]) for row in ordered_rows]
    spearman_rho, spearman_p_value = _spearman_correlation_with_p_value(x_values, y_values)
    x_line = [float(value) for value in range(0, 101)]
    fitted_line = _linear_regression_fit_with_confidence_interval(x_values, y_values, x_line)

    fig, ax = plt.subplots(figsize=(5.6, 4.0))

    if fitted_line is not None:
        line_y, lower_band, upper_band = fitted_line
        ax.plot(
            x_line,
            line_y,
            color=(0.1, 0.1, 0.1, 0.8),
            linewidth=1.2,
            linestyle="--",
        )
        ax.fill_between(
            x_line,
            lower_band,
            upper_band,
            color=(0.1, 0.1, 0.1, 0.12),
        )

    for row, x_value, y_value in zip(ordered_rows, x_values, y_values):
        model_id = str(row.get("model"))
        color = model_palette_map.get(model_id, {}).get("full", (0.3, 0.45, 0.75, 1.0))
        ax.scatter(x_value, y_value, color=color, s=46)

    ax.set_xlabel("Confirmation Bias (%)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="both", labelsize=9)

    legend_handles = [
        Line2D([], [], marker="o", linestyle="None", color="black", markersize=5, label="Models"),
    ]
    if fitted_line is not None:
        legend_handles.append(
            Line2D([], [], color=(0.1, 0.1, 0.1, 0.8), linestyle="--", linewidth=1.2, label="Linear fit")
        )
        legend_handles.append(
            Patch(facecolor=(0.1, 0.1, 0.1, 0.12), edgecolor="none", label="95% CI")
        )

    rho_label = "n/a" if spearman_rho is None else f"{spearman_rho:.3f}"
    p_value_label = _format_p_value_for_legend(spearman_p_value)
    legend = ax.legend(
        handles=legend_handles,
        loc="upper right",
        alignment="left",
        title=f"Spearman rho = {rho_label}\np-value = {p_value_label}",
        title_fontsize=9,
    )
    legend.get_title().set_multialignment("left")

    fig.savefig(output_path, format="png")
    plt.close(fig)
    return spearman_rho


def _plot_positive_testing_bias_vs_conclusive_falsification_rate_scatter(
    rows: List[Dict[str, Any]],
    output_path: Path,
    model_palette_map: Dict[str, Dict[str, Tuple[float, float, float, float]]],
) -> Optional[float]:
    """Plot confirmation bias against conclusive falsification rate with Spearman rho."""
    if len(rows) < 2:
        return None

    canonical_order = _canonical_model_order(rows)
    ordered_rows = _order_entries_by_canonical_model_order(rows, canonical_order, model_key="model")
    if len(ordered_rows) < 2:
        return None

    x_values = [100.0 * float(_metric_value(row, "avg_positive_testing_bias")) for row in ordered_rows]
    y_values = [100.0 * float(_metric_value(row, "avg_conclusive_falsification_rate")) for row in ordered_rows]
    spearman_rho, spearman_p_value = _spearman_correlation_with_p_value(x_values, y_values)
    x_line = [float(value) for value in range(0, 101)]
    fitted_line = _linear_regression_fit_with_confidence_interval(x_values, y_values, x_line)

    fig, ax = plt.subplots(figsize=(PAPER_PAIR_PANEL_WIDTH, PAPER_PAIR_PANEL_HEIGHT))

    if fitted_line is not None:
        line_y, lower_band, upper_band = fitted_line
        ax.plot(
            x_line,
            line_y,
            color=(0.1, 0.1, 0.1, 0.8),
            linewidth=1.2,
            linestyle="--",
        )
        ax.fill_between(
            x_line,
            lower_band,
            upper_band,
            color=(0.1, 0.1, 0.1, 0.12),
        )

    for row, x_value, y_value in zip(ordered_rows, x_values, y_values):
        model_id = str(row.get("model"))
        color = model_palette_map.get(model_id, {}).get("full", (0.3, 0.45, 0.75, 1.0))
        ax.scatter(x_value, y_value, color=color, s=46)

    ax.set_xlabel("Confirmation Bias (%)")
    ax.set_ylabel("Conclusive Falsification Rate (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="both", labelsize=9)

    legend_handles = [
        Line2D([], [], marker="o", linestyle="None", color="black", markersize=5, label="Models"),
    ]
    if fitted_line is not None:
        legend_handles.append(
            Line2D([], [], color=(0.1, 0.1, 0.1, 0.8), linestyle="--", linewidth=1.2, label="Linear fit")
        )
        legend_handles.append(
            Patch(facecolor=(0.1, 0.1, 0.1, 0.12), edgecolor="none", label="95% CI")
        )

    rho_label = "n/a" if spearman_rho is None else f"{spearman_rho:.3f}"
    p_value_label = _format_p_value_for_legend(spearman_p_value)
    legend = ax.legend(
        handles=legend_handles,
        loc="upper right",
        alignment="left",
        title=f"Spearman rho = {rho_label}\np-value = {p_value_label}",
        title_fontsize=9,
    )
    legend.get_title().set_multialignment("left")

    fig.savefig(output_path, format="png")
    plt.close(fig)
    return spearman_rho


def generate_plots(
    results_dir: str = "results",
    output_dir: str = "results/plots",
    max_turns: int = 20,
) -> List[Path]:
    """Generate all model-comparison plots and return written files."""
    configure_plot_style()

    if max_turns <= 0:
        raise ValueError("max_turns must be greater than 0")

    results_path = Path(results_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rows = discover_model_aggregates(results_path)
    target_property_rows = discover_model_target_property_metrics(results_path)
    target_property_relation_rows = discover_model_target_property_relation_metrics(results_path)
    target_property_guess_rows = discover_model_target_property_guess_counts(results_path)
    relation_metric_rows = discover_model_relation_metrics(results_path)
    written: List[Path] = []

    if not rows:
        return written

    model_palette_map = _build_model_palette_map(rows)

    for metric_key, metric_label, _ in METRICS:
        metric_rows = relation_metric_rows if metric_key in RELATION_AWARE_PLOT_METRICS else rows
        if not metric_rows:
            continue

        metric_slug = _output_metric_slug(metric_key)

        overall_png = out_path / f"overall_{metric_slug}.png"
        _plot_overall_metric(metric_rows, metric_key, metric_label, overall_png, max_turns, model_palette_map)
        if overall_png.exists():
            written.append(overall_png)

        target_property_metric_rows = (
            target_property_relation_rows if metric_key in RELATION_AWARE_PLOT_METRICS else target_property_rows
        )
        if target_property_metric_rows:
            by_target_property_png = out_path / f"by_target_property_{metric_slug}.png"
            _plot_target_property_metric(
                target_property_metric_rows,
                metric_key,
                metric_label,
                by_target_property_png,
                max_turns,
            )
            if by_target_property_png.exists():
                written.append(by_target_property_png)

    overall_guesses_rows = discover_model_guess_counts(results_path)
    overall_guesses_png = out_path / "overall_avg_guesses_per_game.png"
    _plot_overall_guess_counts(overall_guesses_rows, overall_guesses_png, model_palette_map)
    if overall_guesses_png.exists():
        written.append(overall_guesses_png)

    if target_property_guess_rows:
        by_target_property_guesses_png = out_path / "by_target_property_avg_guesses_per_game.png"
        _plot_target_property_metric(
            target_property_guess_rows,
            "avg_guess_count",
            "Avg Guesses per Game",
            by_target_property_guesses_png,
            max_turns,
        )
        if by_target_property_guesses_png.exists():
            written.append(by_target_property_guesses_png)

    experiment_1_rows = discover_model_experiment_1_rows(results_path)
    experiment_1_png = out_path / "overall_error_analysis_distributional_difference.png"
    _plot_experiment_1_distributional_difference(experiment_1_rows, experiment_1_png)
    if experiment_1_png.exists():
        written.append(experiment_1_png)

    linguistic_pattern_rows = discover_model_linguistic_pattern_rows(results_path)
    failed_games_with_linguistic_pattern_png = out_path / "overall_error_analysis_linguistic_pattern.png"
    _plot_success_failure_linguistic_pattern_share(
        linguistic_pattern_rows,
        failed_games_with_linguistic_pattern_png,
        model_palette_map,
    )
    if failed_games_with_linguistic_pattern_png.exists():
        written.append(failed_games_with_linguistic_pattern_png)

    pooled_linguistic_pattern_payload = discover_pooled_linguistic_pattern_payload(results_path)
    pooled_linguistic_pattern_rules_png = out_path / "overall_error_analysis_linguistic_pattern_rule_counts.png"
    _plot_pooled_linguistic_pattern_rule_counts(
        pooled_linguistic_pattern_payload,
        pooled_linguistic_pattern_rules_png,
    )
    if pooled_linguistic_pattern_rules_png.exists():
        written.append(pooled_linguistic_pattern_rules_png)

    relation_distribution_rows = discover_model_relation_distributions(results_path)
    relation_distribution_png = out_path / "overall_relation_distribution.png"
    _plot_relation_distribution(relation_distribution_rows, relation_distribution_png)
    if relation_distribution_png.exists():
        written.append(relation_distribution_png)

    successful_relation_distribution_rows = discover_model_relation_distributions_by_outcome(
        results_path,
        success=True,
    )
    successful_relation_distribution_png = out_path / "successful_only_relation_distribution.png"
    _plot_relation_distribution(
        successful_relation_distribution_rows,
        successful_relation_distribution_png,
        title="Successful Games Only",
    )
    if successful_relation_distribution_png.exists():
        written.append(successful_relation_distribution_png)

    unsuccessful_relation_distribution_rows = discover_model_relation_distributions_by_outcome(
        results_path,
        success=False,
    )
    unsuccessful_relation_distribution_png = out_path / "unsuccessful_only_relation_distribution.png"
    _plot_relation_distribution(
        unsuccessful_relation_distribution_rows,
        unsuccessful_relation_distribution_png,
        title="Unsuccessful Games Only",
    )
    if unsuccessful_relation_distribution_png.exists():
        written.append(unsuccessful_relation_distribution_png)

    success_bias_rows = discover_model_success_bias_correlation_rows(results_path)
    success_bias_png = out_path / "overall_confirmation_bias_vs_success_rate.png"
    _plot_positive_testing_bias_vs_success_rate_scatter(success_bias_rows, success_bias_png, model_palette_map)
    if success_bias_png.exists():
        written.append(success_bias_png)

    falsification_bias_rows = discover_model_bias_falsification_correlation_rows(results_path)
    falsification_bias_png = out_path / "overall_confirmation_bias_vs_conclusive_falsification_rate.png"
    _plot_positive_testing_bias_vs_conclusive_falsification_rate_scatter(
        falsification_bias_rows,
        falsification_bias_png,
        model_palette_map,
    )
    if falsification_bias_png.exists():
        written.append(falsification_bias_png)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PNG plots from model evaluation outputs.")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing model subfolders with aggregate_results.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/plots",
        help="Directory where PNG plots are written",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Y-axis maximum for turn-based metrics (default: 20)",
    )
    args = parser.parse_args()

    written = generate_plots(args.results_dir, args.output_dir, args.max_turns)
    if not written:
        print("No model aggregates found. Run evaluation first.")
        return

    print(f"Generated {len(written)} plot(s):")
    for path in written:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
