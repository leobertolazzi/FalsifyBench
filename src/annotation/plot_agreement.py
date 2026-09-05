"""Generate plots for annotation agreement metrics.

Usage examples:
    python -m src.annotation.plot_agreement
    python -m src.annotation.plot_agreement --input-dir results/annotation_pairs_oracle
    python -m src.annotation.plot_agreement --summary-path results/annotation_pairs_oracle/agreement_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.annotation.agreement import compute_agreement_summary
from src.analysis.plots import (
    BAR_PLOT_MODEL_LABEL_SIZE,
    GROUPED_BAR_WIDTH,
    MODEL_SPACING,
    _build_model_palette_map,
    _canonical_model_order,
    _figure_size_for_overall_metric,
    _is_reasoning_display_name,
    _order_entries_by_canonical_model_order,
    _safe_model_name,
    _separator_x_position,
    configure_plot_style,
)


DEFAULT_INPUT_DIR = Path("results/annotation_pairs_oracle")
DEFAULT_OUTPUT_DIR = Path("results/annotation_plots")
PAIR_SPECS: List[Tuple[str, str]] = [
    ("guess_target_equivalence", "GUESS"),
    ("test_turn_target_judgment", "TEST"),
]
METRIC_SPECS: List[Tuple[str, str, str]] = [
    ("percent_agreement", "agreement", "Accuracy"),
    ("cohen_kappa", "kappa", "Cohen's Kappa"),
]
def _load_summary(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _metric_plot_value(metric_key: str, value: Any) -> Optional[float]:
    if not isinstance(value, (int, float)):
        return None
    return float(value) * 100.0 if metric_key == "percent_agreement" else float(value)


def _figure_size_for_annotation_bars(item_count: int, bar_half_width: float) -> Tuple[float, float]:
    if item_count <= 0:
        return (4.8, 3.2)

    span_units = (item_count - 1) * MODEL_SPACING + (2.0 * bar_half_width)
    horizontal_margin_units = max(bar_half_width, 0.08) * 2.0
    total_units = span_units + horizontal_margin_units
    width_inches = max(4.8, min(10.5, total_units * 2.8))
    return (width_inches, 3.2)


def _figure_size_for_grouped_annotation_bars(item_count: int, bar_width: float) -> Tuple[float, float]:
    return _figure_size_for_overall_metric("success_rate", item_count)


def _apply_annotation_axis_limits(ax: plt.Axes, metric_key: str, plotted_values: Sequence[float]) -> None:
    if metric_key == "percent_agreement":
        ax.set_ylim(0.0, 100.0)
        return

    if not plotted_values:
        ax.set_ylim(0.0, 1.0)
        return

    value_min = min(plotted_values)
    value_max = max(plotted_values)
    lower = min(0.0, value_min - 0.05)
    upper = max(1.0, value_max + 0.05)
    if lower == upper:
        upper = lower + 1.0
    ax.set_ylim(lower, upper)


def _format_annotation_value(metric_key: str, value: float) -> str:
    return f"{value:.1f}%" if metric_key == "percent_agreement" else f"{value:.2f}"


def _annotation_metric_axis_label(metric_key: str, metric_label: str) -> str:
    return f"{metric_label} (%)" if metric_key == "percent_agreement" else metric_label


def _annotate_bar_values(ax: plt.Axes, metric_key: str, bars: Any, values: Sequence[float], font_size: int = 6) -> None:
    if not values:
        return

    y_min, y_max = ax.get_ylim()
    offset = (y_max - y_min) * 0.015 if y_max > y_min else 0.05
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + offset,
            _format_annotation_value(metric_key, value),
            ha="center",
            va="bottom",
            fontsize=font_size,
        )


def _plot_model_metric_grouped_by_pair(
    summary: Dict[str, Any],
    metric_key: str,
    metric_label: str,
    output_path: Path,
) -> bool:
    by_model = (summary.get("overall") or {}).get("by_model") or {}
    entries: List[Dict[str, Any]] = []
    for model_name, model_summary in by_model.items():
        pair_values: Dict[str, float] = {}
        for pair_name, _pair_label in PAIR_SPECS:
            pair_summary = (model_summary.get("pair_name") or {}).get(pair_name) or {}
            plot_value = _metric_plot_value(metric_key, pair_summary.get(metric_key))
            if plot_value is not None:
                pair_values[pair_name] = plot_value

        if not pair_values:
            continue

        display_name = _safe_model_name(str(model_name))
        entries.append(
            {
                "model_id": str(model_name),
                "display_name": display_name,
                "pair_values": pair_values,
                "is_reasoning": _is_reasoning_display_name(display_name),
            }
        )

    if not entries:
        return False

    model_palette_map = _build_model_palette_map([{"model": entry["model_id"]} for entry in entries])
    for entry in entries:
        palette = model_palette_map.get(entry["model_id"], {})
        entry["full_color"] = palette.get("full", (0.3, 0.45, 0.75, 1.0))
        entry["light_color"] = palette.get("light", (0.6, 0.72, 0.88, 1.0))

    ordered = _order_entries_by_canonical_model_order(entries, _canonical_model_order([{"model": entry["model_id"]} for entry in entries]))
    if not ordered:
        return False

    model_names = [entry["display_name"] for entry in ordered]
    reasoning_flags = [bool(entry["is_reasoning"]) for entry in ordered]
    x_positions = [index * MODEL_SPACING for index in range(len(model_names))]
    width = GROUPED_BAR_WIDTH

    fig, ax = plt.subplots(figsize=_figure_size_for_grouped_annotation_bars(len(model_names), width))
    plotted_values: List[float] = []
    legend_handles: List[Patch] = []
    value_font_size = 4 if metric_key == "percent_agreement" else 5
    for pair_index, (pair_name, pair_label) in enumerate(PAIR_SPECS):
        values: List[float] = []
        pair_positions: List[float] = []
        pair_colors: List[Tuple[float, float, float, float]] = []
        for x_position, entry in zip(x_positions, ordered):
            value = entry["pair_values"].get(pair_name)
            if value is None:
                continue
            pair_positions.append(x_position + (pair_index - (len(PAIR_SPECS) - 1) / 2.0) * width)
            values.append(float(value))
            pair_colors.append(entry["light_color"] if pair_index == 0 else entry["full_color"])

        if not values:
            continue

        plotted_values.extend(values)
        bars = ax.bar(pair_positions, values, color=pair_colors, width=width, label=pair_label)
        _annotate_bar_values(ax, metric_key, bars, values, font_size=value_font_size)
        legend_handles.append(
            Patch(
                facecolor=(0.75, 0.75, 0.75, 1.0) if pair_index == 0 else (0.35, 0.35, 0.35, 1.0),
                edgecolor="none",
                label=pair_label,
            )
        )

    if not plotted_values:
        plt.close(fig)
        return False

    ax.set_ylabel(_annotation_metric_axis_label(metric_key, metric_label), fontsize=10)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_names, rotation=25)
    ax.tick_params(axis="x", labelsize=BAR_PLOT_MODEL_LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=9)
    if x_positions:
        x_margin = max(width * 1.1, 0.08)
        ax.set_xlim(min(x_positions) - x_margin, max(x_positions) + x_margin)
        separator = _separator_x_position(x_positions, reasoning_flags)
        if separator is not None:
            ax.axvline(separator, color="#000000", linewidth=1.0, alpha=1.0)
    _apply_annotation_axis_limits(ax, metric_key, plotted_values)
    if metric_key != "cohen_kappa":
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.25),
            ncol=len(legend_handles),
            frameon=True,
            facecolor="white",
            edgecolor="#cccccc",
        )

    fig.savefig(output_path, format="png")
    plt.close(fig)
    return True


def generate_annotation_plots(
    input_dir: str = str(DEFAULT_INPUT_DIR),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    summary_path: str = "",
) -> List[Path]:
    configure_plot_style()

    if summary_path:
        summary = _load_summary(Path(summary_path))
    else:
        summary = compute_agreement_summary(Path(input_dir))

    pair_summaries = summary.get("pair_summaries") or []
    if not pair_summaries:
        return []

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    for metric_key, metric_slug, metric_label in METRIC_SPECS:
        output_path = out_path / f"action_{metric_slug}_by_model.png"
        if _plot_model_metric_grouped_by_pair(summary, metric_key, metric_label, output_path):
            written.append(output_path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PNG plots from annotation agreement outputs.")
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing annotation CSV pairs (default: results/annotation_pairs_oracle)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where annotation PNG plots are written (default: results/annotation_plots)",
    )
    parser.add_argument(
        "--summary-path",
        default="",
        help="Optional precomputed agreement summary JSON to plot instead of recomputing",
    )
    args = parser.parse_args()

    written = generate_annotation_plots(args.input_dir, args.output_dir, args.summary_path)
    if not written:
        print("No annotation agreement data found. Compute annotation agreement first.")
        return

    print(f"Generated {len(written)} plot(s):")
    for path in written:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
