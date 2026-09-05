"""Run a standalone game-level mixed-effects logistic analysis.

This script treats each game as one observation, joins in sampled oracle-error
signals from the human annotation CSVs, and fits a mixed-effects logistic model
with a random intercept for model identity.

Usage examples:
    python -m src.analysis.mixed_effects
    python -m src.analysis.mixed_effects --results-dir results --annotation-dir results/annotation_pairs_oracle
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

from src.benchmark.evaluation import GameEvaluator
from src.analysis.plots import (
    PAPER_PAIR_PANEL_HEIGHT,
    PAPER_PAIR_PANEL_WIDTH,
    RELATION_COLORS,
    configure_plot_style,
)


DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_ANNOTATION_DIR = Path("results/annotation_pairs_oracle")
DEFAULT_OUTPUT_DIR = Path("results/mixed_effect_analysis")
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "game_level_mixed_effects_analysis.json"
DEFAULT_ROWS_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "game_level_mixed_effects_rows.csv"
DEFAULT_PLOT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "game_level_mixed_effects_plot.png"
ORACLE_ANNOTATION_TASKS: Tuple[str, ...] = (
    "test_turn_target_judgment",
    "guess_target_equivalence",
)
MISSING_JUDGMENTS = {"", "-", "na", "n/a", "none", "null"}


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, list) else []


def _normalize_judgment(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in MISSING_JUDGMENTS:
        return ""
    return normalized


def _find_annotation_pair_paths(annotation_dir: Path, pair_name: str) -> Tuple[Optional[Path], Optional[Path]]:
    llm_path = annotation_dir / f"{pair_name}_llm.csv"
    human_path = annotation_dir / f"{pair_name}_human.csv"
    if llm_path.exists() and human_path.exists():
        return llm_path, human_path
    return None, None


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _row_key(row: Dict[str, str], fieldnames: Sequence[str]) -> Tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in fieldnames)


def _build_row_index(rows: Iterable[Dict[str, str]], key_fields: Sequence[str]) -> Dict[Tuple[str, ...], Dict[str, str]]:
    index: Dict[Tuple[str, ...], Dict[str, str]] = {}
    for row in rows:
        key = _row_key(row, key_fields)
        if key in index:
            raise ValueError(f"Duplicate annotation row key encountered: {key}")
        index[key] = row
    return index


def _task_error_column(task_name: str) -> str:
    if task_name == "test_turn_target_judgment":
        return "sampled_test_oracle_error"
    if task_name == "guess_target_equivalence":
        return "sampled_guess_oracle_error"
    safe_task_name = task_name.replace("-", "_")
    return f"{safe_task_name}_oracle_error"


def _canonical_game_key(model_name: Any, game_id: Any) -> Tuple[str, str]:
    return str(model_name), str(game_id)


def load_sampled_oracle_error_flags(annotation_dir: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    game_flags: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for task_name in ORACLE_ANNOTATION_TASKS:
        llm_path, human_path = _find_annotation_pair_paths(annotation_dir, task_name)
        if llm_path is None or human_path is None:
            continue

        llm_rows = _read_csv_rows(llm_path)
        human_rows = _read_csv_rows(human_path)
        if not llm_rows or not human_rows:
            continue

        fieldnames = list(llm_rows[0].keys())
        key_fields = [field for field in fieldnames if field != "judgment"]
        llm_index = _build_row_index(llm_rows, key_fields)
        human_index = _build_row_index(human_rows, key_fields)

        for shared_key in sorted(set(llm_index) & set(human_index)):
            llm_row = llm_index[shared_key]
            human_row = human_index[shared_key]
            llm_judgment = _normalize_judgment(llm_row.get("judgment", ""))
            human_judgment = _normalize_judgment(human_row.get("judgment", ""))
            if not llm_judgment or not human_judgment:
                continue

            game_key = _canonical_game_key(llm_row.get("model"), llm_row.get("game_id"))
            entry = game_flags.setdefault(
                game_key,
                {
                    "oracle_annotation_count": 0,
                    "oracle_error_count": 0,
                    "sampled_test_oracle_error": None,
                    "sampled_guess_oracle_error": None,
                },
            )
            is_error = llm_judgment != human_judgment
            entry["oracle_annotation_count"] += 1
            entry["oracle_error_count"] += int(is_error)
            entry[_task_error_column(task_name)] = is_error

    for entry in game_flags.values():
        annotation_count = int(entry.get("oracle_annotation_count", 0))
        if annotation_count > 0:
            entry["oracle_any_error"] = bool(entry.get("oracle_error_count", 0))
            entry["oracle_error_rate"] = float(entry.get("oracle_error_count", 0)) / annotation_count
        else:
            entry["oracle_any_error"] = None
            entry["oracle_error_rate"] = None

    return game_flags


def build_game_level_rows(results_dir: Path, annotation_dir: Path) -> List[Dict[str, Any]]:
    evaluator = GameEvaluator()
    oracle_error_flags = load_sampled_oracle_error_flags(annotation_dir)
    rows: List[Dict[str, Any]] = []

    for game_results_path in sorted(results_dir.glob("*/game_results.json")):
        model_name = game_results_path.parent.name
        for game_result in _load_json_list(game_results_path):
            metrics = evaluator.compute_game_metrics(game_result)
            game_key = _canonical_game_key(model_name, metrics.get("game_id"))
            annotation_entry = oracle_error_flags.get(game_key, {})
            rows.append(
                {
                    "model": model_name,
                    "game_id": metrics.get("game_id"),
                    "success": bool(metrics.get("success", False)),
                    "distance_group": metrics.get("distance_group"),
                    "target_property": metrics.get("target_property"),
                    "distance": metrics.get("distance"),
                    "positive_testing_bias": metrics.get("positive_testing_bias"),
                    "positive_testing_queries": metrics.get("positive_testing_queries"),
                    "negative_testing_queries": metrics.get("negative_testing_queries"),
                    "total_test_queries": metrics.get("total_test_queries"),
                    "oracle_annotation_count": annotation_entry.get("oracle_annotation_count", 0),
                    "oracle_error_count": annotation_entry.get("oracle_error_count", 0),
                    "oracle_any_error": annotation_entry.get("oracle_any_error"),
                    "oracle_error_rate": annotation_entry.get("oracle_error_rate"),
                    "sampled_test_oracle_error": annotation_entry.get("sampled_test_oracle_error"),
                    "sampled_guess_oracle_error": annotation_entry.get("sampled_guess_oracle_error"),
                }
            )

    return rows


def _clean_analysis_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned_rows: List[Dict[str, Any]] = []
    for row in rows:
        positive_testing_bias = row.get("positive_testing_bias")
        oracle_any_error = row.get("oracle_any_error")
        if not isinstance(positive_testing_bias, (int, float)):
            continue
        if oracle_any_error not in {True, False}:
            continue
        cleaned_rows.append(
            {
                **row,
                "success": int(bool(row.get("success", False))),
                "positive_testing_bias": float(positive_testing_bias),
                "oracle_any_error": int(bool(oracle_any_error)),
            }
        )
    return cleaned_rows


def fit_mixed_effects_logit(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    cleaned_rows = _clean_analysis_rows(rows)
    model_ids = sorted({str(row.get("model")) for row in cleaned_rows})
    success_values = [int(row["success"]) for row in cleaned_rows]
    formula = "success ~ positive_testing_bias + oracle_any_error"

    result: Dict[str, Any] = {
        "status": "insufficient_data",
        "formula": formula,
        "complete_case_games": len(cleaned_rows),
        "models_in_fit": model_ids,
    }
    if len(cleaned_rows) < 8 or len(model_ids) < 2:
        return result
    if len(set(success_values)) < 2:
        return result

    data_frame = pd.DataFrame(cleaned_rows)

    try:
        fitted_model = BinomialBayesMixedGLM.from_formula(
            formula,
            {"model": "0 + C(model)"},
            data_frame,
        ).fit_vb()
    except Exception as exc:
        result["status"] = "fit_failed"
        result["error"] = str(exc)
        return result

    fixed_effect_names = list(fitted_model.model.exog_names)
    fixed_effects = {
        name: {
            "posterior_mean": float(mean),
            "posterior_sd": float(sd),
            "odds_ratio": float(math.exp(mean)),
        }
        for name, mean, sd in zip(fixed_effect_names, fitted_model.fe_mean, fitted_model.fe_sd)
    }

    variance_component_names = list(getattr(fitted_model.model, "vcp_names", []))
    random_effects = {
        name: {
            "log_sd_posterior_mean": float(mean),
            "log_sd_posterior_sd": float(sd),
            "implied_sd": float(math.exp(mean)),
        }
        for name, mean, sd in zip(variance_component_names, fitted_model.vcp_mean, fitted_model.vcp_sd)
    }

    result.update(
        {
            "status": "ok",
            "fixed_effects": fixed_effects,
            "random_effects": random_effects,
            "fit_method": "BinomialBayesMixedGLM.fit_vb",
        }
    )
    return result


def _logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _posterior_interval(mean: float, sd: float, z_value: float = 1.96) -> Tuple[float, float]:
    return (mean - z_value * sd, mean + z_value * sd)


def write_interpretation_plot(
    rows: Sequence[Dict[str, Any]],
    fit_summary: Dict[str, Any],
    output_path: Path,
) -> Optional[Path]:
    del rows
    if fit_summary.get("status") != "ok":
        return None

    configure_plot_style()

    fixed_effects = fit_summary.get("fixed_effects") or {}
    intercept_info = fixed_effects.get("Intercept") or {}
    if not isinstance(intercept_info.get("posterior_mean"), (int, float)):
        return None

    intercept = float(intercept_info["posterior_mean"])
    bias_coef = float((fixed_effects.get("positive_testing_bias") or {}).get("posterior_mean", 0.0))
    oracle_coef = float((fixed_effects.get("oracle_any_error") or {}).get("posterior_mean", 0.0))

    x_values = [index / 100.0 for index in range(101)]
    x_percent_values = [100.0 * value for value in x_values]
    scenarios = [
        (0, "Oracle error (No)", RELATION_COLORS["hypothesis_included_in_target"]),
        (1, "Oracle error (Yes)", RELATION_COLORS["identical"]),
    ]

    fig, (ax_coef, ax_curve) = plt.subplots(
        1,
        2,
        figsize=(2.0 * PAPER_PAIR_PANEL_WIDTH, PAPER_PAIR_PANEL_HEIGHT),
        gridspec_kw={"width_ratios": [1.0, 1.35]},
    )

    coefficient_specs = [
        ("positive_testing_bias", "Confirmation bias"),
        ("oracle_any_error", "Oracle error"),
    ]
    y_positions: List[int] = []
    y_labels: List[str] = []
    annotation_x_values: List[float] = []
    for coefficient_key, coefficient_label in coefficient_specs:
        coefficient_info = fixed_effects.get(coefficient_key) or {}
        mean = coefficient_info.get("posterior_mean")
        sd = coefficient_info.get("posterior_sd")
        if not isinstance(mean, (int, float)) or not isinstance(sd, (int, float)):
            continue

        mean_value = float(mean)
        sd_value = float(sd)
        lower, upper = _posterior_interval(mean_value, sd_value, z_value=2.0)
        y_pos = len(y_positions)
        y_positions.append(y_pos)
        y_labels.append(coefficient_label)
        annotation_x_values.extend([lower, upper, mean_value])
        bar_left = min(0.0, mean_value)
        bar_width = abs(mean_value)
        if bar_width > 0.0:
            ax_coef.barh(
                y_pos,
                bar_width,
                left=bar_left,
                height=0.34,
                color=(
                    RELATION_COLORS["hypothesis_included_in_target"]
                    if coefficient_key == "positive_testing_bias"
                    else RELATION_COLORS["identical"]
                ),
                edgecolor="none",
                alpha=0.88,
                zorder=1,
            )
        ax_coef.hlines(
            y=y_pos,
            xmin=lower,
            xmax=upper,
            color=(0.1, 0.1, 0.1, 0.55),
            linewidth=1.2,
            zorder=3,
        )
        ax_coef.vlines(
            x=[lower, upper],
            ymin=y_pos - 0.06,
            ymax=y_pos + 0.06,
            color=(0.1, 0.1, 0.1, 0.55),
            linewidth=1.0,
            zorder=3,
        )
        annotation_x = mean_value
        if coefficient_key == "oracle_any_error":
            annotation_x -= 0.24
        annotation_x_values.append(annotation_x)
        ax_coef.text(
            annotation_x,
            y_pos + 0.22,
            f"{mean_value:.2f}\n± {2.0 * sd_value:.2f}",
            fontsize=7,
            ha="center",
            va="bottom",
            color="#333333",
        )

    ax_coef.axvline(0.0, color=(0.1, 0.1, 0.1, 0.8), linewidth=1.2, linestyle="--")
    if y_positions:
        ax_coef.set_yticks(y_positions)
        ax_coef.set_yticklabels(y_labels)
        ax_coef.set_ylim(-0.6, len(y_positions) - 0.4)
        x_min = min(annotation_x_values)
        x_max = max(annotation_x_values)
        x_padding = max(0.15, 0.12 * max(abs(x_min), abs(x_max), 1.0))
        ax_coef.set_xlim(x_min - x_padding, x_max + x_padding)
    ax_coef.set_xlabel("Posterior mean coefficient (log-odds)")
    #ax_coef.set_title("Fixed Effects")
    ax_coef.tick_params(axis="both", labelsize=9)

    for oracle_error, label, color in scenarios:
        y_values = [
            _logistic(intercept + bias_coef * bias + oracle_coef * oracle_error)
            for bias in x_values
        ]
        y_percent_values = [100.0 * value for value in y_values]
        ax_curve.plot(x_percent_values, y_percent_values, color=color, linewidth=2.4, label=label)

    ax_curve.set_xlabel("Confirmation Bias (%)")
    ax_curve.set_ylabel("Predicted Success Rate (%)")
    ax_curve.set_ylim(0.0, 100.0)
    ax_curve.set_xlim(0.0, 100.0)
    ax_curve.tick_params(axis="both", labelsize=9)
    #ax_curve.set_title("Predicted Probability")
    ax_curve.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="png")
    plt.close(fig)
    return output_path


def write_rows_csv(rows: Sequence[Dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "game_id",
        "success",
        "distance_group",
        "target_property",
        "distance",
        "positive_testing_bias",
        "positive_testing_queries",
        "negative_testing_queries",
        "total_test_queries",
        "oracle_annotation_count",
        "oracle_error_count",
        "oracle_any_error",
        "oracle_error_rate",
        "sampled_test_oracle_error",
        "sampled_guess_oracle_error",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    return output_path


def run_game_level_mixed_effects_analysis(
    results_dir: Path,
    annotation_dir: Path,
    output_path: Path,
    rows_output_path: Path,
    plot_output_path: Path,
) -> Dict[str, Any]:
    rows = build_game_level_rows(results_dir, annotation_dir)
    rows_output_path = write_rows_csv(rows, rows_output_path)
    fit_summary = fit_mixed_effects_logit(rows)
    written_plot_path = write_interpretation_plot(rows, fit_summary, plot_output_path)

    annotated_rows = [row for row in rows if row.get("oracle_any_error") in {True, False}]
    payload = {
        "analysis": "game_level_mixed_effects_logit",
        "outcome": "success",
        "predictors": ["positive_testing_bias", "oracle_any_error"],
        "random_effect": "model",
        "results_dir": str(results_dir),
        "annotation_dir": str(annotation_dir),
        "rows_output_path": str(rows_output_path),
        "plot_output_path": str(written_plot_path) if written_plot_path is not None else None,
        "total_game_rows": len(rows),
        "games_with_oracle_annotations": len(annotated_rows),
        "models_total": len({str(row.get("model")) for row in rows}),
        "models_with_oracle_annotations": len({str(row.get("model")) for row in annotated_rows}),
        "oracle_annotation_note": "oracle_any_error is derived from sampled human-vs-LLM disagreements in guess_target_equivalence and test_turn_target_judgment annotation pairs.",
        "fit": fit_summary,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standalone game-level mixed-effects logistic analysis.")
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory containing per-model game_results.json files (default: results)",
    )
    parser.add_argument(
        "--annotation-dir",
        default=str(DEFAULT_ANNOTATION_DIR),
        help="Directory containing annotation pair CSVs (default: results/annotation_pairs_oracle)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="JSON output path for the model summary (default: results/game_level_mixed_effects_analysis.json)",
    )
    parser.add_argument(
        "--rows-output",
        default=str(DEFAULT_ROWS_OUTPUT_PATH),
        help="CSV output path for per-game rows (default: results/game_level_mixed_effects_rows.csv)",
    )
    parser.add_argument(
        "--plot-output",
        default=str(DEFAULT_PLOT_OUTPUT_PATH),
        help="PNG output path for an interpretation plot (default: results/mixed_effect_analysis/game_level_mixed_effects_plot.png)",
    )
    args = parser.parse_args()

    payload = run_game_level_mixed_effects_analysis(
        results_dir=Path(args.results_dir),
        annotation_dir=Path(args.annotation_dir),
        output_path=Path(args.output),
        rows_output_path=Path(args.rows_output),
        plot_output_path=Path(args.plot_output),
    )
    print(f"Wrote analysis summary: {args.output}")
    print(f"Wrote game-level rows: {args.rows_output}")
    if payload.get("plot_output_path"):
        print(f"Wrote interpretation plot: {payload['plot_output_path']}")
    print(f"Fit status: {payload['fit']['status']}")


if __name__ == "__main__":
    main()
