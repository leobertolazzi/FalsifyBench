"""
Evaluation & Metrics Logger

Analyzes game results and computes metrics including Positive Testing Bias.
"""

import json
import argparse
import math
from typing import List, Dict, Any, Optional
from pathlib import Path
import csv

from scipy import stats


RELATION_LABELS = (
    "identical",
    "disjoint",
    "partial_overlap",
    "hypothesis_included_in_target",
    "target_included_in_hypothesis",
)


CONFIRMATION_BIAS_RELATION = "hypothesis_included_in_target"


def count_guess_actions(game_result: Dict[str, Any]) -> int:
    """Count guess actions taken in a single game."""
    guess_count = 0
    for turn in game_result.get("turns", []):
        action = turn.get("action", {})
        if action.get("action") == "guess":
            guess_count += 1

    return guess_count


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _annotation_turn_key(game_id: Any, turn_number: Any) -> tuple[Any, Any]:
    return (game_id, turn_number)


def _extract_bool(value: Any) -> Optional[bool]:
    return value if isinstance(value, bool) else None


def _normalize_target_property_name(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None

    normalized = value.strip().lower().replace("_", " ")
    normalized = " ".join(normalized.split())
    return normalized or None


def _population_standard_deviation(values: List[float]) -> Optional[float]:
    if not values:
        return None

    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _confidence_interval_95_half_width(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return None

    if all(value == values[0] for value in values):
        return 0.0

    sample_standard_deviation = float(stats.tstd(values))
    if math.isnan(sample_standard_deviation):
        return None

    standard_error = sample_standard_deviation / math.sqrt(len(values))
    critical_t = float(stats.t.ppf(0.975, len(values) - 1))
    return critical_t * standard_error


def _merge_missing_metrics(base_metrics: Dict[str, Any], computed_metrics: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base_metrics)
    for key, value in computed_metrics.items():
        if key not in merged:
            merged[key] = value
    return merged


def _is_conclusive_falsification(
    relation: Any,
    hypothesis_conforms: Optional[bool],
    target_conforms: Optional[bool],
) -> Optional[bool]:
    relation = relation.strip() if isinstance(relation, str) else None
    if relation == "":
        relation = None
    if relation not in RELATION_LABELS:
        return None
    if hypothesis_conforms is None or target_conforms is None:
        return None

    if relation == "identical":
        return False
    if relation == "hypothesis_included_in_target":
        return (not hypothesis_conforms) and target_conforms
    if relation == "target_included_in_hypothesis":
        return hypothesis_conforms and (not target_conforms)
    if relation in {"partial_overlap", "disjoint"}:
        return hypothesis_conforms != target_conforms

    return None


def _load_relation_turn_records(model_dir: Path) -> List[Dict[str, Any]]:
    """Join TEST turns with saved relation annotations for one model directory."""
    game_results_path = model_dir / "game_results.json"
    annotation_path = model_dir / "turn_relation_annotations.json"

    if not game_results_path.exists() or not annotation_path.exists():
        return []

    game_results = _load_json_list(game_results_path)
    annotation_payload = _load_json(annotation_path)
    raw_annotations = annotation_payload.get("annotations") or []
    if not isinstance(raw_annotations, list):
        return []

    annotation_map: Dict[tuple[Any, Any], Dict[str, Any]] = {}
    for annotation in raw_annotations:
        if not isinstance(annotation, dict):
            continue
        if annotation.get("status") != "ok":
            continue

        relation_value = annotation.get("relation")
        relation = relation_value.strip() if isinstance(relation_value, str) else None
        if relation == "":
            relation = None
        if relation not in RELATION_LABELS:
            continue

        normalized_annotation = dict(annotation)
        normalized_annotation["relation"] = relation
        annotation_map[_annotation_turn_key(annotation.get("game_id"), annotation.get("turn_number"))] = normalized_annotation

    turn_records: List[Dict[str, Any]] = []
    for game_result in game_results:
        if not isinstance(game_result, dict):
            continue

        game_id = game_result.get("game_id")
        distance_group = game_result.get("distance_group")
        success = bool(game_result.get("success", False))

        for turn in game_result.get("turns", []):
            if not isinstance(turn, dict):
                continue

            action = turn.get("action") or {}
            if action.get("action") != "test":
                continue

            turn_number = turn.get("turn_number")
            annotation = annotation_map.get(_annotation_turn_key(game_id, turn_number))
            if annotation is None:
                continue

            intention = turn.get("inferred_intention")
            normalized_intention = intention.lower() if isinstance(intention, str) else None
            oracle_response = turn.get("oracle_response") or {}
            hypothesis_oracle_response = turn.get("hypothesis_oracle_response") or {}
            target_conforms = _extract_bool(oracle_response.get("conforms"))
            hypothesis_conforms = _extract_bool(hypothesis_oracle_response.get("conforms"))
            conclusive_falsification = _is_conclusive_falsification(
                annotation.get("relation"),
                hypothesis_conforms,
                target_conforms,
            )

            turn_records.append(
                {
                    "game_id": game_id,
                    "turn_number": turn_number,
                    "distance_group": distance_group,
                    "success": success,
                    "relation": annotation.get("relation"),
                    "inferred_intention": normalized_intention,
                    "target_conforms": target_conforms,
                    "hypothesis_conforms": hypothesis_conforms,
                    "conclusive_falsification": conclusive_falsification,
                }
            )

    return turn_records


def _filtered_turn_records(
    turn_records: List[Dict[str, Any]],
    *,
    distance_group: Optional[str] = None,
    success: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []

    for record in turn_records:
        if distance_group is not None and record.get("distance_group") != distance_group:
            continue
        if success is not None and bool(record.get("success")) is not success:
            continue
        filtered.append(record)

    return filtered


def _per_game_relation_filtered_confirmation_bias_ratios(
    turn_records: List[Dict[str, Any]],
    *,
    distance_group: Optional[str] = None,
    success: Optional[bool] = None,
) -> List[float]:
    by_game: Dict[Any, Dict[str, int]] = {}

    for record in _filtered_turn_records(turn_records, distance_group=distance_group, success=success):
        if record.get("relation") != CONFIRMATION_BIAS_RELATION:
            continue

        intention = record.get("inferred_intention")
        if intention not in {"confirm", "falsify"}:
            continue

        game_id = record.get("game_id")
        counts = by_game.setdefault(game_id, {"confirmatory": 0, "disconfirmatory": 0})
        if intention == "confirm":
            counts["confirmatory"] += 1
        else:
            counts["disconfirmatory"] += 1

    ratios: List[float] = []
    for counts in by_game.values():
        labeled_total = counts["confirmatory"] + counts["disconfirmatory"]
        if labeled_total <= 0:
            continue
        ratios.append(counts["confirmatory"] / labeled_total)

    return ratios


def _relation_filtered_confirmation_bias_ratio(
    turn_records: List[Dict[str, Any]],
    *,
    distance_group: Optional[str] = None,
    success: Optional[bool] = None,
) -> Optional[float]:
    per_game_ratios = _per_game_relation_filtered_confirmation_bias_ratios(
        turn_records,
        distance_group=distance_group,
        success=success,
    )
    if not per_game_ratios:
        return None

    return sum(per_game_ratios) / len(per_game_ratios)


def _per_game_conclusive_falsification_rates(
    turn_records: List[Dict[str, Any]],
    *,
    distance_group: Optional[str] = None,
    success: Optional[bool] = None,
) -> List[float]:
    by_game: Dict[Any, Dict[str, int]] = {}

    for record in _filtered_turn_records(turn_records, distance_group=distance_group, success=success):
        conclusive_falsification = record.get("conclusive_falsification")
        if not isinstance(conclusive_falsification, bool):
            continue

        game_id = record.get("game_id")
        counts = by_game.setdefault(game_id, {"conclusive": 0, "evaluable": 0})
        counts["evaluable"] += 1
        if conclusive_falsification:
            counts["conclusive"] += 1

    rates: List[float] = []
    for counts in by_game.values():
        evaluable = counts["evaluable"]
        if evaluable <= 0:
            continue
        rates.append(counts["conclusive"] / evaluable)

    return rates


def _conclusive_falsification_rate(
    turn_records: List[Dict[str, Any]],
    *,
    distance_group: Optional[str] = None,
    success: Optional[bool] = None,
) -> Optional[float]:
    per_game_rates = _per_game_conclusive_falsification_rates(
        turn_records,
        distance_group=distance_group,
        success=success,
    )
    if not per_game_rates:
        return None

    return sum(per_game_rates) / len(per_game_rates)


def discover_model_aggregates(results_dir: Path) -> List[Dict[str, Any]]:
    """Read aggregate_results.json from each model subdirectory."""
    rows: List[Dict[str, Any]] = []

    for aggregate_path in sorted(results_dir.glob("*/aggregate_results.json")):
        model_dir = aggregate_path.parent
        payload = _load_json(aggregate_path)

        overall = payload.get("overall") or {}
        by_diff = payload.get("by_distance_group") or {}

        game_results_path = model_dir / "game_results.json"
        if game_results_path.exists():
            evaluator = GameEvaluator()
            evaluator.load_results(str(game_results_path))
            overall = _merge_missing_metrics(overall, evaluator.compute_aggregate_metrics())
            easy_metrics = evaluator.compute_aggregate_metrics("close")
            hard_metrics = evaluator.compute_aggregate_metrics("deep")
        else:
            easy_metrics = {}
            hard_metrics = {}

        if not overall:
            continue

        rows.append(
            {
                "model": model_dir.name,
                "overall": overall,
                "close": _merge_missing_metrics(by_diff.get("close") or {}, easy_metrics),
                "deep": _merge_missing_metrics(by_diff.get("deep") or {}, hard_metrics),
            }
        )

    return rows


def discover_model_relation_metrics(results_dir: Path) -> List[Dict[str, Any]]:
    """Compute model metrics that require joined turn annotations, plus public bias outputs."""
    rows: List[Dict[str, Any]] = []

    for game_results_path in sorted(results_dir.glob("*/game_results.json")):
        model_dir = game_results_path.parent
        evaluator = GameEvaluator()
        evaluator.load_results(str(game_results_path))
        evaluator.load_relation_annotations(str(model_dir / "turn_relation_annotations.json"))
        turn_records = _load_relation_turn_records(model_dir)
        overall_metrics = evaluator.compute_aggregate_metrics()
        easy_metrics = evaluator.compute_aggregate_metrics("close")
        hard_metrics = evaluator.compute_aggregate_metrics("deep")
        if not overall_metrics and not turn_records:
            continue

        rows.append(
            {
                "model": model_dir.name,
                "overall": {
                    "avg_positive_testing_bias": overall_metrics.get("avg_positive_testing_bias"),
                    "avg_positive_testing_bias_std": overall_metrics.get("avg_positive_testing_bias_std"),
                    "avg_positive_testing_bias_ci95": overall_metrics.get("avg_positive_testing_bias_ci95"),
                    "avg_conclusive_falsification_rate": overall_metrics.get("avg_conclusive_falsification_rate"),
                    "avg_conclusive_falsification_rate_std": overall_metrics.get("avg_conclusive_falsification_rate_std"),
                    "avg_conclusive_falsification_rate_ci95": overall_metrics.get("avg_conclusive_falsification_rate_ci95"),
                },
                "close": {
                    "avg_positive_testing_bias": easy_metrics.get("avg_positive_testing_bias"),
                    "avg_positive_testing_bias_std": easy_metrics.get("avg_positive_testing_bias_std"),
                    "avg_positive_testing_bias_ci95": easy_metrics.get("avg_positive_testing_bias_ci95"),
                    "avg_conclusive_falsification_rate": easy_metrics.get("avg_conclusive_falsification_rate"),
                    "avg_conclusive_falsification_rate_std": easy_metrics.get("avg_conclusive_falsification_rate_std"),
                    "avg_conclusive_falsification_rate_ci95": easy_metrics.get("avg_conclusive_falsification_rate_ci95"),
                },
                "deep": {
                    "avg_positive_testing_bias": hard_metrics.get("avg_positive_testing_bias"),
                    "avg_positive_testing_bias_std": hard_metrics.get("avg_positive_testing_bias_std"),
                    "avg_positive_testing_bias_ci95": hard_metrics.get("avg_positive_testing_bias_ci95"),
                    "avg_conclusive_falsification_rate": hard_metrics.get("avg_conclusive_falsification_rate"),
                    "avg_conclusive_falsification_rate_std": hard_metrics.get("avg_conclusive_falsification_rate_std"),
                    "avg_conclusive_falsification_rate_ci95": hard_metrics.get("avg_conclusive_falsification_rate_ci95"),
                },
            }
        )

    return rows


def discover_model_relation_distributions(results_dir: Path) -> List[Dict[str, Any]]:
    return discover_model_relation_distributions_by_outcome(results_dir)


def discover_model_relation_distributions_by_outcome(
    results_dir: Path,
    *,
    success: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Summarize relation distributions per model, optionally filtered by game outcome."""
    rows: List[Dict[str, Any]] = []

    for game_results_path in sorted(results_dir.glob("*/game_results.json")):
        model_dir = game_results_path.parent
        turn_records = _load_relation_turn_records(model_dir)
        if success is not None:
            turn_records = _filtered_turn_records(turn_records, success=success)
        if not turn_records:
            continue

        relation_counts = {label: 0 for label in RELATION_LABELS}
        valid_total = 0

        for turn_record in turn_records:
            relation = turn_record.get("relation")
            if relation not in relation_counts:
                continue

            relation_counts[relation] += 1
            valid_total += 1

        if valid_total == 0:
            continue

        rows.append(
            {
                "model": model_dir.name,
                "total_ok_annotations": valid_total,
                "relation_distribution": {
                    label: relation_counts[label] / valid_total
                    for label in RELATION_LABELS
                },
            }
        )

    return rows


def discover_model_guess_counts(results_dir: Path) -> List[Dict[str, Any]]:
    """Read per-game results and summarize overall guess counts per game."""
    rows: List[Dict[str, Any]] = []

    for game_results_path in sorted(results_dir.glob("*/game_results.json")):
        model_dir = game_results_path.parent
        payload = _load_json_list(game_results_path)
        if not payload:
            continue

        guess_counts: List[int] = []
        for game_result in payload:
            if not isinstance(game_result, dict):
                continue
            guess_counts.append(count_guess_actions(game_result))

        guess_count_values = [float(value) for value in guess_counts]
        rows.append(
            {
                "model": model_dir.name,
                "avg_guess_count": (
                    sum(guess_counts) / len(guess_counts)
                    if guess_counts
                    else None
                ),
                "avg_guess_count_std": _population_standard_deviation(guess_count_values),
                "avg_guess_count_ci95": _confidence_interval_95_half_width(guess_count_values),
                "games": len(guess_counts),
            }
        )

    return rows


def discover_model_target_property_guess_counts(results_dir: Path) -> List[Dict[str, Any]]:
    """Read per-game results and summarize guess counts per target property."""
    rows: List[Dict[str, Any]] = []

    for game_results_path in sorted(results_dir.glob("*/game_results.json")):
        model_dir = game_results_path.parent
        payload = _load_json_list(game_results_path)
        if not payload:
            continue

        by_target_property: Dict[str, List[int]] = {}
        for game_result in payload:
            if not isinstance(game_result, dict):
                continue

            target_property = _normalize_target_property_name(game_result.get("target_property"))
            if not target_property:
                continue

            by_target_property.setdefault(target_property, []).append(count_guess_actions(game_result))

        target_property_summary: Dict[str, Dict[str, Any]] = {}
        for target_property, guess_counts in by_target_property.items():
            if not guess_counts:
                continue

            guess_count_values = [float(value) for value in guess_counts]
            target_property_summary[target_property] = {
                "avg_guess_count": round(sum(guess_counts) / len(guess_counts), 2),
                "avg_guess_count_std": _population_standard_deviation(guess_count_values),
                "avg_guess_count_ci95": _confidence_interval_95_half_width(guess_count_values),
                "games": len(guess_counts),
            }

        if target_property_summary:
            rows.append(
                {
                    "model": model_dir.name,
                    "by_target_property": target_property_summary,
                }
            )

    return rows


def discover_model_success_bias_correlation_rows(results_dir: Path) -> List[Dict[str, Any]]:
    """Pair each model's success rate with its Positive Testing Bias."""
    aggregate_rows = discover_model_aggregates(results_dir)
    aggregate_by_model = {str(row.get("model")): row for row in aggregate_rows}

    rows: List[Dict[str, Any]] = []
    for game_results_path in sorted(results_dir.glob("*/game_results.json")):
        model_id = game_results_path.parent.name
        aggregate_row = aggregate_by_model.get(model_id, {})
        success_rate = (aggregate_row.get("overall") or {}).get("success_rate")
        positive_testing_bias = (aggregate_row.get("overall") or {}).get("avg_positive_testing_bias")

        if not isinstance(positive_testing_bias, (int, float)):
            evaluator = GameEvaluator()
            evaluator.load_results(str(game_results_path))
            overall_metrics = evaluator.compute_aggregate_metrics()
            positive_testing_bias = overall_metrics.get("avg_positive_testing_bias")

        if not isinstance(success_rate, (int, float)) or not isinstance(positive_testing_bias, (int, float)):
            continue

        rows.append(
            {
                "model": model_id,
                "success_rate": float(success_rate),
                "avg_positive_testing_bias": float(positive_testing_bias),
            }
        )

    return rows


def discover_model_bias_falsification_correlation_rows(results_dir: Path) -> List[Dict[str, Any]]:
    """Pair each model's Positive Testing Bias with conclusive falsification rate."""
    rows: List[Dict[str, Any]] = []

    for game_results_path in sorted(results_dir.glob("*/game_results.json")):
        model_dir = game_results_path.parent
        annotation_path = model_dir / "turn_relation_annotations.json"
        if not annotation_path.exists():
            continue

        evaluator = GameEvaluator()
        evaluator.load_results(str(game_results_path))
        evaluator.load_relation_annotations(str(annotation_path))
        overall_metrics = evaluator.compute_aggregate_metrics()

        positive_testing_bias = overall_metrics.get("avg_positive_testing_bias")
        conclusive_falsification_rate = overall_metrics.get("avg_conclusive_falsification_rate")
        if not isinstance(positive_testing_bias, (int, float)) or not isinstance(conclusive_falsification_rate, (int, float)):
            continue

        rows.append(
            {
                "model": model_dir.name,
                "avg_positive_testing_bias": float(positive_testing_bias),
                "avg_conclusive_falsification_rate": float(conclusive_falsification_rate),
            }
        )

    return rows


class GameEvaluator:
    """
    Evaluates game results and computes various metrics.
    """

    def __init__(self):
        self.results = []
        self.relation_annotations_by_turn: Dict[tuple[Any, Any], Dict[str, Any]] = {}

    def add_result(self, game_result: Dict[str, Any]):
        """Add a game result to the evaluator."""
        self.results.append(game_result)

    def load_relation_annotations(self, filepath: str):
        """Load optional turn-level relation annotations for filtered bias metrics."""
        path = Path(filepath)
        if not path.exists():
            return

        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        if not isinstance(payload, dict):
            return

        annotations = payload.get("annotations")
        if not isinstance(annotations, list):
            return

        annotation_map: Dict[tuple[Any, Any], Dict[str, Any]] = {}
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            if annotation.get("status") != "ok":
                continue
            relation_value = annotation.get("relation")
            relation = relation_value.strip() if isinstance(relation_value, str) else None
            if relation == "":
                relation = None
            if relation is None:
                continue

            key = (annotation.get("game_id"), annotation.get("turn_number"))
            normalized_annotation = dict(annotation)
            normalized_annotation["relation"] = relation
            annotation_map[key] = normalized_annotation

        self.relation_annotations_by_turn = annotation_map

    def load_results(self, filepath: str):
        """Load game results from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                self.results.extend(data)
            else:
                self.results.append(data)

    def analyze_confirmation_bias(self, game_result: Dict[str, Any]) -> Dict[str, int]:
        """
        Analyze Positive Testing Bias in a single game.

        Counts:
        - Positive Testing queries: Player expects items to fit their hypothesis
        - Negative Testing queries: Player expects items NOT to fit (falsification attempts)

        Returns:
            Dictionary with counts
        """
        confirmatory = 0
        disconfirmatory = 0
        unlabeled = 0

        for turn in game_result.get("turns", []):
            action = turn.get("action", {})

            if action.get("action") == "test":
                intention = turn.get("inferred_intention")
                if isinstance(intention, str):
                    intention_l = intention.lower()
                    if intention_l == "confirm":
                        confirmatory += 1
                    elif intention_l == "falsify":
                        disconfirmatory += 1
                    else:
                        unlabeled += 1
                else:
                    unlabeled += 1

        labeled_total = confirmatory + disconfirmatory
        total_tests = labeled_total + unlabeled

        return {
            "confirmatory_queries": confirmatory,
            "disconfirmatory_queries": disconfirmatory,
            "unlabeled_test_queries": unlabeled,
            "total_test_queries": total_tests,
            "confirmation_bias_ratio": confirmatory / labeled_total if labeled_total > 0 else 0
        }

    def analyze_filtered_confirmation_bias(self, game_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Positive Testing Bias restricted to turns annotated as hypothesis-in-target.

        Falls back to the unfiltered metric when relation annotations are unavailable.
        """
        if not self.relation_annotations_by_turn:
            bias = self.analyze_confirmation_bias(game_result)
            return {
                "filtered_confirmatory_queries": bias["confirmatory_queries"],
                "filtered_disconfirmatory_queries": bias["disconfirmatory_queries"],
                "filtered_labeled_test_queries": bias["confirmatory_queries"] + bias["disconfirmatory_queries"],
                "filtered_confirmation_bias_ratio": bias["confirmation_bias_ratio"],
            }

        confirmatory = 0
        disconfirmatory = 0
        game_id = game_result.get("game_id")

        for index, turn in enumerate(game_result.get("turns", []), start=1):
            action = turn.get("action", {})
            if action.get("action") != "test":
                continue

            turn_number = turn.get("turn_number")
            if not isinstance(turn_number, int) or turn_number <= 0:
                turn_number = index

            annotation = self.relation_annotations_by_turn.get((game_id, turn_number))
            if not annotation:
                continue
            if annotation.get("relation") != CONFIRMATION_BIAS_RELATION:
                continue

            intention = turn.get("inferred_intention")
            if not isinstance(intention, str):
                continue

            intention_l = intention.lower()
            if intention_l == "confirm":
                confirmatory += 1
            elif intention_l == "falsify":
                disconfirmatory += 1

        labeled_total = confirmatory + disconfirmatory
        return {
            "filtered_confirmatory_queries": confirmatory,
            "filtered_disconfirmatory_queries": disconfirmatory,
            "filtered_labeled_test_queries": labeled_total,
            "filtered_confirmation_bias_ratio": confirmatory / labeled_total if labeled_total > 0 else 0,
        }

    def analyze_conclusive_falsification(self, game_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how often annotated TEST turns yield conclusive falsification of the current hypothesis."""
        if not self.relation_annotations_by_turn:
            return {
                "conclusive_falsification_turns": 0,
                "evaluable_conclusive_falsification_turns": 0,
                "conclusive_falsification_rate": None,
            }

        conclusive_turns = 0
        evaluable_turns = 0
        game_id = game_result.get("game_id")

        for index, turn in enumerate(game_result.get("turns", []), start=1):
            action = turn.get("action", {})
            if action.get("action") != "test":
                continue

            turn_number = turn.get("turn_number")
            if not isinstance(turn_number, int) or turn_number <= 0:
                turn_number = index

            annotation = self.relation_annotations_by_turn.get((game_id, turn_number))
            if not annotation:
                continue

            oracle_response = turn.get("oracle_response") or {}
            hypothesis_oracle_response = turn.get("hypothesis_oracle_response") or {}
            conclusive = _is_conclusive_falsification(
                annotation.get("relation"),
                _extract_bool(hypothesis_oracle_response.get("conforms")),
                _extract_bool(oracle_response.get("conforms")),
            )
            if not isinstance(conclusive, bool):
                continue

            evaluable_turns += 1
            if conclusive:
                conclusive_turns += 1

        return {
            "conclusive_falsification_turns": conclusive_turns,
            "evaluable_conclusive_falsification_turns": evaluable_turns,
            "conclusive_falsification_rate": (
                conclusive_turns / evaluable_turns if evaluable_turns > 0 else None
            ),
        }

    def compute_game_metrics(self, game_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute all metrics for a single game.

        Returns:
            Dictionary with all metrics
        """
        bias_metrics = self.analyze_confirmation_bias(game_result)
        filtered_bias_metrics = self.analyze_filtered_confirmation_bias(game_result)
        conclusive_falsification_metrics = self.analyze_conclusive_falsification(game_result)

        return {
            "game_id": game_result.get("game_id"),
            "target_property": game_result.get("target_property"),
            "sampling_hypothesis": game_result.get("sampling_hypothesis"),
            "distance_group": game_result.get("distance_group"),
            "distance": game_result.get("distance"),
            "success": game_result.get("success", False),
            "guess_count": count_guess_actions(game_result),
            "turns_to_solution": game_result.get("turns_to_solution"),
            "total_turns": game_result.get("total_turns", 0),
            "max_turns_reached": game_result.get("max_turns_reached", False),
            "positive_testing_queries": bias_metrics["confirmatory_queries"],
            "negative_testing_queries": bias_metrics["disconfirmatory_queries"],
            "unlabeled_test_queries": bias_metrics["unlabeled_test_queries"],
            "total_test_queries": bias_metrics["total_test_queries"],
            "positive_testing_bias": bias_metrics["confirmation_bias_ratio"],
            "filtered_positive_testing_queries": filtered_bias_metrics["filtered_confirmatory_queries"],
            "filtered_negative_testing_queries": filtered_bias_metrics["filtered_disconfirmatory_queries"],
            "filtered_labeled_test_queries": filtered_bias_metrics["filtered_labeled_test_queries"],
            "filtered_positive_testing_bias": filtered_bias_metrics["filtered_confirmation_bias_ratio"],
            **conclusive_falsification_metrics,
        }

    def compute_aggregate_metrics(
        self,
        distance_group: Optional[str] = None,
        target_property: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute aggregate metrics across all games or filtered by distance_group/target property.

        Args:
            distance_group: Optional filter by distance_group level
            target_property: Optional filter by target property

        Returns:
            Dictionary with aggregate statistics
        """
        filtered_results = self.results
        if distance_group:
            filtered_results = [r for r in self.results if r.get("distance_group") == distance_group]
        if target_property:
            normalized_target_property = _normalize_target_property_name(target_property)
            filtered_results = [
                result
                for result in filtered_results
                if _normalize_target_property_name(result.get("target_property")) == normalized_target_property
            ]

        if not filtered_results:
            return {}

        total_games = len(filtered_results)
        successful_games = [r for r in filtered_results if r.get("success", False)]

        # Success rate
        success_rate = len(successful_games) / total_games if total_games > 0 else 0

        # Average turns to solution (for successful games only)
        turns_to_solution = [r.get("turns_to_solution", 0) for r in successful_games
                            if r.get("turns_to_solution") is not None]
        avg_turns = sum(turns_to_solution) / len(turns_to_solution) if turns_to_solution else None

        # Guess usage across all games
        all_guess_counts = [count_guess_actions(result) for result in filtered_results]
        avg_guesses_per_game = sum(all_guess_counts) / total_games if total_games > 0 else 0

        # Positive Testing Bias metrics
        all_positive_testing_biases = []
        all_conclusive_falsification_rates = []

        for result in filtered_results:
            bias = self.analyze_confirmation_bias(result)
            conclusive_falsification = self.analyze_conclusive_falsification(result)
            labeled_total = bias["confirmatory_queries"] + bias["disconfirmatory_queries"]
            if labeled_total > 0:
                all_positive_testing_biases.append(bias["confirmation_bias_ratio"])
            if isinstance(conclusive_falsification["conclusive_falsification_rate"], float):
                all_conclusive_falsification_rates.append(conclusive_falsification["conclusive_falsification_rate"])

        avg_positive_testing_bias = (
            round(sum(all_positive_testing_biases) / len(all_positive_testing_biases), 3)
            if all_positive_testing_biases
            else 0
        )
        avg_positive_testing_bias_std = _population_standard_deviation(all_positive_testing_biases)
        avg_positive_testing_bias_ci95 = _confidence_interval_95_half_width(all_positive_testing_biases)
        turns_to_solution_values = [float(value) for value in turns_to_solution]
        guess_count_values = [float(value) for value in all_guess_counts]
        avg_turns_std = _population_standard_deviation(turns_to_solution_values)
        avg_turns_ci95 = _confidence_interval_95_half_width(turns_to_solution_values)
        avg_guesses_per_game_std = _population_standard_deviation(guess_count_values)
        avg_guesses_per_game_ci95 = _confidence_interval_95_half_width(guess_count_values)
        avg_conclusive_falsification_rate_std = _population_standard_deviation(all_conclusive_falsification_rates)
        avg_conclusive_falsification_rate_ci95 = _confidence_interval_95_half_width(all_conclusive_falsification_rates)

        return {
            "distance_group": distance_group or "all",
            "total_games": total_games,
            "successful_games": len(successful_games),
            "failed_games": total_games - len(successful_games),
            "success_rate": round(success_rate, 3),
            "avg_guesses_per_game": round(avg_guesses_per_game, 2),
            "avg_guesses_per_game_std": round(avg_guesses_per_game_std, 2) if avg_guesses_per_game_std is not None else None,
            "avg_guesses_per_game_ci95": (
                round(avg_guesses_per_game_ci95, 2)
                if avg_guesses_per_game_ci95 is not None
                else None
            ),
            "avg_turns_to_solution": round(avg_turns, 2) if avg_turns else None,
            "avg_turns_to_solution_std": round(avg_turns_std, 2) if avg_turns_std is not None else None,
            "avg_turns_to_solution_ci95": (
                round(avg_turns_ci95, 2)
                if avg_turns_ci95 is not None
                else None
            ),
            "avg_positive_testing_bias": avg_positive_testing_bias,
            "avg_positive_testing_bias_std": (
                round(avg_positive_testing_bias_std, 3)
                if avg_positive_testing_bias_std is not None
                else None
            ),
            "avg_positive_testing_bias_ci95": (
                round(avg_positive_testing_bias_ci95, 3)
                if avg_positive_testing_bias_ci95 is not None
                else None
            ),
            "avg_conclusive_falsification_rate": (
                round(sum(all_conclusive_falsification_rates) / len(all_conclusive_falsification_rates), 3)
                if all_conclusive_falsification_rates
                else None
            ),
            "avg_conclusive_falsification_rate_std": (
                round(avg_conclusive_falsification_rate_std, 3)
                if avg_conclusive_falsification_rate_std is not None
                else None
            ),
            "avg_conclusive_falsification_rate_ci95": (
                round(avg_conclusive_falsification_rate_ci95, 3)
                if avg_conclusive_falsification_rate_ci95 is not None
                else None
            ),
        }

    def save_detailed_results(self, output_path: str = "results/detailed_results.json"):
        """
        Save detailed results with all metrics for each game.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        detailed = [self.compute_game_metrics(result) for result in self.results]

        with open(output_path, 'w') as f:
            json.dump(detailed, f, indent=2)

        print(f"Saved detailed results to {output_path}")

    def save_aggregate_results(self, output_path: str = "results/aggregate_results.json"):
        """
        Save aggregate statistics overall and by distance_group.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        aggregate = {
            "overall": self.compute_aggregate_metrics(),
            "by_distance_group": {
                "close": self.compute_aggregate_metrics("close"),
                "deep": self.compute_aggregate_metrics("deep")
            }
        }

        with open(output_path, 'w') as f:
            json.dump(aggregate, f, indent=2)

        print(f"Saved aggregate results to {output_path}")

    def save_csv_summary(self, output_path: str = "results/summary.csv"):
        """
        Save a CSV summary of all games for downstream analysis.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.results:
            print("No results to save")
            return

        metrics = [self.compute_game_metrics(result) for result in self.results]

        if not metrics:
            print("No metrics to save")
            return

        fieldnames = list(metrics[0].keys())

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics)

        print(f"Saved CSV summary to {output_path}")

    def print_summary(self):
        """Print a summary of results to console."""
        if not self.results:
            print("No results to summarize")
            return

        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)

        # Overall metrics
        overall = self.compute_aggregate_metrics()
        print("\nOverall Performance:")
        print(f"  Total games: {overall['total_games']}")
        print(f"  Success rate: {overall['success_rate']:.1%}")
        print(f"  Avg guesses per game: {overall['avg_guesses_per_game']}")
        print(f"  Avg turns to solution: {overall['avg_turns_to_solution']}")
        print(f"  Avg Positive Testing Bias: {overall['avg_positive_testing_bias']:.3f}")
        print(f"  Avg conclusive falsification rate: {overall['avg_conclusive_falsification_rate']}")

        # By distance_group
        print("\nPerformance by Taxonomic Distance:")
        for distance_group in ["close", "deep"]:
            metrics = self.compute_aggregate_metrics(distance_group)
            if metrics:
                print(f"\n  {distance_group.upper()}:")
                print(f"    Games: {metrics['total_games']}")
                print(f"    Success rate: {metrics['success_rate']:.1%}")
                print(f"    Avg guesses: {metrics['avg_guesses_per_game']}")
                print(f"    Avg turns: {metrics['avg_turns_to_solution']}")
                print(f"    Positive Testing Bias: {metrics['avg_positive_testing_bias']:.3f}")
                print(f"    Avg conclusive falsification rate: {metrics['avg_conclusive_falsification_rate']}")


def find_model_results_files(results_dir: str = "results") -> List[Path]:
    """Find per-model game_results.json files under results/*/game_results.json."""
    base = Path(results_dir)
    if not base.exists() or not base.is_dir():
        return []

    files = []
    for candidate in base.glob("*/game_results.json"):
        if candidate.is_file():
            files.append(candidate)

    return sorted(files)


def evaluate_single_results_dir(output_dir: Path) -> Dict[str, Any]:
    """Evaluate one model output directory and write evaluation artifacts."""
    output_dir = Path(output_dir)
    results_file = output_dir / "game_results.json"

    if not results_file.exists():
        return {
            "model": output_dir.name,
            "output_dir": str(output_dir),
            "status": "missing_results",
            "message": f"No results file found at {results_file}",
        }

    evaluator = GameEvaluator()
    evaluator.load_results(str(results_file))
    evaluator.load_relation_annotations(str(output_dir / "turn_relation_annotations.json"))

    evaluator.save_detailed_results(str(output_dir / "detailed_results.json"))
    evaluator.save_aggregate_results(str(output_dir / "aggregate_results.json"))
    evaluator.save_csv_summary(str(output_dir / "summary.csv"))

    overall = evaluator.compute_aggregate_metrics()
    close = evaluator.compute_aggregate_metrics("close")
    deep = evaluator.compute_aggregate_metrics("deep")

    return {
        "model": output_dir.name,
        "output_dir": str(output_dir),
        "status": "ok",
        "overall": overall,
        "close": close,
        "deep": deep,
    }


def evaluate_all_models(results_dir: str = "results") -> Dict[str, Any]:
    """Evaluate all model runs found under results/*/game_results.json."""
    results_files = find_model_results_files(results_dir)
    model_dirs = [path.parent for path in results_files]

    evaluations = []
    for model_dir in model_dirs:
        evaluations.append(evaluate_single_results_dir(model_dir))

    summary = {
        "results_dir": str(Path(results_dir)),
        "total_models_found": len(model_dirs),
        "evaluated_models": len([e for e in evaluations if e.get("status") == "ok"]),
        "models": evaluations,
    }

    output_path = Path(results_dir) / "aggregate_by_model.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate game results for one model or all models under a results root."
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help=(
            "Root directory scanned for model runs (expects */game_results.json) when "
            "--output-dir is not provided"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Evaluate a single model run directory (contains game_results.json)",
    )

    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
        single_result = evaluate_single_results_dir(output_dir)

        if single_result.get("status") != "ok":
            print(single_result.get("message", f"No results found in {output_dir}"))
            raise SystemExit(1)

        evaluator = GameEvaluator()
        evaluator.load_results(str(output_dir / "game_results.json"))
        evaluator.load_relation_annotations(str(output_dir / "turn_relation_annotations.json"))
        evaluator.print_summary()
        raise SystemExit(0)

    summary = evaluate_all_models(args.results_dir)
    total = summary.get("total_models_found", 0)
    evaluated = summary.get("evaluated_models", 0)
    results_root = Path(args.results_dir)

    if total == 0:
        print(f"No model results found under {results_root}/*/game_results.json")
        raise SystemExit(1)

    print(f"Evaluated {evaluated}/{total} model directories")
    print(f"Saved cross-model summary to {results_root / 'aggregate_by_model.json'}")
