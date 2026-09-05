"""Run the error-analysis experiments used in the analysis pipeline.

Usage examples:
    python -m src.analysis.failure
    python -m src.analysis.failure --results-dir results
    python -m src.analysis.failure --model-dir results/gpt-5-mini
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from scipy import stats


RELATION_LABELS: Tuple[str, ...] = (
    "identical",
    "disjoint",
    "partial_overlap",
    "hypothesis_included_in_target",
    "target_included_in_hypothesis",
)
TARGET_RELATIONS: Tuple[str, ...] = ("disjoint", "partial_overlap")


DEFAULT_OUTPUT_FILENAME = "error_analysis_results.json"
DEFAULT_PER_GAME_FILENAME = "error_analysis_game_features.csv"
DEFAULT_SUMMARY_FILENAME = "error_analysis_summary.json"
LINGUISTIC_RULE_PATTERNS: Dict[str, re.Pattern[str]] = {
    "syllable": re.compile(r"\bsyllab(?:le|les|ic)\b", re.IGNORECASE),
    "letter": re.compile(r"\bletter(?:s)?\b", re.IGNORECASE),
    "word_count": re.compile(
        r"\b(?:single|one|two|three|four|five|multi|multiple)[-\s]?word(?:ed)?\b",
        re.IGNORECASE,
    ),
    "prefix_suffix": re.compile(r"\b(?:prefix|suffix)(?:es)?\b", re.IGNORECASE),
    "vowel_consonant": re.compile(r"\b(?:vowel|consonant)(?:s)?\b", re.IGNORECASE),
    "spelling": re.compile(r"\b(?:spelling|spelled|spelt|orthograph(?:y|ic))\b", re.IGNORECASE),
    "pronunciation": re.compile(r"\b(?:pronounc(?:e|ed|es|ing)|pronunciation|phonetic|rhym(?:e|es|ing))\b", re.IGNORECASE),
    "naming": re.compile(r"\b(?:common|scientific|latin)\s+names?\b|\bnames?\s+ha(?:s|ve)\b", re.IGNORECASE),
    "alphabetic": re.compile(r"\b(?:alphabet|alphabetic|alphabetical)\b", re.IGNORECASE),
    "initial_final_unit": re.compile(
        r"\b(?:first|last|initial|final)\s+(?:letter|word|syllable)\b",
        re.IGNORECASE,
    ),
}
LINGUISTIC_META_CUE_PATTERN = re.compile(
    r"\b(?:starts?|begins?|ends?|contains?|includes?)\b",
    re.IGNORECASE,
)
LINGUISTIC_META_TARGET_PATTERN = re.compile(
    r"\b(?:letter|word|name|syllable|vowel|consonant|prefix|suffix|alphabet)\b",
    re.IGNORECASE,
)


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    return payload if isinstance(payload, dict) else {}


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    return payload if isinstance(payload, list) else []


def _safe_rate(numerator: float, denominator: float) -> Optional[float]:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def classify_linguistic_pattern_hypothesis(hypothesis: Any) -> Dict[str, Any]:
    """Heuristically flag hypotheses that appeal to linguistic/meta-linguistic form."""
    hypothesis_text = str(hypothesis or "").strip()
    if not hypothesis_text:
        return {"is_linguistic_pattern": False, "matched_rules": []}

    matched_rules = [
        rule_name
        for rule_name, pattern in LINGUISTIC_RULE_PATTERNS.items()
        if pattern.search(hypothesis_text)
    ]

    if (
        LINGUISTIC_META_CUE_PATTERN.search(hypothesis_text)
        and LINGUISTIC_META_TARGET_PATTERN.search(hypothesis_text)
    ):
        matched_rules.append("meta_linguistic_structure")

    return {
        "is_linguistic_pattern": bool(matched_rules),
        "matched_rules": sorted(set(matched_rules)),
    }


def _iter_model_dirs(results_dir: Path) -> Iterable[Path]:
    for results_file in sorted(results_dir.glob("*/game_results.json")):
        yield results_file.parent


def _load_relation_annotation_map(model_dir: Path) -> Dict[Tuple[Any, Any], Dict[str, Any]]:
    annotation_path = model_dir / "turn_relation_annotations.json"
    if not annotation_path.exists():
        return {}

    annotation_payload = _load_json(annotation_path)
    raw_annotations = annotation_payload.get("annotations") or []
    if not isinstance(raw_annotations, list):
        return {}

    annotation_map: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    for annotation in raw_annotations:
        if not isinstance(annotation, dict):
            continue
        if annotation.get("status") != "ok":
            continue
        normalized_annotation = dict(annotation)
        relation_value = annotation.get("relation")
        relation = relation_value.strip() if isinstance(relation_value, str) else None
        if relation == "":
            relation = None
        normalized_annotation["relation"] = relation
        key = (annotation.get("game_id"), annotation.get("turn_number"))
        annotation_map[key] = normalized_annotation

    return annotation_map


def build_game_relation_feature_rows(model_dir: Path) -> List[Dict[str, Any]]:
    """Aggregate per-game relation features from annotated TEST turns."""
    game_results_path = model_dir / "game_results.json"
    if not game_results_path.exists():
        return []

    game_results = _load_json_list(game_results_path)
    annotation_map = _load_relation_annotation_map(model_dir)
    if not annotation_map:
        return []

    rows: List[Dict[str, Any]] = []
    for game_result in game_results:
        if not isinstance(game_result, dict):
            continue

        relation_counts = {relation: 0 for relation in RELATION_LABELS}
        linguistic_relation_counts = {relation: 0 for relation in TARGET_RELATIONS}
        linguistic_rule_counts = {rule_name: 0 for rule_name in sorted(LINGUISTIC_RULE_PATTERNS)}
        linguistic_rule_counts["meta_linguistic_structure"] = 0
        annotated_test_turns = 0
        game_id = game_result.get("game_id")
        annotated_relation_sequence: List[str] = []
        linguistic_test_sequence: List[bool] = []
        linguistic_pattern_turns = 0

        for index, turn in enumerate(game_result.get("turns", []), start=1):
            if not isinstance(turn, dict):
                continue
            action = turn.get("action") or {}
            if action.get("action") != "test":
                continue

            turn_number = turn.get("turn_number")
            if not isinstance(turn_number, int) or turn_number <= 0:
                turn_number = index

            annotation = annotation_map.get((game_id, turn_number))
            if annotation is None:
                continue

            relation = annotation.get("relation")
            if relation in relation_counts:
                relation_counts[relation] += 1
                annotated_relation_sequence.append(relation)
                annotated_test_turns += 1

                action_payload = turn.get("action") or {}
                classification = classify_linguistic_pattern_hypothesis(action_payload.get("hypothesis", ""))
                is_linguistic_pattern = bool(classification["is_linguistic_pattern"])
                linguistic_test_sequence.append(is_linguistic_pattern)
                if is_linguistic_pattern:
                    linguistic_pattern_turns += 1
                    if relation in linguistic_relation_counts:
                        linguistic_relation_counts[relation] += 1
                    for rule_name in classification["matched_rules"]:
                        linguistic_rule_counts[rule_name] = linguistic_rule_counts.get(rule_name, 0) + 1

        first_linguistic_index: Optional[int] = None
        for sequence_index, is_linguistic_pattern in enumerate(linguistic_test_sequence):
            if is_linguistic_pattern:
                first_linguistic_index = sequence_index
                break

        post_linguistic_sequence: List[bool] = []
        if first_linguistic_index is not None:
            post_linguistic_sequence = linguistic_test_sequence[first_linguistic_index + 1 :]

        post_linguistic_recovery_evaluable = bool(post_linguistic_sequence)
        post_linguistic_no_recovery = bool(post_linguistic_sequence) and all(post_linguistic_sequence)
        post_linguistic_recovered = bool(post_linguistic_sequence) and any(
            not is_linguistic_pattern for is_linguistic_pattern in post_linguistic_sequence
        )
        post_linguistic_linguistic_share = (
            sum(1 for is_linguistic_pattern in post_linguistic_sequence if is_linguistic_pattern)
            / len(post_linguistic_sequence)
            if post_linguistic_sequence
            else None
        )

        row = {
            "model": model_dir.name,
            "game_id": game_id,
            "success": bool(game_result.get("success", False)),
            "distance_group": game_result.get("distance_group"),
            "annotated_test_turns": annotated_test_turns,
            "annotated_relation_sequence": annotated_relation_sequence,
            "linguistic_shift_present": linguistic_pattern_turns > 0,
            "linguistic_test_sequence": linguistic_test_sequence,
            "first_linguistic_test_index": (first_linguistic_index + 1) if first_linguistic_index is not None else None,
            "post_linguistic_test_turns": len(post_linguistic_sequence),
            "post_linguistic_recovery_evaluable": post_linguistic_recovery_evaluable,
            "post_linguistic_no_recovery": post_linguistic_no_recovery,
            "post_linguistic_recovered": post_linguistic_recovered,
            "post_linguistic_linguistic_share": post_linguistic_linguistic_share,
            "linguistic_shift_rule_counts": {
                rule_name: count for rule_name, count in linguistic_rule_counts.items() if count > 0
            },
        }
        row.update(relation_counts)

        for relation_label in TARGET_RELATIONS:
            count = relation_counts[relation_label]
            row[f"{relation_label}_count"] = count
            row[f"{relation_label}_proportion"] = (count / annotated_test_turns) if annotated_test_turns > 0 else 0.0
            linguistic_count = linguistic_relation_counts[relation_label]
            row[f"linguistic_{relation_label}_count"] = linguistic_count

        target_relation_turns = sum(relation_counts[relation_label] for relation_label in TARGET_RELATIONS)
        linguistic_target_relation_turns = sum(
            linguistic_relation_counts[relation_label] for relation_label in TARGET_RELATIONS
        )
        row["target_relation_turns"] = target_relation_turns
        row["linguistic_target_relation_turns"] = linguistic_target_relation_turns

        rows.append(row)

    return rows


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _std(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    if len(values) < 2:
        return 0.0
    return float(stats.tstd(values))


def _true_median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(stats.scoreatpercentile(values, 50))


def _cliffs_delta(group_a: Sequence[float], group_b: Sequence[float]) -> Optional[float]:
    if not group_a or not group_b:
        return None

    greater = 0
    lower = 0
    for value_a in group_a:
        for value_b in group_b:
            if value_a > value_b:
                greater += 1
            elif value_a < value_b:
                lower += 1

    return float((greater - lower) / (len(group_a) * len(group_b)))


def _welch_t_test(group_a: Sequence[float], group_b: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    if len(group_a) < 2 or len(group_b) < 2:
        return None, None

    group_a_constant = all(value == group_a[0] for value in group_a)
    group_b_constant = all(value == group_b[0] for value in group_b)
    if group_a_constant and group_b_constant:
        if group_a[0] == group_b[0]:
            return None, None
        return None, 0.0

    statistic, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
    if statistic != statistic or p_value != p_value:
        return None, None
    return float(statistic), float(p_value)


def _mann_whitney_summary(group_a: Sequence[float], group_b: Sequence[float]) -> Dict[str, Optional[float]]:
    if not group_a or not group_b:
        return {
            "mann_whitney_u": None,
            "mann_whitney_p_value": None,
            "cliffs_delta": None,
        }

    statistic, p_value = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
    return {
        "mann_whitney_u": float(statistic),
        "mann_whitney_p_value": float(p_value),
        "cliffs_delta": _cliffs_delta(group_a, group_b),
    }


def _comparison_summary(group_a: Sequence[float], group_b: Sequence[float]) -> Dict[str, Any]:
    mean_a = _mean(group_a)
    mean_b = _mean(group_b)
    median_a = _true_median(group_a)
    median_b = _true_median(group_b)
    welch_t_statistic, welch_t_p_value = _welch_t_test(group_a, group_b)
    mann_whitney = _mann_whitney_summary(group_a, group_b)

    return {
        "group_a_count": len(group_a),
        "group_b_count": len(group_b),
        "group_a_mean": mean_a,
        "group_b_mean": mean_b,
        "mean_difference": (mean_a - mean_b) if mean_a is not None and mean_b is not None else None,
        "group_a_median": median_a,
        "group_b_median": median_b,
        "median_difference": (median_a - median_b) if median_a is not None and median_b is not None else None,
        "welch_t_statistic": welch_t_statistic,
        "welch_t_p_value": welch_t_p_value,
        **mann_whitney,
    }


def _bonferroni_adjust(p_values: Sequence[Optional[float]]) -> List[Optional[float]]:
    valid_p_values = [float(p_value) for p_value in p_values if p_value is not None]
    family_size = len(valid_p_values)
    if family_size == 0:
        return [None for _ in p_values]

    adjusted: List[Optional[float]] = []
    for p_value in p_values:
        if p_value is None:
            adjusted.append(None)
            continue
        adjusted.append(float(min(1.0, p_value * family_size)))
    return adjusted


def _holm_adjust(p_values: Sequence[Optional[float]]) -> List[Optional[float]]:
    indexed_valid_p_values = [
        (index, float(p_value))
        for index, p_value in enumerate(p_values)
        if p_value is not None
    ]
    family_size = len(indexed_valid_p_values)
    if family_size == 0:
        return [None for _ in p_values]

    adjusted: List[Optional[float]] = [None for _ in p_values]
    running_max = 0.0
    for rank, (index, p_value) in enumerate(sorted(indexed_valid_p_values, key=lambda item: item[1]), start=1):
        adjusted_p_value = min(1.0, p_value * (family_size - rank + 1))
        running_max = max(running_max, adjusted_p_value)
        adjusted[index] = float(running_max)
    return adjusted


def evaluate_distributional_difference(
    feature_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compare per-game disjoint/partial-overlap proportions by final outcome."""
    if not feature_rows:
        return {"status": "missing_annotations", "relations": {}}

    successful_rows = [row for row in feature_rows if bool(row.get("success"))]
    unsuccessful_rows = [row for row in feature_rows if not bool(row.get("success"))]

    relation_results: Dict[str, Any] = {}
    for relation_label in TARGET_RELATIONS:
        successful_values = [float(row.get(f"{relation_label}_proportion", 0.0)) for row in successful_rows]
        unsuccessful_values = [float(row.get(f"{relation_label}_proportion", 0.0)) for row in unsuccessful_rows]
        summary = _comparison_summary(unsuccessful_values, successful_values)
        summary["comparison"] = "unsuccessful_minus_successful"
        relation_results[relation_label] = summary

    relation_labels = list(TARGET_RELATIONS)
    mann_whitney_p_values = [
        relation_results[relation_label].get("mann_whitney_p_value")
        for relation_label in relation_labels
    ]
    welch_t_p_values = [
        relation_results[relation_label].get("welch_t_p_value")
        for relation_label in relation_labels
    ]
    adjusted_p_values_by_method = {
        "mann_whitney": {
            "bonferroni": _bonferroni_adjust(mann_whitney_p_values),
            "holm": _holm_adjust(mann_whitney_p_values),
        },
        "welch_t": {
            "bonferroni": _bonferroni_adjust(welch_t_p_values),
            "holm": _holm_adjust(welch_t_p_values),
        },
    }

    for relation_index, relation_label in enumerate(relation_labels):
        relation_results[relation_label]["mann_whitney_p_value_bonferroni"] = adjusted_p_values_by_method[
            "mann_whitney"
        ]["bonferroni"][relation_index]
        relation_results[relation_label]["mann_whitney_p_value_holm"] = adjusted_p_values_by_method[
            "mann_whitney"
        ]["holm"][relation_index]
        relation_results[relation_label]["welch_t_p_value_bonferroni"] = adjusted_p_values_by_method[
            "welch_t"
        ]["bonferroni"][relation_index]
        relation_results[relation_label]["welch_t_p_value_holm"] = adjusted_p_values_by_method[
            "welch_t"
        ]["holm"][relation_index]

    return {
        "status": "ok",
        "successful_games": len(successful_rows),
        "unsuccessful_games": len(unsuccessful_rows),
        "multiple_comparison_correction": {
            "family": list(TARGET_RELATIONS),
            "family_size": len(TARGET_RELATIONS),
            "methods": ["bonferroni", "holm"],
        },
        "relations": relation_results,
    }


def evaluate_linguistic_pattern(feature_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize whether heuristic linguistic/meta-linguistic hypotheses align with failure and target relations."""
    if not feature_rows:
        return {
            "status": "missing_annotations",
            "games": 0,
            "games_with_linguistic_shift": 0,
            "relations": {},
        }

    shifted_rows = [row for row in feature_rows if bool(row.get("linguistic_shift_present"))]
    unshifted_rows = [row for row in feature_rows if not bool(row.get("linguistic_shift_present"))]
    successful_rows = [row for row in feature_rows if bool(row.get("success"))]
    unsuccessful_rows = [row for row in feature_rows if not bool(row.get("success"))]
    trap_evaluable_rows = [row for row in shifted_rows if bool(row.get("post_linguistic_recovery_evaluable"))]
    successful_trap_evaluable_rows = [row for row in trap_evaluable_rows if bool(row.get("success"))]
    unsuccessful_trap_evaluable_rows = [row for row in trap_evaluable_rows if not bool(row.get("success"))]

    shifted_successes = sum(1 for row in shifted_rows if bool(row.get("success")))
    shifted_failures = len(shifted_rows) - shifted_successes
    unshifted_successes = sum(1 for row in unshifted_rows if bool(row.get("success")))
    unshifted_failures = len(unshifted_rows) - unshifted_successes

    fisher_odds_ratio: Optional[float] = None
    fisher_p_value: Optional[float] = None
    if shifted_rows and unshifted_rows:
        fisher_odds_ratio, fisher_p_value = stats.fisher_exact(
            [[shifted_failures, shifted_successes], [unshifted_failures, unshifted_successes]],
            alternative="two-sided",
        )
        fisher_odds_ratio = float(fisher_odds_ratio)
        fisher_p_value = float(fisher_p_value)

    successful_trap_no_recovery = sum(
        1 for row in successful_trap_evaluable_rows if bool(row.get("post_linguistic_no_recovery"))
    )
    unsuccessful_trap_no_recovery = sum(
        1 for row in unsuccessful_trap_evaluable_rows if bool(row.get("post_linguistic_no_recovery"))
    )
    successful_trap_recovered = sum(
        1 for row in successful_trap_evaluable_rows if bool(row.get("post_linguistic_recovered"))
    )
    unsuccessful_trap_recovered = sum(
        1 for row in unsuccessful_trap_evaluable_rows if bool(row.get("post_linguistic_recovered"))
    )

    post_linguistic_fisher_odds_ratio: Optional[float] = None
    post_linguistic_fisher_p_value: Optional[float] = None
    if successful_trap_evaluable_rows and unsuccessful_trap_evaluable_rows:
        post_linguistic_fisher_odds_ratio, post_linguistic_fisher_p_value = stats.fisher_exact(
            [
                [unsuccessful_trap_no_recovery, unsuccessful_trap_recovered],
                [successful_trap_no_recovery, successful_trap_recovered],
            ],
            alternative="two-sided",
        )
        post_linguistic_fisher_odds_ratio = float(post_linguistic_fisher_odds_ratio)
        post_linguistic_fisher_p_value = float(post_linguistic_fisher_p_value)

    relation_summaries: Dict[str, Any] = {}
    for relation_label in TARGET_RELATIONS:
        total_turns = sum(int(row.get(f"{relation_label}_count", 0) or 0) for row in feature_rows)
        linguistic_turns = sum(int(row.get(f"linguistic_{relation_label}_count", 0) or 0) for row in feature_rows)
        relation_summaries[relation_label] = {
            "total_turns": total_turns,
            "linguistic_turns": linguistic_turns,
            "linguistic_share": _safe_rate(linguistic_turns, total_turns),
        }

    target_relation_turns = sum(int(row.get("target_relation_turns", 0) or 0) for row in feature_rows)
    linguistic_target_relation_turns = sum(
        int(row.get("linguistic_target_relation_turns", 0) or 0) for row in feature_rows
    )

    rule_counts: Dict[str, int] = {}
    for row in shifted_rows:
        for rule_name, count in (row.get("linguistic_shift_rule_counts") or {}).items():
            if isinstance(count, int) and count > 0:
                rule_counts[rule_name] = rule_counts.get(rule_name, 0) + count

    return {
        "status": "ok",
        "games": len(feature_rows),
        "games_with_linguistic_shift": len(shifted_rows),
        "games_without_linguistic_shift": len(unshifted_rows),
        "linguistic_shift_rate": _safe_rate(len(shifted_rows), len(feature_rows)),
        "success_rate_with_linguistic_shift": _safe_rate(shifted_successes, len(shifted_rows)),
        "failure_rate_with_linguistic_shift": _safe_rate(shifted_failures, len(shifted_rows)),
        "success_rate_without_linguistic_shift": _safe_rate(unshifted_successes, len(unshifted_rows)),
        "failure_rate_without_linguistic_shift": _safe_rate(unshifted_failures, len(unshifted_rows)),
        "failure_rate_gap_shifted_minus_unshifted": (
            _safe_rate(shifted_failures, len(shifted_rows)) - _safe_rate(unshifted_failures, len(unshifted_rows))
            if shifted_rows and unshifted_rows
            else None
        ),
        "shift_rate_in_successful_games": _safe_rate(
            sum(1 for row in successful_rows if bool(row.get("linguistic_shift_present"))),
            len(successful_rows),
        ),
        "shift_rate_in_unsuccessful_games": _safe_rate(
            sum(1 for row in unsuccessful_rows if bool(row.get("linguistic_shift_present"))),
            len(unsuccessful_rows),
        ),
        "shift_rate_gap_unsuccessful_minus_successful": (
            _safe_rate(
                sum(1 for row in unsuccessful_rows if bool(row.get("linguistic_shift_present"))),
                len(unsuccessful_rows),
            )
            - _safe_rate(
                sum(1 for row in successful_rows if bool(row.get("linguistic_shift_present"))),
                len(successful_rows),
            )
            if successful_rows and unsuccessful_rows
            else None
        ),
        "games_with_post_linguistic_recovery_opportunity": len(trap_evaluable_rows),
        "successful_games_with_post_linguistic_recovery_opportunity": len(successful_trap_evaluable_rows),
        "unsuccessful_games_with_post_linguistic_recovery_opportunity": len(unsuccessful_trap_evaluable_rows),
        "post_linguistic_no_recovery_rate_in_successful_games": _safe_rate(
            successful_trap_no_recovery,
            len(successful_trap_evaluable_rows),
        ),
        "post_linguistic_no_recovery_rate_in_unsuccessful_games": _safe_rate(
            unsuccessful_trap_no_recovery,
            len(unsuccessful_trap_evaluable_rows),
        ),
        "post_linguistic_no_recovery_rate_gap_unsuccessful_minus_successful": (
            _safe_rate(unsuccessful_trap_no_recovery, len(unsuccessful_trap_evaluable_rows))
            - _safe_rate(successful_trap_no_recovery, len(successful_trap_evaluable_rows))
            if successful_trap_evaluable_rows and unsuccessful_trap_evaluable_rows
            else None
        ),
        "post_linguistic_no_recovery_fisher_exact": {
            "odds_ratio": post_linguistic_fisher_odds_ratio,
            "p_value": post_linguistic_fisher_p_value,
            "table": {
                "unsuccessful_no_recovery": unsuccessful_trap_no_recovery,
                "unsuccessful_recovered": unsuccessful_trap_recovered,
                "successful_no_recovery": successful_trap_no_recovery,
                "successful_recovered": successful_trap_recovered,
            },
        },
        "failure_fisher_exact": {
            "odds_ratio": fisher_odds_ratio,
            "p_value": fisher_p_value,
            "table": {
                "shifted_failures": shifted_failures,
                "shifted_successes": shifted_successes,
                "unshifted_failures": unshifted_failures,
                "unshifted_successes": unshifted_successes,
            },
        },
        "relations": {
            **relation_summaries,
            "target_relations_combined": {
                "total_turns": target_relation_turns,
                "linguistic_turns": linguistic_target_relation_turns,
                "linguistic_share": _safe_rate(linguistic_target_relation_turns, target_relation_turns),
            },
        },
        "matched_rule_counts": dict(sorted(rule_counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def evaluate_model_relation_experiments(
    model_dir: Path,
) -> Dict[str, Any]:
    """Run the relation analysis for one model folder."""
    feature_rows = build_game_relation_feature_rows(model_dir)
    if not feature_rows:
        return {
            "model": model_dir.name,
            "status": "missing_annotations",
            "games": 0,
            "successful_games": 0,
            "unsuccessful_games": 0,
            "experiment_1_distributional_difference": {"status": "missing_annotations", "relations": {}},
            "feature_rows": [],
        }

    successful_games = sum(1 for row in feature_rows if bool(row.get("success")))
    unsuccessful_games = len(feature_rows) - successful_games

    payload = {
        "model": model_dir.name,
        "status": "ok",
        "games": len(feature_rows),
        "successful_games": successful_games,
        "unsuccessful_games": unsuccessful_games,
        "experiment_1_distributional_difference": evaluate_distributional_difference(feature_rows),
        "experiment_3_linguistic_shift": evaluate_linguistic_pattern(feature_rows),
        "feature_rows": feature_rows,
    }
    return payload


def _serialize_model_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "feature_rows"}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)


def _write_game_feature_csv(model_dir: Path, evaluation_payload: Dict[str, Any]) -> None:
    feature_rows = evaluation_payload.get("feature_rows") or []

    fieldnames = [
        "model",
        "game_id",
        "success",
        "distance_group",
        "annotated_test_turns",
        "linguistic_shift_present",
        "disjoint_proportion",
        "partial_overlap_proportion",
    ]

    output_path = model_dir / DEFAULT_PER_GAME_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for row in feature_rows:
            csv_row = {
                "model": row.get("model"),
                "game_id": row.get("game_id"),
                "success": row.get("success"),
                "distance_group": row.get("distance_group"),
                "annotated_test_turns": row.get("annotated_test_turns"),
                "linguistic_shift_present": row.get("linguistic_shift_present"),
                "disjoint_proportion": row.get("disjoint_proportion"),
                "partial_overlap_proportion": row.get("partial_overlap_proportion"),
            }
            writer.writerow(csv_row)


def evaluate_all_models(results_dir: Path) -> Dict[str, Any]:
    """Run the relation analysis for all model directories under a results root."""
    evaluations: List[Dict[str, Any]] = []
    feature_rows_by_model: Dict[Path, List[Dict[str, Any]]] = {}

    for model_dir in _iter_model_dirs(results_dir):
        feature_rows_by_model[model_dir] = build_game_relation_feature_rows(model_dir)

    all_feature_rows = [row for rows in feature_rows_by_model.values() for row in rows]
    for model_dir in _iter_model_dirs(results_dir):
        evaluation_payload = evaluate_model_relation_experiments(model_dir)
        evaluations.append(evaluation_payload)
        _write_json(model_dir / DEFAULT_OUTPUT_FILENAME, _serialize_model_payload(evaluation_payload))
        if evaluation_payload.get("status") == "ok":
            _write_game_feature_csv(model_dir, evaluation_payload)

    summary_payload = {
        "results_dir": str(results_dir),
        "evaluated_models": len([payload for payload in evaluations if payload.get("status") == "ok"]),
        "pooled_experiment_3_linguistic_shift": evaluate_linguistic_pattern(all_feature_rows),
        "models": [_serialize_model_payload(payload) for payload in evaluations],
    }
    _write_json(results_dir / DEFAULT_SUMMARY_FILENAME, summary_payload)
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the error-analysis experiments from annotated TEST-turn relations."
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Root directory scanned for model runs (expects */game_results.json)",
    )
    parser.add_argument(
        "--model-dir",
        help="Evaluate a single model directory containing game_results.json",
    )
    args = parser.parse_args()

    if args.model_dir:
        model_dir = Path(args.model_dir)
        evaluation_payload = evaluate_model_relation_experiments(model_dir)
        _write_json(model_dir / DEFAULT_OUTPUT_FILENAME, _serialize_model_payload(evaluation_payload))
        if evaluation_payload.get("status") == "ok":
            _write_game_feature_csv(model_dir, evaluation_payload)
        print(f"Evaluated relation experiments for {model_dir.name}: {evaluation_payload.get('status')}")
        return

    summary_payload = evaluate_all_models(Path(args.results_dir))
    print(f"Evaluated {summary_payload.get('evaluated_models', 0)} model(s)")


if __name__ == "__main__":
    main()
