import json

from src.analysis.failure import (
    build_game_relation_feature_rows,
    classify_linguistic_pattern_hypothesis,
    evaluate_all_models,
    evaluate_distributional_difference,
    evaluate_linguistic_pattern,
)


def _write_game_result_from_relation_sequence(game_id, success, relation_sequence, *, distance_group=None):
    turns = []
    annotations = []
    turn_number = 1

    for relation_label in relation_sequence:
        turns.append({
            "turn_number": turn_number,
            "action": {"action": "test"},
        })
        annotations.append({
            "game_id": game_id,
            "turn_number": turn_number,
            "status": "ok",
            "relation": relation_label,
        })
        turn_number += 1

    return {
        "game_result": {
            "game_id": game_id,
            "success": success,
            "distance_group": distance_group,
            "turns": turns,
        },
        "annotations": annotations,
    }


def test_build_game_relation_feature_rows_computes_proportions(tmp_path):
    model_dir = tmp_path / "results" / "model-a"
    model_dir.mkdir(parents=True)

    payload = _write_game_result_from_relation_sequence(
        1,
        True,
        ["disjoint", "identical", "partial_overlap", "disjoint", "partial_overlap", "partial_overlap"],
    )
    (model_dir / "game_results.json").write_text(json.dumps([payload["game_result"]]), encoding="utf-8")
    (model_dir / "turn_relation_annotations.json").write_text(
        json.dumps({"annotations": payload["annotations"]}),
        encoding="utf-8",
    )

    rows = build_game_relation_feature_rows(model_dir)

    assert rows == [
        {
            "model": "model-a",
            "game_id": 1,
            "success": True,
            "distance_group": None,
            "annotated_test_turns": 6,
            "annotated_relation_sequence": [
                "disjoint",
                "identical",
                "partial_overlap",
                "disjoint",
                "partial_overlap",
                "partial_overlap",
            ],
            "linguistic_shift_present": False,
            "linguistic_test_sequence": [False, False, False, False, False, False],
            "first_linguistic_test_index": None,
            "post_linguistic_test_turns": 0,
            "post_linguistic_recovery_evaluable": False,
            "post_linguistic_no_recovery": False,
            "post_linguistic_recovered": False,
            "post_linguistic_linguistic_share": None,
            "linguistic_shift_rule_counts": {},
            "identical": 1,
            "disjoint": 2,
            "partial_overlap": 3,
            "hypothesis_included_in_target": 0,
            "target_included_in_hypothesis": 0,
            "disjoint_count": 2,
            "disjoint_proportion": 2 / 6,
            "linguistic_disjoint_count": 0,
            "partial_overlap_count": 3,
            "partial_overlap_proportion": 3 / 6,
            "linguistic_partial_overlap_count": 0,
            "target_relation_turns": 5,
            "linguistic_target_relation_turns": 0,
        }
    ]


def test_classify_linguistic_pattern_hypothesis_detects_meta_linguistic_rules():
    payload = classify_linguistic_pattern_hypothesis(
        "The items all have two-word common names and start with the same letter."
    )

    assert payload["is_linguistic_pattern"] is True
    assert payload["matched_rules"] == ["letter", "meta_linguistic_structure", "naming", "word_count"]


def test_evaluate_linguistic_pattern_summarizes_failure_and_target_relation_overlap():
    feature_rows = [
        {
            "success": False,
            "linguistic_shift_present": True,
            "linguistic_shift_rule_counts": {"word_count": 1, "meta_linguistic_structure": 1},
            "disjoint_count": 2,
            "linguistic_disjoint_count": 2,
            "partial_overlap_count": 1,
            "linguistic_partial_overlap_count": 1,
            "target_relation_turns": 3,
            "linguistic_target_relation_turns": 3,
        },
        {
            "success": True,
            "linguistic_shift_present": False,
            "linguistic_shift_rule_counts": {},
            "disjoint_count": 0,
            "linguistic_disjoint_count": 0,
            "partial_overlap_count": 2,
            "linguistic_partial_overlap_count": 0,
            "target_relation_turns": 2,
            "linguistic_target_relation_turns": 0,
        },
        {
            "success": False,
            "linguistic_shift_present": True,
            "linguistic_shift_rule_counts": {"letter": 2},
            "disjoint_count": 1,
            "linguistic_disjoint_count": 0,
            "partial_overlap_count": 1,
            "linguistic_partial_overlap_count": 1,
            "target_relation_turns": 2,
            "linguistic_target_relation_turns": 1,
        },
        {
            "success": True,
            "linguistic_shift_present": False,
            "linguistic_shift_rule_counts": {},
            "disjoint_count": 1,
            "linguistic_disjoint_count": 0,
            "partial_overlap_count": 0,
            "linguistic_partial_overlap_count": 0,
            "target_relation_turns": 1,
            "linguistic_target_relation_turns": 0,
        },
    ]

    payload = evaluate_linguistic_pattern(feature_rows)

    assert payload["status"] == "ok"
    assert payload["games_with_linguistic_shift"] == 2
    assert payload["linguistic_shift_rate"] == 0.5
    assert payload["failure_rate_with_linguistic_shift"] == 1.0
    assert payload["failure_rate_without_linguistic_shift"] == 0.0
    assert payload["failure_rate_gap_shifted_minus_unshifted"] == 1.0
    assert payload["shift_rate_in_unsuccessful_games"] == 1.0
    assert payload["shift_rate_in_successful_games"] == 0.0
    assert payload["games_with_post_linguistic_recovery_opportunity"] == 0
    assert payload["post_linguistic_no_recovery_rate_in_unsuccessful_games"] is None
    assert payload["post_linguistic_no_recovery_rate_in_successful_games"] is None
    assert payload["relations"]["disjoint"]["linguistic_share"] == 2 / 4
    assert payload["relations"]["partial_overlap"]["linguistic_share"] == 2 / 4
    assert payload["relations"]["target_relations_combined"]["linguistic_share"] == 4 / 8
    assert payload["matched_rule_counts"] == {
        "letter": 2,
        "meta_linguistic_structure": 1,
        "word_count": 1,
    }


def test_evaluate_linguistic_pattern_summarizes_post_entry_no_recovery_trap_rates():
    feature_rows = [
        {
            "success": False,
            "linguistic_shift_present": True,
            "post_linguistic_recovery_evaluable": True,
            "post_linguistic_no_recovery": True,
            "post_linguistic_recovered": False,
            "disjoint_count": 0,
            "linguistic_disjoint_count": 0,
            "partial_overlap_count": 0,
            "linguistic_partial_overlap_count": 0,
            "target_relation_turns": 0,
            "linguistic_target_relation_turns": 0,
            "linguistic_shift_rule_counts": {"letter": 1},
        },
        {
            "success": False,
            "linguistic_shift_present": True,
            "post_linguistic_recovery_evaluable": True,
            "post_linguistic_no_recovery": False,
            "post_linguistic_recovered": True,
            "disjoint_count": 0,
            "linguistic_disjoint_count": 0,
            "partial_overlap_count": 0,
            "linguistic_partial_overlap_count": 0,
            "target_relation_turns": 0,
            "linguistic_target_relation_turns": 0,
            "linguistic_shift_rule_counts": {"word_count": 1},
        },
        {
            "success": True,
            "linguistic_shift_present": True,
            "post_linguistic_recovery_evaluable": True,
            "post_linguistic_no_recovery": False,
            "post_linguistic_recovered": True,
            "disjoint_count": 0,
            "linguistic_disjoint_count": 0,
            "partial_overlap_count": 0,
            "linguistic_partial_overlap_count": 0,
            "target_relation_turns": 0,
            "linguistic_target_relation_turns": 0,
            "linguistic_shift_rule_counts": {},
        },
        {
            "success": True,
            "linguistic_shift_present": True,
            "post_linguistic_recovery_evaluable": False,
            "post_linguistic_no_recovery": False,
            "post_linguistic_recovered": False,
            "disjoint_count": 0,
            "linguistic_disjoint_count": 0,
            "partial_overlap_count": 0,
            "linguistic_partial_overlap_count": 0,
            "target_relation_turns": 0,
            "linguistic_target_relation_turns": 0,
            "linguistic_shift_rule_counts": {},
        },
    ]

    payload = evaluate_linguistic_pattern(feature_rows)

    assert payload["games_with_post_linguistic_recovery_opportunity"] == 3
    assert payload["successful_games_with_post_linguistic_recovery_opportunity"] == 1
    assert payload["unsuccessful_games_with_post_linguistic_recovery_opportunity"] == 2
    assert payload["post_linguistic_no_recovery_rate_in_unsuccessful_games"] == 0.5
    assert payload["post_linguistic_no_recovery_rate_in_successful_games"] == 0.0
    assert payload["post_linguistic_no_recovery_rate_gap_unsuccessful_minus_successful"] == 0.5
    assert payload["post_linguistic_no_recovery_fisher_exact"]["table"] == {
        "unsuccessful_no_recovery": 1,
        "unsuccessful_recovered": 1,
        "successful_no_recovery": 0,
        "successful_recovered": 1,
    }


def test_evaluate_distributional_difference_detects_higher_unsuccessful_proportions():
    feature_rows = [
        {"success": True, "disjoint_proportion": 0.0, "partial_overlap_proportion": 0.2},
        {"success": True, "disjoint_proportion": 0.1, "partial_overlap_proportion": 0.1},
        {"success": False, "disjoint_proportion": 0.6, "partial_overlap_proportion": 0.7},
        {"success": False, "disjoint_proportion": 0.5, "partial_overlap_proportion": 0.6},
    ]

    payload = evaluate_distributional_difference(feature_rows)

    assert payload["status"] == "ok"
    assert payload["multiple_comparison_correction"] == {
        "family": ["disjoint", "partial_overlap"],
        "family_size": 2,
        "methods": ["bonferroni", "holm"],
    }
    assert payload["relations"]["disjoint"]["mean_difference"] > 0.0
    assert payload["relations"]["partial_overlap"]["mean_difference"] > 0.0
    assert payload["relations"]["disjoint"]["cliffs_delta"] > 0.0
    assert payload["relations"]["partial_overlap"]["cliffs_delta"] > 0.0
    assert payload["relations"]["disjoint"]["mann_whitney_p_value_bonferroni"] >= payload["relations"]["disjoint"]["mann_whitney_p_value"]
    assert payload["relations"]["partial_overlap"]["mann_whitney_p_value_holm"] >= payload["relations"]["partial_overlap"]["mann_whitney_p_value"]
    assert payload["relations"]["disjoint"]["mann_whitney_p_value_holm"] <= payload["relations"]["disjoint"]["mann_whitney_p_value_bonferroni"]
    assert payload["relations"]["partial_overlap"]["welch_t_p_value_bonferroni"] >= payload["relations"]["partial_overlap"]["welch_t_p_value"]


def test_evaluate_all_models_writes_new_relation_experiment_artifacts(tmp_path):
    results_dir = tmp_path / "results"
    model_dir = results_dir / "model-a"
    model_dir.mkdir(parents=True)

    game_results = []
    annotations = []
    for game_id in range(1, 5):
        success = game_id % 2 == 0
        relation_sequence = ["identical", "partial_overlap"] if success else ["disjoint", "disjoint"]
        payload = _write_game_result_from_relation_sequence(game_id, success, relation_sequence)
        for turn in payload["game_result"]["turns"]:
            turn["action"]["hypothesis"] = (
                "They have two-word names"
                if not success
                else "They are living things"
            )
        game_results.append(payload["game_result"])
        annotations.extend(payload["annotations"])

    (model_dir / "game_results.json").write_text(json.dumps(game_results), encoding="utf-8")
    (model_dir / "turn_relation_annotations.json").write_text(
        json.dumps({"annotations": annotations}),
        encoding="utf-8",
    )

    summary_payload = evaluate_all_models(results_dir)

    assert summary_payload["evaluated_models"] == 1
    assert summary_payload["pooled_experiment_3_linguistic_shift"]["games_with_linguistic_shift"] == 2
    assert summary_payload["pooled_experiment_3_linguistic_shift"]["failure_rate_with_linguistic_shift"] == 1.0
    assert (model_dir / "error_analysis_results.json").exists()
    assert (model_dir / "error_analysis_game_features.csv").exists()
    assert (results_dir / "error_analysis_summary.json").exists()

    csv_text = (model_dir / "error_analysis_game_features.csv").read_text(encoding="utf-8")
    assert "model" in csv_text
    assert "disjoint_proportion" in csv_text
    assert "partial_overlap_proportion" in csv_text
    assert "linguistic_shift_present" in csv_text
