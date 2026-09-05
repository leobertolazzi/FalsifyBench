import json

from src.analysis.plots import (
    _linear_regression_fit_with_confidence_interval,
    _percentage_label_lower_bound,
    _rounded_percentage_label_value,
    _safe_model_name,
    _spearman_correlation_with_p_value,
    discover_model_bias_falsification_correlation_rows,
    discover_model_linguistic_pattern_rows,
    discover_model_relation_distributions_by_outcome,
    discover_model_relation_metrics,
    discover_model_success_bias_correlation_rows,
    discover_model_target_property_metrics,
    discover_model_target_property_relation_metrics,
    discover_pooled_linguistic_pattern_payload,
    generate_plots,
)
from src.benchmark.evaluation import discover_model_target_property_guess_counts


def test_rounded_percentage_label_value_uses_python_rounding_for_display():
    assert _rounded_percentage_label_value(4.49) == 4
    assert _rounded_percentage_label_value(4.5) == 4
    assert _rounded_percentage_label_value(5.5) == 6


def test_percentage_label_lower_bound_uses_floor_for_visibility_thresholds():
    assert _percentage_label_lower_bound(4.7) == 4
    assert _percentage_label_lower_bound(5.0) == 5
    assert _percentage_label_lower_bound(5.9) == 5


def test_safe_model_name_uses_requested_plot_display_names():
    assert _safe_model_name("meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8") == "Llama-4-Maverick"
    assert _safe_model_name("openai_gpt-oss-120b") == "GPT-OSS-120B"
    assert _safe_model_name("openai_gpt-oss-20b") == "GPT-OSS-20B"
    assert _safe_model_name("gpt-5-mini") == "GPT-5-Mini"
    assert _safe_model_name("gpt-5-nano") == "GPT-5-Nano"
    assert _safe_model_name("gpt-5.2-chat") == "GPT-5.2-Chat"


def test_discover_model_linguistic_pattern_rows_reads_new_error_analysis_payload(tmp_path):
    results_dir = tmp_path / "results"
    model_dir = results_dir / "model-a"
    model_dir.mkdir(parents=True)

    (model_dir / "error_analysis_results.json").write_text(
        json.dumps(
            {
                "experiment_3_linguistic_shift": {
                    "status": "ok",
                    "shift_rate_in_unsuccessful_games": 0.7,
                    "matched_rule_counts": {"word_count": 4},
                    "relations": {
                        "target_relations_combined": {"linguistic_share": 0.625},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    rows = discover_model_linguistic_pattern_rows(results_dir)

    assert rows == [
        {
            "model": "model-a",
            "shift_rate_in_successful_games": None,
            "shift_rate_in_unsuccessful_games": 0.7,
            "post_linguistic_no_recovery_rate_in_successful_games": None,
            "post_linguistic_no_recovery_rate_in_unsuccessful_games": None,
            "games_with_post_linguistic_recovery_opportunity": None,
            "post_linguistic_no_recovery_fisher_exact": {},
            "failure_fisher_exact": {},
            "matched_rule_counts": {"word_count": 4},
            "target_relations_combined_linguistic_share": 0.625,
        }
    ]


def test_discover_model_linguistic_pattern_rows_reads_success_rates_and_fisher_table(tmp_path):
    results_dir = tmp_path / "results"
    model_dir = results_dir / "model-a"
    model_dir.mkdir(parents=True)

    (model_dir / "error_analysis_results.json").write_text(
        json.dumps(
            {
                "experiment_3_linguistic_shift": {
                    "status": "ok",
                    "shift_rate_in_successful_games": 0.25,
                    "shift_rate_in_unsuccessful_games": 0.75,
                    "post_linguistic_no_recovery_rate_in_successful_games": 0.0,
                    "post_linguistic_no_recovery_rate_in_unsuccessful_games": 0.5,
                    "games_with_post_linguistic_recovery_opportunity": 3,
                    "post_linguistic_no_recovery_fisher_exact": {
                        "table": {
                            "unsuccessful_no_recovery": 1,
                            "unsuccessful_recovered": 1,
                            "successful_no_recovery": 0,
                            "successful_recovered": 1,
                        }
                    },
                    "failure_fisher_exact": {
                        "table": {
                            "shifted_failures": 3,
                            "shifted_successes": 1,
                            "unshifted_failures": 1,
                            "unshifted_successes": 3,
                        }
                    },
                    "matched_rule_counts": {"word_count": 4},
                    "relations": {
                        "target_relations_combined": {"linguistic_share": 0.625},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    rows = discover_model_linguistic_pattern_rows(results_dir)

    assert rows == [
        {
            "model": "model-a",
            "shift_rate_in_successful_games": 0.25,
            "shift_rate_in_unsuccessful_games": 0.75,
            "post_linguistic_no_recovery_rate_in_successful_games": 0.0,
            "post_linguistic_no_recovery_rate_in_unsuccessful_games": 0.5,
            "games_with_post_linguistic_recovery_opportunity": 3,
            "post_linguistic_no_recovery_fisher_exact": {
                "table": {
                    "unsuccessful_no_recovery": 1,
                    "unsuccessful_recovered": 1,
                    "successful_no_recovery": 0,
                    "successful_recovered": 1,
                }
            },
            "failure_fisher_exact": {
                "table": {
                    "shifted_failures": 3,
                    "shifted_successes": 1,
                    "unshifted_failures": 1,
                    "unshifted_successes": 3,
                }
            },
            "matched_rule_counts": {"word_count": 4},
            "target_relations_combined_linguistic_share": 0.625,
        }
    ]


def test_discover_pooled_linguistic_pattern_payload_reads_summary_artifact(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "error_analysis_summary.json").write_text(
        json.dumps(
            {
                "pooled_experiment_3_linguistic_shift": {
                    "status": "ok",
                    "matched_rule_counts": {"letter": 7, "word_count": 5},
                }
            }
        ),
        encoding="utf-8",
    )

    payload = discover_pooled_linguistic_pattern_payload(results_dir)

    assert payload == {
        "status": "ok",
        "matched_rule_counts": {"letter": 7, "word_count": 5},
    }


def test_discover_model_relation_metrics_uses_relation_aware_rules(tmp_path):
    model_dir = tmp_path / "results" / "model-a"
    model_dir.mkdir(parents=True)

    (model_dir / "game_results.json").write_text(
        json.dumps(
            [
                {
                    "game_id": 1,
                    "distance_group": "close",
                    "success": True,
                    "turns": [
                        {
                            "turn_number": 1,
                            "action": {"action": "test"},
                            "inferred_intention": "confirm",
                            "oracle_response": {"conforms": True},
                            "hypothesis_oracle_response": {"conforms": False},
                        },
                        {
                            "turn_number": 2,
                            "action": {"action": "test"},
                            "inferred_intention": "falsify",
                            "oracle_response": {"conforms": True},
                            "hypothesis_oracle_response": {"conforms": True},
                        },
                        {
                            "turn_number": 3,
                            "action": {"action": "test"},
                            "inferred_intention": "confirm",
                            "oracle_response": {"conforms": False},
                            "hypothesis_oracle_response": {"conforms": True},
                        },
                        {
                            "turn_number": 4,
                            "action": {"action": "test"},
                            "inferred_intention": "falsify",
                            "oracle_response": {"conforms": False},
                            "hypothesis_oracle_response": {"conforms": False},
                        },
                    ],
                },
                {
                    "game_id": 2,
                    "distance_group": "deep",
                    "success": False,
                    "turns": [
                        {
                            "turn_number": 1,
                            "action": {"action": "test"},
                            "inferred_intention": "confirm",
                            "oracle_response": {"conforms": False},
                            "hypothesis_oracle_response": {"conforms": True},
                        },
                        {
                            "turn_number": 2,
                            "action": {"action": "test"},
                            "inferred_intention": "confirm",
                            "oracle_response": {"conforms": True},
                            "hypothesis_oracle_response": {"conforms": True},
                        },
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )

    (model_dir / "turn_relation_annotations.json").write_text(
        json.dumps(
            {
                "annotations": [
                    {"game_id": 1, "turn_number": 1, "status": "ok", "relation": "hypothesis_included_in_target"},
                    {"game_id": 1, "turn_number": 2, "status": "ok", "relation": "hypothesis_included_in_target"},
                    {"game_id": 1, "turn_number": 3, "status": "ok", "relation": "partial_overlap"},
                    {"game_id": 1, "turn_number": 4, "status": "ok", "relation": "disjoint"},
                    {"game_id": 2, "turn_number": 1, "status": "ok", "relation": "target_included_in_hypothesis"},
                    {"game_id": 2, "turn_number": 2, "status": "ok", "relation": "hypothesis_included_in_target"},
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = discover_model_relation_metrics(tmp_path / "results")

    assert len(rows) == 1
    row = rows[0]
    assert row["model"] == "model-a"
    assert row["overall"]["avg_positive_testing_bias"] == 0.75
    assert row["overall"]["avg_positive_testing_bias_std"] == 0.25
    assert row["close"]["avg_positive_testing_bias"] == 0.5
    assert row["close"]["avg_positive_testing_bias_std"] == 0.0
    assert row["deep"]["avg_positive_testing_bias"] == 1.0
    assert row["deep"]["avg_positive_testing_bias_std"] == 0.0
    assert row["overall"]["avg_conclusive_falsification_rate"] == 0.5
    assert row["overall"]["avg_conclusive_falsification_rate_std"] == 0.0
    assert row["close"]["avg_conclusive_falsification_rate"] == 0.5
    assert row["close"]["avg_conclusive_falsification_rate_std"] == 0.0
    assert row["deep"]["avg_conclusive_falsification_rate"] == 0.5
    assert row["deep"]["avg_conclusive_falsification_rate_std"] == 0.0


def test_discover_model_relation_distributions_by_outcome_filters_games(tmp_path):
    model_dir = tmp_path / "results" / "model-a"
    model_dir.mkdir(parents=True)

    (model_dir / "game_results.json").write_text(
        json.dumps(
            [
                {
                    "game_id": 1,
                    "distance_group": "close",
                    "success": True,
                    "turns": [
                        {"turn_number": 1, "action": {"action": "test"}},
                        {"turn_number": 2, "action": {"action": "test"}},
                    ],
                },
                {
                    "game_id": 2,
                    "distance_group": "deep",
                    "success": False,
                    "turns": [
                        {"turn_number": 1, "action": {"action": "test"}},
                        {"turn_number": 2, "action": {"action": "test"}},
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )

    (model_dir / "turn_relation_annotations.json").write_text(
        json.dumps(
            {
                "annotations": [
                    {"game_id": 1, "turn_number": 1, "status": "ok", "relation": "identical"},
                    {"game_id": 1, "turn_number": 2, "status": "ok", "relation": "partial_overlap"},
                    {"game_id": 2, "turn_number": 1, "status": "ok", "relation": "disjoint"},
                    {"game_id": 2, "turn_number": 2, "status": "ok", "relation": "disjoint"},
                ]
            }
        ),
        encoding="utf-8",
    )

    successful_rows = discover_model_relation_distributions_by_outcome(tmp_path / "results", success=True)
    unsuccessful_rows = discover_model_relation_distributions_by_outcome(tmp_path / "results", success=False)

    assert successful_rows == [
        {
            "model": "model-a",
            "total_ok_annotations": 2,
            "relation_distribution": {
                "identical": 0.5,
                "disjoint": 0.0,
                "partial_overlap": 0.5,
                "hypothesis_included_in_target": 0.0,
                "target_included_in_hypothesis": 0.0,
            },
        }
    ]
    assert unsuccessful_rows == [
        {
            "model": "model-a",
            "total_ok_annotations": 2,
            "relation_distribution": {
                "identical": 0.0,
                "disjoint": 1.0,
                "partial_overlap": 0.0,
                "hypothesis_included_in_target": 0.0,
                "target_included_in_hypothesis": 0.0,
            },
        }
    ]


def test_discover_model_success_bias_correlation_rows_pairs_success_and_positive_testing_bias(tmp_path):
    model_dir = tmp_path / "results" / "model-a"
    model_dir.mkdir(parents=True)

    (model_dir / "aggregate_results.json").write_text(
        json.dumps(
            {
                "overall": {"success_rate": 0.75},
                "by_distance_group": {},
            }
        ),
        encoding="utf-8",
    )

    (model_dir / "game_results.json").write_text(
        json.dumps(
            [
                {
                    "game_id": 1,
                    "distance_group": "close",
                    "success": True,
                    "turns": [
                        {
                            "turn_number": 1,
                            "action": {"action": "test"},
                            "inferred_intention": "confirm",
                        },
                        {
                            "turn_number": 2,
                            "action": {"action": "test"},
                            "inferred_intention": "falsify",
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    (model_dir / "turn_relation_annotations.json").write_text(
        json.dumps(
            {
                "annotations": [
                    {"game_id": 1, "turn_number": 1, "status": "ok", "relation": "hypothesis_included_in_target"},
                    {"game_id": 1, "turn_number": 2, "status": "ok", "relation": "hypothesis_included_in_target"},
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = discover_model_success_bias_correlation_rows(tmp_path / "results")

    assert rows == [
        {
            "model": "model-a",
            "success_rate": 0.75,
            "avg_positive_testing_bias": 0.5,
        }
    ]


def test_discover_model_bias_falsification_correlation_rows_pairs_bias_and_falsification(tmp_path):
    model_dir = tmp_path / "results" / "model-a"
    model_dir.mkdir(parents=True)

    (model_dir / "game_results.json").write_text(
        json.dumps(
            [
                {
                    "game_id": 1,
                    "distance_group": "close",
                    "success": True,
                    "turns": [
                        {
                            "turn_number": 1,
                            "action": {"action": "test"},
                            "inferred_intention": "confirm",
                            "oracle_response": {"conforms": True},
                            "hypothesis_oracle_response": {"conforms": False},
                        },
                        {
                            "turn_number": 2,
                            "action": {"action": "test"},
                            "inferred_intention": "falsify",
                            "oracle_response": {"conforms": True},
                            "hypothesis_oracle_response": {"conforms": True},
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    (model_dir / "turn_relation_annotations.json").write_text(
        json.dumps(
            {
                "annotations": [
                    {"game_id": 1, "turn_number": 1, "status": "ok", "relation": "hypothesis_included_in_target"},
                    {"game_id": 1, "turn_number": 2, "status": "ok", "relation": "hypothesis_included_in_target"},
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = discover_model_bias_falsification_correlation_rows(tmp_path / "results")

    assert rows == [
        {
            "model": "model-a",
            "avg_positive_testing_bias": 0.5,
            "avg_conclusive_falsification_rate": 0.5,
        }
    ]


def test_discover_model_target_property_metrics_returns_core_metric_slices(tmp_path):
    model_dir = tmp_path / "results" / "model-a"
    model_dir.mkdir(parents=True)

    (model_dir / "game_results.json").write_text(
        json.dumps(
            [
                {
                    "game_id": 1,
                    "target_property": "animal",
                    "distance_group": "close",
                    "success": True,
                    "turns_to_solution": 3,
                    "turns": [
                        {
                            "turn_number": 1,
                            "action": {"action": "test"},
                            "inferred_intention": "falsify",
                        },
                        {
                            "turn_number": 2,
                            "action": {"action": "guess"},
                        },
                    ],
                },
                {
                    "game_id": 2,
                    "target_property": "animal",
                    "distance_group": "deep",
                    "success": False,
                    "turns": [
                        {
                            "turn_number": 1,
                            "action": {"action": "guess"},
                        },
                    ],
                },
                {
                    "game_id": 3,
                    "target_property": "plant",
                    "distance_group": "close",
                    "success": True,
                    "turns_to_solution": 2,
                    "turns": [
                        {
                            "turn_number": 1,
                            "action": {"action": "test"},
                            "inferred_intention": "confirm",
                        },
                        {
                            "turn_number": 2,
                            "action": {"action": "guess"},
                        },
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )

    rows = discover_model_target_property_metrics(tmp_path / "results")

    assert len(rows) == 1
    row = rows[0]
    assert row["model"] == "model-a"
    assert row["by_target_property"]["animal"]["total_games"] == 2
    assert row["by_target_property"]["animal"]["success_rate"] == 0.5
    assert row["by_target_property"]["animal"]["avg_turns_to_solution_std"] == 0.0
    assert row["by_target_property"]["plant"]["total_games"] == 1
    assert row["by_target_property"]["plant"]["success_rate"] == 1.0
    assert row["by_target_property"]["plant"]["avg_turns_to_solution_std"] == 0.0
    assert "food" not in row["by_target_property"]


def test_discover_model_target_property_relation_metrics_returns_conclusive_falsification_slices(tmp_path):
    model_dir = tmp_path / "results" / "model-a"
    model_dir.mkdir(parents=True)

    (model_dir / "game_results.json").write_text(
        json.dumps(
            [
                {
                    "game_id": 1,
                    "target_property": "animal",
                    "distance_group": "close",
                    "success": True,
                    "turns": [
                        {
                            "turn_number": 1,
                            "action": {"action": "test"},
                            "inferred_intention": "falsify",
                            "oracle_response": {"conforms": True},
                            "hypothesis_oracle_response": {"conforms": False},
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    (model_dir / "turn_relation_annotations.json").write_text(
        json.dumps(
            {
                "annotations": [
                    {
                        "game_id": 1,
                        "turn_number": 1,
                        "status": "ok",
                        "relation": "hypothesis_included_in_target",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = discover_model_target_property_relation_metrics(tmp_path / "results")

    assert len(rows) == 1
    row = rows[0]
    assert row["by_target_property"]["animal"]["avg_conclusive_falsification_rate"] == 1.0
    assert row["by_target_property"]["animal"]["avg_conclusive_falsification_rate_std"] == 0.0


def test_discover_model_target_property_guess_counts_returns_target_property_slices(tmp_path):
    model_dir = tmp_path / "results" / "model-a"
    model_dir.mkdir(parents=True)

    (model_dir / "game_results.json").write_text(
        json.dumps(
            [
                {
                    "game_id": 1,
                    "target_property": "animal",
                    "distance_group": "close",
                    "success": True,
                    "turns": [
                        {"turn_number": 1, "action": {"action": "guess"}},
                    ],
                },
                {
                    "game_id": 2,
                    "target_property": "animal",
                    "distance_group": "deep",
                    "success": False,
                    "turns": [
                        {"turn_number": 1, "action": {"action": "guess"}},
                        {"turn_number": 2, "action": {"action": "guess"}},
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )

    rows = discover_model_target_property_guess_counts(tmp_path / "results")

    assert len(rows) == 1
    row = rows[0]
    assert row["by_target_property"]["animal"]["avg_guess_count"] == 1.5
    assert row["by_target_property"]["animal"]["avg_guess_count_std"] == 0.5


def test_generate_plots_writes_success_and_unsuccessful_relation_distribution_plots(tmp_path):
    results_dir = tmp_path / "results"
    model_dir = results_dir / "model-a"
    model_dir.mkdir(parents=True)

    (model_dir / "aggregate_results.json").write_text(
        json.dumps(
            {
                "overall": {
                    "success_rate": 0.5,
                    "avg_turns_to_solution": 5.0,
                    "avg_turns_to_first_negative_test": 2.0,
                    "negative_test_before_first_guess_rate": 0.5,
                    "avg_positive_testing_bias": 0.5,
                    "avg_positive_testing_queries": 2.0,
                    "avg_negative_testing_queries": 2.0,
                    "avg_conclusive_falsification_rate": 0.5,
                },
                "by_distance_group": {
                    "close": {
                        "success_rate": 1.0,
                        "avg_turns_to_solution": 4.0,
                        "avg_turns_to_first_negative_test": 1.0,
                        "negative_test_before_first_guess_rate": 1.0,
                        "avg_positive_testing_bias": 0.5,
                        "avg_positive_testing_queries": 2.0,
                        "avg_negative_testing_queries": 2.0,
                        "avg_conclusive_falsification_rate": 1.0,
                    },
                    "deep": {
                        "success_rate": 0.0,
                        "avg_turns_to_solution": 6.0,
                        "avg_turns_to_first_negative_test": 3.0,
                        "negative_test_before_first_guess_rate": 0.0,
                        "avg_positive_testing_bias": 0.5,
                        "avg_positive_testing_queries": 2.0,
                        "avg_negative_testing_queries": 2.0,
                        "avg_conclusive_falsification_rate": 0.0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    (model_dir / "game_results.json").write_text(
        json.dumps(
            [
                {
                    "game_id": 1,
                    "target_property": "animal",
                    "distance_group": "close",
                    "success": True,
                    "turns": [
                        {
                            "turn_number": 1,
                            "action": {"action": "test"},
                            "inferred_intention": "confirm",
                            "oracle_response": {"conforms": True},
                            "hypothesis_oracle_response": {"conforms": False},
                        },
                        {
                            "turn_number": 2,
                            "action": {"action": "guess"},
                        },
                    ],
                },
                {
                    "game_id": 2,
                    "target_property": "plant",
                    "distance_group": "deep",
                    "success": False,
                    "turns": [
                        {
                            "turn_number": 1,
                            "action": {"action": "test"},
                            "inferred_intention": "falsify",
                            "oracle_response": {"conforms": False},
                            "hypothesis_oracle_response": {"conforms": True},
                        },
                        {
                            "turn_number": 2,
                            "action": {"action": "test"},
                            "inferred_intention": "confirm",
                            "oracle_response": {"conforms": False},
                            "hypothesis_oracle_response": {"conforms": False},
                        },
                        {"turn_number": 3, "action": {"action": "guess"}},
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )

    (model_dir / "turn_relation_annotations.json").write_text(
        json.dumps(
            {
                "annotations": [
                    {
                        "game_id": 1,
                        "turn_number": 1,
                        "status": "ok",
                        "relation": "hypothesis_included_in_target",
                    },
                    {
                        "game_id": 2,
                        "turn_number": 1,
                        "status": "ok",
                        "relation": "hypothesis_included_in_target",
                    },
                    {
                        "game_id": 2,
                        "turn_number": 2,
                        "status": "ok",
                        "relation": "disjoint",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    (model_dir / "error_analysis_results.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "status": "ok",
                "experiment_1_distributional_difference": {
                    "status": "ok",
                    "relations": {
                        "disjoint": {"mean_difference": 0.25, "cliffs_delta": 0.50, "mann_whitney_p_value": 0.01},
                        "partial_overlap": {"mean_difference": 0.10, "cliffs_delta": 0.20, "mann_whitney_p_value": 0.22},
                    },
                },
                "experiment_3_linguistic_shift": {
                    "status": "ok",
                    "shift_rate_in_successful_games": 0.0,
                    "shift_rate_in_unsuccessful_games": 1.0,
                    "post_linguistic_no_recovery_rate_in_successful_games": 0.0,
                    "post_linguistic_no_recovery_rate_in_unsuccessful_games": 1.0,
                    "games_with_post_linguistic_recovery_opportunity": 2,
                    "post_linguistic_no_recovery_fisher_exact": {
                        "table": {
                            "unsuccessful_no_recovery": 1,
                            "unsuccessful_recovered": 0,
                            "successful_no_recovery": 0,
                            "successful_recovered": 1,
                        }
                    },
                    "matched_rule_counts": {"word_count": 2},
                    "relations": {
                        "target_relations_combined": {"linguistic_share": 0.5},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    written = generate_plots(str(results_dir), str(tmp_path / "plots"))

    assert tmp_path.joinpath("plots", "successful_only_relation_distribution.png") in written
    assert tmp_path.joinpath("plots", "unsuccessful_only_relation_distribution.png") in written
    assert tmp_path.joinpath("plots", "overall_error_analysis_distributional_difference.png") in written
    assert tmp_path.joinpath("plots", "by_target_property_success_rate.png") in written
    assert tmp_path.joinpath("plots", "by_target_property_avg_confirmation_bias.png") in written
    assert tmp_path.joinpath("plots", "by_target_property_avg_conclusive_falsification_rate.png") in written
    assert tmp_path.joinpath("plots", "by_target_property_avg_guesses_per_game.png") in written
    assert tmp_path.joinpath("plots", "overall_avg_confirmation_bias.png") in written
    assert tmp_path.joinpath("plots", "overall_avg_conclusive_falsification_rate.png") in written
    assert tmp_path.joinpath("plots", "overall_error_analysis_linguistic_pattern.png") in written


def test_generate_plots_writes_bias_vs_conclusive_falsification_scatter_when_two_models_exist(tmp_path):
    results_dir = tmp_path / "results"
    for model_name, fixture_spec in {
        "model-a": {
            "success_rate": 0.5,
            "turns": [
                ("confirm", True, False),
                ("falsify", True, True),
            ],
        },
        "model-b": {
            "success_rate": 0.75,
            "turns": [
                ("confirm", False, False),
                ("confirm", True, True),
            ],
        },
    }.items():
        model_dir = results_dir / model_name
        model_dir.mkdir(parents=True)

        (model_dir / "aggregate_results.json").write_text(
            json.dumps(
                {
                    "overall": {"success_rate": fixture_spec["success_rate"]},
                    "by_distance_group": {},
                }
            ),
            encoding="utf-8",
        )

        turns = []
        annotations = []
        for index, (intention, target_conforms, hypothesis_conforms) in enumerate(fixture_spec["turns"], start=1):
            turns.append(
                {
                    "turn_number": index,
                    "action": {"action": "test"},
                    "inferred_intention": intention,
                    "oracle_response": {"conforms": target_conforms},
                    "hypothesis_oracle_response": {"conforms": hypothesis_conforms},
                }
            )
            annotations.append(
                {
                    "game_id": 1,
                    "turn_number": index,
                    "status": "ok",
                    "relation": "hypothesis_included_in_target",
                }
            )

        (model_dir / "game_results.json").write_text(
            json.dumps(
                [
                    {
                        "game_id": 1,
                        "distance_group": "close",
                        "success": True,
                        "turns": turns,
                    }
                ]
            ),
            encoding="utf-8",
        )

        (model_dir / "turn_relation_annotations.json").write_text(
            json.dumps({"annotations": annotations}),
            encoding="utf-8",
        )

    written = generate_plots(str(results_dir), str(tmp_path / "plots"))

    assert tmp_path.joinpath("plots", "overall_confirmation_bias_vs_conclusive_falsification_rate.png") in written


def test_spearman_correlation_with_p_value_returns_perfect_monotonic_result():
    rho, p_value = _spearman_correlation_with_p_value([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])

    assert rho == 1.0
    assert isinstance(p_value, float)
    assert p_value >= 0.0


def test_linear_regression_fit_with_confidence_interval_returns_full_band():
    fit = _linear_regression_fit_with_confidence_interval(
        [10.0, 20.0, 30.0, 40.0],
        [15.0, 25.0, 35.0, 45.0],
        [0.0, 50.0, 100.0],
    )

    assert fit is not None
    fitted_y, lower_band, upper_band = fit
    assert len(fitted_y) == 3
    assert len(lower_band) == 3
    assert len(upper_band) == 3
    assert all(lower <= fitted <= upper for lower, fitted, upper in zip(lower_band, fitted_y, upper_band))
