import json

from src.benchmark.evaluation import (
    GameEvaluator,
)


def test_aggregate_metrics_use_all_test_turns_for_positive_testing_bias(tmp_path):
    annotation_file = tmp_path / "turn_relation_annotations.json"
    annotation_file.write_text(
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
                        "game_id": 1,
                        "turn_number": 2,
                        "status": "ok",
                        "relation": "partial_overlap",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    evaluator = GameEvaluator()
    evaluator.load_relation_annotations(str(annotation_file))
    evaluator.add_result(
        {
            "game_id": 1,
            "distance_group": "close",
            "success": True,
            "turns_to_solution": 4,
            "turns": [
                {
                    "turn_number": 1,
                    "action": {"action": "test"},
                    "inferred_intention": "falsify",
                    "oracle_response": {"conforms": True},
                    "hypothesis_oracle_response": {"conforms": False},
                },
                {
                    "turn_number": 2,
                    "action": {"action": "test"},
                    "inferred_intention": "confirm",
                    "oracle_response": {"conforms": False},
                    "hypothesis_oracle_response": {"conforms": False},
                },
                {
                    "turn_number": 3,
                    "action": {"action": "guess"},
                },
            ],
        }
    )

    aggregate = evaluator.compute_aggregate_metrics("close")
    metrics = evaluator.compute_game_metrics(evaluator.results[0])
    assert metrics["positive_testing_bias"] == 0.5
    assert metrics["filtered_positive_testing_bias"] == 0.0
    assert metrics["conclusive_falsification_rate"] == 0.5
    assert aggregate["avg_positive_testing_bias"] == 0.5
    assert aggregate["avg_positive_testing_bias_std"] == 0.0
    assert aggregate["avg_conclusive_falsification_rate"] == 0.5
    assert aggregate["avg_conclusive_falsification_rate_std"] == 0.0


def test_aggregate_metrics_include_population_std_for_average_metrics():
    evaluator = GameEvaluator()
    evaluator.add_result(
        {
            "game_id": 1,
            "distance_group": "close",
            "success": True,
            "turns_to_solution": 4,
            "turns": [
                {"turn_number": 1, "action": {"action": "test"}, "inferred_intention": "confirm"},
                {"turn_number": 2, "action": {"action": "guess"}},
            ],
        }
    )
    evaluator.add_result(
        {
            "game_id": 2,
            "distance_group": "close",
            "success": True,
            "turns_to_solution": 8,
            "turns": [
                {"turn_number": 1, "action": {"action": "test"}, "inferred_intention": "falsify"},
                {"turn_number": 2, "action": {"action": "test"}, "inferred_intention": "falsify"},
                {"turn_number": 3, "action": {"action": "guess"}},
            ],
        }
    )

    aggregate = evaluator.compute_aggregate_metrics("close")

    assert aggregate["avg_turns_to_solution"] == 6.0
    assert aggregate["avg_turns_to_solution_std"] == 2.0
    assert aggregate["avg_positive_testing_bias"] == 0.5
    assert aggregate["avg_positive_testing_bias_std"] == 0.5
