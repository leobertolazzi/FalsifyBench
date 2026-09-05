import csv
import json

from src.annotation.export_pairs import build_annotation_rows, export_annotation_pairs


def _write_model_results(model_dir, game_results):
    model_dir.mkdir(parents=True)
    (model_dir / "game_results.json").write_text(json.dumps(game_results), encoding="utf-8")


def _read_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_build_annotation_rows_is_deterministic_for_fixed_seed(tmp_path):
    results_dir = tmp_path / "results"
    model_dir = results_dir / "model-a"
    _write_model_results(
        model_dir,
        [
            {
                "game_id": 10,
                "target_property": "animal",
                "turns": [
                    {
                        "turn_number": 1,
                        "action": {
                            "action": "test",
                            "items": ["sparrow", "eagle", "hawk"],
                            "hypothesis": "bird",
                        },
                        "oracle_response": {"conforms": True},
                        "hypothesis_oracle_response": {"conforms": True},
                    },
                    {
                        "turn_number": 2,
                        "action": {
                            "action": "test",
                            "items": ["salmon", "oak", "whale"],
                            "hypothesis": "bird",
                        },
                        "oracle_response": {"conforms": False},
                        "hypothesis_oracle_response": {"conforms": False},
                    },
                    {
                        "turn_number": 3,
                        "action": {
                            "action": "guess",
                            "property": "animal",
                        },
                        "oracle_response": {"correct": True},
                    },
                    {
                        "turn_number": 4,
                        "action": {
                            "action": "guess",
                            "property": "bird",
                        },
                        "oracle_response": {"correct": False},
                    },
                ],
            }
        ],
    )

    results_files = [model_dir / "game_results.json"]
    first = build_annotation_rows(results_files, seed=11)
    second = build_annotation_rows(results_files, seed=11)
    third = build_annotation_rows(results_files, seed=12)

    assert first == second
    assert first["test_turn_target_judgment"][0]["turn_number"] == 2
    assert first["guess_target_equivalence"][0]["turn_number"] == 4
    assert third != first
    assert third["guess_target_equivalence"][0]["turn_number"] == 3


def test_export_annotation_pairs_writes_llm_and_human_csv_pairs(tmp_path):
    results_dir = tmp_path / "results"
    model_a = results_dir / "model-a"
    model_b = results_dir / "model-b"

    _write_model_results(
        model_a,
        [
            {
                "game_id": 1,
                "target_property": "animal",
                "turns": [
                    {
                        "turn_number": 1,
                        "action": {
                            "action": "test",
                            "items": ["dog", "cat", "otter"],
                            "hypothesis": "mammal",
                        },
                        "oracle_response": {"conforms": True},
                        "hypothesis_oracle_response": {"conforms": True},
                    },
                    {
                        "turn_number": 2,
                        "action": {"action": "guess", "property": "animal"},
                        "oracle_response": {"correct": True},
                    },
                ],
            }
        ],
    )
    _write_model_results(
        model_b,
        [
            {
                "game_id": 2,
                "target_property": "plant",
                "turns": [
                    {
                        "turn_number": 5,
                        "action": {
                            "action": "test",
                            "items": ["oak", "rose", "moss"],
                            "hypothesis": "tree",
                        },
                        "oracle_response": {"conforms": True},
                        "hypothesis_oracle_response": {"conforms": False},
                    },
                    {
                        "turn_number": 6,
                        "action": {"action": "guess", "property": "tree"},
                        "oracle_response": {"correct": False},
                    },
                ],
            }
        ],
    )

    output_dir = tmp_path / "annotation_pairs"
    written = export_annotation_pairs(results_dir, output_dir, seed=7, recursive=False)

    assert len(written) == 6

    test_llm_rows = _read_csv_rows(output_dir / "test_turn_target_judgment_llm.csv")
    test_human_rows = _read_csv_rows(output_dir / "test_turn_target_judgment_human.csv")
    hypothesis_llm_rows = _read_csv_rows(output_dir / "test_turn_hypothesis_judgment_llm.csv")
    guess_llm_rows = _read_csv_rows(output_dir / "guess_target_equivalence_llm.csv")
    guess_human_rows = _read_csv_rows(output_dir / "guess_target_equivalence_human.csv")

    assert [row["model"] for row in test_llm_rows] == ["model-a", "model-b"]
    assert test_llm_rows[0]["triple"] == "dog | cat | otter"
    assert test_llm_rows[0]["target_property"] == "animal"
    assert test_llm_rows[0]["judgment"] == "yes"
    assert test_human_rows[0]["judgment"] == ""

    assert hypothesis_llm_rows[1]["current_hypothesis"] == "tree"
    assert hypothesis_llm_rows[1]["judgment"] == "no"

    assert guess_llm_rows[0]["guess"] == "animal"
    assert guess_llm_rows[1]["guess"] == "tree"
    assert guess_llm_rows[1]["judgment"] == "no"
    assert guess_human_rows[1]["judgment"] == ""