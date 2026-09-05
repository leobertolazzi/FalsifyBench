import csv
import json

from src.annotation.agreement import compute_agreement_summary, compute_pair_agreement


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_compute_pair_agreement_handles_missing_and_mismatched_labels(tmp_path):
    fieldnames = ["model", "game_id", "turn_number", "guess", "target_property", "judgment"]
    llm_path = tmp_path / "guess_target_equivalence_llm.csv"
    human_path = tmp_path / "guess_target_equivalence_human.csv"

    llm_rows = [
        {
            "model": "model-a",
            "game_id": "1",
            "turn_number": "1",
            "guess": "animal",
            "target_property": "animal",
            "judgment": "yes",
        },
        {
            "model": "model-a",
            "game_id": "2",
            "turn_number": "1",
            "guess": "bird",
            "target_property": "animal",
            "judgment": "no",
        },
        {
            "model": "model-a",
            "game_id": "3",
            "turn_number": "1",
            "guess": "artifact",
            "target_property": "artifact",
            "judgment": "yes",
        },
    ]
    human_rows = [
        {
            "model": "model-a",
            "game_id": "1",
            "turn_number": "1",
            "guess": "animal",
            "target_property": "animal",
            "judgment": "yes",
        },
        {
            "model": "model-a",
            "game_id": "2",
            "turn_number": "1",
            "guess": "bird",
            "target_property": "animal",
            "judgment": "yes",
        },
        {
            "model": "model-a",
            "game_id": "3",
            "turn_number": "1",
            "guess": "artifact",
            "target_property": "artifact",
            "judgment": "-",
        },
    ]

    _write_csv(llm_path, fieldnames, llm_rows)
    _write_csv(human_path, fieldnames, human_rows)

    summary = compute_pair_agreement(llm_path, human_path)

    assert summary["pair_name"] == "guess_target_equivalence"
    assert summary["rows_compared"] == 2
    assert summary["matches"] == 1
    assert summary["mismatches"] == 1
    assert summary["rows_with_missing_human_judgment"] == 1
    assert summary["percent_agreement"] == 0.5
    assert summary["cohen_kappa"] == 0.0
    assert summary["by_model"] == {
        "model-a": {
            "rows_compared": 2,
            "matches": 1,
            "mismatches": 1,
            "percent_agreement": 0.5,
            "cohen_kappa": 0.0,
            "llm_label_counts": {"no": 1, "yes": 1},
            "human_label_counts": {"yes": 2},
        }
    }
    assert summary["by_target_property"] == {
        "animal": {
            "rows_compared": 2,
            "matches": 1,
            "mismatches": 1,
            "percent_agreement": 0.5,
            "cohen_kappa": 0.0,
            "llm_label_counts": {"no": 1, "yes": 1},
            "human_label_counts": {"yes": 2},
        }
    }
    assert summary["disagreements"] == [
        {
            "model": "model-a",
            "game_id": "2",
            "turn_number": "1",
            "guess": "bird",
            "target_property": "animal",
            "llm_judgment": "no",
            "human_judgment": "yes",
        }
    ]


def test_compute_agreement_summary_aggregates_multiple_pairs(tmp_path):
    fieldnames = ["model", "game_id", "turn_number", "triple", "target_property", "judgment"]
    _write_csv(
        tmp_path / "test_turn_target_judgment_llm.csv",
        fieldnames,
        [
            {
                "model": "model-a",
                "game_id": "1",
                "turn_number": "1",
                "triple": "dog | cat | otter",
                "target_property": "animal",
                "judgment": "yes",
            }
        ],
    )
    _write_csv(
        tmp_path / "test_turn_target_judgment_human.csv",
        fieldnames,
        [
            {
                "model": "model-a",
                "game_id": "1",
                "turn_number": "1",
                "triple": "dog | cat | otter",
                "target_property": "animal",
                "judgment": "yes",
            }
        ],
    )

    guess_fieldnames = ["model", "game_id", "turn_number", "guess", "target_property", "judgment"]
    _write_csv(
        tmp_path / "guess_target_equivalence_llm.csv",
        guess_fieldnames,
        [
            {
                "model": "model-b",
                "game_id": "2",
                "turn_number": "4",
                "guess": "tree",
                "target_property": "plant",
                "judgment": "no",
            }
        ],
    )
    _write_csv(
        tmp_path / "guess_target_equivalence_human.csv",
        guess_fieldnames,
        [
            {
                "model": "model-b",
                "game_id": "2",
                "turn_number": "4",
                "guess": "tree",
                "target_property": "plant",
                "judgment": "yes",
            }
        ],
    )

    summary = compute_agreement_summary(tmp_path)

    assert [pair["pair_name"] for pair in summary["pair_summaries"]] == [
        "guess_target_equivalence",
        "test_turn_target_judgment",
    ]
    assert summary["overall"]["pairs_found"] == 2
    assert summary["overall"]["rows_compared"] == 2
    assert summary["overall"]["matches"] == 1
    assert summary["overall"]["mismatches"] == 1
    assert summary["overall"]["micro_percent_agreement"] == 0.5
    assert summary["overall"]["macro_percent_agreement"] == 0.5
    assert summary["overall"]["macro_cohen_kappa"] == 0.5
    assert summary["overall"]["by_file"] == {
        "guess_target_equivalence": {
            "rows_compared": 1,
            "matches": 0,
            "mismatches": 1,
            "percent_agreement": 0.0,
            "cohen_kappa": 0.0,
            "llm_label_counts": {"no": 1},
            "human_label_counts": {"yes": 1},
        },
        "test_turn_target_judgment": {
            "rows_compared": 1,
            "matches": 1,
            "mismatches": 0,
            "percent_agreement": 1.0,
            "cohen_kappa": 1.0,
            "llm_label_counts": {"yes": 1},
            "human_label_counts": {"yes": 1},
        },
    }
    assert summary["overall"]["by_model"] == {
        "model-a": {
            "rows_compared": 1,
            "matches": 1,
            "mismatches": 0,
            "percent_agreement": 1.0,
            "cohen_kappa": 1.0,
            "llm_label_counts": {"yes": 1},
            "human_label_counts": {"yes": 1},
            "pair_name": {
                "test_turn_target_judgment": {
                    "rows_compared": 1,
                    "matches": 1,
                    "mismatches": 0,
                    "percent_agreement": 1.0,
                    "cohen_kappa": 1.0,
                    "llm_label_counts": {"yes": 1},
                    "human_label_counts": {"yes": 1},
                }
            },
        },
        "model-b": {
            "rows_compared": 1,
            "matches": 0,
            "mismatches": 1,
            "percent_agreement": 0.0,
            "cohen_kappa": 0.0,
            "llm_label_counts": {"no": 1},
            "human_label_counts": {"yes": 1},
            "pair_name": {
                "guess_target_equivalence": {
                    "rows_compared": 1,
                    "matches": 0,
                    "mismatches": 1,
                    "percent_agreement": 0.0,
                    "cohen_kappa": 0.0,
                    "llm_label_counts": {"no": 1},
                    "human_label_counts": {"yes": 1},
                }
            },
        },
    }
    assert summary["overall"]["by_target_property"] == {
        "animal": {
            "rows_compared": 1,
            "matches": 1,
            "mismatches": 0,
            "percent_agreement": 1.0,
            "cohen_kappa": 1.0,
            "llm_label_counts": {"yes": 1},
            "human_label_counts": {"yes": 1},
        },
        "plant": {
            "rows_compared": 1,
            "matches": 0,
            "mismatches": 1,
            "percent_agreement": 0.0,
            "cohen_kappa": 0.0,
            "llm_label_counts": {"no": 1},
            "human_label_counts": {"yes": 1},
        },
    }


def test_compute_agreement_summary_skips_target_property_breakdown_when_column_missing(tmp_path):
    fieldnames = ["model", "game_id", "turn_number", "triple", "current_hypothesis", "judgment"]
    _write_csv(
        tmp_path / "test_turn_hypothesis_judgment_llm.csv",
        fieldnames,
        [
            {
                "model": "model-c",
                "game_id": "9",
                "turn_number": "2",
                "triple": "oak | birch | pine",
                "current_hypothesis": "tree",
                "judgment": "yes",
            }
        ],
    )
    _write_csv(
        tmp_path / "test_turn_hypothesis_judgment_human.csv",
        fieldnames,
        [
            {
                "model": "model-c",
                "game_id": "9",
                "turn_number": "2",
                "triple": "oak | birch | pine",
                "current_hypothesis": "tree",
                "judgment": "no",
            }
        ],
    )

    summary = compute_agreement_summary(tmp_path)

    assert summary["pair_summaries"][0]["by_target_property"] == {}
    assert summary["overall"]["by_target_property"] == {}
    assert summary["overall"]["by_file"] == {
        "test_turn_hypothesis_judgment": {
            "rows_compared": 1,
            "matches": 0,
            "mismatches": 1,
            "percent_agreement": 0.0,
            "cohen_kappa": 0.0,
            "llm_label_counts": {"yes": 1},
            "human_label_counts": {"no": 1},
        }
    }
    assert summary["overall"]["by_model"] == {
        "model-c": {
            "rows_compared": 1,
            "matches": 0,
            "mismatches": 1,
            "percent_agreement": 0.0,
            "cohen_kappa": 0.0,
            "llm_label_counts": {"yes": 1},
            "human_label_counts": {"no": 1},
            "pair_name": {
                "test_turn_hypothesis_judgment": {
                    "rows_compared": 1,
                    "matches": 0,
                    "mismatches": 1,
                    "percent_agreement": 0.0,
                    "cohen_kappa": 0.0,
                    "llm_label_counts": {"yes": 1},
                    "human_label_counts": {"no": 1},
                }
            },
        }
    }