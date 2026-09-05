import json

from src.annotation.plot_agreement import generate_annotation_plots


def test_generate_annotation_plots_writes_expected_pngs_from_summary(tmp_path):
    summary = {
        "input_dir": str(tmp_path / "annotation_pairs"),
        "pair_summaries": [
            {
                "pair_name": "guess_target_equivalence",
                "percent_agreement": 0.9,
                "cohen_kappa": 0.8,
                "by_target_property": {
                    "animal": {"percent_agreement": 0.95, "cohen_kappa": 0.9},
                    "artifact": {"percent_agreement": 0.85, "cohen_kappa": 0.7},
                },
            },
            {
                "pair_name": "test_turn_target_judgment",
                "percent_agreement": 0.8,
                "cohen_kappa": 0.6,
                "by_target_property": {
                    "animal": {"percent_agreement": 0.75, "cohen_kappa": 0.55},
                    "artifact": {"percent_agreement": 0.9, "cohen_kappa": 0.7},
                },
            },
        ],
        "overall": {
            "by_model": {
                "model-a": {
                    "percent_agreement": 0.865,
                    "cohen_kappa": 0.715,
                    "pair_name": {
                        "guess_target_equivalence": {
                            "percent_agreement": 0.92,
                            "cohen_kappa": 0.82,
                        },
                        "test_turn_target_judgment": {
                            "percent_agreement": 0.81,
                            "cohen_kappa": 0.61,
                        },
                    }
                },
                "model-b": {
                    "percent_agreement": 0.835,
                    "cohen_kappa": 0.655,
                    "pair_name": {
                        "guess_target_equivalence": {
                            "percent_agreement": 0.88,
                            "cohen_kappa": 0.73,
                        },
                        "test_turn_target_judgment": {
                            "percent_agreement": 0.79,
                            "cohen_kappa": 0.58,
                        },
                    }
                },
            }
        },
    }

    summary_path = tmp_path / "agreement_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    results_dir = tmp_path / "results"
    for model_name, success_rate in (("model-a", 0.7), ("model-b", 0.5)):
        model_dir = results_dir / model_name
        model_dir.mkdir(parents=True)
        (model_dir / "aggregate_results.json").write_text(
            json.dumps({"overall": {"success_rate": success_rate}, "by_distance_group": {}}),
            encoding="utf-8",
        )

    output_dir = tmp_path / "annotation_plots"
    written = generate_annotation_plots(
        summary_path=str(summary_path),
        output_dir=str(output_dir),
    )

    expected_names = {
        "action_agreement_by_model.png",
        "action_kappa_by_model.png",
    }
    assert {path.name for path in written} == expected_names
    assert all(path.exists() for path in written)


def test_generate_annotation_plots_returns_empty_when_no_pair_summaries(tmp_path):
    summary_path = tmp_path / "agreement_summary.json"
    summary_path.write_text(json.dumps({"pair_summaries": []}), encoding="utf-8")

    written = generate_annotation_plots(summary_path=str(summary_path), output_dir=str(tmp_path / "annotation_plots"))

    assert written == []
