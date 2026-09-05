import csv
import json

from src.analysis.mixed_effects import run_game_level_mixed_effects_analysis


def _write_csv(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_run_game_level_mixed_effects_analysis_writes_joined_rows_and_summary(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    annotation_dir = tmp_path / "annotation_pairs"
    annotation_dir.mkdir()

    for model_name, games in {
        "model-a": [
            {
                "game_id": 1,
                "target_property": "animal",
                "distance_group": "close",
                "distance": 2,
                "success": True,
                "turns": [
                    {"turn_number": 1, "action": {"action": "test"}, "inferred_intention": "confirm"},
                    {"turn_number": 2, "action": {"action": "test"}, "inferred_intention": "falsify"},
                ],
            },
            {
                "game_id": 2,
                "target_property": "plant",
                "distance_group": "deep",
                "distance": 4,
                "success": False,
                "turns": [
                    {"turn_number": 1, "action": {"action": "test"}, "inferred_intention": "confirm"},
                ],
            },
        ],
        "model-b": [
            {
                "game_id": 1,
                "target_property": "animal",
                "distance_group": "close",
                "distance": 2,
                "success": True,
                "turns": [
                    {"turn_number": 1, "action": {"action": "test"}, "inferred_intention": "falsify"},
                    {"turn_number": 2, "action": {"action": "test"}, "inferred_intention": "falsify"},
                ],
            },
            {
                "game_id": 2,
                "target_property": "plant",
                "distance_group": "deep",
                "distance": 4,
                "success": False,
                "turns": [
                    {"turn_number": 1, "action": {"action": "test"}, "inferred_intention": "confirm"},
                    {"turn_number": 2, "action": {"action": "test"}, "inferred_intention": "confirm"},
                ],
            },
        ],
    }.items():
        model_dir = results_dir / model_name
        model_dir.mkdir(parents=True)
        (model_dir / "game_results.json").write_text(json.dumps(games), encoding="utf-8")

    fieldnames_test = ["model", "game_id", "turn_number", "triple", "target_property", "judgment"]
    _write_csv(
        annotation_dir / "test_turn_target_judgment_llm.csv",
        fieldnames_test,
        [
            {"model": "model-a", "game_id": 1, "turn_number": 1, "triple": "a | b | c", "target_property": "animal", "judgment": "yes"},
            {"model": "model-a", "game_id": 2, "turn_number": 1, "triple": "d | e | f", "target_property": "plant", "judgment": "no"},
            {"model": "model-b", "game_id": 1, "turn_number": 1, "triple": "g | h | i", "target_property": "animal", "judgment": "yes"},
            {"model": "model-b", "game_id": 2, "turn_number": 1, "triple": "j | k | l", "target_property": "plant", "judgment": "no"},
        ],
    )
    _write_csv(
        annotation_dir / "test_turn_target_judgment_human.csv",
        fieldnames_test,
        [
            {"model": "model-a", "game_id": 1, "turn_number": 1, "triple": "a | b | c", "target_property": "animal", "judgment": "no"},
            {"model": "model-a", "game_id": 2, "turn_number": 1, "triple": "d | e | f", "target_property": "plant", "judgment": "no"},
            {"model": "model-b", "game_id": 1, "turn_number": 1, "triple": "g | h | i", "target_property": "animal", "judgment": "yes"},
            {"model": "model-b", "game_id": 2, "turn_number": 1, "triple": "j | k | l", "target_property": "plant", "judgment": "yes"},
        ],
    )

    fieldnames_guess = ["model", "game_id", "turn_number", "guess", "target_property", "judgment"]
    _write_csv(
        annotation_dir / "guess_target_equivalence_llm.csv",
        fieldnames_guess,
        [
            {"model": "model-a", "game_id": 1, "turn_number": 3, "guess": "animal", "target_property": "animal", "judgment": "yes"},
            {"model": "model-b", "game_id": 1, "turn_number": 3, "guess": "animal", "target_property": "animal", "judgment": "yes"},
        ],
    )
    _write_csv(
        annotation_dir / "guess_target_equivalence_human.csv",
        fieldnames_guess,
        [
            {"model": "model-a", "game_id": 1, "turn_number": 3, "guess": "animal", "target_property": "animal", "judgment": "yes"},
            {"model": "model-b", "game_id": 1, "turn_number": 3, "guess": "animal", "target_property": "animal", "judgment": "no"},
        ],
    )

    def _fake_fit(rows):
        assert len(rows) == 4
        return {
            "status": "ok",
            "formula": "success ~ positive_testing_bias + oracle_any_error",
            "complete_case_games": 4,
            "models_in_fit": ["model-a", "model-b"],
            "fixed_effects": {
                "Intercept": {"posterior_mean": 0.1, "posterior_sd": 0.2, "odds_ratio": 1.105},
            },
            "random_effects": {
                "model": {"log_sd_posterior_mean": -1.0, "log_sd_posterior_sd": 0.1, "implied_sd": 0.368},
            },
        }

    monkeypatch.setattr("src.analysis.mixed_effects.fit_mixed_effects_logit", _fake_fit)

    output_path = tmp_path / "analysis.json"
    rows_output_path = tmp_path / "analysis_rows.csv"
    plot_output_path = tmp_path / "analysis_plot.png"
    payload = run_game_level_mixed_effects_analysis(
        results_dir=results_dir,
        annotation_dir=annotation_dir,
        output_path=output_path,
        rows_output_path=rows_output_path,
        plot_output_path=plot_output_path,
    )

    assert payload["fit"]["status"] == "ok"
    assert payload["games_with_oracle_annotations"] == 4
    assert output_path.exists()
    assert rows_output_path.exists()
    assert plot_output_path.exists()

    rows = list(csv.DictReader(rows_output_path.open("r", encoding="utf-8")))
    rows_by_key = {(row["model"], row["game_id"]): row for row in rows}
    assert rows_by_key[("model-a", "1")]["oracle_any_error"] == "True"
    assert rows_by_key[("model-a", "1")]["sampled_test_oracle_error"] == "True"
    assert rows_by_key[("model-a", "1")]["sampled_guess_oracle_error"] == "False"
    assert rows_by_key[("model-a", "2")]["oracle_any_error"] == "False"
    assert rows_by_key[("model-b", "1")]["oracle_any_error"] == "True"
    assert rows_by_key[("model-b", "2")]["oracle_any_error"] == "True"