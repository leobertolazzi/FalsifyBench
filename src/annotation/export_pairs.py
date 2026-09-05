"""Export paired CSV files for human annotation against saved oracle judgments.

Usage examples:
    python -m src.annotation.export_pairs
    python -m src.annotation.export_pairs --input results --seed 13
    python -m src.annotation.export_pairs --input results/gpt-5-mini
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class AnnotationTask:
    """Configuration for one CSV-pair annotation export."""

    name: str
    row_fields: Sequence[str]
    judgment_field: str


TEST_TARGET_TASK = AnnotationTask(
    name="test_turn_target_judgment",
    row_fields=("triple", "target_property", "judgment"),
    judgment_field="judgment",
)

TEST_HYPOTHESIS_TASK = AnnotationTask(
    name="test_turn_hypothesis_judgment",
    row_fields=("triple", "current_hypothesis", "judgment"),
    judgment_field="judgment",
)

GUESS_TARGET_TASK = AnnotationTask(
    name="guess_target_equivalence",
    row_fields=("guess", "target_property", "judgment"),
    judgment_field="judgment",
)


def load_results_file(results_file: Path) -> List[Dict[str, Any]]:
    """Load game results from JSON; return [] when the file is missing or invalid."""
    if not results_file.exists():
        return []

    try:
        with open(results_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    return []


def find_results_files(input_path: Path, recursive: bool = True) -> List[Path]:
    """Find game_results.json files from an input file or directory."""
    if input_path.is_file():
        if input_path.name != "game_results.json":
            raise ValueError("If --input is a file, it must be named game_results.json")
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    files: List[Path] = []
    direct = input_path / "game_results.json"
    if direct.exists():
        files.append(direct)

    if recursive:
        files.extend(sorted(input_path.rglob("game_results.json")))
    else:
        files.extend(sorted(input_path.glob("*/game_results.json")))

    return sorted(set(files))


def _normalize_model_name(results_file: Path) -> str:
    """Use the parent folder name as the model identifier."""
    return results_file.parent.name


def _format_items(items: Iterable[Any]) -> str:
    """Serialize a triple as a stable pipe-separated string."""
    return " | ".join(str(item) for item in items)


def _sample_test_turn(game_result: Dict[str, Any], rng: random.Random) -> Dict[str, Any] | None:
    """Sample one test turn from a game result."""
    test_turns = [
        turn
        for turn in game_result.get("turns", [])
        if (turn.get("action") or {}).get("action") == "test"
    ]
    if not test_turns:
        return None
    return rng.choice(test_turns)


def _sample_guess_turn(game_result: Dict[str, Any], rng: random.Random) -> Dict[str, Any] | None:
    """Sample one guess turn from a game result."""
    guess_turns = [
        turn
        for turn in game_result.get("turns", [])
        if (turn.get("action") or {}).get("action") == "guess"
    ]
    if not guess_turns:
        return None
    return rng.choice(guess_turns)


def _bool_to_judgment(value: Any) -> str:
    """Normalize saved boolean judgments into CSV-friendly labels."""
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return ""


def _build_test_target_row(model_name: str, game_result: Dict[str, Any], turn: Dict[str, Any]) -> Dict[str, Any]:
    action = turn.get("action") or {}
    oracle_response = turn.get("oracle_response") or {}
    return {
        "model": model_name,
        "game_id": game_result.get("game_id"),
        "turn_number": turn.get("turn_number"),
        "triple": _format_items(action.get("items", [])),
        "target_property": str(game_result.get("target_property", "")),
        "judgment": _bool_to_judgment(oracle_response.get("conforms")),
    }


def _build_test_hypothesis_row(model_name: str, game_result: Dict[str, Any], turn: Dict[str, Any]) -> Dict[str, Any]:
    action = turn.get("action") or {}
    hypothesis_response = turn.get("hypothesis_oracle_response") or {}
    return {
        "model": model_name,
        "game_id": game_result.get("game_id"),
        "turn_number": turn.get("turn_number"),
        "triple": _format_items(action.get("items", [])),
        "current_hypothesis": str(action.get("hypothesis", "")),
        "judgment": _bool_to_judgment(hypothesis_response.get("conforms")),
    }


def _build_guess_target_row(model_name: str, game_result: Dict[str, Any], turn: Dict[str, Any]) -> Dict[str, Any]:
    action = turn.get("action") or {}
    oracle_response = turn.get("oracle_response") or {}
    return {
        "model": model_name,
        "game_id": game_result.get("game_id"),
        "turn_number": turn.get("turn_number"),
        "guess": str(action.get("property", "")),
        "target_property": str(game_result.get("target_property", "")),
        "judgment": _bool_to_judgment(oracle_response.get("correct")),
    }


def build_annotation_rows(results_files: Sequence[Path], seed: int) -> Dict[str, List[Dict[str, Any]]]:
    """Build export rows for all annotation tasks across model results files."""
    test_target_rows: List[Dict[str, Any]] = []
    test_hypothesis_rows: List[Dict[str, Any]] = []
    guess_target_rows: List[Dict[str, Any]] = []

    for file_index, results_file in enumerate(sorted(results_files)):
        model_name = _normalize_model_name(results_file)
        results = load_results_file(results_file)

        for game_result in results:
            game_id = int(game_result.get("game_id", 0)) if game_result.get("game_id") is not None else 0
            game_rng = random.Random((seed * 1000003) + (file_index * 10007) + game_id)

            sampled_test_turn = _sample_test_turn(game_result, game_rng)
            if sampled_test_turn is not None:
                test_target_rows.append(
                    _build_test_target_row(model_name, game_result, sampled_test_turn)
                )
                test_hypothesis_rows.append(
                    _build_test_hypothesis_row(model_name, game_result, sampled_test_turn)
                )

            sampled_guess_turn = _sample_guess_turn(game_result, game_rng)
            if sampled_guess_turn is not None:
                guess_target_rows.append(
                    _build_guess_target_row(model_name, game_result, sampled_guess_turn)
                )

    return {
        TEST_TARGET_TASK.name: sorted(test_target_rows, key=_row_sort_key),
        TEST_HYPOTHESIS_TASK.name: sorted(test_hypothesis_rows, key=_row_sort_key),
        GUESS_TARGET_TASK.name: sorted(guess_target_rows, key=_row_sort_key),
    }


def _row_sort_key(row: Dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("model", ""), row.get("game_id", -1), row.get("turn_number", -1))


def _human_row_from_llm_row(row: Dict[str, Any], judgment_field: str) -> Dict[str, Any]:
    """Copy a row but blank out the judgment column for human annotation."""
    human_row = dict(row)
    human_row[judgment_field] = ""
    return human_row


def write_csv_pair(output_dir: Path, task: AnnotationTask, rows: Sequence[Dict[str, Any]]) -> tuple[Path, Path]:
    """Write one machine-labeled CSV and one human-annotation CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    llm_path = output_dir / f"{task.name}_llm.csv"
    human_path = output_dir / f"{task.name}_human.csv"

    fieldnames = ["model", "game_id", "turn_number", *task.row_fields]
    with open(llm_path, "w", encoding="utf-8", newline="") as llm_handle:
        writer = csv.DictWriter(llm_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(human_path, "w", encoding="utf-8", newline="") as human_handle:
        writer = csv.DictWriter(human_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_human_row_from_llm_row(row, task.judgment_field) for row in rows)

    return llm_path, human_path


def export_annotation_pairs(input_path: Path, output_dir: Path, seed: int, recursive: bool = True) -> List[Path]:
    """Export paired annotation CSVs from discovered result files."""
    results_files = find_results_files(input_path, recursive=recursive)
    if not results_files:
        return []

    rows_by_task = build_annotation_rows(results_files, seed)
    written: List[Path] = []
    for task in (TEST_TARGET_TASK, TEST_HYPOTHESIS_TASK, GUESS_TARGET_TASK):
        llm_path, human_path = write_csv_pair(output_dir, task, rows_by_task[task.name])
        written.extend([llm_path, human_path])
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export paired CSV files for human annotation from saved game results"
    )
    parser.add_argument(
        "--input",
        default="results",
        help="Path to a results directory, a model folder, or a specific game_results.json (default: results)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/annotation_pairs_oracle",
        help="Directory where paired CSV files will be written (default: results/annotation_pairs_oracle)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Fixed random seed for deterministic per-game sampling (default: 42)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also process nested subdirectories under --input",
    )

    args = parser.parse_args()

    try:
        written = export_annotation_pairs(
            input_path=Path(args.input),
            output_dir=Path(args.output_dir),
            seed=args.seed,
            recursive=args.recursive,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return

    if not written:
        print("No game_results.json files found.")
        return

    print(f"Wrote {len(written)} files:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()