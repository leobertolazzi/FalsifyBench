"""Annotate test-turn hypothesis/target relations across saved game results.

Usage examples:
    python -m src.annotation.annotate_relations
    python -m src.annotation.annotate_relations --input results --overwrite
    python -m src.annotation.annotate_relations --input results/gpt-5-mini
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm.auto import tqdm

from ..benchmark.engine import (
    OracleAgent,
    _create_provider_client,
    _detect_provider_for_model,
    _is_api_timeout_error,
)
from ..main import _load_env_from_dotenv


RELATION_LABELS: Tuple[str, ...] = (
    "identical",
    "disjoint",
    "partial_overlap",
    "hypothesis_included_in_target",
    "target_included_in_hypothesis",
)


DEFAULT_ANNOTATION_MODEL = "gpt-5-mini"
DEFAULT_OUTPUT_NAME = "turn_relation_annotations.json"
DEFAULT_MAX_COMPLETION_TOKENS = 8000


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


def load_annotation_file(annotation_file: Path) -> Optional[Dict[str, Any]]:
    """Load an existing annotation checkpoint file when it is valid."""
    if not annotation_file.exists():
        return None

    try:
        with open(annotation_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    return None


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically write JSON so checkpoint files are never partially written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


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


def iter_test_turn_entries(results: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    """Yield flat metadata records for every TEST turn in the result set."""
    for game_result in results:
        for turn in game_result.get("turns", []):
            action = turn.get("action", {})
            if action.get("action") != "test":
                continue

            yield {
                "game_id": game_result.get("game_id"),
                "turn_number": turn.get("turn_number"),
                "target_property": game_result.get("target_property"),
                "hypothesis": action.get("hypothesis"),
            }


def _build_annotation_prompt(
    target_property: str,
    hypothesis: str,
    *,
    include_explanation: bool,
) -> str:
    """Build the prompt for semantic set-relation annotation."""
    labels = "\n".join(
        [
            "- identical: the two sets are the same.",
            "- disjoint: the two sets share no members.",
            "- partial_overlap: the two sets overlap, but neither includes the other.",
            "- hypothesis_included_in_target: the hypothesis set is a proper subset of the target set.",
            "- target_included_in_hypothesis: the target set is a proper subset of the hypothesis set.",
        ]
    )
    examples = "\n".join(
        [
            '1. Target: "animal" | Hypothesis: "animal" -> identical',
            '2. Target: "animal" | Hypothesis: "mammal" -> hypothesis_included_in_target',
            '3. Target: "animal" | Hypothesis: "living thing" -> target_included_in_hypothesis',
            '4. Target: "animal" | Hypothesis: "carnivore" -> partial_overlap',
            '5. Target: "animal" | Hypothesis: "mineral" -> disjoint',
        ]
    )

    response_format = (
        '{\n  "relation": "one of ' + ", ".join(RELATION_LABELS) + '",\n  "explanation": "at most 12 words"\n}'
        if include_explanation
        else '{\n  "relation": "one of ' + ", ".join(RELATION_LABELS) + '"\n}'
    )

    return f"""You are annotating saved game turns from an inductive reasoning experiment.

Treat the target property and the current hypothesis as semantic sets or classes of entities.
Decide the relation between those two sets using exactly one label from this closed set:
{labels}

Interpret terms by their normal semantic meaning, not by string overlap.
If one label is more specific than another, choose the more specific label:
- Prefer identical over either inclusion label when the sets are the same.
- Prefer an inclusion label over partial_overlap when one set fully contains the other.

Examples using target property "animal":
{examples}

Now annotate this case.
Target property: "{target_property}"
Current hypothesis: "{hypothesis}"

Respond with JSON only in this format:
{response_format}

Do not add any extra keys, markdown, or prose outside the JSON object."""


def _extract_response_text(content: Any) -> str:
    """Normalize response content into a plain string for JSON extraction."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def _annotation_entry_key(entry: Dict[str, Any]) -> Tuple[Any, ...]:
    """Return a stable key for matching saved annotations to source turns."""
    return (
        entry.get("game_id"),
        entry.get("turn_number"),
    )


def _build_saved_annotation_entry(entry: Dict[str, Any], annotation: Dict[str, Any]) -> Dict[str, Any]:
    """Build the compact persisted annotation record."""
    relation_value = annotation.get("relation")
    relation = relation_value.strip() if isinstance(relation_value, str) else None
    if relation == "":
        relation = None
    return {
        "game_id": entry.get("game_id"),
        "turn_number": entry.get("turn_number"),
        "target_property": entry.get("target_property"),
        "hypothesis": entry.get("hypothesis"),
        "status": annotation.get("status"),
        "relation": relation,
        "explanation": annotation.get("explanation"),
        "error": annotation.get("error"),
    }


def _build_annotation_payload(
    *,
    results_file: Path,
    annotator: TurnRelationAnnotator,
    annotations: List[Dict[str, Any]],
    games_processed: int,
    test_turns_processed: int,
    annotated_turns: int,
    error_count: int,
    completed: bool,
) -> Dict[str, Any]:
    """Build the persisted annotation payload."""
    return {
        "source_results_file": str(results_file),
        "annotation_model": annotator.model,
        "annotation_provider": annotator.provider,
        "relation_labels": list(RELATION_LABELS),
        "games_processed": games_processed,
        "test_turns_processed": test_turns_processed,
        "annotated_turns": annotated_turns,
        "error_count": error_count,
        "completed": completed,
        "annotations": annotations,
    }


def _summarize_annotation_entries(annotations: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Count successful and errored annotation entries."""
    annotated_turns = 0
    error_count = 0
    for annotation in annotations:
        if annotation.get("status") == "ok":
            annotated_turns += 1
        else:
            error_count += 1
    return annotated_turns, error_count


def _count_test_turns_by_game(entries: List[Dict[str, Any]]) -> Dict[Any, int]:
    """Count how many TEST turns each game contributes to the annotation job."""
    counts: Dict[Any, int] = {}
    for entry in entries:
        game_id = entry.get("game_id")
        counts[game_id] = counts.get(game_id, 0) + 1
    return counts


def _count_completed_games(
    results: List[Dict[str, Any]],
    annotations: List[Dict[str, Any]],
    test_turns_by_game: Dict[Any, int],
) -> int:
    """Count games whose TEST turns are fully annotated, including zero-test games."""
    annotated_turns_by_game: Dict[Any, int] = {}
    for annotation in annotations:
        game_id = annotation.get("game_id")
        annotated_turns_by_game[game_id] = annotated_turns_by_game.get(game_id, 0) + 1

    completed = 0
    for game_result in results:
        game_id = game_result.get("game_id")
        required_turns = test_turns_by_game.get(game_id, 0)
        if annotated_turns_by_game.get(game_id, 0) >= required_turns:
            completed += 1
    return completed


def _create_game_progress_bar(model_label: str, completed: int, total: int):
    """Create a tqdm progress bar for game-level progress."""
    return tqdm(
        total=total,
        initial=completed,
        desc=f"{model_label} games",
        unit="game",
        file=sys.stdout,
        leave=True,
    )


class TurnRelationAnnotator:
    """LLM-backed annotator for hypothesis/target set relations."""

    def __init__(
        self,
        model: str = DEFAULT_ANNOTATION_MODEL,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_retry_attempts: int = 3,
        timeout_retry_initial_delay: float = 1.5,
        timeout_retry_backoff_factor: float = 2.0,
    ):
        self.model = model
        self.timeout_retry_attempts = max(0, int(timeout_retry_attempts))
        self.timeout_retry_initial_delay = max(0.0, float(timeout_retry_initial_delay))
        self.timeout_retry_backoff_factor = max(1.0, float(timeout_retry_backoff_factor))

        self.provider = provider or _detect_provider_for_model(model)
        client_config = _create_provider_client(self.provider, model, api_key=api_key)
        self.api_key = client_config["api_key"]
        self.client = client_config["client"]
        self.api_version = client_config["api_version"]
        self.azure_endpoint = client_config["azure_endpoint"]

    def _call_with_timeout_retries(self, call_fn):
        """Reuse the same timeout retry behavior as the game engine clients."""
        attempts = self.timeout_retry_attempts + 1
        delay = self.timeout_retry_initial_delay
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                return call_fn()
            except Exception as error:
                last_error = error
                if (not _is_api_timeout_error(error)) or attempt >= attempts:
                    raise
                if delay > 0:
                    time.sleep(delay)
                delay *= self.timeout_retry_backoff_factor
        if last_error is not None:
            raise last_error

    def _chat_create(
        self,
        *,
        prompt: str,
        max_completion_tokens: int,
        force_json: bool,
        reasoning_effort: Optional[str] = None,
    ) -> Any:
        """Delegate to the shared OpenAI-compatible request wrapper."""
        return OracleAgent._chat_create(
            self,
            prompt=prompt,
            max_completion_tokens=max_completion_tokens,
            force_json=force_json,
            reasoning_effort=reasoning_effort,
        )

    def annotate_relation(self, target_property: str, hypothesis: str) -> Dict[str, Any]:
        """Annotate the semantic relation between a hypothesis and target property."""
        attempts = [
            {
                "include_explanation": True,
                "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            },
            {
                "include_explanation": False,
                "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            },
        ]

        last_raw_response: Optional[str] = None
        last_error = "Response did not contain a valid JSON object."

        for attempt in attempts:
            prompt = _build_annotation_prompt(
                target_property=target_property,
                hypothesis=hypothesis,
                include_explanation=attempt["include_explanation"],
            )
            response = self._chat_create(
                prompt=prompt,
                max_completion_tokens=attempt["max_completion_tokens"],
                force_json=True,
                reasoning_effort="minimal",
            )
            raw_content = _extract_response_text(response.choices[0].message.content)
            last_raw_response = raw_content
            payload = OracleAgent._extract_first_json_object(raw_content)
            if payload is None:
                last_error = "Response did not contain a valid JSON object."
                continue

            relation_value = payload.get("relation")
            relation = relation_value.strip() if isinstance(relation_value, str) else None
            if relation == "":
                relation = None
            if relation not in RELATION_LABELS:
                last_error = f"Invalid relation label: {relation!r}"
                continue

            explanation = payload.get("explanation")
            if not isinstance(explanation, str):
                explanation = None

            return {
                "status": "ok",
                "relation": relation,
                "explanation": explanation,
                "raw_response": raw_content,
                "error": None,
            }

        return {
            "status": "annotation_error",
            "relation": None,
            "explanation": None,
            "raw_response": last_raw_response,
            "error": last_error,
        }


def annotate_results_file(
    results_file: Path,
    annotator: TurnRelationAnnotator,
    output_name: str = DEFAULT_OUTPUT_NAME,
    verbose: bool = False,
    progress: bool = False,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Annotate one results file and write a sibling JSON artifact."""
    results = load_results_file(results_file)
    output_path = results_file.parent / output_name
    model_label = results_file.parent.name
    all_entries = list(iter_test_turn_entries(results))
    test_turns_by_game = _count_test_turns_by_game(all_entries)
    existing_payload = None if overwrite else load_annotation_file(output_path)
    existing_annotations: List[Dict[str, Any]] = []
    existing_keys = set()

    if existing_payload:
        raw_annotations = existing_payload.get("annotations", [])
        if isinstance(raw_annotations, list):
            existing_annotations = []
            for item in raw_annotations:
                if not isinstance(item, dict):
                    continue
                normalized_item = dict(item)
                relation_value = item.get("relation")
                relation = relation_value.strip() if isinstance(relation_value, str) else None
                if relation == "":
                    relation = None
                normalized_item["relation"] = relation
                existing_annotations.append(normalized_item)
            existing_keys = {_annotation_entry_key(item) for item in existing_annotations}

    annotations = list(existing_annotations)
    resumed_turns = len(existing_annotations)

    if verbose and resumed_turns:
        print(f"[{model_label}] resuming from {output_path} with {resumed_turns} saved turn annotation(s)")

    last_completed_games = _count_completed_games(results, annotations, test_turns_by_game)
    progress_bar = _create_game_progress_bar(model_label, last_completed_games, len(results)) if progress else None

    def persist_checkpoint(completed: bool) -> Dict[str, Any]:
        annotated_count, error_count = _summarize_annotation_entries(annotations)
        payload = _build_annotation_payload(
            results_file=results_file,
            annotator=annotator,
            annotations=annotations,
            games_processed=len(results),
            test_turns_processed=len(annotations),
            annotated_turns=annotated_count,
            error_count=error_count,
            completed=completed,
        )
        _atomic_write_json(output_path, payload)
        return payload

    if existing_annotations:
        persist_checkpoint(completed=False)

    try:
        for entry in all_entries:
            if _annotation_entry_key(entry) in existing_keys:
                continue

            hypothesis = entry.get("hypothesis")
            target_property = entry.get("target_property")

            if not isinstance(hypothesis, str) or not hypothesis.strip():
                annotation = {
                    "status": "annotation_error",
                    "relation": None,
                    "explanation": None,
                    "raw_response": None,
                    "error": "Missing or empty test-turn hypothesis.",
                }
            elif not isinstance(target_property, str) or not target_property.strip():
                annotation = {
                    "status": "annotation_error",
                    "relation": None,
                    "explanation": None,
                    "raw_response": None,
                    "error": "Missing or empty target property.",
                }
            else:
                annotation = annotator.annotate_relation(
                    target_property=target_property,
                    hypothesis=hypothesis,
                )

            if verbose:
                game_id = entry.get("game_id")
                turn_number = entry.get("turn_number")
                hypothesis_preview = (hypothesis or "").strip() if isinstance(hypothesis, str) else ""
                if len(hypothesis_preview) > 120:
                    hypothesis_preview = hypothesis_preview[:117] + "..."
                if annotation["status"] == "ok":
                    explanation = annotation.get("explanation")
                    explanation_suffix = f" | {explanation}" if explanation else ""
                    print(
                        f"[{model_label}] game {game_id} turn {turn_number}: "
                        f"{hypothesis_preview!r} -> {annotation['relation']}{explanation_suffix}"
                    )
                else:
                    print(
                        f"[{model_label}] game {game_id} turn {turn_number}: "
                        f"{hypothesis_preview!r} -> ERROR: {annotation['error']}"
                    )

            saved_annotation = _build_saved_annotation_entry(entry, annotation)
            annotations.append(saved_annotation)
            existing_keys.add(_annotation_entry_key(saved_annotation))
            persist_checkpoint(completed=False)

            if progress:
                completed_games = _count_completed_games(results, annotations, test_turns_by_game)
                if completed_games != last_completed_games:
                    progress_bar.update(completed_games - last_completed_games)
                    last_completed_games = completed_games
    finally:
        if progress_bar is not None:
            progress_bar.close()

    payload = persist_checkpoint(completed=True)

    return {
        "results_file": str(results_file),
        "output_path": str(output_path),
        "games_processed": len(results),
        "test_turns_processed": payload["test_turns_processed"],
        "annotated_turns": payload["annotated_turns"],
        "error_count": payload["error_count"],
        "resumed_turns": resumed_turns,
        "completed": payload["completed"],
        "status": "ok",
    }


def annotate_many(
    input_path: Path,
    output_name: str = DEFAULT_OUTPUT_NAME,
    annotation_model: str = DEFAULT_ANNOTATION_MODEL,
    recursive: bool = True,
    overwrite: bool = False,
    verbose: bool = False,
    progress: bool = False,
) -> Dict[str, Any]:
    """Scan a results root and annotate every discovered game_results.json file."""
    results_files = find_results_files(input_path, recursive=recursive)
    if not results_files:
        return {
            "status": "no_results",
            "input_path": str(input_path),
            "processed_files": 0,
            "processed_models": [],
        }
    processed_models: List[Dict[str, Any]] = []
    skipped_files = 0
    files_to_process: List[Path] = []

    for results_file in results_files:
        output_path = results_file.parent / output_name
        existing_payload = None if overwrite else load_annotation_file(output_path)
        if existing_payload and existing_payload.get("completed") is True:
            processed_models.append(
                {
                    "results_file": str(results_file),
                    "output_path": str(output_path),
                    "status": "skipped_existing",
                }
            )
            skipped_files += 1
            print(f"Skipping existing annotation file: {output_path}")
            continue

        files_to_process.append(results_file)

    if not files_to_process:
        return {
            "status": "ok",
            "input_path": str(input_path),
            "processed_files": 0,
            "skipped_files": skipped_files,
            "processed_models": processed_models,
        }

    annotator = TurnRelationAnnotator(model=annotation_model)

    for results_file in files_to_process:
        print(f"Annotating: {results_file}")
        processed_models.append(
            annotate_results_file(
                results_file=results_file,
                annotator=annotator,
                output_name=output_name,
                verbose=verbose,
                progress=progress,
                overwrite=overwrite,
            )
        )

    return {
        "status": "ok",
        "input_path": str(input_path),
        "processed_files": len(results_files) - skipped_files,
        "skipped_files": skipped_files,
        "processed_models": processed_models,
    }


def main() -> None:
    _load_env_from_dotenv()

    parser = argparse.ArgumentParser(
        description="Annotate TEST turns in saved game results with semantic set relations"
    )
    parser.add_argument(
        "--input",
        default="results",
        help="Path to a results directory, a model folder, or a specific game_results.json (default: results)",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Output filename to write next to each game_results.json (default: {DEFAULT_OUTPUT_NAME})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_ANNOTATION_MODEL,
        help=f"OpenAI-compatible model used for annotation (default: {DEFAULT_ANNOTATION_MODEL})",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan the immediate model directories under --input instead of recursing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing annotation files instead of skipping them",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each turn annotation live while processing",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print a game-level progress bar while processing each results file",
    )

    args = parser.parse_args()
    input_path = Path(args.input)

    try:
        summary = annotate_many(
            input_path=input_path,
            output_name=args.output_name,
            annotation_model=args.model,
            recursive=not args.no_recursive,
            overwrite=args.overwrite,
            verbose=args.verbose,
            progress=args.progress,
        )
    except Exception as error:
        print(f"Error: {error}")
        raise SystemExit(1)

    if summary.get("status") == "no_results":
        print("No game_results.json files found.")
        raise SystemExit(1)

    processed_files = summary.get("processed_files", 0)
    skipped_files = summary.get("skipped_files", 0)
    print(f"Processed {processed_files} result file(s).")
    if skipped_files:
        print(f"Skipped {skipped_files} existing annotation file(s).")

    total_turns = 0
    total_errors = 0
    for model_result in summary.get("processed_models", []):
        if model_result.get("status") != "ok":
            continue
        total_turns += int(model_result.get("test_turns_processed", 0))
        total_errors += int(model_result.get("error_count", 0))

    print(f"Annotated {total_turns} test turn(s) with {total_errors} error(s).")


if __name__ == "__main__":
    main()