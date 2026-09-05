"""
Main script to run the inductive reasoning game simulation.

Usage:
    python -m src.main --mode prepare     # Prepare the dataset
    python -m src.main --mode play        # Play games
    python -m src.main --mode evaluate    # Evaluate results
    python -m src.main --mode all         # Run complete pipeline
"""

import argparse
import json
import random
from pathlib import Path
import re
import os
from typing import Any, Dict, List

from .benchmark.construction import GameDatasetBuilder
from .benchmark.engine import GameEngine
from .benchmark.evaluation import GameEvaluator, evaluate_all_models, evaluate_single_results_dir


def _load_env_from_dotenv() -> None:
    """Load environment variables from a project-root .env if present.

    This keeps CLI behavior aligned with README instructions (cp .env.example .env).
    """
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return

    project_root = Path(__file__).resolve().parents[1]
    dotenv_path = project_root / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path, override=False)

        
def _normalize_model_name_for_output_dir(model_name: str) -> str:
    """Create a stable, filesystem-safe model name for output directories."""
    normalized = model_name.replace("/", "_").replace(":", "_").strip("_-")

    date_suffix_patterns = [
        r"(?:[-_](?:19|20)\d{2}[-_]\d{2}[-_]\d{2})$",  # -2025-08-07
        r"(?:[-_](?:19|20)\d{6})$",                     # -20250807 or -20241022
        r"(?:[-_]\d{8})$",                               # generic 8-digit suffix
    ]

    previous = None
    while previous != normalized:
        previous = normalized
        for pattern in date_suffix_patterns:
            normalized = re.sub(pattern, "", normalized)

    normalized = re.sub(r"[-_]{2,}", "-", normalized).strip("_-")
    return normalized or "model"


def _resolve_output_dir(output_dir_arg, player_model, resume: bool = False):
    """Resolve output directory, defaulting to results/{normalized_model}."""
    if output_dir_arg:
        return Path(output_dir_arg)

    safe_model = _normalize_model_name_for_output_dir(player_model)
    resolved = Path("results") / safe_model

    if resume:
        print(f"Resuming from model output directory: {resolved}")
    else:
        print(f"Using model output directory: {resolved}")

    return resolved


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically write JSON to avoid partial/corrupted checkpoint files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(path)


def _load_results_file(results_file: Path) -> List[Dict[str, Any]]:
    """Load an existing results file if valid, otherwise return empty list."""
    if not results_file.exists():
        return []

    try:
        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    return []


def _instance_key(instance: Dict[str, Any]) -> str:
    """Stable key used to match dataset instances with saved game results."""
    target = instance.get("target_hypothesis") or instance.get("target_property") or ""
    sampling = instance.get("sampling_hypothesis") or ""
    distance_group = instance.get("distance_group") or ""
    distance = instance.get("distance")
    examples = tuple(instance.get("initial_examples", []))
    return f"{target}|{sampling}|{distance_group}|{distance}|{examples}"


def prepare_dataset(apply_filters: bool = True, sample_per_group: int | None = None, seed: int = 42):
    """
    Step 1: Prepare and stratify the dataset.

    Args:
        apply_filters: If True, apply quality filters during generation
        sample_per_group: If set, downsample to N instances per distance group
        seed: Random seed for deterministic sampling
    """
    print("\n" + "="*60)
    print("STEP 1: DATASET PREPARATION")
    print("="*60)

    stratifier = GameDatasetBuilder("raw_data/wordnet_triples.json", apply_filters=apply_filters)
    stratifier.load_data()
    instances = stratifier.create_game_instances()

    # Print statistics
    stats = stratifier.get_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Total instances: {stats['total_instances']}")
    print(f"  Close: {stats['close']} ({stats['close_pct']}%)")
    print(f"  Deep: {stats['deep']} ({stats['deep_pct']}%)")

    # Print filtering statistics if enabled
    if stats['filtering']['enabled']:
        print(f"\nQuality Filtering:")
        print(f"  Total created: {stats['filtering']['total_created']}")
        print(f"  Filtered out: {stats['filtering']['total_filtered']} ({stats['filtering']['filter_rate']}%)")
        print(f"  Kept: {stats['filtering']['total_kept']}")
        print(f"  Reasons:")
        print(f"    - G standalone: {stats['filtering']['reasons']['G_standalone']}")
        print(f"    - G in examples: {stats['filtering']['reasons']['G_in_examples']}")
        print(f"    - Overly specific: {stats['filtering']['reasons']['overly_specific']}")
    else:
        print(f"\nQuality Filtering: Disabled")

    # Apply downsampling if requested
    if sample_per_group:
        print(f"\nDownsampling to {sample_per_group} instances per distance group (seed={seed})...")
        random.seed(seed)

        sampled_instances = []
        for distance_group in ["close", "deep"]:
            diff_instances = [inst for inst in instances if inst['distance_group'] == distance_group]
            if len(diff_instances) > sample_per_group:
                sampled = random.sample(diff_instances, sample_per_group)
            else:
                sampled = diff_instances
            sampled_instances.extend(sampled)
            print(f"  {distance_group.capitalize()}: {len(sampled)} instances (from {len(diff_instances)})")

        instances = sampled_instances
        print(f"Total after sampling: {len(instances)} instances")

    # Save datasets
    stratifier.save_dataset("data/game_instances.json")
    stratifier.save_by_distance_group("data")

    print("\n✓ Dataset preparation complete!")
    return instances


def play_games(
    distance_group=None, max_turns=20, sample_per_group=None, seed=42,
    verbose=False, max_tests=None,
    player_model="gpt-5-nano-2025-08-07",
    oracle_model="gpt-5-nano-2025-08-07",
    use_curated=False,
    output_dir=None,
    resume=False,
):
    """
    Step 2: Play games.

    Args:
        distance_group: Filter by taxonomic distance group (close/deep) or None for all
        max_turns: Maximum turns per game
        sample_per_group: Number of instances per distance group to play
        seed: Random seed for deterministic sampling
        verbose: Enable verbose per-turn logging
        max_tests: Hard cap on number of TEST actions per game
        player_model: Model name for the player agent
        oracle_model: Model name for the oracle agent
        use_curated: If True, load from curated game files instead of random sampling
        output_dir: Path where results will be saved (already resolved by caller)
        resume: If True, restore from existing results in output_dir and continue
    """
    print("\n" + "="*60)
    print("STEP 2: PLAYING GAMES")
    print("="*60)

    # Load game instances
    if use_curated:
        if distance_group:
            data_file = f"data/curated_games_{distance_group}.json"
        else:
            data_file = "data/curated_games.json"

        data_path = Path(data_file)
        if not data_path.exists():
            print(f"Curated games file not found: {data_file}")
            print("Run --mode curate or ensure data/curated_games*.json exists")
            return []

        with open(data_path, 'r') as f:
            instances = json.load(f)

        print(f"\nLoading curated games from {data_file}...")
        games_to_play = instances
        print(f"Total games to play: {len(games_to_play)}")
    else:
        if sample_per_group is None:
            print("Error: --sample-per-group is required")
            print("Example: python -m src.main --mode play --sample-per-group 100")
            return []

        if distance_group:
            data_file = f"data/game_instances_{distance_group}.json"
        else:
            data_file = "data/game_instances.json"

        data_path = Path(data_file)
        if not data_path.exists():
            print(f"Game instances file not found: {data_file}")
            print("Run with --mode prepare first")
            return []

        with open(data_path, 'r') as f:
            instances = json.load(f)

        # Apply sampling
        if not distance_group:
            # Balanced sampling across all distance groups
            print(f"\nSampling {sample_per_group} instances per distance group (seed={seed})...")
            random.seed(seed)

            sampled_instances = []
            for diff in ["close", "deep"]:
                diff_instances = [inst for inst in instances if inst['distance_group'] == diff]
                if len(diff_instances) > sample_per_group:
                    sampled = random.sample(diff_instances, sample_per_group)
                else:
                    sampled = diff_instances
                sampled_instances.extend(sampled)
                print(f"  {diff.capitalize()}: {len(sampled)} instances (from {len(diff_instances)})")

            instances = sampled_instances
            print(f"Total games to play: {len(instances)}")
        else:
            # Single distance-group sampling
            print(f"\nSampling {sample_per_group} {distance_group} instances (seed={seed})...")
            random.seed(seed)
            if len(instances) > sample_per_group:
                instances = random.sample(instances, sample_per_group)
            print(f"Total games to play: {len(instances)}")

        games_to_play = instances

    if output_dir is None:
        output_dir = Path("results")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "game_results.json"

    if resume:
        results = _load_results_file(results_file)
        completed_keys = {_instance_key(r) for r in results}
        if results and len(completed_keys) != len(results):
            print("Warning: duplicate game keys found in existing results; duplicates will be ignored for resume matching.")
        games_to_run = [g for g in games_to_play if _instance_key(g) not in completed_keys]
        print(f"\nResume mode: loaded {len(results)} completed games from {results_file}")
    else:
        results = []
        games_to_run = games_to_play
        if results_file.exists():
            print(f"\nOverwriting previous results in {results_file} (use --resume to continue instead).")

    print(f"\nPlaying {len(games_to_run)} games...")
    if distance_group:
        print(f"Distance group: {distance_group}")
    print(f"Max turns per game: {max_turns}")
    print(f"Player model: {player_model}")
    print(f"Oracle model: {oracle_model}")

    if not games_to_run:
        print("No remaining games to play. Existing results are already complete for this selection.")
        return results

    # Preflight: fail fast on missing API keys (avoid repeating the same error per game)
    try:
        _ = GameEngine(
            games_to_run[0], max_turns=max_turns, verbose=verbose, max_tests=max_tests,
            player_model=player_model, oracle_model=oracle_model,
        )
    except Exception as e:
        print(f"\nCannot start games due to configuration error: {e}")
        print("Tip: If you're using .env, ensure it exists at the project root and contains the right key.")
        print(
            "Expected env vars: AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT for gpt-5.2-chat; "
            "OPENAI_API_KEY for gpt-5-mini and gpt-5-nano; TOGETHER_API_KEY for all other paper models."
        )
        return []

    try:
        for i, instance in enumerate(games_to_run, 1):
            absolute_index = len(results) + 1
            print(f"\n{'─'*60}")
            print(f"Game {absolute_index}/{len(games_to_play)} (session {i}/{len(games_to_run)})")
            print(f"Target: {instance['target_hypothesis']}")
            print(f"Sampling: {instance['sampling_hypothesis']}")
            print(f"Distance group: {instance['distance_group']} (distance: {instance['distance']})")
            print(f"Examples: {', '.join(instance['initial_examples'])}")

            try:
                engine = GameEngine(
                    instance, max_turns=max_turns, verbose=verbose, max_tests=max_tests,
                    player_model=player_model, oracle_model=oracle_model,
                )
                result = engine.play_game()
                result['initial_examples'] = instance['initial_examples']
                results.append(result)

                _atomic_write_json(results_file, results)

                print(f"Result: {'SUCCESS ✓' if result['success'] else 'FAILED ✗'}")
                print(f"Turns: {result['total_turns']}")
                print(f"Checkpoint saved ({len(results)}/{len(games_to_play)}): {results_file}")

            except Exception as e:
                print(f"Error playing game: {e}")
                import traceback
                traceback.print_exc()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Preserving progress and exiting gracefully...")
    finally:
        _atomic_write_json(results_file, results)

    print(f"\n✓ Games complete! Results saved to {results_file}")
    return results


def evaluate_results(output_dir=None, results_dir: str = "results"):
    """Step 3: Evaluate and analyze results."""
    print("\n" + "="*60)
    print("STEP 3: EVALUATION")
    print("="*60)

    if output_dir is not None:
        output_dir = Path(output_dir)
        single_result = evaluate_single_results_dir(output_dir)

        if single_result.get("status") != "ok":
            print(single_result.get("message", f"No results found in {output_dir}"))
            print("Run with --mode play first")
            return

        evaluator = GameEvaluator()
        evaluator.load_results(str(output_dir / "game_results.json"))
        evaluator.print_summary()
    else:
        summary = evaluate_all_models(results_dir)
        total = summary.get("total_models_found", 0)
        evaluated = summary.get("evaluated_models", 0)
        results_root = Path(results_dir)

        if total == 0:
            print(f"No model results found under {results_root}/*/game_results.json")
            print("Run with --mode play first")
            return

        print(f"Evaluated {evaluated}/{total} model directories")
        print(f"Saved cross-model summary to {results_root / 'aggregate_by_model.json'}")

    print("\n✓ Evaluation complete!")


def main():
    _load_env_from_dotenv()

    parser = argparse.ArgumentParser(
        description="Inductive Reasoning Game Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare the dataset (with quality filtering enabled by default)
  python -m src.main --mode prepare

  # Prepare dataset without filtering
  python -m src.main --mode prepare --no-filter

  # Play curated games (hand-picked)
  python -m src.main --mode play --curated

  # Play balanced games (100 per distance group = 200 total)
  python -m src.main --mode play --sample-per-group 100

  # Play 50 close-category games only
  python -m src.main --mode play --distance-group close --sample-per-group 50

  # Play with a specific model, saving to a named directory
  python -m src.main --mode play --curated --player-model gpt-5-mini --output-dir results/gpt-5-mini

  # Default output dir is results/{normalized_player_model}
  # e.g. gpt-5-mini -> results/gpt-5-mini
  python -m src.main --mode play --curated --player-model gpt-5-mini

  # Resume an interrupted run without specifying output-dir
  python -m src.main --mode play --curated --player-model gpt-5-mini --resume

  # Evaluate all model runs under results/*/game_results.json
  python -m src.main --mode evaluate

  # Evaluate all model runs under a custom results root
  python -m src.main --mode evaluate --results-dir custom_results

  # Evaluate results from a specific run
  python -m src.main --mode evaluate --output-dir results/mymodel

  # Run complete pipeline (50 per distance group = 100 total)
  python -m src.main --mode all --sample-per-group 50 --seed 42
        """
    )

    parser.add_argument(
        "--mode",
        choices=["prepare", "play", "evaluate", "all"],
        default="all",
        help="Operation mode"
    )

    parser.add_argument(
        "--curated",
        action="store_true",
        help="Load from data/curated_games.json instead of random downsampling"
    )

    parser.add_argument(
        "--player-model",
        default="gpt-5-nano-2025-08-07",
        help="Model name for the player agent (default: gpt-5-nano-2025-08-07)"
    )

    parser.add_argument(
        "--oracle-model",
        default=None,
        help="Model name for the oracle agent (default: same as --player-model)"
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for play outputs; for evaluate mode, setting this evaluates a single run directory "
            "instead of scanning all model subdirectories"
        )
    )

    parser.add_argument(
        "--results-dir",
        default="results",
        help=(
            "Root directory scanned by --mode evaluate when --output-dir is omitted "
            "(expects model subdirectories with game_results.json)"
        )
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted play run from existing game_results.json in --output-dir (or results/{normalized_player_model} if --output-dir is omitted)"
    )

    parser.add_argument(
        "--sample-per-group",
        type=int,
        default=None,
        help="Number of instances per distance group (close/deep; required unless --curated)"
    )
    parser.add_argument(
        "--sample-per-difficulty",
        dest="sample_per_group",
        type=int,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--distance-group",
        choices=["close", "deep"],
        default=None,
        help="Filter games by taxonomic distance group (close/deep)"
    )
    parser.add_argument(
        "--difficulty",
        dest="legacy_distance_group",
        choices=["easy", "hard"],
        default=None,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Maximum turns per game (default: 20)"
    )

    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable quality filtering during dataset preparation (default: filters enabled)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling (default: 42)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose per-turn logging during game play"
    )

    parser.add_argument(
        "--max-tests",
        type=int,
        default=None,
        help="Hard cap on number of TEST actions allowed per game (after reaching it, player must GUESS)"
    )

    args = parser.parse_args()

    # Accept pre-publication flags without exposing them in the public CLI.
    if args.distance_group is not None and args.legacy_distance_group is not None:
        parser.error("use either --distance-group or the legacy --difficulty flag, not both")
    if args.legacy_distance_group is not None:
        args.distance_group = {"easy": "close", "hard": "deep"}[args.legacy_distance_group]

    # Default oracle model to player model when not specified
    if args.oracle_model is None:
        args.oracle_model = args.player_model

    # Validation
    if args.curated and args.sample_per_group is not None:
        print("Warning: --curated ignores --sample-per-group.")
        args.sample_per_group = None

    if args.mode in ("play", "all") and not args.curated and args.sample_per_group is None:
        parser.error("--sample-per-group is required unless --curated is set.")

    resolved_output_dir = None

    if args.mode in ("prepare", "all"):
        prepare_dataset(
            apply_filters=not args.no_filter,
            sample_per_group=args.sample_per_group,
            seed=args.seed
        )

    if args.mode in ("play", "all"):
        resolved_output_dir = _resolve_output_dir(args.output_dir, args.player_model, resume=args.resume)
        results = play_games(
            distance_group=args.distance_group,
            max_turns=args.max_turns,
            sample_per_group=args.sample_per_group,
            seed=args.seed,
            verbose=args.verbose,
            max_tests=args.max_tests,
            player_model=args.player_model,
            oracle_model=args.oracle_model,
            use_curated=args.curated,
            output_dir=resolved_output_dir,
            resume=args.resume,
        )
        if not results and args.mode == "all":
            print("Skipping evaluation: no games were played.")
            return

    if args.mode in ("evaluate", "all"):
        evaluate_results(output_dir=resolved_output_dir
                         or (Path(args.output_dir) if args.output_dir else None),
                         results_dir=args.results_dir)

    print("\n" + "="*60)
    print("ALL TASKS COMPLETE!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
