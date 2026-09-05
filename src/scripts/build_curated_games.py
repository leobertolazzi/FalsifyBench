import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_hypotheses(hypotheses: Dict) -> None:
    required_groups = {"close", "deep"}
    found_groups = set(hypotheses.keys())
    missing_groups = required_groups - found_groups
    if missing_groups:
        missing = ", ".join(sorted(missing_groups))
        raise ValueError(f"Missing distance-group sections in curated hypotheses: {missing}")

    for distance_group in required_groups:
        if not isinstance(hypotheses[distance_group], dict):
            raise ValueError(f"curated_hypotheses[{distance_group}] must be an object")
        for target, samples in hypotheses[distance_group].items():
            if not isinstance(samples, list):
                raise ValueError(
                    f"curated_hypotheses[{distance_group}][{target}] must be a list of sampling hypotheses"
                )


def _filter_candidates(
    instances: List[dict],
    allowed_by_target: Dict[str, List[str]],
    distance_group: str,
) -> Dict[str, List[dict]]:
    allowed = {target: set(samples) for target, samples in allowed_by_target.items()}
    buckets: Dict[str, List[dict]] = {target: [] for target in allowed_by_target}

    for instance in instances:
        target = instance.get("target_hypothesis")
        sample = instance.get("sampling_hypothesis")
        if target not in allowed:
            continue
        if sample not in allowed[target]:
            continue
        if instance.get("distance_group") != distance_group:
            continue
        buckets[target].append(instance)

    return buckets


def _balanced_counts(targets: List[str], total: int) -> Dict[str, int]:
    if not targets:
        return {}
    base = total // len(targets)
    remainder = total % len(targets)
    counts = {target: base for target in targets}
    for target in sorted(targets)[:remainder]:
        counts[target] += 1
    return counts


def _sample_balanced(
    buckets: Dict[str, List[dict]],
    total: int,
    rng: random.Random,
    distance_group: str,
) -> List[dict]:
    targets = sorted(buckets.keys())
    if not targets:
        raise ValueError(f"No target properties configured for {distance_group}")

    desired = _balanced_counts(targets, total)

    shortages = []
    for target, needed in desired.items():
        available = len(buckets[target])
        if available < needed:
            shortages.append((target, needed, available))

    if shortages:
        details = "; ".join(
            f"{target}: need {needed}, available {available}"
            for target, needed, available in shortages
        )
        raise ValueError(
            f"Insufficient eligible instances for {distance_group} under balanced allocation ({total} requested): {details}"
        )

    selected: List[dict] = []
    for target in targets:
        selected.extend(rng.sample(buckets[target], desired[target]))

    rng.shuffle(selected)
    return selected


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build(
    close_instances_path: Path,
    deep_instances_path: Path,
    curated_hypotheses_path: Path,
    output_dir: Path,
    per_group: int,
    seed: int,
) -> None:
    curated_hypotheses = _load_json(curated_hypotheses_path)
    _validate_hypotheses(curated_hypotheses)

    close_instances = _load_json(close_instances_path)
    deep_instances = _load_json(deep_instances_path)

    rng = random.Random(seed)

    close_buckets = _filter_candidates(
        close_instances,
        curated_hypotheses["close"],
        "close",
    )
    deep_buckets = _filter_candidates(
        deep_instances,
        curated_hypotheses["deep"],
        "deep",
    )

    close_selected = _sample_balanced(close_buckets, per_group, rng, "close")
    deep_selected = _sample_balanced(deep_buckets, per_group, rng, "deep")

    combined = close_selected + deep_selected
    rng.shuffle(combined)

    close_out = output_dir / "curated_games_close.json"
    deep_out = output_dir / "curated_games_deep.json"
    combined_out = output_dir / "curated_games.json"

    _write_json(close_out, close_selected)
    _write_json(deep_out, deep_selected)
    _write_json(combined_out, combined)

    print("Curated games created successfully")
    print(f"  Seed: {seed}")
    print(f"  Close: {len(close_selected)} -> {close_out}")
    print(f"  Deep: {len(deep_selected)} -> {deep_out}")
    print(f"  Total: {len(combined)} -> {combined_out}")

    close_breakdown = {target: 0 for target in sorted(close_buckets)}
    for instance in close_selected:
        close_breakdown[instance["target_hypothesis"]] += 1

    deep_breakdown = {target: 0 for target in sorted(deep_buckets)}
    for instance in deep_selected:
        deep_breakdown[instance["target_hypothesis"]] += 1

    print("\nBreakdown by target_hypothesis")
    print("  close:")
    for target, count in close_breakdown.items():
        print(f"    - {target}: {count}")

    print("  deep:")
    for target, count in deep_breakdown.items():
        print(f"    - {target}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build curated game files (close/deep/combined) from curated hypotheses "
            "with balanced sampling across target properties."
        )
    )
    parser.add_argument(
        "--close-instances",
        type=Path,
        default=Path("data/game_instances_close.json"),
        help="Path to close-category game instances JSON",
    )
    parser.add_argument(
        "--deep-instances",
        type=Path,
        default=Path("data/game_instances_deep.json"),
        help="Path to deep-category game instances JSON",
    )
    parser.add_argument(
        "--curated-hypotheses",
        type=Path,
        default=Path("data/curated_hypotheses.json"),
        help="Path to curated hypotheses JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for curated game files",
    )
    parser.add_argument(
        "--per-group",
        dest="per_group",
        type=int,
        default=50,
        help="Number of curated instances per distance group",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling",
    )

    args = parser.parse_args()

    if args.per_group <= 0:
        raise ValueError("--per-group must be > 0")

    _build(
        close_instances_path=args.close_instances,
        deep_instances_path=args.deep_instances,
        curated_hypotheses_path=args.curated_hypotheses,
        output_dir=args.output_dir,
        per_group=args.per_group,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
