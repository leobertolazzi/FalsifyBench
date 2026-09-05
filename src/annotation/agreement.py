"""Compute agreement metrics between paired LLM and human annotation CSV files.

Usage examples:
    python -m src.annotation.agreement
    python -m src.annotation.agreement --input-dir results/annotation_pairs_oracle
    python -m src.annotation.agreement --output results/annotation_pairs_oracle/agreement_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


DEFAULT_INPUT_DIR = Path("results/annotation_pairs_oracle")
JUDGMENT_FIELD = "judgment"
MISSING_JUDGMENTS = {"", "-", "na", "n/a", "none", "null"}


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize_judgment(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in MISSING_JUDGMENTS:
        return ""
    return normalized


def _pair_name_from_path(path: Path) -> str:
    name = path.stem
    if name.endswith("_llm"):
        return name[:-4]
    if name.endswith("_human"):
        return name[:-6]
    return name


def _find_annotation_pairs(input_dir: Path) -> List[Tuple[str, Path, Path]]:
    llm_files = sorted(input_dir.glob("*_llm.csv"))
    pairs: List[Tuple[str, Path, Path]] = []
    for llm_path in llm_files:
        pair_name = _pair_name_from_path(llm_path)
        human_path = input_dir / f"{pair_name}_human.csv"
        if human_path.exists():
            pairs.append((pair_name, llm_path, human_path))
    return pairs


def _row_key(row: Dict[str, str], fieldnames: Sequence[str]) -> Tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in fieldnames)


def _build_row_index(rows: Iterable[Dict[str, str]], key_fields: Sequence[str]) -> Dict[Tuple[str, ...], Dict[str, str]]:
    index: Dict[Tuple[str, ...], Dict[str, str]] = {}
    for row in rows:
        key = _row_key(row, key_fields)
        if key in index:
            raise ValueError(f"Duplicate row key encountered: {key}")
        index[key] = row
    return index


def _cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float | None:
    if not labels_a or len(labels_a) != len(labels_b):
        return None

    categories = sorted(set(labels_a) | set(labels_b))
    if len(categories) <= 1:
        return 1.0

    total = len(labels_a)
    observed = sum(1 for left, right in zip(labels_a, labels_b) if left == right) / total

    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    expected = sum((counts_a[category] / total) * (counts_b[category] / total) for category in categories)

    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def _agreement_metrics(labels_a: Sequence[str], labels_b: Sequence[str]) -> Dict[str, Any]:
    compared_count = len(labels_a)
    matches = sum(1 for left, right in zip(labels_a, labels_b) if left == right)
    percent_agreement = (matches / compared_count) if compared_count else None
    return {
        "rows_compared": compared_count,
        "matches": matches,
        "mismatches": compared_count - matches,
        "percent_agreement": percent_agreement,
        "cohen_kappa": _cohen_kappa(labels_a, labels_b),
        "llm_label_counts": dict(sorted(Counter(labels_a).items())),
        "human_label_counts": dict(sorted(Counter(labels_b).items())),
    }


def _group_comparable_rows(rows: Sequence[Dict[str, str]], group_field: str) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        group_value = str(row.get(group_field, "")).strip()
        if not group_value:
            continue
        grouped.setdefault(group_value, []).append(row)

    summaries: Dict[str, Dict[str, Any]] = {}
    for group_value in sorted(grouped):
        group_rows = grouped[group_value]
        summaries[group_value] = _agreement_metrics(
            [row["llm_judgment"] for row in group_rows],
            [row["human_judgment"] for row in group_rows],
        )
    return summaries


def _group_rows_by_fields(
    rows: Sequence[Dict[str, str]],
    primary_field: str,
    secondary_field: str,
) -> Dict[str, Dict[str, Any]]:
    primary_summaries = _group_comparable_rows(rows, primary_field)
    grouped_rows: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        primary_value = str(row.get(primary_field, "")).strip()
        if not primary_value:
            continue
        grouped_rows.setdefault(primary_value, []).append(row)

    for primary_value, summary in primary_summaries.items():
        summary[secondary_field] = _group_comparable_rows(
            grouped_rows.get(primary_value, []),
            secondary_field,
        )

    return primary_summaries


def _compute_pair_agreement_details(
    llm_path: Path,
    human_path: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, str]], List[str]]:
    llm_rows = _read_csv_rows(llm_path)
    human_rows = _read_csv_rows(human_path)

    llm_fieldnames = list(llm_rows[0].keys()) if llm_rows else []
    human_fieldnames = list(human_rows[0].keys()) if human_rows else []
    if llm_fieldnames != human_fieldnames:
        raise ValueError(
            f"Header mismatch between {llm_path.name} and {human_path.name}: "
            f"{llm_fieldnames} != {human_fieldnames}"
        )

    key_fields = [field for field in llm_fieldnames if field != JUDGMENT_FIELD]
    llm_index = _build_row_index(llm_rows, key_fields)
    human_index = _build_row_index(human_rows, key_fields)

    llm_keys = set(llm_index)
    human_keys = set(human_index)
    shared_keys = sorted(llm_keys & human_keys)

    llm_only = sorted(llm_keys - human_keys)
    human_only = sorted(human_keys - llm_keys)

    comparable_rows: List[Dict[str, str]] = []
    disagreements: List[Dict[str, str]] = []
    missing_human_count = 0

    for key in shared_keys:
        llm_row = llm_index[key]
        human_row = human_index[key]
        llm_judgment = _normalize_judgment(llm_row.get(JUDGMENT_FIELD, ""))
        human_judgment = _normalize_judgment(human_row.get(JUDGMENT_FIELD, ""))

        if not human_judgment:
            missing_human_count += 1
            continue

        if not llm_judgment:
            continue

        comparable_row = {
            **{field: str(llm_row.get(field, "")) for field in key_fields},
            "llm_judgment": llm_judgment,
            "human_judgment": human_judgment,
        }
        comparable_rows.append(comparable_row)

        if llm_judgment != human_judgment:
            disagreements.append(comparable_row)

    metrics = _agreement_metrics(
        [row["llm_judgment"] for row in comparable_rows],
        [row["human_judgment"] for row in comparable_rows],
    )

    summary = {
        "pair_name": _pair_name_from_path(llm_path),
        "llm_file": str(llm_path),
        "human_file": str(human_path),
        "rows_in_llm": len(llm_rows),
        "rows_in_human": len(human_rows),
        "rows_shared": len(shared_keys),
        "rows_only_in_llm": len(llm_only),
        "rows_only_in_human": len(human_only),
        "rows_with_missing_human_judgment": missing_human_count,
        **metrics,
        "by_model": _group_comparable_rows(comparable_rows, "model"),
        "by_target_property": (
            _group_comparable_rows(comparable_rows, "target_property")
            if "target_property" in key_fields
            else {}
        ),
        "disagreements": disagreements,
    }
    return summary, comparable_rows, key_fields


def compute_pair_agreement(llm_path: Path, human_path: Path) -> Dict[str, Any]:
    summary, _, _ = _compute_pair_agreement_details(llm_path, human_path)
    return summary


def compute_agreement_summary(input_dir: Path) -> Dict[str, Any]:
    pairs = _find_annotation_pairs(input_dir)
    if not pairs:
        return {"input_dir": str(input_dir), "pair_summaries": []}

    pair_summaries: List[Dict[str, Any]] = []
    all_comparable_rows: List[Dict[str, str]] = []
    comparable_rows_with_target_property: List[Dict[str, str]] = []
    for pair_name, llm_path, human_path in pairs:
        pair_summary, comparable_rows, key_fields = _compute_pair_agreement_details(llm_path, human_path)
        pair_summaries.append(pair_summary)
        all_comparable_rows.extend({**row, "pair_name": pair_name} for row in comparable_rows)
        if "target_property" in key_fields:
            comparable_rows_with_target_property.extend(
                {**row, "pair_name": pair_name} for row in comparable_rows
            )

    compared_counts = [summary["rows_compared"] for summary in pair_summaries]
    percent_values = [summary["percent_agreement"] for summary in pair_summaries if summary["percent_agreement"] is not None]
    kappa_values = [summary["cohen_kappa"] for summary in pair_summaries if summary["cohen_kappa"] is not None]

    overall_compared = sum(compared_counts)
    overall_matches = sum(summary["matches"] for summary in pair_summaries)

    return {
        "input_dir": str(input_dir),
        "pair_summaries": pair_summaries,
        "overall": {
            "pairs_found": len(pair_summaries),
            "rows_compared": overall_compared,
            "matches": overall_matches,
            "mismatches": overall_compared - overall_matches,
            "micro_percent_agreement": (overall_matches / overall_compared) if overall_compared else None,
            "macro_percent_agreement": (sum(percent_values) / len(percent_values)) if percent_values else None,
            "macro_cohen_kappa": (sum(kappa_values) / len(kappa_values)) if kappa_values else None,
            "by_file": _group_comparable_rows(all_comparable_rows, "pair_name"),
            "by_model": _group_rows_by_fields(all_comparable_rows, "model", "pair_name"),
            "by_target_property": _group_comparable_rows(
                comparable_rows_with_target_property,
                "target_property",
            ),
        },
    }


def _print_summary(summary: Dict[str, Any]) -> None:
    pair_summaries = summary.get("pair_summaries", [])
    if not pair_summaries:
        print("No paired annotation CSV files found.")
        return

    print(f"Found {len(pair_summaries)} annotation pair(s) in {summary['input_dir']}")
    for pair_summary in pair_summaries:
        percent = pair_summary["percent_agreement"]
        kappa = pair_summary["cohen_kappa"]
        percent_text = f"{percent:.1%}" if percent is not None else "n/a"
        kappa_text = f"{kappa:.3f}" if kappa is not None else "n/a"
        print(
            f"- {pair_summary['pair_name']}: compared={pair_summary['rows_compared']}, "
            f"agreement={percent_text}, kappa={kappa_text}, "
            f"missing_human={pair_summary['rows_with_missing_human_judgment']}"
        )

    overall = summary["overall"]
    micro = overall["micro_percent_agreement"]
    macro = overall["macro_percent_agreement"]
    macro_kappa = overall["macro_cohen_kappa"]
    print(
        "Overall: "
        f"rows_compared={overall['rows_compared']}, "
        f"micro_agreement={(f'{micro:.1%}' if micro is not None else 'n/a')}, "
        f"macro_agreement={(f'{macro:.1%}' if macro is not None else 'n/a')}, "
        f"macro_kappa={(f'{macro_kappa:.3f}' if macro_kappa is not None else 'n/a')}"
    )

    by_model = overall.get("by_model", {})
    if by_model:
        print("By model:")
        for model_name, model_summary in by_model.items():
            percent = model_summary["percent_agreement"]
            kappa = model_summary["cohen_kappa"]
            print(
                f"- {model_name}: compared={model_summary['rows_compared']}, "
                f"agreement={(f'{percent:.1%}' if percent is not None else 'n/a')}, "
                f"kappa={(f'{kappa:.3f}' if kappa is not None else 'n/a')}"
            )
            by_file = model_summary.get("pair_name", {})
            for pair_name, pair_summary in by_file.items():
                pair_percent = pair_summary["percent_agreement"]
                pair_kappa = pair_summary["cohen_kappa"]
                print(
                    f"  {pair_name}: compared={pair_summary['rows_compared']}, "
                    f"agreement={(f'{pair_percent:.1%}' if pair_percent is not None else 'n/a')}, "
                    f"kappa={(f'{pair_kappa:.3f}' if pair_kappa is not None else 'n/a')}"
                )

    by_file = overall.get("by_file", {})
    if by_file:
        print("By file:")
        for pair_name, pair_summary in by_file.items():
            percent = pair_summary["percent_agreement"]
            kappa = pair_summary["cohen_kappa"]
            print(
                f"- {pair_name}: compared={pair_summary['rows_compared']}, "
                f"agreement={(f'{percent:.1%}' if percent is not None else 'n/a')}, "
                f"kappa={(f'{kappa:.3f}' if kappa is not None else 'n/a')}"
            )

    by_target_property = overall.get("by_target_property", {})
    if by_target_property:
        print("By target property:")
        for target_property, target_summary in by_target_property.items():
            percent = target_summary["percent_agreement"]
            kappa = target_summary["cohen_kappa"]
            print(
                f"- {target_property}: compared={target_summary['rows_compared']}, "
                f"agreement={(f'{percent:.1%}' if percent is not None else 'n/a')}, "
                f"kappa={(f'{kappa:.3f}' if kappa is not None else 'n/a')}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute agreement between paired LLM and human annotation CSVs"
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing *_llm.csv and *_human.csv files (default: results/annotation_pairs_oracle)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON file to write the agreement summary",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    summary = compute_agreement_summary(input_dir)
    _print_summary(summary)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()