from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


TIER_BUCKETS: list[tuple[str, int, int | None]] = [
    ("Tier 1", 0, 2),
    ("Tier 2", 3, 5),
    ("Tier 3", 6, 10),
    ("Tier 4", 11, None),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute execution metrics with tiered selection quality, success rates, "
            "and recall-vs-success correlation."
        )
    )
    parser.add_argument(
        "--input-path",
        action="append",
        default=None,
        help=(
            "Path to judged JSON results. Can be passed multiple times for correlation across pipelines. "
            "Default: replanner_execution_results_judged.json"
        ),
    )
    parser.add_argument(
        "--input-glob",
        default=None,
        help="Optional glob pattern for multiple input files (e.g. all_scripts/**/results/*_judged.json)",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional path to save computed metrics as JSON",
    )
    return parser.parse_args()


def _safe_bool(dct: dict[str, Any], key: str) -> bool:
    return bool(dct.get(key, False))


def _safe_list(dct: dict[str, Any], key: str) -> list[Any]:
    value = dct.get(key, [])
    return value if isinstance(value, list) else []


def _safe_tool_names(raw_tools: list[Any]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for tool in raw_tools:
        if isinstance(tool, str):
            name = tool.strip()
        elif isinstance(tool, dict):
            name = str(tool.get("tool", "")).strip()
        else:
            name = ""
        if not name or name in seen:
            continue
        names.append(name)
        seen.add(name)
    return names


def _pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(100.0 * numerator / denominator, 2)


def _tier_by_reference_count(reference_count: int) -> str:
    for tier_name, lower, upper in TIER_BUCKETS:
        if upper is None and reference_count >= lower:
            return tier_name
        if upper is not None and lower <= reference_count <= upper:
            return tier_name
    return "Unknown"


def _calc_prf1(selected: set[str], reference: set[str]) -> tuple[float, float, float]:
    if not selected and not reference:
        return 1.0, 1.0, 1.0

    tp = len(selected & reference)
    fp = len(selected - reference)
    fn = len(reference - selected)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = _avg(xs)
    mean_y = _avg(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=False))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def _rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2
        for k in range(i, j + 1):
            idx = indexed[k][0]
            ranks[idx] = avg_rank
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    return _pearson(_rank(xs), _rank(ys))


def _extract_reference(item: dict[str, Any]) -> list[str]:
    if "reference" in item:
        return _safe_tool_names(_safe_list(item, "reference"))
    return []


def _extract_selected(item: dict[str, Any]) -> list[str]:
    if "selected_tools" in item:
        return _safe_tool_names(_safe_list(item, "selected_tools"))
    if "selected" in item:
        return _safe_tool_names(_safe_list(item, "selected"))
    return []


def compute_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(items)

    success = 0
    failed = 0
    no_tool_selected = 0
    selection_failed = 0
    has_runtime_error = 0

    fail_reasons = Counter()
    fail_reason_combo = Counter()
    fail_with_empty_output = 0

    selected_tool_counts: list[int] = []

    per_tier: dict[str, dict[str, Any]] = {
        tier_name: {
            "count": 0,
            "precision_values": [],
            "recall_values": [],
            "f1_values": [],
            "success_count": 0,
        }
        for tier_name, _, _ in TIER_BUCKETS
    }

    recall_all: list[float] = []
    success_all: list[float] = []
    recall_backend_filtered: list[float] = []
    success_backend_filtered: list[float] = []

    for item in items:
        judges = item.get("llm_judges", {}) if isinstance(item.get("llm_judges"), dict) else {}

        fully_answered = _safe_bool(judges, "response_fully_answers_question")
        needs_more_info = _safe_bool(judges, "response_requires_more_input_data")
        backend_failed = _safe_bool(judges, "response_failed_backend_problems")
        tools_useless = _safe_bool(judges, "response_failed_tools_useless")

        selected_tools = _extract_selected(item)
        selected_tool_counts.append(len(selected_tools))

        if item.get("error") == "selection_failed":
            selection_failed += 1
        if item.get("error"):
            has_runtime_error += 1

        if len(selected_tools) == 0:
            no_tool_selected += 1

        if fully_answered:
            success += 1
        else:
            failed += 1
            current_reasons: list[str] = []
            if needs_more_info:
                fail_reasons["needs_more_input_data"] += 1
                current_reasons.append("needs_more_input_data")
            if backend_failed:
                fail_reasons["backend_problems"] += 1
                current_reasons.append("backend_problems")
            if tools_useless:
                fail_reasons["tools_useless_or_insufficient"] += 1
                current_reasons.append("tools_useless_or_insufficient")

            if not current_reasons:
                fail_reasons["other_or_unclassified"] += 1
                current_reasons.append("other_or_unclassified")

            fail_reason_combo[" + ".join(sorted(current_reasons))] += 1

            output = item.get("execution_output")
            if output is None or str(output).strip() == "":
                fail_with_empty_output += 1

        reference_tools = _extract_reference(item)
        if reference_tools:
            selected_set = set(selected_tools)
            reference_set = set(reference_tools)
            precision, recall, f1 = _calc_prf1(selected_set, reference_set)
            tier = _tier_by_reference_count(len(reference_set))

            if tier in per_tier:
                per_tier[tier]["count"] += 1
                per_tier[tier]["precision_values"].append(precision)
                per_tier[tier]["recall_values"].append(recall)
                per_tier[tier]["f1_values"].append(f1)
                if fully_answered:
                    per_tier[tier]["success_count"] += 1

            recall_all.append(recall)
            success_all.append(1.0 if fully_answered else 0.0)

            if not backend_failed:
                recall_backend_filtered.append(recall)
                success_backend_filtered.append(1.0 if fully_answered else 0.0)

    failed_non_success = failed

    failed_and_no_tool_selected_count = sum(
        1
        for item in items
        if not _safe_bool(item.get("llm_judges", {}), "response_fully_answers_question")
        and len(_extract_selected(item)) == 0
    )

    tier_metrics: dict[str, Any] = {}
    for tier_name, values in per_tier.items():
        tier_count = values["count"]
        tier_metrics[tier_name] = {
            "count": tier_count,
            "selection_metrics": {
                "precision": round(_avg(values["precision_values"]), 4),
                "recall": round(_avg(values["recall_values"]), 4),
                "f1": round(_avg(values["f1_values"]), 4),
            },
            "success_rate_percent": round(100.0 * values["success_count"] / tier_count, 2)
            if tier_count
            else 0.0,
        }

    correlation = {
        "all_items": {
            "count": len(recall_all),
            "avg_recall": round(_avg(recall_all), 4),
            "success_rate_percent": round(100.0 * _avg(success_all), 2),
            "pearson": round(_pearson(recall_all, success_all), 4),
            "spearman": round(_spearman(recall_all, success_all), 4),
        },
        "backend_filtered": {
            "count": len(recall_backend_filtered),
            "avg_recall": round(_avg(recall_backend_filtered), 4),
            "success_rate_percent": round(100.0 * _avg(success_backend_filtered), 2),
            "pearson": round(_pearson(recall_backend_filtered, success_backend_filtered), 4),
            "spearman": round(_spearman(recall_backend_filtered, success_backend_filtered), 4),
        },
    }

    metrics = {
        "total_items": total,
        "tier_definition": {
            "Tier 1": "1-2 reference tools",
            "Tier 2": "3-5 reference tools",
            "Tier 3": "6-10 reference tools",
            "Tier 4": "11+ reference tools",
        },
        "success": {
            "count": success,
            "rate_percent": _pct(success, total),
        },
        "failed": {
            "count": failed,
            "rate_percent": _pct(failed, total),
        },
        "tool_selection": {
            "no_tool_selected_count": no_tool_selected,
            "no_tool_selected_rate_percent": _pct(no_tool_selected, total),
            "selection_failed_count": selection_failed,
            "selection_failed_rate_percent": _pct(selection_failed, total),
            "avg_selected_tool_count": round(sum(selected_tool_counts) / total, 3) if total else 0.0,
            "median_selected_tool_count": median(selected_tool_counts) if total else 0,
        },
        "errors": {
            "items_with_error_field_count": has_runtime_error,
            "items_with_error_field_rate_percent": _pct(has_runtime_error, total),
            "failed_with_empty_execution_output_count": fail_with_empty_output,
            "failed_with_empty_execution_output_rate_percent_of_failed": _pct(
                fail_with_empty_output,
                failed_non_success,
            ),
        },
        "failure_reasons_among_failed": {
            "counts": dict(fail_reasons),
            "rates_percent_of_failed": {
                reason: _pct(count, failed_non_success) for reason, count in fail_reasons.items()
            },
            "reason_combinations": dict(fail_reason_combo),
        },
        "conditional_metrics": {
            "failed_due_to_needs_more_info": {
                "count": fail_reasons.get("needs_more_input_data", 0),
                "rate_percent_of_failed": _pct(
                    fail_reasons.get("needs_more_input_data", 0),
                    failed_non_success,
                ),
            },
            "failed_due_to_backend": {
                "count": fail_reasons.get("backend_problems", 0),
                "rate_percent_of_failed": _pct(
                    fail_reasons.get("backend_problems", 0),
                    failed_non_success,
                ),
            },
            "failed_due_to_tools_useless": {
                "count": fail_reasons.get("tools_useless_or_insufficient", 0),
                "rate_percent_of_failed": _pct(
                    fail_reasons.get("tools_useless_or_insufficient", 0),
                    failed_non_success,
                ),
            },
            "failed_and_no_tool_selected": {
                "count": failed_and_no_tool_selected_count,
                "rate_percent_of_failed": _pct(
                    failed_and_no_tool_selected_count,
                    failed_non_success,
                ),
            },
        },
        "selection_metrics_by_tier": tier_metrics,
        "recall_success_correlation": correlation,
        "scatter_points": {
            "all_items": [
                {"recall": round(recall, 6), "success": int(success_bin)}
                for recall, success_bin in zip(recall_all, success_all, strict=False)
            ],
            "backend_filtered": [
                {"recall": round(recall, 6), "success": int(success_bin)}
                for recall, success_bin in zip(
                    recall_backend_filtered,
                    success_backend_filtered,
                    strict=False,
                )
            ],
        },
    }

    return metrics


def _resolve_input_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []

    if args.input_path:
        paths.extend(Path(value) for value in args.input_path)

    if args.input_glob:
        paths.extend(sorted(Path().glob(args.input_glob)))

    if not paths:
        paths = [Path(__file__).parent / "results/replanner_execution_results_judged.json"]

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        deduped.append(path)
        seen.add(resolved)

    return deduped


def _load_items(input_path: Path) -> list[dict[str, Any]]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Input JSON must be a list: {input_path}")
    return [item for item in data if isinstance(item, dict)]


def _compute_across_files(file_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    points_all = []
    points_backend = []

    for file_path, metrics in file_metrics.items():
        corr = metrics.get("recall_success_correlation", {})
        all_items = corr.get("all_items", {})
        backend_filtered = corr.get("backend_filtered", {})

        points_all.append(
            {
                "file": file_path,
                "avg_recall": all_items.get("avg_recall", 0.0),
                "success_rate_percent": all_items.get("success_rate_percent", 0.0),
                "count": all_items.get("count", 0),
            }
        )
        points_backend.append(
            {
                "file": file_path,
                "avg_recall": backend_filtered.get("avg_recall", 0.0),
                "success_rate_percent": backend_filtered.get("success_rate_percent", 0.0),
                "count": backend_filtered.get("count", 0),
            }
        )

    xs_all = [point["avg_recall"] for point in points_all]
    ys_all = [point["success_rate_percent"] for point in points_all]
    xs_backend = [point["avg_recall"] for point in points_backend]
    ys_backend = [point["success_rate_percent"] for point in points_backend]

    return {
        "points_all_items": points_all,
        "points_backend_filtered": points_backend,
        "pearson_all_items": round(_pearson(xs_all, ys_all), 4),
        "spearman_all_items": round(_spearman(xs_all, ys_all), 4),
        "pearson_backend_filtered": round(_pearson(xs_backend, ys_backend), 4),
        "spearman_backend_filtered": round(_spearman(xs_backend, ys_backend), 4),
    }


def main() -> None:
    args = parse_args()
    input_paths = _resolve_input_paths(args)

    per_file_metrics: dict[str, dict[str, Any]] = {}
    for input_path in input_paths:
        items = _load_items(input_path)
        per_file_metrics[str(input_path)] = compute_metrics(items)

    output: dict[str, Any]
    if len(per_file_metrics) == 1:
        output = next(iter(per_file_metrics.values()))
    else:
        output = {
            "per_file": per_file_metrics,
            "cross_file_correlation": _compute_across_files(per_file_metrics),
        }

    print(json.dumps(output, ensure_ascii=False, indent=2))

    if args.output_path:
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
