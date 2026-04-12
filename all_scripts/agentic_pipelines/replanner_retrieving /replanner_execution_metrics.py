from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute aggregate metrics from replanner_execution_results_judged.json"
    )
    parser.add_argument(
        "--input-path",
        default=str(Path(__file__).parent / "results/replanner_execution_results_judged.json"),
        help="Path to judged JSON results",
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


def _pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(100.0 * numerator / denominator, 2)


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

    for item in items:
        judges = item.get("llm_judges", {}) if isinstance(item.get("llm_judges"), dict) else {}

        fully_answered = _safe_bool(judges, "response_fully_answers_question")
        needs_more_info = _safe_bool(judges, "response_requires_more_input_data")
        backend_failed = _safe_bool(judges, "response_failed_backend_problems")
        tools_useless = _safe_bool(judges, "response_failed_tools_useless")

        selected_tools = _safe_list(item, "selected_tools")
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

    failed_non_success = failed

    failed_and_no_tool_selected_count = sum(
        1
        for item in items
        if not _safe_bool(item.get("llm_judges", {}), "response_fully_answers_question")
        and len(_safe_list(item, "selected_tools")) == 0
    )

    metrics = {
        "total_items": total,
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
    }

    return metrics


def main() -> None:
    args = parse_args()
    items = json.loads(Path(args.input_path).read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("Input JSON must be a list")

    metrics = compute_metrics(items)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.output_path:
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
