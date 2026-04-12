import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from utils.config import (  # noqa: E402
    OPENROUTER_API_KEY,
    OPENROUTER_URL,
    SUBTASK_K,
)

WORKSPACE_ROOT = Path(__file__).parent.parent.parent
DEFAULT_TEST_BENCHMARKS_PATH1 = str(WORKSPACE_ROOT / "data/stabletoolbench/top_benchmarks_enriched.json")
DEFAULT_TEST_BENCHMARKS_PATH2 = str(WORKSPACE_ROOT / "data/stabletoolbench/without_top_benchmarks_enriched.json")

BASELINE_LLM_MODEL = "openai/gpt-4o"
OPENROUTER_API_KEY = "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"
DECOMPOSER_SYSTEM_PROMPT = """
You are a task decomposer.

Goal:
Convert the original user request into the set of executable subtasks.

Inputs:
- Original query
- Ground-truth tool names as JSON list (tools that are truly required for successful execution)

Rules:
1. Preserve the original user intent, entities, constraints, and dependency order. Fulfill request in full.
2. Do not add exploratory, supporting, or background subtasks unless they are required for execution.
3. Each subtask must correspond to one concrete retrievable action.
4. Use the ground-truth tool list only as hidden execution guidance to choose viable decomposition paths.
5. Do not create subtasks for trivial reasoning, generic date logic, formatting, comparison, summarization, or other actions that do not require tools.
6. Each subtask must describe exactly one simple action.
7. Do not mention tool names or APIs.
8. Do not add explanations, markdown, numbering, or extra keys.

Output format:
Return ONLY a JSON array of strings.

Example: 
Original query:
"Please order a Spicy Hot Pot for me at the restaurant, add two extra servings of beef and a plate of hand-torn cabbage, then place the order using my table ID 10, and help me check out."

Best output:
["Obtain Spicy Hot Pot ID", "Select Spicy Hot Pot (Dish ID: <Spicy Hot Pot ID>, Quantity: '1')", "Obtain beef ID", "Add beef (Dish ID: <Beef ID>, Quantity: '2')", "Obtain hand-torn cabbage ID", "Add hand-torn cabbage (Dish ID: <Hand-torn Cabbage ID>, Quantity: '1')", "Place order using table ID (Order ID: Updated Order Number, Table ID: '10')", "Complete payment operation (Order ID: Updated Order Number, Payment Amount: Order Amount)"]
""".strip()

AGENT_SYSTEM_PROMPT = """
You are a tool selector.

Goal:
For the current subtask, select the single best tool from the provided toolset candidates.

Available context:
- Original query
- Current subtask
- Executed history
- Toolset candidates as a JSON list of objects

Toolset notes:
- Each tool has fields such as name, description, input_structure, arguments, and results.
- Use the exact `name` field as the tool identifier in the output.

Selection rules:
1. Select tool only from the provided toolset candidates.
2. Choose a tool only if it can directly help execute the current subtask.
3. Prefer semantic fit, input compatibility, and result compatibility over general topical similarity.
4. Do not choose a tool just because it is loosely related to the topic.
5. If no candidate can directly support the subtask, return an empty list.
6. Return at most 1 unique tool name.
7. Keep consistency with previously selected tools only when that tool is still the best fit.
8. Return tool names exactly as written in the `name` field.
9. Do not invent tools or alter tool names.


Output format:
Return ONLY a JSON array of tool name strings.
Examples:
["tool_a"]
""".strip()
REPLANNER_SYSTEM_PROMPT = """
You are a replanner.

Goal:
Rebuild only the remaining executable subtasks for the original query.

Inputs:
- Original query
- Executed history
- Subtasks to replan
- Ground-truth tool names as JSON list (tools that are truly required for successful execution)

Replan only:
- pending subtasks not executed yet;
- executed subtasks whose selected_tools was empty;
- later subtasks that depended on those failed subtasks.

Rules:
1. Preserve the original intent and constraints.
2. Never include subtasks that were already executed successfully.
3. A rewritten failed subtask must not be a wording variant of the failed one.
4. A valid rewrite must change the method, the intermediate target, or the decomposition.
5. Before proposing a subtask, consider implicitly:
   - whether it needs a tool;
   - whether it is executable under the provided ground-truth tool constraints;
   - whether it can be replaced by smaller executable steps.
6. If no ground-truth-supported route can support a failed subtask, do not restate it.
   Instead:
   - replace it with an alternative executable route, or
   - decompose it into executable intermediate steps, or
   - abandon that route.
7. Prefer subtasks that target observable intermediate facts retrievable with the ground-truth tool knowledge.
8. Do not create subtasks for trivial reasoning, date inference, simple comparison, summarization, or formatting.
9. Each subtask must contain exactly one action.
10. Keep subtasks concise, self-contained, and logically ordered.
11. Do not mention tool names, APIs, explanations, numbering, or metadata.

Output format:
Return ONLY a JSON array of strings.
""".strip()
LOGGER = logging.getLogger(__name__)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_json_strict(text: str):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        parsed = json.loads(text)
    except Exception as e:
        parsed = None
        parse_error = str(e)
    else:
        parse_error = None

    if isinstance(parsed, list):
        return parsed, None

    if isinstance(parsed, dict):
        for key in ("subtasks", "tools", "selected_tools", "result", "items"):
            value = parsed.get(key)
            if isinstance(value, list):
                return value, None

    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            parsed_list = json.loads(match.group(0))
            if isinstance(parsed_list, list):
                return parsed_list, None
        except Exception:
            pass

    return None, parse_error or "Response is not a JSON array"


def prepare_dataset_for_model(data):
    print("Подготовка данных: toolset-кандидаты + reference GT...")
    prepared_items = []

    for i, item in enumerate(data):
        query = item.get("question", "")

        gt_names = []
        seen = set()
        if "reference" in item:
            for ref in item["reference"]:
                tool_name = ref.get("tool")
                if tool_name and tool_name not in seen:
                    gt_names.append(tool_name)
                    seen.add(tool_name)

        if not gt_names:
            continue

        toolset_candidates = item.get("toolset", [])
        available_tool_names = {tool.get("name") for tool in toolset_candidates if tool.get("name")}

        gt_names = [name for name in gt_names if name in available_tool_names]
        if not gt_names:
            continue

        count = len(gt_names)
        if count <= 2:
            tier = "Tier 1"
        elif 3 <= count <= 5:
            tier = "Tier 2"
        elif 6 <= count <= 10:
            tier = "Tier 3"
        else:
            tier = "Tier 4+"

        prepared_items.append(
            {
                "id": i,
                "query": query,
                "candidate_tools": toolset_candidates,
                "gt_names": gt_names,
                "tier": tier,
                "raw_item": item,
            }
        )

    return prepared_items


def format_toolset_json(candidate_tools: list[dict[str, Any]]):
    return json.dumps(candidate_tools, ensure_ascii=False, indent=2)


def format_ground_truth_tools_json(gt_tool_names: list[str]):
    return json.dumps(gt_tool_names, ensure_ascii=False, indent=2)


def normalize_tool_name(raw_name: str):
    cleaned = raw_name.strip().strip("`\"'")
    cleaned = re.sub(r"^[-*\d.\)\s]+", "", cleaned).strip()
    return cleaned


def map_to_allowed_tools(candidates: list[Any], allowed_tools: set[str]):
    allowed_map = {name.lower(): name for name in allowed_tools}
    selected = []
    for item in candidates:
        if not isinstance(item, str):
            continue
        normalized = normalize_tool_name(item)
        matched = None
        if normalized in allowed_tools:
            matched = normalized
        else:
            matched = allowed_map.get(normalized.lower())
        if matched and matched not in selected:
            selected.append(matched)
    return selected


def invoke_decomposer(llm, query: str, ground_truth_tools_json: str):
    # Logical block: Decomposition LLM call
    ground_truth_tools_json = ground_truth_tools_json.replace("{", "{{").replace("}", "}}")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", DECOMPOSER_SYSTEM_PROMPT),
            (
                "user",
                "Ground-truth tools (JSON):\n"
                + ground_truth_tools_json
                + "\n\nOriginal query:\n{query}",
            ),
        ]
    ).format_prompt(query=query)
    response = llm.invoke(prompt).content
    response = response.replace("<|fim_middle|>", "").replace("<|fim_prefix|>", "").replace("<|fim_suffix|>", "").strip()
    parsed, err = parse_json_strict(response)
    if err or not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, str) and item.strip()]




def try_repair_agent_response(llm, raw_response: str, allowed_tools: set[str]):
    repair_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract tool names from the text and return ONLY a JSON array of strings. "
                "Use only allowed tool names.",
            ),
            (
                "user",
                "Allowed tool names:\n{allowed_tools}\n\nText to repair:\n{raw_response}",
            ),
        ]
    ).format_prompt(
        allowed_tools=json.dumps(sorted(list(allowed_tools)), ensure_ascii=False),
        raw_response=raw_response[:8000],
    )
    repaired = llm.invoke(repair_prompt).content.strip()
    parsed, err = parse_json_strict(repaired)
    if err or not isinstance(parsed, list):
        return []
    return map_to_allowed_tools(parsed, allowed_tools)[:SUBTASK_K]



def tokenize_for_match(text: str):
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())



def invoke_agent(
    llm,
    query: str,
    current_subtask: str,
    history: list[dict[str, Any]],
    toolset_json: str,
    allowed_tools: set[str],
):
    # Logical block: Tool-selection agent for current subtask
    toolset_json = toolset_json.replace("{", "{{").replace("}", "}}")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", AGENT_SYSTEM_PROMPT),
            (
                "user",
                "Original query:\n{query}\n\n"
                "Current subtask:\n{current_subtask}\n\n"
                "Executed subtasks and selected tools:\n{history}\n\n"
                "Toolset candidates (JSON):\n"
                + toolset_json,
            ),
        ]
    ).format_prompt(
        query=query,
        current_subtask=current_subtask,
        history=json.dumps(history, ensure_ascii=False, indent=2),
        subtask_k=SUBTASK_K,
    )

    response = llm.invoke(prompt).content
    response = response.replace("<|fim_middle|>", "").replace("<|fim_prefix|>", "").replace("<|fim_suffix|>", "").strip()
    parsed, err = parse_json_strict(response)
    if err or not isinstance(parsed, list):
        LOGGER.warning(
            "Agent returned non-list JSON. subtask=%r parse_error=%s raw_response=%r",
            current_subtask,
            err,
            response,
        )
        repaired_selection = try_repair_agent_response(llm, response, allowed_tools)
        if repaired_selection:
            LOGGER.info(
                "Recovered tool selection after repair | subtask=%r | selected=%s",
                current_subtask,
                repaired_selection,
            )
            return repaired_selection


        return []

    selected = map_to_allowed_tools(parsed, allowed_tools)
    normalized_selected = {normalize_tool_name(item) for item in selected}
    dropped_tools = [
        item
        for item in parsed
        if isinstance(item, str) and normalize_tool_name(item) not in normalized_selected
    ]
    LOGGER.info(
        "Tool selection | subtask=%r | model_candidates=%s | selected=%s | dropped=%s",
        current_subtask,
        parsed,
        selected[:SUBTASK_K],
        dropped_tools,
    )
    return selected[:SUBTASK_K]


def invoke_replanner(
    llm,
    query: str,
    pending_subtasks: list[str],
    history: list[dict[str, Any]],
    ground_truth_tools_json: str,
):
    # Logical block: Replanner LLM call for subtasks requiring re-execution
    ground_truth_tools_json = ground_truth_tools_json.replace("{", "{{").replace("}", "}}")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REPLANNER_SYSTEM_PROMPT),
            (
                "user",
                "Original query:\n{query}\n\n"
                "Executed history:\n{history}\n\n"
                "Subtasks to replan (pending + failed with empty selected_tools):\n{pending_subtasks}"
                "\n\nGround-truth tools (JSON):\n"
                + ground_truth_tools_json,
            ),
        ]
    ).format_prompt(
        query=query,
        history=json.dumps(history, ensure_ascii=False, indent=2),
        pending_subtasks=json.dumps(pending_subtasks, ensure_ascii=False, indent=2),

    )

    response = llm.invoke(prompt).content
    response = response.replace("<|fim_middle|>", "").replace("<|fim_prefix|>", "").replace("<|fim_suffix|>", "").strip()
    parsed, err = parse_json_strict(response)
    if err or not isinstance(parsed, list):
        return pending_subtasks
    replanned = [item for item in parsed if isinstance(item, str) and item.strip()]
    return replanned if replanned else pending_subtasks


def run_pipeline_for_benchmark(
    llm_decomposer,
    llm_agent,
    llm_replanner,
    benchmark,
):
    query = benchmark.get("query", "")
    print(query)
    if not query:
        return None

    reference_tool_names = set(benchmark.get("gt_names", []))
    candidate_tools = benchmark.get("candidate_tools", [])
    reference_candidates = [tool["name"] for tool in candidate_tools if tool.get("name")]

    if not reference_tool_names or not reference_candidates:
        return None

    toolset_json = format_toolset_json(candidate_tools)
    ground_truth_tools_json = format_ground_truth_tools_json(sorted(list(reference_tool_names)))
    LOGGER.debug("Prepared compact toolset JSON for prompt | tools=%s | chars=%s", len(candidate_tools), len(toolset_json))
    LOGGER.debug(
        "Prepared ground-truth tools JSON for decomposition/replanning | tools=%s | chars=%s",
        len(reference_tool_names),
        len(ground_truth_tools_json),
    )

    subtasks = invoke_decomposer(
        llm_decomposer,
        query=query,
        ground_truth_tools_json=ground_truth_tools_json,
    )
    if not subtasks:
        subtasks = [query]

    history = []
    pending_subtasks = list(subtasks)
    max_iterations = max(1, len(subtasks) * 3)
    iterations = 0

    while pending_subtasks and iterations < max_iterations:
        iterations += 1
        current_subtask = pending_subtasks.pop(0)
        selected_tools = invoke_agent(
            llm_agent,
            query=query,
            current_subtask=current_subtask,
            history=history,
            toolset_json=toolset_json,
            allowed_tools=set(reference_candidates),
        )

        history.append(
            {
                "subtask": current_subtask,
                "selected_tools": selected_tools,
            }
        )
        LOGGER.debug(
            "Benchmark step | query=%r | subtask=%r | selected_tools=%s",
            query,
            current_subtask,
            selected_tools,
        )

        needs_replan = not selected_tools
        if needs_replan:
            pending_subtasks = [current_subtask] + pending_subtasks

        if pending_subtasks:
            pending_subtasks = invoke_replanner(
                llm_replanner,
                query=query,
                pending_subtasks=pending_subtasks,
                history=history,
                ground_truth_tools_json=ground_truth_tools_json,
            )

    all_selected_tools = {tool for item in history for tool in item["selected_tools"]}
    selected_list = sorted(list(all_selected_tools))

    tp = len(all_selected_tools & reference_tool_names)
    precision = tp / len(selected_list) if selected_list else 0.0
    recall = tp / len(reference_tool_names) if reference_tool_names else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    zero_metric_reasons = []
    if not selected_list:
        zero_metric_reasons.append("no_tools_selected")
    if selected_list and tp == 0:
        zero_metric_reasons.append("selected_tools_do_not_intersect_reference")
    if not reference_tool_names:
        zero_metric_reasons.append("reference_is_empty")

    if precision == 0.0 or recall == 0.0:
        LOGGER.warning(
            "Zero metric detected | query=%r | reasons=%s | selected=%s | reference=%s | tp=%s",
            query,
            zero_metric_reasons or ["unknown_reason"],
            selected_list,
            sorted(list(reference_tool_names)),
            tp,
        )

    return {
        "question": query,
        "subtasks": history,
        "reference": sorted(list(reference_tool_names)),
        "selected": selected_list,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "zero_metric_reasons": zero_metric_reasons,
    }


def evaluate_pipeline(
    model_name: str,
    benchmarks,
    llm_decomposer,
    llm_agent,
    llm_replanner,
):
    precisions, recalls, f1s = [], [], []
    detail_table = []

    for idx, benchmark in enumerate(tqdm(benchmarks, desc=f"Evaluating {model_name}")):
        print("question:", benchmark)
        result = run_pipeline_for_benchmark(
            llm_decomposer=llm_decomposer,
            llm_agent=llm_agent,
            llm_replanner=llm_replanner,
            benchmark=benchmark,
        )
        if result is None:
            continue

        precisions.append(result["precision"])
        recalls.append(result["recall"])
        f1s.append(result["f1"])

        detail_table.append(
            {
                "idx": idx,
                **result,
            }
        )

    return {
        "precision": float(np.mean(precisions) if precisions else 0.0),
        "recall": float(np.mean(recalls) if recalls else 0.0),
        "f1": float(np.mean(f1s) if f1s else 0.0),
        "detail_table": detail_table,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Replanner-based decomposed retrieval pipeline")
    parser.add_argument(
        "--benchmarks-path1",
        default=DEFAULT_TEST_BENCHMARKS_PATH1,
        help="Path to benchmark questions json",
    )
    parser.add_argument(
        "--benchmarks-path2",
        default=DEFAULT_TEST_BENCHMARKS_PATH2,
        help="Path to benchmark questions json",
    )
    parser.add_argument(
        "--results-dir",
        default=str(Path(__file__).parent / "results"),
        help="Directory for output files",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    print("=" * 80)
    print("Replanner Retrieval Evaluation")
    print("=" * 80)
    print(f"Benchmarks: {args.benchmarks_path1}")
    print(f"Results dir: {args.results_dir}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    benchmarks1 = load_json(args.benchmarks_path1)
    benchmarks2 = load_json(args.benchmarks_path1)
    benchmarks = benchmarks1+benchmarks2
    del benchmarks1, benchmarks2

    benchmarks=benchmarks[0:5]
    print(f"Loaded {len(benchmarks)} benchmarks")

    prepared_benchmarks = prepare_dataset_for_model(benchmarks)
    filtered_benchmarks = [
        item for item in prepared_benchmarks if item["tier"] in {"Tier 3", "Tier 4+"}
    ]
    print(f"Prepared benchmarks: {len(prepared_benchmarks)}")
    print(f"Filtered benchmarks (Tier 3 & Tier 4+): {len(filtered_benchmarks)}")

    llm_decomposer = ChatOpenAI(
        model=BASELINE_LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_URL,
        http_client=None,
        temperature=0.0,
    )
    llm_agent = ChatOpenAI(
        model=BASELINE_LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_URL,
        http_client=None,
        temperature=0.0,
    )
    llm_replanner = ChatOpenAI(
        model=BASELINE_LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_URL,
        http_client=None,
        temperature=0.0,
    )

    all_results = {}

    base_results = evaluate_pipeline(
        model_name="baseline_4o-mini",
        benchmarks=filtered_benchmarks,
        llm_decomposer=llm_decomposer,
        llm_agent=llm_agent,
        llm_replanner=llm_replanner,
    )
    all_results["base"] = base_results

    base_result_file = results_dir / "replanner_base_results.json"
    with open(base_result_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "completed": [item["idx"] for item in base_results["detail_table"]],
                "detail_table": base_results["detail_table"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Model':<30} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 80)
    for model_key, model_label in [("base", "Baseline (qwen2.5-coder-7b-instruct)")]:
        if model_key in all_results:
            res = all_results[model_key]
            print(
                f"{model_label:<30} {res['precision']:<12.4f} {res['recall']:<12.4f} {res['f1']:<12.4f}"
            )


if __name__ == "__main__":
    main()
