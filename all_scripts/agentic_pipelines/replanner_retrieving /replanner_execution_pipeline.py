import argparse
import json
import logging
import os
import sys
from pathlib import Path

from langchain_openai import ChatOpenAI

WORKSPACE_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(WORKSPACE_ROOT))
os.environ.setdefault("OPENAI_API_KEY", "dummy")

from all_scripts.execution_agent import ExecutionRequest, ToolCard, run_execution_agent
from all_scripts.agentic_pipelines.replanner_retrieving.replanner_retrieving import (
    BASELINE_LLM_MODEL,
    OPENROUTER_API_KEY,
    prepare_dataset_for_model,
    run_pipeline_for_benchmark,
)
from all_scripts.utils.config import OPENROUTER_URL

LOGGER = logging.getLogger(__name__)
DEFAULT_BENCHMARK_PATH = WORKSPACE_ROOT / "data/stabletoolbench/benchmark_cleared_api.json"
DEFAULT_TOOL_INDEX_PATH = WORKSPACE_ROOT / "data/stabletoolbench/tools_expanded_with_categories_checked.json"


def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=BASELINE_LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_URL,
        http_client=None,
        temperature=0.0,
    )


def run_item(item: dict, tool_index_path: Path, model: str, use_openrouter: bool) -> dict:
    selection_result = run_pipeline_for_benchmark(
        llm_decomposer=build_llm(),
        llm_agent=build_llm(),
        llm_replanner=build_llm(),
        benchmark=item,
    )
    if not selection_result:
        return {"question": item.get("query", ""), "error": "selection_failed"}

    selected_names = set(selection_result.get("selected", []))
    selected_tool_cards = [
        ToolCard.model_validate(tool)
        for tool in item.get("candidate_tools", [])
        if tool.get("name") in selected_names
    ]

    execution_output = run_execution_agent(
        ExecutionRequest(query=item.get("query", ""), toolset=selected_tool_cards),
        tool_index_path=tool_index_path,
        model=model,
        use_openrouter=use_openrouter,
    )

    return {
        "question": item.get("query", ""),
        "selected_tools": sorted(selected_names),
        "selected_tool_count": len(selected_tool_cards),
        "subtasks": selection_result.get("subtasks", []),
        "execution_output": execution_output,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select tools with replanner_retrieving and execute query via pydantic_execution_agent"
    )
    parser.add_argument("--benchmark-path", default=str(DEFAULT_BENCHMARK_PATH))
    parser.add_argument("--tool-index", default=str(DEFAULT_TOOL_INDEX_PATH))
    parser.add_argument("--results-path", default=str(Path(__file__).parent / "results/replanner_execution_results.json"))
    parser.add_argument("--item-index", type=int, default=None, help="Run only one benchmark item by original index")
    parser.add_argument("--limit", type=int, default=5, help="How many prepared benchmarks to process when --item-index is not set")
    parser.add_argument("--model", default="openai:gpt-4o-mini", help="Model for execution agent")
    parser.add_argument("--use-openrouter", action="store_true", help="Use OpenRouter in execution agent")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    raw_benchmarks = json.loads(Path(args.benchmark_path).read_text(encoding="utf-8"))
    prepared = prepare_dataset_for_model(raw_benchmarks)

    if args.item_index is not None:
        items = [item for item in prepared if item.get("id") == args.item_index]
    else:
        items = prepared[: args.limit]

    LOGGER.info("Prepared benchmark items: %d | selected for run: %d", len(prepared), len(items))

    results = []
    for item in items:
        LOGGER.info("Running benchmark id=%s", item.get("id"))
        try:
            result = run_item(
                item=item,
                tool_index_path=Path(args.tool_index),
                model=args.model,
                use_openrouter=args.use_openrouter,
            )
        except Exception as exc:
            LOGGER.exception("Pipeline failed for id=%s", item.get("id"))
            result = {
                "question": item.get("query", ""),
                "error": str(exc),
            }

        result["id"] = item.get("id")
        results.append(result)

    out_path = Path(args.results_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved %d results to %s", len(results), out_path)


if __name__ == "__main__":
    main()
