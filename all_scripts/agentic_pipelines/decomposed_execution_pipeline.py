import argparse
import importlib.metadata
import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

WORKSPACE_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(WORKSPACE_ROOT))
from dotenv import load_dotenv
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

from all_scripts.retrieving.decomposed_retrieving import (
    retrieve_tools_for_subtask as retrieve_tools_for_subtask_shared,
    BASE_MODEL,
    DECOMPOSITIONS_PATH,
    DEVICE,
    PASSAGE_PREFIX,
    SUBTASK_K,
    TEST_BENCHMARKS_PATH,
    TOOLS_PATH,
    load_json,
)

from all_scripts.utils.data_utils import structure_tool

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

LOGGER = logging.getLogger(__name__)
DEFAULT_RESULTS_PATH = Path(__file__).parent / "results/decomposed_execution_results.json"


def resolve_sentence_transformer_cls() -> Any:
    hub_version = importlib.metadata.version("huggingface-hub")
    if hub_version.startswith("1."):
        raise RuntimeError(
            "Incompatible dependency detected: huggingface-hub=="
            f"{hub_version}. Please install a 0.x version required by current transformers/sentence-transformers, "
            "for example: pip install \"huggingface-hub>=0.34.0,<1.0\""
        )

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer


def build_tool_embeddings(
    model: "SentenceTransformer",
    tools_dict: dict[str, dict],
    tool_names: list[str],
    use_prefixes: bool,
) -> torch.Tensor:
    if use_prefixes:
        tool_passages = [structure_tool(name, tools_dict[name], PASSAGE_PREFIX) for name in tool_names]
    else:
        tool_passages = [structure_tool(name, tools_dict[name], passage_prefix="") for name in tool_names]

    LOGGER.info("Encoding %d tools", len(tool_passages))
    tool_embeddings = model.encode(
        tool_passages,
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=32,
    )
    return tool_embeddings.cpu()




def get_reference_tool_names(benchmark: dict) -> set[str]:
    if "reference" in benchmark:
        return {ref["tool"] for ref in benchmark["reference"]}

    reference_toolset = benchmark.get("toolset", [])
    return {tool["name"] for tool in reference_toolset}


def build_subtasks_index(decompositions: list[dict], decomposition_model_key: str) -> dict[str, list[str]]:
    selected = [item for item in decompositions if item.get("model") == decomposition_model_key]
    return {item.get("query", ""): item.get("subtasks", []) for item in selected}


def get_subtasks_for_query(subtasks_by_query: dict[str, list[str]], query: str) -> list[str]:
    subtasks = subtasks_by_query.get(query)
    if subtasks is not None:
        return subtasks

    normalized_query = " ".join(query.split())
    for decomp_query, decomp_subtasks in subtasks_by_query.items():
        if " ".join(decomp_query.split()) == normalized_query:
            return decomp_subtasks
    return []


def load_execution_agent_components():
    from all_scripts.execution_agent import ExecutionRequest, ToolCard, run_execution_agent

    return ExecutionRequest, ToolCard, run_execution_agent


def run_item(
    *,
    item: dict,
    subtasks: list[str],
    model: "SentenceTransformer",
    tool_embeddings: torch.Tensor,
    tools_dict: dict[str, dict],
    tool_names: list[str],
    top_k: int,
    use_prefixes: bool,
    tool_index_path: Path,
    execution_model: str,
    use_openrouter: bool,
) -> dict:
    query = item.get("question") or item.get("query", "")
    if not query:
        return {"question": "", "error": "missing_query"}

    if not subtasks:
        return {"question": query, "error": "missing_subtasks"}

    selected_tools_set: set[str] = set()
    subtask_details = []
    for subtask in subtasks:
        retrieved = retrieve_tools_for_subtask_shared(
            subtask,
            model,
            tool_embeddings,
            tool_names,
            use_prefixes,
            top_k,
        )
        selected_tools_set.update(retrieved)
        subtask_details.append({"subtask": subtask, "selected": retrieved})

    selected_tool_names = sorted(selected_tools_set)
    ExecutionRequest, ToolCard, run_execution_agent = load_execution_agent_components()
    selected_tool_cards = [ToolCard.model_validate(tools_dict[name]) for name in selected_tool_names if name in tools_dict]

    execution_output = run_execution_agent(
        ExecutionRequest(query=query, toolset=selected_tool_cards),
        tool_index_path=tool_index_path,
        model=execution_model,
        use_openrouter=use_openrouter,
    )

    return {
        "question": query,
        "subtasks": subtask_details,
        "reference": sorted(list(get_reference_tool_names(item))),
        "selected_tools": selected_tool_names,
        "selected_tool_count": len(selected_tool_cards),
        "execution_output": execution_output,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve tools with decomposed RAG and execute query via pydantic_execution_agent"
    )
    parser.add_argument("--benchmark-path", default=TEST_BENCHMARKS_PATH)
    parser.add_argument("--decompositions-path", default=DECOMPOSITIONS_PATH)
    parser.add_argument("--decomposition-model-key", default="base")
    parser.add_argument("--tools-path", default=TOOLS_PATH)
    parser.add_argument("--tool-index", default=TOOLS_PATH)
    parser.add_argument("--results-path", default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--item-index", type=int, default=None, help="Run only one benchmark item by index")
    parser.add_argument("--limit", type=int, default=5, help="How many benchmark items to process")
    parser.add_argument("--top-k", type=int, default=SUBTASK_K)
    parser.add_argument("--retrieval-model", default=BASE_MODEL)
    parser.add_argument("--use-prefixes", action="store_true", help="Use query/passage prefixes")
    parser.add_argument("--execution-model", default=r"openai/gpt-4o-mini", help="Model for execution agent")
    parser.add_argument("--use-openrouter", action="store_true", help="Use OpenRouter in execution agent")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    benchmarks = load_json(args.benchmark_path)
    if args.item_index is not None:
        indexed_items = (
            [(args.item_index, benchmarks[args.item_index])]
            if 0 <= args.item_index < len(benchmarks)
            else []
        )
    else:
        indexed_items = list(enumerate(benchmarks[: args.limit]))

    decompositions = load_json(args.decompositions_path)
    subtasks_by_query = build_subtasks_index(decompositions, args.decomposition_model_key)

    tools = load_json(args.tools_path)
    tools_dict = {tool["name"]: tool for tool in tools if tool.get("name")}
    tool_names = list(tools_dict.keys())

    LOGGER.info("Device: %s", DEVICE)
    LOGGER.info(
        "Loaded benchmarks=%d | selected=%d | tools=%d | decompositions=%d",
        len(benchmarks),
        len(indexed_items),
        len(tool_names),
        len(subtasks_by_query),
    )

    sentence_transformer_cls = resolve_sentence_transformer_cls()
    retrieval_model = sentence_transformer_cls(args.retrieval_model, device=DEVICE)
    tool_embeddings = build_tool_embeddings(
        retrieval_model,
        tools_dict=tools_dict,
        tool_names=tool_names,
        use_prefixes=args.use_prefixes,
    )

    results = []
    for original_idx, item in indexed_items:
        query = item.get("question") or item.get("query", "")
        LOGGER.info("Running item idx=%d", original_idx)
        try:
            result = run_item(
                item=item,
                subtasks=get_subtasks_for_query(subtasks_by_query, query),
                model=retrieval_model,
                tool_embeddings=tool_embeddings,
                tools_dict=tools_dict,
                tool_names=tool_names,
                top_k=args.top_k,
                use_prefixes=args.use_prefixes,
                tool_index_path=Path(args.tool_index),
                execution_model=args.execution_model,
                use_openrouter=args.use_openrouter,
            )
        except Exception as exc:
            LOGGER.exception("Pipeline failed for item idx=%d", original_idx)
            result = {
                "question": query,
                "error": str(exc),
            }

        result["idx"] = original_idx
        results.append(result)

    out_path = Path(args.results_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved %d results to %s", len(results), out_path)


if __name__ == "__main__":
    main()
