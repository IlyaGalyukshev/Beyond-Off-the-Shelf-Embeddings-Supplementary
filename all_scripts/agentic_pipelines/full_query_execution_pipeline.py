import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

WORKSPACE_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(WORKSPACE_ROOT))
from all_scripts.replanner_retrieving.replanner_retrieving import (
    BASELINE_LLM_MODEL,
    prepare_dataset_for_model,
    run_pipeline_for_benchmark,
)
from dotenv import load_dotenv
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
from all_scripts.execution_agent import ExecutionRequest, ToolCard, run_execution_agent
from all_scripts.retrieving.full_query_retrieving import (
    BASE_MODEL,
    DEVICE,
    PASSAGE_PREFIX,
    TASK_K,
    TOOLS_PATH,
    TEST_BENCHMARKS_PATH,
    load_json,
)
from all_scripts.utils.data_utils import structure_tool

LOGGER = logging.getLogger(__name__)
DEFAULT_RESULTS_PATH = Path(__file__).parent / "results/full_query_execution_results.json"


def build_tool_embeddings(
    model: SentenceTransformer,
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


def retrieve_tools_for_query(
    *,
    query: str,
    model: SentenceTransformer,
    tool_embeddings: torch.Tensor,
    tool_names: list[str],
    use_prefixes: bool,
    top_k: int,
) -> list[str]:
    if not query:
        return []

    query_text = f"query: {query}" if use_prefixes else query
    query_emb = model.encode(query_text, convert_to_tensor=True, show_progress_bar=False)

    query_emb_device = query_emb.device
    tool_embeddings_device = tool_embeddings.to(query_emb_device)
    similarities = torch.nn.functional.cosine_similarity(
        query_emb.unsqueeze(0),
        tool_embeddings_device,
        dim=1,
    )

    top_k_indices = torch.topk(similarities, k=min(top_k, len(tool_names))).indices
    retrieved_tools = [tool_names[idx] for idx in top_k_indices.cpu().numpy()]
    tool_embeddings_device = tool_embeddings_device.cpu()
    return retrieved_tools


def run_item(
    *,
    item: dict,
    model: SentenceTransformer,
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

    retrieved_tool_names = retrieve_tools_for_query(
        query=query,
        model=model,
        tool_embeddings=tool_embeddings,
        tool_names=tool_names,
        use_prefixes=use_prefixes,
        top_k=top_k,
    )

    selected_tool_cards = [
        ToolCard.model_validate(tools_dict[name])
        for name in retrieved_tool_names
        if name in tools_dict
    ]

    execution_output = run_execution_agent(
        ExecutionRequest(query=query, toolset=selected_tool_cards),
        tool_index_path=tool_index_path,
        model=execution_model,
        use_openrouter=use_openrouter,
    )

    reference_tool_names = get_reference_tool_names(item)
    return {
        "question": query,
        "reference": sorted(list(reference_tool_names)),
        "selected_tools": retrieved_tool_names,
        "selected_tool_count": len(selected_tool_cards),
        "execution_output": execution_output,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve tools with full-query RAG and execute query via pydantic_execution_agent"
    )
    parser.add_argument("--benchmark-path", default=TEST_BENCHMARKS_PATH)
    parser.add_argument("--tools-path", default=TOOLS_PATH)
    parser.add_argument("--tool-index", default=TOOLS_PATH)
    parser.add_argument("--results-path", default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--item-index", type=int, default=None, help="Run only one benchmark item by index")
    parser.add_argument("--limit", type=int, default=5, help="How many benchmark items to process")
    parser.add_argument("--top-k", type=int, default=TASK_K)
    parser.add_argument("--retrieval-model", default=BASE_MODEL)
    parser.add_argument("--use-prefixes", action="store_true", help="Use query/passage prefixes")
    parser.add_argument("--execution-model", default="openai/gpt-4o-mini", help="Model for execution agent")
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

    tools = load_json(args.tools_path)
    tools_dict = {tool["name"]: tool for tool in tools if tool.get("name")}
    tool_names = list(tools_dict.keys())

    LOGGER.info("Device: %s", DEVICE)
    LOGGER.info(
        "Loaded benchmarks=%d | selected=%d | tools=%d",
        len(benchmarks),
        len(indexed_items),
        len(tool_names),
    )

    retrieval_model = SentenceTransformer(args.retrieval_model, device=DEVICE)
    tool_embeddings = build_tool_embeddings(
        retrieval_model,
        tools_dict=tools_dict,
        tool_names=tool_names,
        use_prefixes=args.use_prefixes,
    )

    results = []
    for original_idx, item in indexed_items:
        LOGGER.info("Running item idx=%d", original_idx)
        try:
            result = run_item(
                item=item,
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
                "question": item.get("question") or item.get("query", ""),
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
