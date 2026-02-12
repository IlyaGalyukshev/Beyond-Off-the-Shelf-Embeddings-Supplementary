import json
import sys
from pathlib import Path
from typing import List

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from utils.config import (
    DATASET,
    PLANNER_AGENT_SYSTEM_PROMPT,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    OPENROUTER_URL,
    PLANNER_K,
    TEST_BENCHMARKS_PATH,
    TOOLS_PATH,
)
from utils.data_utils import structure_tool

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATASET_TAG = Path(TOOLS_PATH).parent.name if TOOLS_PATH else DATASET
TRAINED_MODEL_STAGE2 = str(
    Path(__file__).parent.parent / "train_embed" / f"checkpoints-{DATASET_TAG}" / "minilm-stage2"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


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
        return parsed, None
    except Exception as e:
        return None, str(e)


def retrieve_tools_for_query(
    query: str, model, tool_embeddings, tool_names, use_prefixes: bool, k: int
):
    query_text = QUERY_PREFIX + query if use_prefixes else query
    query_emb = model.encode(
        query_text, convert_to_tensor=True, show_progress_bar=False
    )

    query_emb_device = query_emb.device
    tool_embeddings_device = tool_embeddings.to(query_emb_device)
    similarities = cos_sim(query_emb, tool_embeddings_device)[0]

    top_k_indices = torch.topk(similarities, k=min(k, len(tool_names))).indices
    retrieved_tools = [tool_names[idx] for idx in top_k_indices.cpu().numpy()]
    tool_embeddings_device = tool_embeddings_device.cpu()

    return retrieved_tools


def format_tool_descriptions(tools_dict, tool_names):
    lines = []
    for name in tool_names:
        tool = tools_dict[name]
        desc = tool.get("description_expanded", tool.get("description", ""))
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def invoke_planner(llm, user_request: str, tool_desc_block: str) -> List[str]:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PLANNER_AGENT_SYSTEM_PROMPT),
            (
                "user",
                "Available tools:\n{tool_desc_block}\n\nUser request:\n{user_request}",
            ),
        ]
    ).format_prompt(user_request=user_request, tool_desc_block=tool_desc_block)

    response = llm.invoke(prompt).content
    parsed, err = parse_json_strict(response)

    if err or not isinstance(parsed, list):
        return []

    subtasks = [s for s in parsed if isinstance(s, str)]
    return subtasks


def process_decomposition(
    model_name: str,
    model,
    tool_embeddings,
    tool_names,
    tools_dict,
    benchmarks,
    llm,
    use_prefixes: bool,
):
    print(f"\n{'='*80}")
    print(f"Processing with {model_name} (prefixes: {'YES' if use_prefixes else 'NO'})")
    print(f"{'='*80}")

    results = []

    for benchmark in tqdm(benchmarks, desc=f"{model_name}"):
        query = benchmark.get("question", "")
        if not query:
            continue

        retrieved_tool_names = retrieve_tools_for_query(
            query, model, tool_embeddings, tool_names, use_prefixes, PLANNER_K
        )

        tool_desc_block = format_tool_descriptions(tools_dict, retrieved_tool_names)
        subtasks = invoke_planner(llm, query, tool_desc_block)

        results.append(
            {
                "query": query,
                "retrieved_tools": retrieved_tool_names,
                "subtasks": subtasks,
                "model": model_name,
                "use_prefixes": use_prefixes,
            }
        )

    return results


def main():
    print("=" * 80)
    print("Query Decomposition Generation")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Planner K (top-K tools): {PLANNER_K}")
    print(f"Test benchmarks: {TEST_BENCHMARKS_PATH}")
    print(f"Tools: {TOOLS_PATH}")
    output_path = str(Path(TOOLS_PATH).parent / "decompositions.json")
    print(f"Output: {output_path}")

    tools = load_json(TOOLS_PATH)
    tools_dict = {tool["name"]: tool for tool in tools}
    tool_names = list(tools_dict.keys())
    benchmarks = load_json(TEST_BENCHMARKS_PATH)

    print(f"\nLoaded {len(tools)} tools and {len(benchmarks)} benchmarks")

    llm = ChatOpenAI(
        model=OPENROUTER_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_URL,
        http_client=None,
        temperature=0.0,
    )

    all_decompositions = []

    print("\n" + "=" * 80)
    print("Loading BASE model (no prefixes)")
    print("=" * 80)
    base_model = SentenceTransformer(BASE_MODEL, device=DEVICE)

    tool_passages_base = [
        structure_tool(name, tools_dict[name], passage_prefix="") for name in tool_names
    ]
    print("Encoding tools with BASE model...")
    tool_embeddings_base = base_model.encode(
        tool_passages_base,
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=32,
    )
    tool_embeddings_base = tool_embeddings_base.cpu()

    base_results = process_decomposition(
        "base",
        base_model,
        tool_embeddings_base,
        tool_names,
        tools_dict,
        benchmarks,
        llm,
        use_prefixes=False,
    )
    all_decompositions.extend(base_results)

    del base_model, tool_embeddings_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("Loading TRAINED model - Stage 2 (with prefixes)")
    print("=" * 80)

    try:
        trained_model = SentenceTransformer(TRAINED_MODEL_STAGE2, device=DEVICE)

        tool_passages_trained = [
            structure_tool(name, tools_dict[name], PASSAGE_PREFIX)
            for name in tool_names
        ]
        print("Encoding tools with TRAINED model...")
        tool_embeddings_trained = trained_model.encode(
            tool_passages_trained,
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=32,
        )
        tool_embeddings_trained = tool_embeddings_trained.cpu()

        trained_results = process_decomposition(
            "trained_stage2",
            trained_model,
            tool_embeddings_trained,
            tool_names,
            tools_dict,
            benchmarks,
            llm,
            use_prefixes=True,
        )
        all_decompositions.extend(trained_results)

        del trained_model, tool_embeddings_trained
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error loading trained model: {e}")

    print(f"\n{'='*80}")
    print("Saving results")
    print(f"{'='*80}")
    print(f"Total decompositions: {len(all_decompositions)}")
    save_json(all_decompositions, output_path)
    print(f"Saved to: {output_path}")

    print("\n" + "=" * 80)
    print("Statistics")
    print("=" * 80)

    for model_key in ["base", "trained_stage2"]:
        model_results = [r for r in all_decompositions if r["model"] == model_key]
        if model_results:
            avg_subtasks = sum(len(r["subtasks"]) for r in model_results) / len(
                model_results
            )
            empty_decomp = sum(1 for r in model_results if len(r["subtasks"]) == 0)
            print(f"\n{model_key}:")
            print(f"  Total queries: {len(model_results)}")
            print(f"  Avg subtasks per query: {avg_subtasks:.2f}")
            print(f"  Empty decompositions: {empty_decomp}")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
