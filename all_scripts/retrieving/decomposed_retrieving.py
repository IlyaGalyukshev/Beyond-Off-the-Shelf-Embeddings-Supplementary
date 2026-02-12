import json
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from utils.data_utils import structure_tool
from utils.config import (
    DATASET,
    SUBTASK_K as SUBTASK_K_CONFIG,
    TEST_BENCHMARKS_PATH,
    TOOLS_PATH,
)

DECOMPOSITIONS_PATH = str(Path(TOOLS_PATH).parent / "decompositions.json")
SUBTASK_K = SUBTASK_K_CONFIG

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

DATASET_TAG = Path(TOOLS_PATH).parent.name if TOOLS_PATH else DATASET


def resolve_stage2_model_dir(dataset_tag: str) -> str:
    candidates = [
        Path.cwd() / f"checkpoints-{dataset_tag}" / "minilm-stage2",
        Path(__file__).parent.parent
        / "train_embed"
        / f"checkpoints-{dataset_tag}"
        / "minilm-stage2",
    ]
    for p in candidates:
        if (p / "config.json").exists():
            return str(p)
    return str(candidates[0])


TRAINED_MODEL_STAGE2 = resolve_stage2_model_dir(DATASET_TAG)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve_tools_for_subtask(
    subtask, model, tool_embeddings, tool_names, use_prefixes, k
):
    query_text = QUERY_PREFIX + subtask if use_prefixes else subtask
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


def evaluate_decomposed_retrieval(
    model,
    decompositions,
    benchmarks_dict,
    tool_embeddings,
    tool_names,
    use_prefixes,
    model_key,
    k=SUBTASK_K,
):

    model_decomps = [d for d in decompositions if d["model"] == model_key]

    precisions, recalls, f1s = [], [], []
    detail_table = []

    for idx, decomp in enumerate(tqdm(model_decomps, desc=f"Evaluating {model_key}")):
        query = decomp["query"]
        subtasks = decomp["subtasks"]

        if query not in benchmarks_dict:
            continue

        benchmark = benchmarks_dict[query]
        # Handle both old toolset format and new reference format
        if "reference" in benchmark:
            reference_tool_names = {ref["tool"] for ref in benchmark["reference"]}
        else:
            reference_toolset = benchmark.get("toolset", [])
            reference_tool_names = {tool["name"] for tool in reference_toolset}

        if not reference_tool_names or not subtasks:
            continue

        all_retrieved_tools = set()
        subtask_details = []

        for subtask in subtasks:
            retrieved = retrieve_tools_for_subtask(
                subtask, model, tool_embeddings, tool_names, use_prefixes, k
            )
            all_retrieved_tools.update(retrieved)
            subtask_details.append({"subtask": subtask, "retrieved_tools": retrieved})

        retrieved_list = list(all_retrieved_tools)

        tp = len(all_retrieved_tools & reference_tool_names)
        precision = tp / len(retrieved_list) if retrieved_list else 0.0
        recall = tp / len(reference_tool_names) if reference_tool_names else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        # Add to detail table in REACT format
        detail_table.append(
            {
                "idx": idx,
                "question": query,
                "subtasks": subtask_details,
                "reference": sorted(list(reference_tool_names)),
                "selected": sorted(retrieved_list),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    results = {
        "precision": float(np.mean(precisions) if precisions else 0.0),
        "recall": float(np.mean(recalls) if recalls else 0.0),
        "f1": float(np.mean(f1s) if f1s else 0.0),
        "detail_table": detail_table,
    }

    return results


def main():
    print("=" * 80)
    print("Decomposed Query Retrieval Evaluation")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATASET_TAG}")
    print(f"K (top-K per subtask): {SUBTASK_K}")
    print(f"Test benchmarks: {TEST_BENCHMARKS_PATH}")
    print(f"Tools: {TOOLS_PATH}")
    print(f"Decompositions: {DECOMPOSITIONS_PATH}")
    print(f"Stage 2 model: {TRAINED_MODEL_STAGE2}")

    # Create results directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    tools = load_json(TOOLS_PATH)
    tools_dict = {tool["name"]: tool for tool in tools}
    tool_names = list(tools_dict.keys())

    benchmarks = load_json(TEST_BENCHMARKS_PATH)
    benchmarks_dict = {b["question"]: b for b in benchmarks}

    decompositions = load_json(DECOMPOSITIONS_PATH)

    print(
        f"\nLoaded {len(tools)} tools, {len(benchmarks)} benchmarks, {len(decompositions)} decompositions"
    )

    all_results = {}

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

    print("\n" + "=" * 80)
    print("Evaluating BASE model with decomposed retrieval")
    print("=" * 80)
    base_results = evaluate_decomposed_retrieval(
        base_model,
        decompositions,
        benchmarks_dict,
        tool_embeddings_base,
        tool_names,
        use_prefixes=False,
        model_key="base",
        k=SUBTASK_K,
    )
    all_results["base"] = base_results

    # Save base results
    base_result_file = results_dir / "decomposed_base_results.json"
    completed_indices = [item["idx"] for item in base_results["detail_table"]]
    with open(base_result_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "completed": completed_indices,
                "detail_table": base_results["detail_table"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n✅ Saved base results to {base_result_file}")

    print(f"\nBase Model Results:")
    print(f"  Precision: {base_results['precision']:.4f}")
    print(f"  Recall:    {base_results['recall']:.4f}")
    print(f"  F1:        {base_results['f1']:.4f}")

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

        print("\n" + "=" * 80)
        print("Evaluating TRAINED model with decomposed retrieval")
        print("=" * 80)
        trained_results = evaluate_decomposed_retrieval(
            trained_model,
            decompositions,
            benchmarks_dict,
            tool_embeddings_trained,
            tool_names,
            use_prefixes=True,
            model_key="trained_stage2",
            k=SUBTASK_K,
        )
        all_results["trained_stage2"] = trained_results

        # Save trained results
        trained_result_file = results_dir / "decomposed_trained_stage2_results.json"
        completed_indices = [item["idx"] for item in trained_results["detail_table"]]
        with open(trained_result_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "completed": completed_indices,
                    "detail_table": trained_results["detail_table"],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\n✅ Saved trained results to {trained_result_file}")

        print(f"\nTrained Model Results:")
        print(f"  Precision: {trained_results['precision']:.4f}")
        print(f"  Recall:    {trained_results['recall']:.4f}")
        print(f"  F1:        {trained_results['f1']:.4f}")

        del trained_model, tool_embeddings_trained
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error loading trained model: {e}")

    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Model':<30} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 80)

    for model_key, model_label in [
        ("base", "Base (no prefixes)"),
        ("trained_stage2", "Stage 2 (with prefixes)"),
    ]:
        if model_key in all_results:
            res = all_results[model_key]
            print(
                f"{model_label:<30} {res['precision']:<12.4f} {res['recall']:<12.4f} {res['f1']:<12.4f}"
            )

    print("=" * 80)
    print(f"\n✅ Evaluation complete! Results saved to {results_dir}")
    print(f"   - decomposed_base_results.json")
    print(f"   - decomposed_trained_stage2_results.json")


if __name__ == "__main__":
    main()
