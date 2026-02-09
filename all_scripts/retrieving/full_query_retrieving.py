import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from utils.data_utils import structure_tool

WORKSPACE_ROOT = Path(__file__).parent.parent.parent
TOOLS_PATH = str(WORKSPACE_ROOT / "data/ultratool/tools_expanded.json")
TEST_BENCHMARKS_PATH = str(WORKSPACE_ROOT / "data/ultratool/top_benchmarks_enriched.json")
TASK_K = 10

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TRAINED_MODEL_STAGE1 = "/workspace/all_scripts/train_embed/checkpoints-adv/minilm-stage1"
TRAINED_MODEL_STAGE2 = "/workspace/all_scripts/train_embed/checkpoints-adv/minilm-stage2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_ndcg(retrieved_tools: list[str], reference_tools: set, k: int = 10) -> float:
    dcg = 0.0
    for rank, tool_name in enumerate(retrieved_tools[:k], start=1):
        rel = 1.0 if tool_name in reference_tools else 0.0
        if rel > 0:
            dcg += rel / math.log2(rank + 1)
    ideal_len = min(k, len(reference_tools))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_len + 1))
    return dcg / idcg if idcg > 0 else 0.0


def compute_mrr(retrieved_tools: list[str], reference_tools: set[str]) -> float:
    for rank, tool_name in enumerate(retrieved_tools, start=1):
        if tool_name in reference_tools:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(model, test_benchmarks_path, tools_path, use_prefixes=False, model_name="Model"):
    print(f"\n{'='*80}")
    print(f"Evaluating {model_name} | Prefixes: {'YES' if use_prefixes else 'NO'}")
    print(f"{'='*80}")
    
    tools = load_json(tools_path)
    tools_dict = {tool["name"]: tool for tool in tools}
    tool_names = list(tools_dict.keys())
    
    if use_prefixes:
        tool_passages = [structure_tool(name, tools_dict[name], PASSAGE_PREFIX) for name in tool_names]
    else:
        tool_passages = [structure_tool(name, tools_dict[name], passage_prefix="") for name in tool_names]
    
    print("Encoding tools...")
    tool_embeddings = model.encode(tool_passages, convert_to_tensor=True, show_progress_bar=True, batch_size=32)
    tool_embeddings = tool_embeddings.cpu()
    
    benchmarks = load_json(test_benchmarks_path)
    
    precisions, recalls, f1s, ndcgs, mrrs = [], [], [], [], []
    detail_table = []
    
    print(f"Processing {len(benchmarks)} benchmarks...")
    for idx, benchmark in enumerate(tqdm(benchmarks, desc="Evaluating")):
        query = benchmark.get("question", "")
        if not query:
            continue
        
        # Handle both old toolset format and new reference format
        if "reference" in benchmark:
            reference_tool_names = {ref["tool"] for ref in benchmark["reference"]}
        else:
            reference_toolset = benchmark.get("toolset", [])
            reference_tool_names = {tool["name"] for tool in reference_toolset}
        
        if not reference_tool_names:
            continue
        
        query_text = QUERY_PREFIX + query if use_prefixes else query
        query_emb = model.encode(query_text, convert_to_tensor=True, show_progress_bar=False)
        
        query_emb_device = query_emb.device
        tool_embeddings_device = tool_embeddings.to(query_emb_device)
        similarities = cos_sim(query_emb, tool_embeddings_device)[0]
        
        top_k_indices = torch.topk(similarities, k=min(TASK_K, len(tool_names))).indices
        retrieved_tools = [tool_names[idx] for idx in top_k_indices.cpu().numpy()]
        tool_embeddings_device = tool_embeddings_device.cpu()
        
        tp = len(set(retrieved_tools) & reference_tool_names)
        precision = tp / len(retrieved_tools) if retrieved_tools else 0.0
        recall = tp / len(reference_tool_names) if reference_tool_names else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        ndcg = compute_ndcg(retrieved_tools, reference_tool_names, k=TASK_K)
        mrr = compute_mrr(retrieved_tools, reference_tool_names)
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        ndcgs.append(ndcg)
        mrrs.append(mrr)
        
        # Add to detail table in REACT format
        detail_table.append({
            "idx": idx,
            "question": query,
            "subtasks": [],  # No subtasks for full query retrieval
            "reference": sorted(list(reference_tool_names)),
            "selected": retrieved_tools,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ndcg": ndcg,
            "mrr": mrr
        })
    
    results = {
        "precision": float(np.mean(precisions) if precisions else 0.0),
        "recall": float(np.mean(recalls) if recalls else 0.0),
        "f1": float(np.mean(f1s) if f1s else 0.0),
        "ndcg": float(np.mean(ndcgs) if ndcgs else 0.0),
        "mrr": float(np.mean(mrrs) if mrrs else 0.0),
        "detail_table": detail_table
    }
    
    print(f"\n{model_name} Results:")
    print(f"  Precision@{TASK_K}: {results['precision']:.4f}")
    print(f"  Recall@{TASK_K}:    {results['recall']:.4f}")
    print(f"  F1@{TASK_K}:        {results['f1']:.4f}")
    print(f"  nDCG@{TASK_K}:      {results['ndcg']:.4f}")
    print(f"  MRR@{TASK_K}:       {results['mrr']:.4f}")
    
    return results


def main():
    print("="*80)
    print("Full Query Retrieval Evaluation")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"K (top-K retrieval): {TASK_K}")
    print(f"Test benchmarks: {TEST_BENCHMARKS_PATH}")
    print(f"Tools: {TOOLS_PATH}")
    
    # Create results directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    all_results = {}
    
    print("\n" + "="*80)
    print("Loading BASE model (no prefixes)...")
    print("="*80)
    base_model = SentenceTransformer(BASE_MODEL, device=DEVICE)
    base_results = evaluate_retrieval(base_model, TEST_BENCHMARKS_PATH, TOOLS_PATH, use_prefixes=False, model_name="Base Model")
    all_results["base"] = base_results
    
    # Save base results
    base_result_file = results_dir / "full_query_base_results.json"
    completed_indices = [item["idx"] for item in base_results["detail_table"]]
    with open(base_result_file, "w", encoding="utf-8") as f:
        json.dump({
            "completed": completed_indices,
            "detail_table": base_results["detail_table"]
        }, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved base results to {base_result_file}")
    
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n" + "="*80)
    print("Loading TRAINED model - Stage 1 (with prefixes)...")
    print("="*80)
    try:
        stage1_model = SentenceTransformer(TRAINED_MODEL_STAGE1, device=DEVICE)
        stage1_results = evaluate_retrieval(stage1_model, TEST_BENCHMARKS_PATH, TOOLS_PATH, use_prefixes=True, model_name="Trained - Stage 1")
        all_results["stage1"] = stage1_results
        
        # Save stage1 results
        stage1_result_file = results_dir / "full_query_stage1_results.json"
        completed_indices = [item["idx"] for item in stage1_results["detail_table"]]
        with open(stage1_result_file, "w", encoding="utf-8") as f:
            json.dump({
                "completed": completed_indices,
                "detail_table": stage1_results["detail_table"]
            }, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved stage1 results to {stage1_result_file}")
        
        del stage1_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error loading Stage 1: {e}")
    
    print("\n" + "="*80)
    print("Loading TRAINED model - Stage 2 (with prefixes)...")
    print("="*80)
    try:
        stage2_model = SentenceTransformer(TRAINED_MODEL_STAGE2, device=DEVICE)
        stage2_results = evaluate_retrieval(stage2_model, TEST_BENCHMARKS_PATH, TOOLS_PATH, use_prefixes=True, model_name="Trained - Stage 2")
        all_results["stage2"] = stage2_results
        
        # Save stage2 results
        stage2_result_file = results_dir / "full_query_stage2_results.json"
        completed_indices = [item["idx"] for item in stage2_results["detail_table"]]
        with open(stage2_result_file, "w", encoding="utf-8") as f:
            json.dump({
                "completed": completed_indices,
                "detail_table": stage2_results["detail_table"]
            }, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved stage2 results to {stage2_result_file}")
        
        del stage2_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error loading Stage 2: {e}")
    
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(f"{'Model':<30} {'Precision':<12} {'Recall':<12} {'F1':<12} {'nDCG':<12} {'MRR':<12}")
    print("-"*80)
    
    for model_key, model_label in [("base", "Base (no prefixes)"), ("stage1", "Stage 1 (with prefixes)"), ("stage2", "Stage 2 (with prefixes)")]:
        if model_key in all_results:
            res = all_results[model_key]
            print(f"{model_label:<30} {res['precision']:<12.4f} {res['recall']:<12.4f} {res['f1']:<12.4f} {res['ndcg']:<12.4f} {res['mrr']:<12.4f}")
    
    print("="*80)
    print(f"\n✅ Evaluation complete! Results saved to {results_dir}")
    print(f"   - full_query_base_results.json")
    print(f"   - full_query_stage1_results.json")
    print(f"   - full_query_stage2_results.json")


if __name__ == "__main__":
    main()
