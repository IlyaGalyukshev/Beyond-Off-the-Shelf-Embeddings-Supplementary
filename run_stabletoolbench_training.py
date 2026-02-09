#!/usr/bin/env python3
"""
Wrapper to run generate_pairs.py and train_script.py for StableToolBench.
"""

import sys
import os
from pathlib import Path

# Add paths in correct order
sys.path.insert(0, '/workspace/all_scripts/utils')
sys.path.insert(0, '/workspace/all_scripts/train_embed')

# Import and patch config
import config as cfg
cfg.TOOLS_PATH = '/workspace/data/stabletoolbench/tools_expanded.json'
cfg.TRAIN_BENCHMARKS_PATH = '/workspace/data/stabletoolbench/without_top_benchmarks_enriched.json'
cfg.TEST_BENCHMARKS_PATH = '/workspace/data/stabletoolbench/top_benchmarks_enriched.json'
cfg.PAIRS_PATH = '/workspace/data/stabletoolbench/pairs_augmented.json'

# Import data_utils as utils for compatibility
import data_utils
sys.modules['utils'] = data_utils

# Import train_script and override save directories
import train_script
train_script.SAVE_DIR_STAGE1 = '/workspace/all_scripts/train_embed/checkpoints-stabletoolbench/minilm-stage1'
train_script.SAVE_DIR_STAGE2 = '/workspace/all_scripts/train_embed/checkpoints-stabletoolbench/minilm-stage2'

# Patch generate_pairs to handle 'reference' format
import generate_pairs
_original_generate_pairs = generate_pairs.generate_pairs

def patched_generate_pairs():
    """Patched version that handles both toolset and reference formats."""
    import json
    from generate_pairs import QUERY_PREFIX, PASSAGE_PREFIX
    from data_utils import structure_tool
    import random
    
    print("Loading tools...")
    with open(cfg.TOOLS_PATH, 'r', encoding='utf-8') as f:
        tools = json.load(f)
    
    print(f"Loaded {len(tools)} tools")
    
    print("Loading benchmarks...")
    with open(cfg.TRAIN_BENCHMARKS_PATH, 'r', encoding='utf-8') as f:
        benchmarks = json.load(f)
    
    print(f"Loaded {len(benchmarks)} benchmarks")
    
    tools_dict = {tool['name']: tool for tool in tools}
    
    pairs = []
    
    for idx, benchmark in enumerate(benchmarks):
        if (idx + 1) % 100 == 0:
            print(f"Processing benchmark {idx + 1}/{len(benchmarks)}...")
        
        query = benchmark.get('question', '')
        if not query:
            continue
        
        query_str = QUERY_PREFIX + query
        
        # Handle both formats: toolset (old) and reference (new unified)
        if 'reference' in benchmark:
            reference_tool_names = {ref['tool'] for ref in benchmark['reference']}
        else:
            reference_toolset = benchmark.get('toolset', [])
            reference_tool_names = {tool['name'] for tool in reference_toolset}
        
        for tool_name in reference_tool_names:
            if tool_name in tools_dict:
                tool = tools_dict[tool_name]
                tool_passage = structure_tool(tool_name, tool, PASSAGE_PREFIX)
                
                pairs.append({
                    'query': query_str,
                    'tool': tool_passage,
                    'reference': True
                })
        
        negative_tool_names = [name for name in tools_dict.keys() if name not in reference_tool_names]

        for tool_name in negative_tool_names:
            tool = tools_dict[tool_name]
            tool_passage = structure_tool(tool_name, tool, PASSAGE_PREFIX)
            
            pairs.append({
                'query': query_str,
                'tool': tool_passage,
                'reference': False
            })
    
    print(f"\nGenerated {len(pairs)} pairs")
    print(f"Saving to {cfg.PAIRS_PATH}...")
    
    with open(cfg.PAIRS_PATH, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    
    positive_pairs = sum(1 for p in pairs if p['reference'])
    negative_pairs = sum(1 for p in pairs if not p['reference'])
    
    print(f"\nStatistics:")
    print(f"  Total pairs: {len(pairs)}")
    print(f"  Positive pairs: {positive_pairs}")
    print(f"  Negative pairs: {negative_pairs}")
    print(f"  Positive/Negative ratio: {positive_pairs / negative_pairs:.2f}")

generate_pairs.generate_pairs = patched_generate_pairs

# Patch train_script evaluate_model to handle 'reference' format
_original_evaluate = train_script.evaluate_model

def patched_evaluate_model(model, test_benchmarks_path, tools_path):
    """Patched version that handles both toolset and reference formats."""
    import json
    import math
    import torch
    from tqdm import tqdm
    from sentence_transformers.util import cos_sim
    from train_script import (
        EvalResults, QUERY_PREFIX, PASSAGE_PREFIX, TASK_K,
        maybe_add_prefix, load_json, compute_ndcg, compute_mrr
    )
    from data_utils import structure_tool
    import numpy as np
    
    print(f"\nEvaluating on {test_benchmarks_path}...")
    
    tools = load_json(tools_path)
    tools_dict = {tool["name"]: tool for tool in tools}
    tool_names = list(tools_dict.keys())
    
    tool_passages = [
        structure_tool(name, tools_dict[name], PASSAGE_PREFIX) for name in tool_names
    ]
    
    print("Encoding tools...")
    tool_embeddings = model.encode(
        tool_passages, convert_to_tensor=True, show_progress_bar=True, batch_size=32
    )
    tool_embeddings = tool_embeddings.cpu()
    
    benchmarks = load_json(test_benchmarks_path)
    
    precisions = []
    recalls = []
    f1s = []
    ndcgs = []
    mrrs = []
    
    for benchmark in tqdm(benchmarks, desc="Evaluating"):
        query = benchmark.get("question", "")
        if not query:
            continue
        
        # Handle both formats
        if 'reference' in benchmark:
            reference_tool_names = {ref['tool'] for ref in benchmark['reference']}
        else:
            reference_toolset = benchmark.get("toolset", [])
            reference_tool_names = {tool["name"] for tool in reference_toolset}
        
        if not reference_tool_names:
            continue
        
        query_for_model = maybe_add_prefix(query, QUERY_PREFIX)
        query_emb = model.encode(
            query_for_model, convert_to_tensor=True, show_progress_bar=False
        )
        query_emb_device = query_emb.device
        tool_embeddings_device = tool_embeddings.to(query_emb_device)
        similarities = cos_sim(query_emb, tool_embeddings_device)[0]
        top_k_indices = torch.topk(
            similarities, k=min(TASK_K, len(tool_names))
        ).indices
        retrieved_tools = [tool_names[idx] for idx in top_k_indices.cpu().numpy()]
        tool_embeddings_device = tool_embeddings_device.cpu()
        
        tp = len(set(retrieved_tools) & reference_tool_names)
        precision = tp / len(retrieved_tools) if retrieved_tools else 0.0
        recall = tp / len(reference_tool_names) if reference_tool_names else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        
        ndcg = compute_ndcg(retrieved_tools, reference_tool_names, k=TASK_K)
        mrr = compute_mrr(retrieved_tools, reference_tool_names)
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        ndcgs.append(ndcg)
        mrrs.append(mrr)
    
    return EvalResults(
        precision=float(np.mean(precisions) if precisions else 0.0),
        recall=float(np.mean(recalls) if recalls else 0.0),
        f1=float(np.mean(f1s) if f1s else 0.0),
        ndcg=float(np.mean(ndcgs) if ndcgs else 0.0),
        mrr=float(np.mean(mrrs) if mrrs else 0.0),
    )

train_script.evaluate_model = patched_evaluate_model

print("=" * 100)
print("StableToolBench Training Pipeline")
print("=" * 100)
print(f"\nConfiguration:")
print(f"  Tools: {cfg.TOOLS_PATH}")
print(f"  Train benchmarks: {cfg.TRAIN_BENCHMARKS_PATH}")
print(f"  Test benchmarks: {cfg.TEST_BENCHMARKS_PATH}")
print(f"  Pairs output: {cfg.PAIRS_PATH}")
print(f"  Stage1 checkpoint: {train_script.SAVE_DIR_STAGE1}")
print(f"  Stage2 checkpoint: {train_script.SAVE_DIR_STAGE2}")
print("\n" + "=" * 100)

# Check if pairs already exist
pairs_path = Path(cfg.PAIRS_PATH)
if pairs_path.exists():
    print(f"\n✅ Pairs file already exists: {cfg.PAIRS_PATH}")
    print("   Skipping generate_pairs step.")
else:
    print(f"\n📝 Step 1: Generating training pairs...")
    print("=" * 100)
    generate_pairs.generate_pairs()
    print("\n✅ Pairs generated successfully!")

print(f"\n🚀 Step 2: Training MiniLM embedder (2 stages)...")
print("=" * 100)
train_script.main()

print("\n" + "=" * 100)
print("✅ StableToolBench training completed!")
print("=" * 100)
