import json
import math
import os
import random
from collections import defaultdict
from typing import Optional
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
)
from sentence_transformers.util import cos_sim
from tqdm import tqdm

from config import (
    PAIRS_PATH,
    TEST_BENCHMARKS_PATH,
    TOOLS_PATH,
    SEED,
    TASK_K,
)
from utils import structure_tool

MINILM_BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STAGE1_BATCH_SIZE = 128
STAGE1_EPOCHS = 4
STAGE1_LR = 2e-5
STAGE1_TEMPERATURE = 0.05
STAGE1_WARMUP_STEPS_RATIO = 0.1
STAGE1_MAX_NEG_PER_POS = 4  

STAGE2_BATCH_SIZE = 64
STAGE2_SAMPLES = 4000 
STAGE2_EPOCHS = 3
STAGE2_LR = 1e-5 
STAGE2_TEMPERATURE = 0.03
STAGE2_WARMUP_STEPS_RATIO = 0.1
STAGE2_MAX_HARD_NEG_PER_POS = 6
STAGE2_MAX_CANDIDATES_PER_QUERY = 64
STAGE2_MARGIN = 0.1 

USE_FP16 = True  

WEIGHT_DECAY = 0.01
SAVE_DIR_STAGE1 = "./checkpoints-adv/minilm-stage1"
SAVE_DIR_STAGE2 = "./checkpoints-adv/minilm-stage2"

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def maybe_add_prefix(text: str, prefix: str) -> str:
    """
    Добавить префикс, если его еще нет.
    """
    if not prefix:
        return text
    stripped = text.lstrip()
    if stripped.lower().startswith(prefix.lower()):
        return text
    return prefix + text


def load_pairs_grouped(
    pairs_path: str,
) -> tuple[list[dict], dict[str, dict[str, list[str]]]]:
    """
    Загружаем все пары и группируем по query:
    groups[query] = {"positive": [tool...], "negative": [tool...]}
    """
    print(f"Loading pairs from {pairs_path}...")
    pairs = load_json(pairs_path)

    groups: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"positive": [], "negative": []}
    )

    num_positives = 0
    num_negatives = 0

    for p in pairs:
        query = p["query"]
        tool = p["tool"]
        if p.get("reference", False):
            groups[query]["positive"].append(tool)
            num_positives += 1
        else:
            groups[query]["negative"].append(tool)
            num_negatives += 1

    print(
        f"Total pairs: {len(pairs)} "
        f"(positives: {num_positives}, negatives: {num_negatives})"
    )
    print(f"Unique queries: {len(groups)}")

    return pairs, groups


def build_stage1_examples(
    groups: dict[str, dict[str, list[str]]],
    max_negatives_per_positive: int = STAGE1_MAX_NEG_PER_POS,
) -> list[InputExample]:
    """
    Stage 1: строим примеры (query, positive, [negatives...]) с размеченными негативами.
    """
    print("\nPreparing Stage 1 training examples with labeled negatives...")
    examples: list[InputExample] = []

    for query, data in groups.items():
        positives = data["positive"]
        negatives = data["negative"]

        if not positives:
            continue

        for pos_tool in positives:
            sampled_negatives: list[str] = []
            if negatives:
                k = min(max_negatives_per_positive, len(negatives))
                sampled_negatives = random.sample(negatives, k=k)

            texts = [query, pos_tool] + sampled_negatives
            examples.append(InputExample(texts=texts))

    random.shuffle(examples)
    print(f"Stage 1: constructed {len(examples)} training examples")
    return examples


def mine_hard_negatives(
    model: SentenceTransformer,
    groups: dict[str, dict[str, list[str]]],
    num_samples: int = STAGE2_SAMPLES,
    max_hard_negatives_per_positive: int = STAGE2_MAX_HARD_NEG_PER_POS,
    max_candidates_per_query: int = STAGE2_MAX_CANDIDATES_PER_QUERY,
    margin: float = STAGE2_MARGIN,
) -> list[InputExample]:
    """
    Stage 2: positive-aware hard negatives.
    """
    print(f"\nMining positive-aware hard negatives (up to {num_samples} positives)...")

    positive_triples: list[tuple[str, str, list[str]]] = []
    for query, data in groups.items():
        positives = data["positive"]
        negatives = data["negative"]
        if not positives or not negatives:
            continue
        for pos_tool in positives:
            positive_triples.append((query, pos_tool, negatives))

    if not positive_triples:
        print("No positive triples with negatives found. Skipping Stage 2 mining.")
        return []

    random.shuffle(positive_triples)
    if num_samples > 0:
        positive_triples = positive_triples[: min(num_samples, len(positive_triples))]

    query_emb_cache: dict[str, torch.Tensor] = {}
    tool_emb_cache: dict[str, torch.Tensor] = {}

    def get_query_emb(q: str) -> torch.Tensor:
        if q not in query_emb_cache:
            emb = model.encode(
                q, convert_to_tensor=True, show_progress_bar=False
            )
            query_emb_cache[q] = emb.cpu()
        return query_emb_cache[q]

    def get_tool_emb(t: str) -> torch.Tensor:
        if t not in tool_emb_cache:
            emb = model.encode(
                t, convert_to_tensor=True, show_progress_bar=False
            )
            tool_emb_cache[t] = emb.cpu()
        return tool_emb_cache[t]

    examples: list[InputExample] = []

    for idx, (query, pos_tool, negatives) in enumerate(tqdm(
        positive_triples, desc="Mining hard negatives"
    )):
        if not negatives:
            continue

        q_emb = get_query_emb(query).to(DEVICE)
        pos_emb = get_tool_emb(pos_tool).to(DEVICE)

        if len(negatives) > max_candidates_per_query:
            candidate_negs = random.sample(negatives, max_candidates_per_query)
        else:
            candidate_negs = list(negatives)

        if not candidate_negs:
            continue

        neg_embs_list: list[torch.Tensor] = [get_tool_emb(t).to(DEVICE) for t in candidate_negs]
        neg_embs = torch.stack(neg_embs_list)

        q_emb_2d = q_emb.unsqueeze(0)
        pos_emb_2d = pos_emb.unsqueeze(0)

        sim_pos = cos_sim(q_emb_2d, pos_emb_2d)[0][0]
        sims_negs = cos_sim(q_emb_2d, neg_embs)[0]

        semi_hard_mask = (sims_negs < sim_pos) & (sims_negs > (sim_pos - margin))
        semi_hard_indices = torch.nonzero(semi_hard_mask, as_tuple=False).view(-1)

        if semi_hard_indices.numel() > 0:
            semi_hard_sims = sims_negs[semi_hard_indices]
            sorted_idx = torch.argsort(semi_hard_sims, descending=True)
            selected_indices = semi_hard_indices[sorted_idx][
                :max_hard_negatives_per_positive
            ]
        else:
            k = min(max_hard_negatives_per_positive, sims_negs.size(0))
            sorted_all = torch.argsort(sims_negs, descending=True)
            selected_indices = sorted_all[:k]

        if selected_indices.numel() == 0:
            continue

        selected_negatives: list[str] = [
            candidate_negs[i] for i in selected_indices.tolist()
        ]

        texts = [query, pos_tool] + selected_negatives
        examples.append(InputExample(texts=texts))
        
        if idx % 100 == 0:
            del q_emb, pos_emb, neg_embs, q_emb_2d, pos_emb_2d, sim_pos, sims_negs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    query_emb_cache.clear()
    tool_emb_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"Created {len(examples)} Stage 2 training examples with hard negatives")
    return examples


def compute_ndcg(retrieved_tools: list[str], reference_tools: set, k: int = 10) -> float:
    """
    Binary nDCG@k: relevance is 1 if tool in reference else 0.
    DCG = sum(rel_i / log2(rank_i+1)); IDCG = ideal DCG (all relevant at top); nDCG = DCG/IDCG.
    """
    dcg = 0.0
    for rank, tool_name in enumerate(retrieved_tools[:k], start=1):
        rel = 1.0 if tool_name in reference_tools else 0.0
        if rel > 0:
            dcg += rel / math.log2(rank + 1)

    ideal_len = min(k, len(reference_tools))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_len + 1))

    return dcg / idcg if idcg > 0 else 0.0


def compute_mrr(retrieved_tools: list[str], reference_tools: set[str]) -> float:
    """
    Reciprocal rank for one query (MRR@K): 1 / (1-based rank of first relevant tool in list), or 0 if none.
    Typically list has length TASK_K (top-K retrieved).
    """
    for rank, tool_name in enumerate(retrieved_tools, start=1):
        if tool_name in reference_tools:
            return 1.0 / rank
    return 0.0


@dataclass
class EvalResults:
    precision: float
    recall: float
    f1: float
    ndcg: float
    mrr: float

    def __str__(self):
        return (
            f"Precision: {self.precision:.4f} | Recall: {self.recall:.4f} | "
            f"F1: {self.f1:.4f} | nDCG: {self.ndcg:.4f} | MRR: {self.mrr:.4f}"
        )


def evaluate_model(
    model: SentenceTransformer,
    test_benchmarks_path: str,
    tools_path: str,
) -> EvalResults:
    """
    Evaluate model on test benchmarks (retrieval only, no planner).
    """
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

    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    ndcgs: list[float] = []
    mrrs: list[float] = []

    for benchmark in tqdm(benchmarks, desc="Evaluating"):
        query = benchmark.get("question", "")
        if not query:
            continue

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


def train_stage1(
    model: SentenceTransformer,
    train_examples: list[InputExample],
    save_dir: str,
) -> SentenceTransformer:
    """
    Stage 1: MNRLoss с размеченными негативами + in-batch negatives.
    """
    print("\n" + "=" * 80)
    print("STAGE 1: Main Contrastive Training with Labeled Negatives")
    print("=" * 80)

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=STAGE1_BATCH_SIZE,
    )

    train_loss = losses.MultipleNegativesRankingLoss(
        model=model,
        scale=1.0 / STAGE1_TEMPERATURE,
    )

    num_train_steps = len(train_dataloader) * STAGE1_EPOCHS
    warmup_steps = max(1, int(num_train_steps * STAGE1_WARMUP_STEPS_RATIO))

    print(f"\nTraining parameters:")
    print(f"  Batch size: {STAGE1_BATCH_SIZE}")
    print(f"  Epochs: {STAGE1_EPOCHS}")
    print(f"  Learning rate: {STAGE1_LR}")
    print(f"  Temperature: {STAGE1_TEMPERATURE}")
    print(f"  Mixed precision (FP16): {USE_FP16}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Total training steps: {num_train_steps}")
    print(f"  Training samples: {len(train_examples)}")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=STAGE1_EPOCHS,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": STAGE1_LR},
        weight_decay=WEIGHT_DECAY,
        scheduler="warmupcosine",
        show_progress_bar=True,
        use_amp=USE_FP16, 
    )

    print(f"\nSaving Stage 1 model to {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    model.save(save_dir)

    return model


def train_stage2(
    model: SentenceTransformer,
    train_examples: list[InputExample],
    save_dir: str,
) -> SentenceTransformer:
    """
    Stage 2: Fine-tuning with positive-aware hard negatives.
    """
    print("\n" + "=" * 80)
    print("STAGE 2: Fine Contrastive Training with Positive-Aware Hard Negatives")
    print("=" * 80)

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=STAGE2_BATCH_SIZE,
    )

    train_loss = losses.MultipleNegativesRankingLoss(
        model=model,
        scale=1.0 / STAGE2_TEMPERATURE,
    )

    num_train_steps = len(train_dataloader) * STAGE2_EPOCHS
    warmup_steps = max(1, int(num_train_steps * STAGE2_WARMUP_STEPS_RATIO))

    print(f"\nTraining parameters:")
    print(f"  Batch size: {STAGE2_BATCH_SIZE}")
    print(f"  Epochs: {STAGE2_EPOCHS}")
    print(f"  Learning rate: {STAGE2_LR}")
    print(f"  Temperature: {STAGE2_TEMPERATURE}")
    print(f"  Mixed precision (FP16): {USE_FP16}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Total training steps: {num_train_steps}")
    print(f"  Training samples: {len(train_examples)}")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=STAGE2_EPOCHS,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": STAGE2_LR},
        weight_decay=WEIGHT_DECAY,
        scheduler="warmupcosine",
        show_progress_bar=True,
        use_amp=USE_FP16, 
    )

    print(f"\nSaving Stage 2 model to {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    model.save(save_dir)

    return model


def main():
    set_seed(SEED)

    print("=" * 80)
    print("Two-Stage MiniLM Fine-tuning for Tool Retrieval (Improved)")
    print("=" * 80)
    print(f"\nDevice: {DEVICE}")
    print(f"Base model: {MINILM_BASE_MODEL}")

    stage1_exists = os.path.exists(SAVE_DIR_STAGE1) and os.path.exists(
        os.path.join(SAVE_DIR_STAGE1, "config.json")
    )
    stage2_exists = os.path.exists(SAVE_DIR_STAGE2) and os.path.exists(
        os.path.join(SAVE_DIR_STAGE2, "config.json")
    )

    print(f"\nModel status:")
    print(f"  Stage 1 model exists: {stage1_exists} ({SAVE_DIR_STAGE1})")
    print(f"  Stage 2 model exists: {stage2_exists} ({SAVE_DIR_STAGE2})")

    _, groups = load_pairs_grouped(PAIRS_PATH)

    if not stage1_exists:
        print("\n" + "=" * 80)
        print("Baseline Evaluation (before training)")
        print("=" * 80)

        base_model = SentenceTransformer(MINILM_BASE_MODEL, device=DEVICE)
        baseline_results = evaluate_model(
            model=base_model,
            test_benchmarks_path=TEST_BENCHMARKS_PATH,
            tools_path=TOOLS_PATH,
        )
        print(f"Baseline: {baseline_results}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("\n" + "=" * 80)
        print("Skipping baseline evaluation (Stage 1 model already exists)")
        print("=" * 80)
        base_model = None


    if not stage1_exists:
        print("\n" + "=" * 80)
        print("Preparing Stage 1 Training Data")
        print("=" * 80)

        stage1_examples = build_stage1_examples(
            groups=groups,
            max_negatives_per_positive=STAGE1_MAX_NEG_PER_POS,
        )
        print(f"Loaded {len(stage1_examples)} examples for Stage 1")

        stage1_model = train_stage1(
            model=base_model,  
            train_examples=stage1_examples,
            save_dir=SAVE_DIR_STAGE1,
        )
    else:
        print("\n" + "=" * 80)
        print("Loading existing Stage 1 model")
        print("=" * 80)
        print(f"Loading from {SAVE_DIR_STAGE1}...")
        stage1_model = SentenceTransformer(SAVE_DIR_STAGE1, device=DEVICE)
        print("Stage 1 model loaded successfully!")


    print("\n" + "=" * 80)
    print("Evaluation after Stage 1")
    print("=" * 80)

    stage1_results = evaluate_model(
        model=stage1_model,
        test_benchmarks_path=TEST_BENCHMARKS_PATH,
        tools_path=TOOLS_PATH,
    )
    print(f"Stage 1: {stage1_results}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not stage2_exists:
        print("\n" + "=" * 80)
        print("Preparing Stage 2 Training Data")
        print("=" * 80)

        stage2_examples = mine_hard_negatives(
            model=stage1_model,
            groups=groups,
            num_samples=STAGE2_SAMPLES,
            max_hard_negatives_per_positive=STAGE2_MAX_HARD_NEG_PER_POS,
            max_candidates_per_query=STAGE2_MAX_CANDIDATES_PER_QUERY,
            margin=STAGE2_MARGIN,
        )

        stage2_model = train_stage2(
            model=stage1_model,
            train_examples=stage2_examples,
            save_dir=SAVE_DIR_STAGE2,
        )
    else:
        print("\n" + "=" * 80)
        print("Loading existing Stage 2 model")
        print("=" * 80)
        print(f"Loading from {SAVE_DIR_STAGE2}...")
        stage2_model = SentenceTransformer(SAVE_DIR_STAGE2, device=DEVICE)
        print("Stage 2 model loaded successfully!")

    print("\n" + "=" * 80)
    print("Evaluation after Stage 2")
    print("=" * 80)

    stage2_results = evaluate_model(
        model=stage2_model,
        test_benchmarks_path=TEST_BENCHMARKS_PATH,
        tools_path=TOOLS_PATH,
    )
    print(f"Stage 2: {stage2_results}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

