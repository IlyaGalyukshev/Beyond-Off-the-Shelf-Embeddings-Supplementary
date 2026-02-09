import os
from dotenv import load_dotenv

load_dotenv()

TOOLS_PATH = "../../data/ultratool/tools_expanded.json"
TRAIN_BENCHMARKS_PATH = "../../data/ultratool/without_top_benchmarks_enriched.json"
TEST_BENCHMARKS_PATH = "../../data/ultratool/top_benchmarks_enriched.json"
PAIRS_PATH = "../../data/ultratool/pairs_augmented.json"
SEED = 42

OPENROUTER_API_KEY = os.getenv("OPENROUTER_KEY")
HTTP_PROXY = os.getenv("HTTP_PROXY")
OPENROUTER_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "openai/gpt-4o-mini"

TASK_K = 10
SUBTASK_K = 5
PLANNER_K = 15


PLANNER_AGENT_SYSTEM_PROMPT = """
Rewrite the USER REQUEST as the smallest sequence of independent, solvable sub-requests.

Context: you have access to 15 tools provided in a separate context section (via RAG). Use them to
think how the request can be decomposed so that every sub-request can be solved by SOME tool
from the list. Split as aggressively as possible – the more fine-grained the better (but preserve
logical order). 

Rules:
1. Each sub-request must be a self-contained natural-language instruction; no code or tool names.
2. Preserve order and any quoted literals (file names, texts, numbers).
3. Return ONLY a JSON array of strings. No keys, no commentary.

User request: "{user_request}"
"""
