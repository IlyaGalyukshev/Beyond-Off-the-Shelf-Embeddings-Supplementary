#!/usr/bin/env python3
"""
Wrapper to run REACT pipeline for StableToolBench with stage1 embedder.
"""

import sys
import os
from pathlib import Path

# Set up paths
sys.path.insert(0, '/workspace/all_scripts/react')

# Import and configure
import config
config.TOOLS_PATH = '/workspace/data/stabletoolbench/tools_expanded.json'
config.TEST_BENCHMARKS_PATH = '/workspace/data/stabletoolbench/top_benchmarks_enriched.json'

# Patch SentenceTransformer to use stage1 model with prefixes
from sentence_transformers import SentenceTransformer as OriginalSentenceTransformer

TRAINED_MODEL_PATH = '/workspace/all_scripts/train_embed/checkpoints-stabletoolbench/minilm-stage1'
QUERY_PREFIX = 'query: '
PASSAGE_PREFIX = 'passage: '

class PatchedSentenceTransformer:
    def __init__(self, model_name_or_path, *args, **kwargs):
        # Force stage1 model
        self.inner = OriginalSentenceTransformer(TRAINED_MODEL_PATH, *args, **kwargs)
        self.use_prefixes = True
    
    def encode(self, sentences, *args, **kwargs):
        # Add prefixes based on context
        if isinstance(sentences, str):
            sentences = [sentences]
            was_single = True
        else:
            was_single = False
        
        # Heuristic: if sentence is short and looks like a query, add query prefix
        # Otherwise add passage prefix (for tools)
        prefixed = []
        for s in sentences:
            s_stripped = s.strip()
            # If already has prefix, don't add again
            if s_stripped.startswith(QUERY_PREFIX) or s_stripped.startswith(PASSAGE_PREFIX):
                prefixed.append(s)
            # If looks like a tool description (contains ':'), add passage prefix
            elif ':' in s_stripped[:100] or len(s_stripped) > 200:
                prefixed.append(PASSAGE_PREFIX + s)
            # Otherwise assume it's a query
            else:
                prefixed.append(QUERY_PREFIX + s)
        
        result = self.inner.encode(prefixed, *args, **kwargs)
        
        if was_single:
            return result[0] if len(result) == 1 else result
        return result
    
    def __getattr__(self, name):
        return getattr(self.inner, name)

# Replace SentenceTransformer
sys.modules['sentence_transformers'].SentenceTransformer = PatchedSentenceTransformer

# Set unique progress file path
PROGRESS_FILE = Path('/workspace/all_scripts/react/results/tmp_stabletoolbench_stage1_progress.json')
PROGRESS_FILE.parent.mkdir(exist_ok=True, parents=True)

# Import run_react and patch
import run_react
run_react.PROGRESS_FILE = PROGRESS_FILE
run_react.BENCHMARK_PATH = '/workspace/data/stabletoolbench/top_benchmarks_enriched.json'
run_react.TOOLS_PATH = '/workspace/data/stabletoolbench/tools_expanded.json'

# Patch tool_utils.format_tool_descriptions_for_llm to escape curly braces
from tool_utils import format_tool_descriptions_for_llm as _original_format
import tool_utils

def patched_format_tool_descriptions_for_llm(tools):
    """Escape curly braces in tool names/descriptions to avoid template errors."""
    from typing import Dict
    from schemas import ToolSchema
    
    # Escape curly braces in all tool names and descriptions
    escaped_tools = {}
    for name, schema in tools.items():
        # Escape name
        escaped_name = name.replace("{", "{{").replace("}", "}}")
        
        # Create new schema with escaped fields
        escaped_schema = ToolSchema(
            name=escaped_name,
            description=schema.description.replace("{", "{{").replace("}", "}}") if schema.description else "",
            arguments=schema.arguments,
            results=schema.results,
            synthetic_questions=schema.synthetic_questions
        )
        escaped_tools[escaped_name] = escaped_schema
    
    # Call original function with escaped data
    return _original_format(escaped_tools)

tool_utils.format_tool_descriptions_for_llm = patched_format_tool_descriptions_for_llm
run_react.format_tool_descriptions_for_llm = patched_format_tool_descriptions_for_llm

# Redirect logging
import logging
log_file = '/workspace/logs/react_stabletoolbench_stage1.log'
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# Clear existing handlers and add new ones
run_react.logger.handlers.clear()
run_react.logger.addHandler(logging.FileHandler(log_file))
run_react.logger.addHandler(logging.StreamHandler(sys.stdout))
run_react.logger.setLevel(logging.INFO)

print("=" * 100)
print("REACT Pipeline: StableToolBench with Stage1 Embedder")
print("=" * 100)
print(f"Tools: {config.TOOLS_PATH}")
print(f"Benchmarks: {config.TEST_BENCHMARKS_PATH}")
print(f"Embedder: {TRAINED_MODEL_PATH} (stage1, with prefixes)")
print(f"Progress file: {PROGRESS_FILE}")
print(f"Log file: {log_file}")
print("=" * 100)

# Run REACT
run_react.main()

print("\n" + "=" * 100)
print("✅ REACT Pipeline Completed!")
print("=" * 100)
