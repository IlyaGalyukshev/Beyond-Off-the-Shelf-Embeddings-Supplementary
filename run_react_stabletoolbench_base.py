#!/usr/bin/env python3
"""
Wrapper to run REACT pipeline for StableToolBench with base embedder.
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

# Patch SentenceTransformer to use base model without prefixes
from sentence_transformers import SentenceTransformer as OriginalSentenceTransformer

class PatchedSentenceTransformer:
    def __init__(self, model_name_or_path, *args, **kwargs):
        # Force base model
        self.inner = OriginalSentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', *args, **kwargs)
        self.use_prefixes = False
    
    def encode(self, sentences, *args, **kwargs):
        # No prefixes for base model
        return self.inner.encode(sentences, *args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self.inner, name)

# Replace SentenceTransformer
sys.modules['sentence_transformers'].SentenceTransformer = PatchedSentenceTransformer

# Set unique progress file path
PROGRESS_FILE = Path('/workspace/all_scripts/react/results/tmp_stabletoolbench_base_progress.json')
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
log_file = '/workspace/logs/react_stabletoolbench_base.log'
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# Clear existing handlers and add new ones
run_react.logger.handlers.clear()
run_react.logger.addHandler(logging.FileHandler(log_file))
run_react.logger.addHandler(logging.StreamHandler(sys.stdout))
run_react.logger.setLevel(logging.INFO)

print("=" * 100)
print("REACT Pipeline: StableToolBench with Base Embedder")
print("=" * 100)
print(f"Tools: {config.TOOLS_PATH}")
print(f"Benchmarks: {config.TEST_BENCHMARKS_PATH}")
print(f"Embedder: sentence-transformers/all-MiniLM-L6-v2 (base, no prefixes)")
print(f"Progress file: {PROGRESS_FILE}")
print(f"Log file: {log_file}")
print("=" * 100)

# Run REACT
run_react.main()

print("\n" + "=" * 100)
print("✅ REACT Pipeline Completed!")
print("=" * 100)
