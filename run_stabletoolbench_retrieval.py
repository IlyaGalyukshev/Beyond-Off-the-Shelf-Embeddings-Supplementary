#!/usr/bin/env python3
"""
Wrapper to run full_query_retrieving.py for StableToolBench.
"""

import sys
import os
from pathlib import Path

# Set up paths correctly - the script expects parent directory in path
sys.path.insert(0, str(Path('/workspace/all_scripts/retrieving').parent))
sys.path.insert(0, '/workspace/all_scripts/retrieving')

# Import full_query_retrieving and override paths
import full_query_retrieving as fq

# Override paths
fq.TOOLS_PATH = '/workspace/data/stabletoolbench/tools_expanded.json'
fq.TEST_BENCHMARKS_PATH = '/workspace/data/stabletoolbench/top_benchmarks_enriched.json'
fq.TRAINED_MODEL_STAGE1 = '/workspace/all_scripts/train_embed/checkpoints-stabletoolbench/minilm-stage1'
fq.TRAINED_MODEL_STAGE2 = '/workspace/all_scripts/train_embed/checkpoints-stabletoolbench/minilm-stage2'

# Patch open to rename result files with dataset prefix
import builtins
_original_open = builtins.open

def patched_open(file, mode='r', *args, **kwargs):
    if mode.startswith('w') and isinstance(file, Path):
        file_path = Path(file)
        if file_path.parent.name == 'results' and file_path.suffix == '.json':
            # Rename to include stabletoolbench prefix
            new_name = file_path.stem
            if 'full_query_' in new_name and 'stabletoolbench' not in new_name:
                new_name = new_name.replace('full_query_', 'full_query_stabletoolbench_')
            file = file_path.parent / f"{new_name}{file_path.suffix}"
    return _original_open(file, mode, *args, **kwargs)

builtins.open = patched_open

print("=" * 100)
print("StableToolBench Full Query Retrieval Evaluation")
print("=" * 100)
print(f"\nConfiguration:")
print(f"  Tools: {fq.TOOLS_PATH}")
print(f"  Test benchmarks: {fq.TEST_BENCHMARKS_PATH}")
print(f"  Base model: {fq.BASE_MODEL}")
print(f"  Stage1 model: {fq.TRAINED_MODEL_STAGE1}")
print(f"  Stage2 model: {fq.TRAINED_MODEL_STAGE2}")
print("\n" + "=" * 100)

# Run main
fq.main()

print("\n" + "=" * 100)
print("✅ StableToolBench evaluation completed!")
print("=" * 100)
