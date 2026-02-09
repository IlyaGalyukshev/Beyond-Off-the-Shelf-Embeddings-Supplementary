#!/usr/bin/env python3
"""
Wrapper to run gen_decomposition.py and decomposed_retrieving.py for StableToolBench.
"""

import sys
from pathlib import Path

# Set up paths correctly
sys.path.insert(0, str(Path('/workspace/all_scripts/retrieving').parent))
sys.path.insert(0, '/workspace/all_scripts/retrieving')

print("=" * 100)
print("StableToolBench Decomposed Retrieval Pipeline")
print("=" * 100)

# Step 1: Generate decompositions
print("\n" + "=" * 100)
print("STEP 1: Generating Decompositions")
print("=" * 100)

import gen_decomposition as gen_dec

# Override paths for StableToolBench
gen_dec.TOOLS_PATH_ABS = '/workspace/data/stabletoolbench/tools_expanded.json'
gen_dec.TEST_BENCHMARKS_PATH_ABS = '/workspace/data/stabletoolbench/top_benchmarks_enriched.json'
gen_dec.OUTPUT_PATH = '/workspace/data/stabletoolbench/decompositions.json'
gen_dec.TRAINED_MODEL_STAGE2 = '/workspace/all_scripts/train_embed/checkpoints-stabletoolbench/minilm-stage1'

# Patch format_tool_descriptions to escape curly braces
_original_format_tool_descriptions = gen_dec.format_tool_descriptions

def patched_format_tool_descriptions(tools_dict, tool_names):
    """Escape curly braces in tool descriptions to avoid template errors."""
    lines = []
    for name in tool_names:
        tool = tools_dict[name]
        desc = tool.get("description_expanded", tool.get("description", ""))
        # Escape curly braces by doubling them
        name_escaped = name.replace("{", "{{").replace("}", "}}")
        desc_escaped = desc.replace("{", "{{").replace("}", "}}")
        lines.append(f"- {name_escaped}: {desc_escaped}")
    return "\n".join(lines)

gen_dec.format_tool_descriptions = patched_format_tool_descriptions

print(f"\nConfiguration:")
print(f"  Tools: {gen_dec.TOOLS_PATH_ABS}")
print(f"  Benchmarks: {gen_dec.TEST_BENCHMARKS_PATH_ABS}")
print(f"  Output: {gen_dec.OUTPUT_PATH}")
print(f"  Base model: {gen_dec.BASE_MODEL}")
print(f"  Trained model (Stage 1): {gen_dec.TRAINED_MODEL_STAGE2}")

# Check if decompositions already exist
from pathlib import Path
decomp_path = Path(gen_dec.OUTPUT_PATH)
if decomp_path.exists():
    print(f"\n✅ Decompositions already exist: {gen_dec.OUTPUT_PATH}")
    print("   Skipping generation step.")
else:
    print("\n🚀 Generating decompositions...")
    gen_dec.main()
    print("\n✅ Decompositions generated!")

# Step 2: Evaluate decomposed retrieval
print("\n" + "=" * 100)
print("STEP 2: Evaluating Decomposed Retrieval")
print("=" * 100)

import decomposed_retrieving as dec_ret

# Override paths for StableToolBench
dec_ret.TOOLS_PATH = '/workspace/data/stabletoolbench/tools_expanded.json'
dec_ret.TEST_BENCHMARKS_PATH = '/workspace/data/stabletoolbench/top_benchmarks_enriched.json'
dec_ret.DECOMPOSITIONS_PATH = '/workspace/data/stabletoolbench/decompositions.json'
dec_ret.TRAINED_MODEL_STAGE2 = '/workspace/all_scripts/train_embed/checkpoints-stabletoolbench/minilm-stage1'

# Patch open to rename result files with dataset prefix
import builtins
_original_open = builtins.open

def patched_open(file, mode='r', *args, **kwargs):
    if mode.startswith('w') and isinstance(file, Path):
        file_path = Path(file)
        if file_path.parent.name == 'results' and file_path.suffix == '.json':
            # Rename to include stabletoolbench prefix
            new_name = file_path.stem
            if 'decomposed_' in new_name and 'stabletoolbench' not in new_name:
                new_name = new_name.replace('decomposed_', 'decomposed_stabletoolbench_')
            file = file_path.parent / f"{new_name}{file_path.suffix}"
    return _original_open(file, mode, *args, **kwargs)

builtins.open = patched_open

print(f"\nConfiguration:")
print(f"  Tools: {dec_ret.TOOLS_PATH}")
print(f"  Benchmarks: {dec_ret.TEST_BENCHMARKS_PATH}")
print(f"  Decompositions: {dec_ret.DECOMPOSITIONS_PATH}")
print(f"  Base model: {dec_ret.BASE_MODEL}")
print(f"  Trained model (Stage 1): {dec_ret.TRAINED_MODEL_STAGE2}")

print("\n🚀 Running decomposed retrieval evaluation...")
dec_ret.main()

print("\n" + "=" * 100)
print("✅ StableToolBench Decomposed Retrieval Pipeline Completed!")
print("=" * 100)
