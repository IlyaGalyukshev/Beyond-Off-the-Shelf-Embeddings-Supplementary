#!/usr/bin/env python3
"""Run all retrieval evaluations for UltraTool and ToolLinkOS."""

import subprocess
import sys
from pathlib import Path

def run_with_config(script_name, dataset, config):
    """Run script with monkey-patched configuration."""
    
    script_path = Path(__file__).parent / script_name
    log_name = f"{dataset}_{script_name.replace('.py', '')}"
    log_path = Path(__file__).parent / "logs" / f"{log_name}.log"
    
    # Create Python wrapper that patches the config
    wrapper_code = f"""
import sys
sys.path.insert(0, '/workspace/all_scripts/retrieving')

# Import and patch configuration
if '{script_name}' == 'full_query_retrieving.py':
    import full_query_retrieving as module
else:
    import decomposed_retrieving as module

# Override paths
module.TOOLS_PATH = '{config['tools']}'
module.TEST_BENCHMARKS_PATH = '{config['benchmarks']}'
if hasattr(module, 'DECOMPOSITIONS_PATH'):
    module.DECOMPOSITIONS_PATH = '{config.get('decompositions', '')}'
if hasattr(module, 'TRAINED_MODEL_STAGE1'):
    module.TRAINED_MODEL_STAGE1 = '{config.get('stage1', '')}'
if hasattr(module, 'TRAINED_MODEL_STAGE2'):
    module.TRAINED_MODEL_STAGE2 = '{config.get('stage2', '')}'

# Rename output files to include dataset name
import json
from pathlib import Path
_original_open = open

def patched_open(file, mode='r', *args, **kwargs):
    if isinstance(file, (str, Path)):
        file_str = str(file)
        if 'results/' in file_str and mode == 'w':
            # Rename result files to include dataset name
            file_path = Path(file_str)
            new_name = file_path.stem.replace('full_query_', f'full_query_{dataset}_')
            new_name = new_name.replace('decomposed_', f'decomposed_{dataset}_')
            file = file_path.parent / (new_name + file_path.suffix)
    return _original_open(file, mode, *args, **kwargs)

import builtins
builtins.open = patched_open

# Run main
module.main()
"""
    
    print(f"\n{'='*100}")
    print(f"Running {script_name} for {dataset}")
    print(f"{'='*100}")
    print(f"Log: {log_path}")
    
    # Run the wrapper
    result = subprocess.run(
        [sys.executable, '-c', wrapper_code],
        stdout=open(log_path, 'w'),
        stderr=subprocess.STDOUT,
        cwd='/workspace/all_scripts/retrieving'
    )
    
    if result.returncode == 0:
        print(f"✅ Completed successfully")
    else:
        print(f"❌ Failed with return code {result.returncode}")
        print(f"Check log: {log_path}")
    
    return result.returncode


def main():
    # Create directories
    retrieving_dir = Path('/workspace/all_scripts/retrieving')
    (retrieving_dir / 'logs').mkdir(exist_ok=True)
    (retrieving_dir / 'results').mkdir(exist_ok=True)
    
    configs = {
        'ultratool': {
            'tools': '/workspace/data/ultratool/tools_expanded.json',
            'benchmarks': '/workspace/data/ultratool/top_benchmarks_enriched.json',
            'decompositions': '/workspace/data/ultratool/decompositions.json',
            'stage1': '/workspace/all_scripts/train_embed/checkpoints-adv/minilm-stage1',
            'stage2': '/workspace/all_scripts/train_embed/checkpoints-adv/minilm-stage2',
        },
        'toollinkos': {
            'tools': '/workspace/data/toollinkos/tools_expanded.json',
            'benchmarks': '/workspace/data/toollinkos/top_benchmarks_enriched.json',
            'decompositions': '/workspace/data/toollinkos/decompositions.json',
            'stage1': '/workspace/all_scripts/train_embed/checkpoints-toollinkos/minilm-stage1',
            'stage2': '/workspace/all_scripts/train_embed/checkpoints-toollinkos/minilm-stage2',
        }
    }
    
    scripts = [
        'full_query_retrieving.py',
        'decomposed_retrieving.py'
    ]
    
    total = len(configs) * len(scripts)
    current = 0
    
    for dataset, config in configs.items():
        for script in scripts:
            current += 1
            print(f"\n{'='*100}")
            print(f"Progress: {current}/{total}")
            print(f"{'='*100}")
            run_with_config(script, dataset, config)
    
    print("\n" + "="*100)
    print("✅ ALL RETRIEVAL EVALUATIONS COMPLETED!")
    print("="*100)
    print("\nResults directory: /workspace/all_scripts/retrieving/results/")
    
    # List all result files
    import os
    result_files = sorted([f for f in os.listdir('/workspace/all_scripts/retrieving/results') if f.endswith('.json')])
    print("\nGenerated files:")
    for f in result_files:
        print(f"  - {f}")


if __name__ == '__main__':
    main()
