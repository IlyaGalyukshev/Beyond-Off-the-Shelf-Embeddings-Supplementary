#!/usr/bin/env python3
"""
Run decomposed postprocessing for multiple dataset/model combinations in parallel.
"""

import subprocess
import sys
from pathlib import Path

# Configurations for each run
configs = [
    {
        'name': 'toollinkos_stage1_decomposed',
        'retrieval_result': '/workspace/all_scripts/retrieving/results/decomposed_toollinkos_trained_stage1_results.json',
        'react_result': '/workspace/all_scripts/react/results/tmp_toollinkos_stage1_progress.json',
        'tools': '/workspace/data/toollinkos/tools_expanded.json',
        'output': '/workspace/all_scripts/postprocessing/results/toollinkos_stage1_decomposed_postprocessed.json',
        'log': '/workspace/all_scripts/postprocessing/logs/toollinkos_stage1_decomposed.log'
    },
    {
        'name': 'ultratool_stage2_decomposed',
        'retrieval_result': '/workspace/all_scripts/retrieving/results/decomposed_ultratool_trained_stage2_results.json',
        'react_result': '/workspace/all_scripts/react/results/tmp_ultratool_stage2_progress.json',
        'tools': '/workspace/data/ultratool/tools_expanded.json',
        'output': '/workspace/all_scripts/postprocessing/results/ultratool_stage2_decomposed_postprocessed.json',
        'log': '/workspace/all_scripts/postprocessing/logs/ultratool_stage2_decomposed.log'
    },
    {
        'name': 'toollinkos_stage1_react',
        'retrieval_result': None,  # Use only REACT results
        'react_result': '/workspace/all_scripts/react/results/tmp_toollinkos_stage1_progress.json',
        'tools': '/workspace/data/toollinkos/tools_expanded.json',
        'output': '/workspace/all_scripts/postprocessing/results/toollinkos_stage1_react_postprocessed.json',
        'log': '/workspace/all_scripts/postprocessing/logs/toollinkos_stage1_react.log'
    },
    {
        'name': 'ultratool_stage2_react',
        'retrieval_result': None,  # Use only REACT results
        'react_result': '/workspace/all_scripts/react/results/tmp_ultratool_stage2_progress.json',
        'tools': '/workspace/data/ultratool/tools_expanded.json',
        'output': '/workspace/all_scripts/postprocessing/results/ultratool_stage2_react_postprocessed.json',
        'log': '/workspace/all_scripts/postprocessing/logs/ultratool_stage2_react.log'
    }
]


def run_postprocessing(config):
    """Run postprocessing with specific configuration."""
    
    wrapper_code = f"""
import sys
sys.path.insert(0, '/workspace/all_scripts/postprocessing')

# Import and patch
import decomposed_postprocessing as module
from pathlib import Path

# Override configuration
module.TOOLS_PATH = '{config['tools']}'
module.PROGRESS_FILE = Path('{config['react_result']}')
module.POSTPROCESSING_RESULTS_FILE = Path('{config['output']}')

# Override logging
import logging
module.logger.handlers.clear()
module.logger.addHandler(logging.FileHandler('{config['log']}'))
module.logger.addHandler(logging.StreamHandler(sys.stdout))

# Run main
module.main()
"""
    
    print(f"\nStarting: {config['name']}")
    
    # Run in subprocess
    process = subprocess.Popen(
        [sys.executable, '-c', wrapper_code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd='/workspace/all_scripts/postprocessing'
    )
    
    return process


def main():
    # Create directories
    postprocessing_dir = Path('/workspace/all_scripts/postprocessing')
    (postprocessing_dir / 'logs').mkdir(exist_ok=True)
    (postprocessing_dir / 'results').mkdir(exist_ok=True)
    
    print("="*100)
    print("Starting Postprocessing for 4 configurations")
    print("="*100)
    
    processes = []
    for config in configs:
        proc = run_postprocessing(config)
        processes.append((config['name'], proc))
        print(f"✅ Started {config['name']} (PID: {proc.pid})")
    
    print("\n" + "="*100)
    print("All 4 postprocessing jobs started in parallel")
    print("="*100)
    print("\nLogs:")
    for config in configs:
        print(f"  - {config['log']}")
    
    print("\nOutputs:")
    for config in configs:
        print(f"  - {config['output']}")
    
    print("\nProcesses are running in nohup mode. You can close IDE.")
    print("="*100)


if __name__ == '__main__':
    main()
