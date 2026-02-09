#!/usr/bin/env python3
"""
Wrapper to run decomposed_postprocessing.py for StableToolBench configurations.
"""

import subprocess
import sys
from pathlib import Path

# Configurations for each run
configs = [
    {
        'name': 'stabletoolbench_base_react',
        'progress': '/workspace/all_scripts/react/results/tmp_stabletoolbench_base_progress.json',
        'tools': '/workspace/data/stabletoolbench/tools_expanded.json',
        'output': '/workspace/all_scripts/postprocessing/results/stabletoolbench_base_react_postprocessed.json',
        'log': '/workspace/all_scripts/postprocessing/logs/stabletoolbench_base_react.log'
    },
    {
        'name': 'stabletoolbench_stage1_react',
        'progress': '/workspace/all_scripts/react/results/tmp_stabletoolbench_stage1_progress.json',
        'tools': '/workspace/data/stabletoolbench/tools_expanded.json',
        'output': '/workspace/all_scripts/postprocessing/results/stabletoolbench_stage1_react_postprocessed.json',
        'log': '/workspace/all_scripts/postprocessing/logs/stabletoolbench_stage1_react.log'
    },
    {
        'name': 'stabletoolbench_base_decomposed',
        'progress': '/workspace/all_scripts/retrieving/results/decomposed_stabletoolbench_base_results.json',
        'tools': '/workspace/data/stabletoolbench/tools_expanded.json',
        'output': '/workspace/all_scripts/postprocessing/results/stabletoolbench_base_decomposed_postprocessed.json',
        'log': '/workspace/all_scripts/postprocessing/logs/stabletoolbench_base_decomposed.log'
    },
    {
        'name': 'stabletoolbench_stage1_decomposed',
        'progress': '/workspace/all_scripts/retrieving/results/decomposed_stabletoolbench_trained_stage1_results.json',
        'tools': '/workspace/data/stabletoolbench/tools_expanded.json',
        'output': '/workspace/all_scripts/postprocessing/results/stabletoolbench_stage1_decomposed_postprocessed.json',
        'log': '/workspace/all_scripts/postprocessing/logs/stabletoolbench_stage1_decomposed.log'
    }
]


def run_postprocessing(config):
    """Run postprocessing with specific configuration."""
    
    wrapper_code = f"""
import sys
sys.path.insert(0, '/workspace/all_scripts/react')
sys.path.insert(0, '/workspace/all_scripts/postprocessing')

# Import and patch
import decomposed_postprocessing as module
from pathlib import Path

# Override configuration
module.TOOLS_PATH = '{config['tools']}'
module.PROGRESS_FILE = Path('{config['progress']}')
module.POSTPROCESSING_RESULTS_FILE = Path('{config['output']}')

# Override logging
import logging
module.logger.handlers.clear()
module.logger.addHandler(logging.FileHandler('{config['log']}'))
module.logger.addHandler(logging.StreamHandler(sys.stdout))

# Run main
module.main()
"""
    
    print(f"Starting: {config['name']}")
    
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
    print("Starting Postprocessing for StableToolBench (4 configurations)")
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
