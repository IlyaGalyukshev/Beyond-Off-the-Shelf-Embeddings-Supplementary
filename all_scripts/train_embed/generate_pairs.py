import json
import random
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ALL_SCRIPTS_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _ALL_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _ALL_SCRIPTS_DIR)

from utils.config import TOOLS_PATH, TRAIN_BENCHMARKS_PATH, PAIRS_PATH, SEED
from utils import structure_tool

QUERY_PREFIX = 'query: '
PASSAGE_PREFIX = 'passage: '

random.seed(SEED)


def generate_pairs():
    """Generate query-tool pairs for training."""
    print("Loading tools...")
    with open(TOOLS_PATH, 'r', encoding='utf-8') as f:
        tools = json.load(f)
    
    print(f"Loaded {len(tools)} tools")
    
    print("Loading benchmarks...")
    with open(TRAIN_BENCHMARKS_PATH, 'r', encoding='utf-8') as f:
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
    print(f"Saving to {PAIRS_PATH}...")
    
    with open(PAIRS_PATH, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    
    positive_pairs = sum(1 for p in pairs if p['reference'])
    negative_pairs = sum(1 for p in pairs if not p['reference'])
    
    print(f"\nStatistics:")
    print(f"  Total pairs: {len(pairs)}")
    print(f"  Positive pairs: {positive_pairs}")
    print(f"  Negative pairs: {negative_pairs}")
    print(f"  Positive/Negative ratio: {positive_pairs / negative_pairs:.2f}")


if __name__ == "__main__":
    generate_pairs()

