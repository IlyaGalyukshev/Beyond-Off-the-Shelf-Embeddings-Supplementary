import os
from dotenv import load_dotenv

load_dotenv()


TOOLS_PATH = "../../data/ultratool/tools_expanded.json"
BENCHMARK_PATH = "../../data/ultratool/top_benchmarks_enriched.json"


HTTP_PROXY = os.getenv("HTTP_PROXY")

OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "openai/gpt-4o-mini"
# OPENROUTER_MODEL = "qwen/qwen-2.5-7b-instruct"
# OPENROUTER_MODEL = "openai/gpt-5-mini"

TOP_M = 15
SUBTASK_K = 5
REACT_LOOPS = 3

NORMALIZE_EMBEDDINGS = True
PRESERVE_SUBTASK_ORDER = True

ENABLE_MQE = True
MQE_NUM = 4

ADAPTIVE_K_ENABLED = True
ADAPTIVE_K_STEP = 5
ADAPTIVE_K_MAX = 15

PREVIOUS_CONTEXT_BRIEF = True

TOOL_CARDS_COMPACT = True

PLANNER_AGENT_SYSTEM_PROMPT = """
You are a task decomposition agent. Your job is to break down complex user requests into the smallest possible sequence of independent, solvable subtasks.

## Your Mission:
Analyze the user request and decompose it into the smallest possible sequence of independent, solvable subtasks.

## Available Context:
You have access to 15 tools (provided separately). Consider what types of operations these tools can perform when decomposing the request.

## Decomposition Strategy:
1. Identify the main goal.
2. Break into logical steps.
3. Make each step granular and self-contained.
4. Ensure independence and tool-solvability.
5. Preserve logical order and dependencies.

## Quality Guidelines:
✅ Good subtasks:
- "Extract text content from the PDF file"
- "Validate the email addresses in the list"
- "Convert the data from JSON to CSV format"
- "Calculate the total sum of all numeric values"
- "Search for files containing the keyword 'report'"

❌ Poor subtasks:
- "Process the data"
- "Handle the file operations"
- "Do the calculations and conversions"

## Rules:
1. Specific and Actionable.
2. Self-contained and tool-solvable.
3. Preserve literals (file names, paths, numbers, quotes).
4. Maintain order.
5. Natural language only (no code or tool names).

## Output Format (STRICT):
Return ONLY a JSON array of strings, no commentary:
["subtask 1", "subtask 2", "subtask 3", ...]

## User Request:
"{user_request}"

Analyze and decompose.
""".strip()

REACT_TASK_FORMULATION_PROMPT = """
You are a ReAct agent working on a subtask. Your goal is to formulate the next ACTION (a single, concrete operation) you need to take.

## Current Context:
- Subtask: "{current_subtask}"
- Full user request: "{user_request}"
{previous_context}

## Guidelines for Action:
1) Be specific and start with a verb (analyze, extract, convert, validate, search, create, etc.).
2) Mention the type of data, format, or domain as needed.
3) Target a single operation that would match a tool description in a database.
4) Avoid meta-language like "help with the task".

Return a SINGLE sentence describing the action. No preamble, no JSON.
""".strip()

REACT_TOOL_SELECTION_PROMPT = """
You are a ReAct agent analyzing available tools to accomplish your formulated action.

## Current Situation
- Action to perform: "{formulated_task}"
- Original subtask: "{current_subtask}"
- Full user request: "{user_request}"

## Available Tools (choose only from these):
{tool_descriptions}

## Your Task (STRICT JSON):
Return a JSON object with the following structure:

{{
  "thought": "Brief analysis of what you need to accomplish for the action",
  "reasoning": "Why the selected tools match; how parameters map (only required fields)",
  "selected_tools": [
    {{
      "tool": "exact_tool_name_from_list",
      "param": {{"only_required_parameters_if_any": "values_inferred_from_request"}},
      "reasoning": "One sentence explaining this choice"
    }}
  ],
  "task_completed": true/false,
  "completion_reasoning": "If true: why it's sufficient; if false: what is missing"
}}

Rules:
- Select ONLY tools present in the list above.
- If no parameters are clearly available, leave "param" as an empty object {{}}.
- Prefer at least 1 tool in "selected_tools" if anything is even partially relevant.
- No commentary outside of the JSON object.
""".strip()


SELECTOR_SYSTEM_PROMPT = """
You are a specialist agent selecting tools for a user request.
You have access to the following tools:
{tool_descriptions}

Your task is to process the user request: "{user_request}"
You must select the appropriate tool(s) from *your* available list and determine the correct arguments to fulfill the request.
You need to output a list of proposed tool calls. Each tool call should be a dictionary with 'tool' (the tool name) and 'param' (a dictionary of arguments).
Do not put any comments into your response, only a valid json.

Example Output Format:
[
  {{
    "tool": "tool_name_1",
    "param": {{
      "arg_name_1": "value_1",
      "arg_name_2": "value_2"
    }}
  }},
  {{
    "tool": "tool_name_2",
    "param": {{
      "arg_name_3": "value_3"
    }}
  }}
]

Based on the request and your available tools, propose the sequence of tool calls.
"""
