import json
from pathlib import Path
from typing import Optional, Any, Dict, List

from langchain_core.documents import Document

from schemas import ToolSchema, BenchmarkItem
from config import TOOL_CARDS_COMPACT


def load_json(path: Path) -> Any:
    """Load JSON data from a file, return Python object or None on error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {path}: {e}")
        return None


def load_tools(path: Path) -> Optional[Dict[str, ToolSchema]]:
    """
    Load and validate tool descriptions, returning a dict keyed by tool name.
    """
    data = load_json(path)
    if not data:
        return None
    try:
        tools_list = [ToolSchema.model_validate(item) for item in data]
        return {tool.name: tool for tool in tools_list}
    except Exception as e:
        print(f"Error validating tool descriptions: {e}")
        return None


def load_benchmark(path: Path) -> Optional[List[BenchmarkItem]]:
    """
    Load a list of benchmark items from a JSON file.
    """
    data = load_json(path)
    if not data:
        return None
    if not isinstance(data, list):
        print(f"Error: Benchmark file at {path} does not contain a JSON list.")
        return None
    try:
        benchmark_items = [BenchmarkItem.model_validate(item) for item in data]
        print(f"Successfully loaded {len(benchmark_items)} benchmark items.")
        return benchmark_items
    except Exception as e:
        print(f"Error validating benchmark items: {e}")
        return None


def _truncate(text: Optional[str], max_chars: int) -> str:
    """Truncate text to max_chars with ellipsis."""
    if not text:
        return ""
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _format_required_args(
    schema: ToolSchema, max_args: int = 6, compact: bool = True
) -> str:
    """
    Return a human-readable list of required arguments with types and (brief) descriptions.
    """
    if not schema.arguments or not schema.arguments.properties:
        return "None"

    required = schema.arguments.required or []
    props = schema.arguments.properties or {}
    if not required:
        return "None"

    lines: List[str] = []
    for arg_name in required:
        prop = props.get(arg_name)
        if not prop:
            continue
        arg_type = getattr(prop, "type", "any")
        desc = getattr(prop, "description", "") or ""
        if compact:
            desc = _truncate(desc, 80)
        lines.append(
            f"{arg_name}: {arg_type} — {desc}" if desc else f"{arg_name}: {arg_type}"
        )
        if len(lines) >= max_args:
            break

    return " | ".join(lines) if lines else "None"


def _format_optional_args(
    schema: ToolSchema, max_args: int = 6, compact: bool = True
) -> str:
    """
    Optional arguments list (for reference; not always shown in compact mode).
    """
    if not schema.arguments or not schema.arguments.properties:
        return "None"

    required = set(schema.arguments.required or [])
    props = schema.arguments.properties or {}
    opt_names = [n for n in props.keys() if n not in required]
    if not opt_names:
        return "None"

    lines: List[str] = []
    for name in opt_names[:max_args]:
        prop = props.get(name)
        if not prop:
            continue
        arg_type = getattr(prop, "type", "any")
        desc = getattr(prop, "description", "") or ""
        if compact:
            desc = _truncate(desc, 80)
        lines.append(f"{name}: {arg_type} — {desc}" if desc else f"{name}: {arg_type}")

    return " | ".join(lines) if lines else "None"


def _format_results(
    schema: ToolSchema, max_fields: int = 6, compact: bool = True
) -> str:
    """
    Return a compact string of result fields with types and brief descriptions.
    """
    if not schema.results or not schema.results.properties:
        return "None"

    props = schema.results.properties or {}
    lines: List[str] = []
    for i, (name, res) in enumerate(props.items()):
        if i >= max_fields:
            break
        rtype = getattr(res, "type", "any")
        desc = getattr(res, "description", "") or ""
        if compact:
            desc = _truncate(desc, 80)
        lines.append(f"{name}: {rtype} — {desc}" if desc else f"{name}: {rtype}")

    return " | ".join(lines) if lines else "None"


def _pick_capabilities(schema: ToolSchema, compact: bool = True) -> str:
    """
    Choose the best single-line capability summary for a tool.
    Prefer description_expanded, then description. Keep it short in compact mode.
    """
    cap = schema.description_expanded or schema.description or ""
    cap = cap.replace("\n", " ").strip()
    return _truncate(cap, 240) if compact else _truncate(cap, 600)


def _pick_synth_questions(schema: ToolSchema, k: int = 2) -> List[str]:
    sq = schema.synthetic_questions or []
    out: List[str] = []
    for q in sq:
        if isinstance(q, str):
            out.append(_truncate(q.strip(), 140))
        if len(out) >= k:
            break
    return out


def format_tool_descriptions_for_llm(tools: Dict[str, ToolSchema]) -> str:
    """
    Produce a **compact** block of tool descriptions (LLM-friendly "cards").
    This replaces the old verbose format to reduce cognitive load.

    Card structure:
    - {name}: {Capabilities (1 line)}
      Required: {arg: type — desc | ...}   # or "None"
      Examples: "q1"; "q2"                  # if available

    Notes:
    - Results section is omitted in compact mode to save tokens.
    - If TOOL_CARDS_COMPACT=False, we add Optional/Returns sections.
    """
    if not tools:
        return "No tools available for this category."

    lines: List[str] = []
    for name, schema in tools.items():
        cap = _pick_capabilities(schema, compact=TOOL_CARDS_COMPACT)

        examples = _pick_synth_questions(schema, k=2)

        lines.append(f"- {name}: {cap}")

        if not TOOL_CARDS_COMPACT:
            opt = _format_optional_args(schema, compact=False)
            ret = _format_results(schema, compact=False)
            req = _format_required_args(schema, compact=TOOL_CARDS_COMPACT)
            lines.append(f"  Required: {req}")
            lines.append(f"  Optional: {opt}")
            lines.append(f"  Returns: {ret}")

        if examples:
            ex_str = '"; "'.join(examples)
            lines.append(f'  Examples: "{ex_str}"')

    return "\n".join(lines)


def build_docs(
    tool_names: set[str], tools_schema: Dict[str, ToolSchema]
) -> List[Document]:
    """
    Convert each tool into a LangChain Document for vector search.

    Store ALL information about each tool in the vector database.
    The 'USE_AUGMENTED_DOCS' flag controls whether to include expanded descriptions and synthetic questions.

    Document content schema:
      Tool Name: {name}
      Basic Description: {schema.description}
      Expanded Description: {schema.description_expanded or ''}
      Arguments: {flat list}
      Results: {flat list}
      Synthetic Questions: {list or []}

    This keeps the index rich for retrieval, while the prompt-facing cards stay compact.
    """
    docs: List[Document] = []
    for name in tool_names:
        schema = tools_schema[name]

        flat_args = _flat_arguments(schema)
        flat_results = _flat_results(schema)

        page_text = (
            f"Tool Name: {name}\n"
            f"Basic Description: {schema.description}\n"
            f"Expanded Description: {schema.description_expanded}\n"
            f"Arguments: {flat_args}\n"
            f"Results: {flat_results}\n"
            f"Synthetic Questions: {schema.synthetic_questions}"
        )

        docs.append(
            Document(
                page_content=page_text,
                metadata={"tool_name": name},
            )
        )
    return docs


def _flat_arguments(schema: ToolSchema) -> str:
    """
    Flat, readable representation of all arguments (required and optional).
    """
    if not schema.arguments or not schema.arguments.properties:
        return "None"

    props = schema.arguments.properties or {}
    required = set(schema.arguments.required or [])
    parts: List[str] = []
    for arg_name, arg_props in props.items():
        a_type = getattr(arg_props, "type", "any")
        desc = getattr(arg_props, "description", "") or ""
        req_marker = " (required)" if arg_name in required else ""
        if desc:
            parts.append(f"{arg_name}: {a_type}{req_marker} — {desc}")
        else:
            parts.append(f"{arg_name}: {a_type}{req_marker}")
    return " | ".join(parts) if parts else "None"


def _flat_results(schema: ToolSchema) -> str:
    """
    Flat representation of results fields.
    """
    if not schema.results or not schema.results.properties:
        return "None"

    parts: List[str] = []
    for name, res in (schema.results.properties or {}).items():
        rtype = getattr(res, "type", "any")
        desc = getattr(res, "description", "") or ""
        if desc:
            parts.append(f"{name}: {rtype} — {desc}")
        else:
            parts.append(f"{name}: {rtype}")
    return " | ".join(parts) if parts else "None"
