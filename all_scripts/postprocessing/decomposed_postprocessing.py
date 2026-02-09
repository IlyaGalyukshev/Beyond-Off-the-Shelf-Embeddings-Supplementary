import json
import logging
import sys
import time
from pathlib import Path

from httpx import Client
import dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import (
    TOOLS_PATH,
    OPENROUTER_KEY,
    OPENROUTER_MODEL,
    OPENROUTER_URL,
    HTTP_PROXY,
    SELECTOR_SYSTEM_PROMPT,
)
from tool_utils import load_tools
from schemas import ToolSchema

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("postprocessing_decomposed.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

PROGRESS_FILE = Path("results/tmp_unified_progress_minilm_trained.json")
POSTPROCESSING_RESULTS_FILE = Path(
    "results/tmp_unified_progress_minilm_trained_triplet_decomposed_results_qwen3.json"
)


def parse_json_strict(text: str, prefer_array: bool = False):
    """
    Parse JSON with error handling.
    Returns (obj, err_msg_or_None).
    """
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        text = text.strip()
        if prefer_array and text.startswith("[") and text.endswith("]"):
            try:
                return json.loads(text), None
            except json.JSONDecodeError:
                pass
        elif not prefer_array and text.startswith("{") and text.endswith("}"):
            try:
                return json.loads(text), None
            except json.JSONDecodeError:
                pass

        if '"' not in text and "'" in text:
            repaired = text.replace("'", '"')
            try:
                return json.loads(repaired), None
            except json.JSONDecodeError as e2:
                return (
                    None,
                    f"{type(e).__name__}: {e} | repair: {type(e2).__name__}: {e2}",
                )

        return None, f"{type(e).__name__}: {e}"


def llm_init() -> ChatOpenAI:
    """Initialize the LLM client."""
    llm = ChatOpenAI(
        model=OPENROUTER_MODEL,
        api_key=OPENROUTER_KEY,
        base_url=OPENROUTER_URL,
        temperature=0,
        http_client=Client(proxy=HTTP_PROXY),
        extra_body={
            "reasoning": {
                "max_tokens": 0,
                # "effort": "none",
                "exclude": True,
            },
            # "provider": {
            #     "only": ["fireworks"],
            # }
        },
    )
    return llm


def compute_metrics(ref: set[str], selected: set[str]) -> tuple[float, float, float]:
    """Return (precision, recall, f1) for one query."""
    tp = len(ref & selected)
    precision = tp / len(selected) if selected else (1.0 if not ref else 0.0)
    recall = tp / len(ref) if ref else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def get_tool_descriptions_for_selector(
    tools: dict[str, ToolSchema], tool_names: list[str]
) -> str:
    """
    Format tool descriptions for the selector agent with full details.
    """
    if not tool_names or not tools:
        return "No tools available."

    lines: list[str] = []
    for tool_name in tool_names:
        if tool_name not in tools:
            continue

        schema = tools[tool_name]
        lines.append(f"Tool: {tool_name}")
        lines.append(f"Description: {schema.description}")

        if schema.description_expanded:
            lines.append(f"Expanded Description: {schema.description_expanded}")

        if schema.arguments and schema.arguments.properties:
            lines.append("Arguments:")
            required = set(schema.arguments.required or [])
            for arg_name, arg_props in schema.arguments.properties.items():
                arg_type = getattr(arg_props, "type", "any")
                desc = getattr(arg_props, "description", "")
                req_marker = " (required)" if arg_name in required else " (optional)"
                lines.append(f"  - {arg_name}: {arg_type}{req_marker} - {desc}")
        else:
            lines.append("Arguments: None")

        if schema.results and schema.results.properties:
            lines.append("Returns:")
            for res_name, res_props in schema.results.properties.items():
                res_type = getattr(res_props, "type", "any")
                desc = getattr(res_props, "description", "")
                lines.append(f"  - {res_name}: {res_type} - {desc}")
        else:
            lines.append("Returns: None")

        lines.append("")

    return "\n".join(lines)


def invoke_selector_agent(
    llm: ChatOpenAI,
    user_request: str,
    available_tools: dict[str, ToolSchema],
    selected_tools: list[str],
    max_retries: int = 3,
) -> list[str]:
    """
    Invoke the selector agent to refine tool selection with retry logic.
    """
    logger.info("🔧 Invoking selector agent for request: %s...", user_request[:100])
    logger.info("Available tools for selection: %s", selected_tools)

    tool_descriptions = get_tool_descriptions_for_selector(
        available_tools, selected_tools
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "user",
                SELECTOR_SYSTEM_PROMPT,
            )
        ]
    )
    messages = prompt.format_messages(
        tool_descriptions=tool_descriptions,
        user_request=user_request,
    )
    rendered_text = "".join(m.content for m in messages)

    logger.info("📤 SELECTOR PROMPT:")
    logger.info(rendered_text)

    # Retry logic with exponential backoff
    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages).content
            logger.info(
                "📥 SELECTOR RESPONSE (attempt %d/%d):", attempt + 1, max_retries
            )
            logger.info(response)

            parsed, err = parse_json_strict(response, prefer_array=True)
            if err or not isinstance(parsed, list):
                logger.error(
                    "❌ Selector JSON parse error (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    err,
                )

                # If not the last attempt, retry with exponential backoff
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # 1s, 2s, 4s...
                    logger.warning("⏳ Retrying in %d seconds...", wait_time)
                    time.sleep(wait_time)
                    continue
                else:
                    logger.warning(
                        "❌ All retries exhausted. Falling back to original tools: %s",
                        selected_tools,
                    )
                    return selected_tools

            refined_tools = []
            for tool_call in parsed:
                if isinstance(tool_call, dict) and "tool" in tool_call:
                    tool_name = tool_call["tool"]
                    if tool_name in available_tools:
                        refined_tools.append(tool_name)

            logger.info("✅ Selector refined tools: %s", refined_tools)
            return refined_tools

        except Exception as e:
            logger.error(
                "❌ Selector invocation error (attempt %d/%d): %s",
                attempt + 1,
                max_retries,
                str(e),
            )

            # If not the last attempt, retry with exponential backoff
            if attempt < max_retries - 1:
                wait_time = 2**attempt
                logger.warning("⏳ Retrying in %d seconds...", wait_time)
                time.sleep(wait_time)
                continue
            else:
                logger.warning(
                    "❌ All retries exhausted. Falling back to original tools: %s",
                    selected_tools,
                )
                return selected_tools

    # Fallback (should not reach here)
    logger.warning(
        "❌ Unexpected state. Falling back to original tools: %s", selected_tools
    )
    return selected_tools


def process_decomposed_subtasks(
    llm: ChatOpenAI,
    tools_schema: dict[str, ToolSchema],
    subtask_results: list[dict],
    original_selected_tools: list[str],
) -> set[str]:
    """
    Process each subtask through selector and collect all refined tools.
    """
    all_refined_tools = set()

    for subtask_result in subtask_results:
        subtask = subtask_result["subtask"]

        logger.info("\n🎯 Processing subtask: %s...", subtask[:100])
        logger.info("Available tools: %s", original_selected_tools)

        if not original_selected_tools:
            logger.warning("No original tools to process, skipping selector")
            continue

        refined_tools = invoke_selector_agent(
            llm=llm,
            user_request=subtask,
            available_tools=tools_schema,
            selected_tools=original_selected_tools,
        )

        all_refined_tools.update(refined_tools)

        logger.info(
            "✅ Subtask processed: %d → %d tools",
            len(original_selected_tools),
            len(refined_tools),
        )

    return all_refined_tools


def main() -> None:
    logger.info("🚀 Starting Decomposed Postprocessing with Selector Agent")
    logger.info("=" * 80)

    tools_schema = load_tools(Path(TOOLS_PATH))
    if not tools_schema:
        logger.error("❌ Failed to load tools JSON.")
        sys.exit("Failed to load tools JSON.")

    logger.info("🛠️ Loaded %d tools", len(tools_schema))

    if not PROGRESS_FILE.exists():
        logger.error("❌ Progress file not found.")
        sys.exit("Progress file not found.")

    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        progress_data = json.load(f)

    detail_table = progress_data.get("detail_table", [])
    logger.info("📊 Loaded %d benchmark results", len(detail_table))

    llm = llm_init()
    logger.info("✅ LLM initialized: %s", OPENROUTER_MODEL)

    # Load existing results if available
    postprocessed_results = []
    start_idx = 0
    if POSTPROCESSING_RESULTS_FILE.exists():
        with open(POSTPROCESSING_RESULTS_FILE, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            postprocessed_results = existing_data.get("results", [])
            start_idx = len(postprocessed_results)
            logger.info(
                "📂 Loaded %d existing results, resuming from idx %d",
                start_idx,
                start_idx,
            )
    else:
        logger.info("📂 No existing results found, starting from scratch")

    for idx, item in enumerate(detail_table):
        # Skip already processed items
        if idx < start_idx:
            continue
        logger.info("\n%s", "=" * 100)
        logger.info("📋 PROCESSING ITEM %d/%d", idx + 1, len(detail_table))
        logger.info("%s", "=" * 100)

        user_request = item["question"]
        reference_tools = set(item["reference"])
        original_selected_tools = set(item["selected"])
        subtask_results = item["subtasks"]

        logger.info("❓ Question: %s...", user_request[:100])
        logger.info("🎯 Reference tools: %s", sorted(reference_tools))
        logger.info("🔧 Original selected tools: %s", sorted(original_selected_tools))

        refined_selected_tools = process_decomposed_subtasks(
            llm=llm,
            tools_schema=tools_schema,
            subtask_results=subtask_results,
            original_selected_tools=list(original_selected_tools),
        )

        original_precision, original_recall, original_f1 = compute_metrics(
            reference_tools, original_selected_tools
        )
        refined_precision, refined_recall, refined_f1 = compute_metrics(
            reference_tools, refined_selected_tools
        )

        logger.info("\n📊 METRICS COMPARISON:")
        logger.info(
            "Original - P: %.3f, R: %.3f, F1: %.3f",
            original_precision,
            original_recall,
            original_f1,
        )
        logger.info(
            "Refined  - P: %.3f, R: %.3f, F1: %.3f",
            refined_precision,
            refined_recall,
            refined_f1,
        )

        result = {
            "idx": idx,
            "question": user_request,
            "reference_tools": sorted(reference_tools),
            "original_selected_tools": sorted(original_selected_tools),
            "refined_selected_tools": sorted(refined_selected_tools),
            "original_metrics": {
                "precision": original_precision,
                "recall": original_recall,
                "f1": original_f1,
            },
            "refined_metrics": {
                "precision": refined_precision,
                "recall": refined_recall,
                "f1": refined_f1,
            },
        }

        postprocessed_results.append(result)

        with open(POSTPROCESSING_RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "completed_items": len(postprocessed_results),
                    "results": postprocessed_results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info("✅ ITEM %d COMPLETED", idx + 1)

    if postprocessed_results:
        original_precisions = [
            r["original_metrics"]["precision"] for r in postprocessed_results
        ]
        original_recalls = [
            r["original_metrics"]["recall"] for r in postprocessed_results
        ]
        original_f1s = [r["original_metrics"]["f1"] for r in postprocessed_results]

        refined_precisions = [
            r["refined_metrics"]["precision"] for r in postprocessed_results
        ]
        refined_recalls = [
            r["refined_metrics"]["recall"] for r in postprocessed_results
        ]
        refined_f1s = [r["refined_metrics"]["f1"] for r in postprocessed_results]

        logger.info("\n%s", "=" * 80)
        logger.info("🏆 FINAL DECOMPOSED POSTPROCESSING RESULTS")
        logger.info("=" * 80)
        logger.info("Total questions processed: %d", len(postprocessed_results))
        logger.info(
            "Original  - Avg P: %.3f, Avg R: %.3f, Avg F1: %.3f",
            sum(original_precisions) / len(original_precisions),
            sum(original_recalls) / len(original_recalls),
            sum(original_f1s) / len(original_f1s),
        )
        logger.info(
            "Refined   - Avg P: %.3f, Avg R: %.3f, Avg F1: %.3f",
            sum(refined_precisions) / len(refined_precisions),
            sum(refined_recalls) / len(refined_recalls),
            sum(refined_f1s) / len(refined_f1s),
        )
        logger.info("=" * 80)

    logger.info("🎉 Decomposed postprocessing completed successfully!")


if __name__ == "__main__":
    main()
