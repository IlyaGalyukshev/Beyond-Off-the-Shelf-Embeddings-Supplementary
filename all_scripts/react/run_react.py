import json
import logging
import statistics
import sys
import tempfile
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

from httpx import Client
import dotenv
from sentence_transformers import SentenceTransformer

from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

# Override config paths
BENCHMARK_PATH = "../../data/ultratool/top_benchmarks_enriched.json"
TOOLS_PATH = "../../data/ultratool/tools_expanded.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

from config import (
    SUBTASK_K,
    TOP_M,
    REACT_LOOPS,
    OPENROUTER_KEY,
    OPENROUTER_MODEL,
    OPENROUTER_URL,
    HTTP_PROXY,
    PLANNER_AGENT_SYSTEM_PROMPT,
    REACT_TASK_FORMULATION_PROMPT,
    REACT_TOOL_SELECTION_PROMPT,
    NORMALIZE_EMBEDDINGS,
    PRESERVE_SUBTASK_ORDER,
    ENABLE_MQE,
    MQE_NUM,
    ADAPTIVE_K_ENABLED,
    ADAPTIVE_K_STEP,
    ADAPTIVE_K_MAX,
    PREVIOUS_CONTEXT_BRIEF,
)
from tool_utils import (
    load_benchmark,
    load_tools,
    format_tool_descriptions_for_llm,
    build_docs,
)
from schemas import (
    ToolSchema,
    ReactToolCall,
    ReactIteration,
    ReactSubtaskResult,
    ReactBenchmarkResult,
)

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("react_ultratool_base.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

PROGRESS_FILE = Path(
    "/workspace/all_scripts/react/results/tmp_ultratool_base_progress.json"
)


class STEmbeddings(Embeddings):
    """Adapter so LangChain can use SentenceTransformers inside Chroma."""

    def __init__(
        self,
        st_model: SentenceTransformer,
        *,
        query_prompt_name: str = "query",
        normalize: bool = True,
    ):
        self.model = st_model
        self.query_prompt_name = query_prompt_name
        self.normalize = normalize

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=self.normalize).tolist()
        return vecs

    def embed_query(self, text: str) -> list[float]:
        vec = self.model.encode(
            [text],
            prompt_name=self.query_prompt_name,
            normalize_embeddings=self.normalize,
        )[0]
        return vec.tolist()


def compute_metrics(ref: Set[str], selected: Set[str]) -> Tuple[float, float, float]:
    """Return (precision, recall, f1) for one query."""
    tp = len(ref & selected)
    precision = tp / len(selected) if selected else (1.0 if not ref else 0.0)
    recall = tp / len(ref) if ref else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _extract_balanced(text: str, start: int) -> Optional[str]:
    """
    Return a balanced JSON substring starting at index 'start' (either '{' or '[').
    Handles quotes and escapes so braces/brackets inside strings don't break counting.
    """
    if start < 0 or start >= len(text):
        return None
    open_ch = text[start]
    if open_ch not in "{[":
        return None
    close_ch = "}" if open_ch == "{" else "]"

    depth = 0
    i = start
    in_str = False
    esc = False
    quote_ch = ""

    while i < len(text):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote_ch:
                in_str = False
        else:
            if ch == '"' or ch == "'":
                in_str = True
                quote_ch = ch
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        i += 1
    return None


def extract_json_block(text: str, prefer_array: bool = False) -> str:
    """
    Extract a JSON object/array by scanning for the first plausible balanced block.
    Try preferred type first (array or object), then the other type. If nothing found, return minimal stub.
    """
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    primary = "[" if prefer_array else "{"
    secondary = "{" if prefer_array else "["

    for idx, ch in enumerate(text):
        if ch == primary:
            blk = _extract_balanced(text, idx)
            if blk is not None:
                return blk

    for idx, ch in enumerate(text):
        if ch == secondary:
            blk = _extract_balanced(text, idx)
            if blk is not None:
                return blk

    return "[]" if prefer_array else "{}"


def parse_json_strict(text: str, prefer_array: bool = False):
    """
    Strict parse with very light repair. No regex recursion; uses balanced scanning.
    Returns (obj, err_msg_or_None).
    """
    block = extract_json_block(text, prefer_array=prefer_array)
    try:
        return json.loads(block), None
    except Exception as e:
        repaired = block
        if '"' not in repaired and "'" in repaired:
            repaired = repaired.replace("'", '"')
            try:
                return json.loads(repaired), None
            except Exception as e2:
                return (
                    None,
                    f"{type(e).__name__}: {e} | repair: {type(e2).__name__}: {e2}",
                )
        return None, f"{type(e).__name__}: {e}"


def llm_init() -> ChatOpenAI:
    llm = ChatOpenAI(
        model=OPENROUTER_MODEL,
        api_key=OPENROUTER_KEY,
        base_url=OPENROUTER_URL,
        temperature=0,
        http_client=Client(proxy=HTTP_PROXY),
    )
    return llm


def planner_prompt(user_request: str, tool_desc_block: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PLANNER_AGENT_SYSTEM_PROMPT),
            (
                "user",
                "Available tools:\n"
                + tool_desc_block
                + "\n\nUser request:\n{user_request}",
            ),
        ]
    ).format_prompt(user_request=user_request)
    return prompt


def task_formulation_prompt(
    subtask: str, user_request: str, previous_context: str
) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a careful ReAct agent. Return a SINGLE actionable sentence. No preamble.",
            ),
            ("user", REACT_TASK_FORMULATION_PROMPT),
        ]
    ).format_prompt(
        current_subtask=subtask,
        user_request=user_request,
        previous_context=previous_context,
    )
    return prompt


def tool_selection_prompt(
    formulated_task: str,
    subtask: str,
    user_request: str,
    tool_desc_block: str,
    previous_context: str,
) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Return STRICT JSON matching the schema. No extra text."),
            ("user", REACT_TOOL_SELECTION_PROMPT),
        ]
    ).format_prompt(
        formulated_task=formulated_task,
        current_subtask=subtask,
        user_request=user_request,
        tool_descriptions=tool_desc_block,
    )
    return prompt


def invoke_planner(
    llm: ChatOpenAI, user_request: str, tool_desc_block: str
) -> List[str]:
    """Return an ordered list of subtasks produced by the planner LLM."""
    logger.info("=" * 80)
    logger.info("🎯 PLANNER INVOCATION")
    logger.info("=" * 80)
    logger.info(f"User Request: {user_request}")
    logger.info(f"Available Tools in Context: {len(tool_desc_block.split('- ')) - 1}")

    prompt = planner_prompt(user_request, tool_desc_block)

    logger.info("📤 PLANNER PROMPT:")
    logger.info("-" * 40)
    logger.info(prompt.to_string())
    logger.info("-" * 40)

    response = llm.invoke(prompt).content

    logger.info("📥 PLANNER RESPONSE:")
    logger.info("-" * 40)
    logger.info(response)
    logger.info("-" * 40)

    parsed, err = parse_json_strict(response, prefer_array=True)
    if err or not isinstance(parsed, list):
        logger.error(f"❌ Planner JSON parse error: {err}")
        subtasks = []
    else:
        subtasks = [s for s in parsed if isinstance(s, str)]

    if not PRESERVE_SUBTASK_ORDER:
        subtasks = list(sorted(set(subtasks)))
    else:
        seen = set()
        ordered = []
        for s in subtasks:
            if s not in seen:
                ordered.append(s)
                seen.add(s)
        subtasks = ordered

    logger.info(
        f"✅ Parsed {len(subtasks)} subtasks (order preserved={PRESERVE_SUBTASK_ORDER}):"
    )
    for i, subtask in enumerate(subtasks, 1):
        logger.info(f"  {i}. {subtask}")
    return subtasks


def mqe_queries(llm: ChatOpenAI, action: str, n: int) -> List[str]:
    """Generate n paraphrases for the action and return [action] + paraphrases."""
    if not ENABLE_MQE or n <= 0:
        return [action]

    mqe_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Return STRICT JSON array of strings with diverse, concrete paraphrases. No commentary.",
            ),
            (
                "user",
                'Generate {n} diverse paraphrases for this action to retrieve tools:\n"{action}"\nReturn JSON array.',
            ),
        ]
    ).format_prompt(action=action, n=n)

    resp = llm.invoke(mqe_template).content
    arr, err = parse_json_strict(resp, prefer_array=True)
    if isinstance(arr, list):
        out = [action] + [s for s in arr if isinstance(s, str)]
        seen = set()
        uniq = []
        for s in out:
            s_clean = s.strip()
            if s_clean and s_clean not in seen:
                uniq.append(s_clean)
                seen.add(s_clean)
        return uniq[: 1 + n]
    return [action]


def invoke_react_agent(
    llm: ChatOpenAI,
    vectordb: Chroma,
    tools_schema: Dict[str, ToolSchema],
    user_request: str,
    subtask: str,
    previous_selected_tools: Set[str],
) -> ReactSubtaskResult:
    """
    ReAct agent that iteratively formulates tasks and selects tools for a subtask.
    """
    logger.info("\n" + "=" * 80)
    logger.info("🔄 REACT AGENT STARTED")
    logger.info("=" * 80)
    logger.info(f"Subtask: {subtask}")
    logger.info(f"Max iterations: {REACT_LOOPS}")

    iterations: List[ReactIteration] = []
    selected_tools_history: Set[str] = set()
    formulated_tasks_history: Set[str] = set()

    adaptive_k = SUBTASK_K

    for iteration_num in range(1, REACT_LOOPS + 1):
        logger.info(f"\n🔄 REACT ITERATION {iteration_num}/{REACT_LOOPS}")
        logger.info("-" * 60)

        previous_context = ""
        if PREVIOUS_CONTEXT_BRIEF:
            if previous_selected_tools:
                previous_context = f"- Already selected in earlier steps: {sorted(previous_selected_tools)}"
        else:
            previous_context = ""

        logger.info("🎯 STEP 1: Task Formulation")
        task_prompt = task_formulation_prompt(subtask, user_request, previous_context)

        logger.info("📤 TASK FORMULATION PROMPT:")
        logger.info(task_prompt.to_string())

        formulated_task = llm.invoke(task_prompt).content.strip()

        formulated_task = re.sub(
            r"\(iteration\s*\d+\)\s*$", "", formulated_task
        ).strip()

        logger.info("📥 TASK FORMULATION RESPONSE:")
        logger.info(f"Formulated Task: {formulated_task}")

        if formulated_task in formulated_tasks_history:
            logger.info(
                "🔁 Formulated task repeated; will proceed but retrieval uses MQE to diversify."
            )
        formulated_tasks_history.add(formulated_task)

        logger.info("🔍 STEP 2: Tool Retrieval")
        queries = (
            mqe_queries(llm, formulated_task, MQE_NUM)
            if ENABLE_MQE
            else [formulated_task]
        )
        logger.info(f"Retrieval queries (MQE={ENABLE_MQE}): {queries}")

        retrieved_docs = []
        seen_tools = set()
        for q in queries:
            docs = vectordb.similarity_search(q, k=adaptive_k)
            for d in docs:
                tn = d.metadata.get("tool_name")
                if tn and tn not in seen_tools and tn in tools_schema:
                    retrieved_docs.append(d)
                    seen_tools.add(tn)

        subtask_tools: Dict[str, ToolSchema] = {
            d.metadata["tool_name"]: tools_schema[d.metadata["tool_name"]]
            for d in retrieved_docs
        }

        if not subtask_tools:
            logger.warning(
                "⚠️ No tools retrieved for this subtask; will fallback to first K tools from schema (one-off)."
            )
            for name in list(tools_schema.keys())[:adaptive_k]:
                subtask_tools[name] = tools_schema[name]

        logger.info(
            f"📦 Retrieved {len(subtask_tools)} unique tools: {list(subtask_tools.keys())[:10]}{'...' if len(subtask_tools)>10 else ''}"
        )

        logger.info("⚡ STEP 3: Tool Selection")
        desc_block = format_tool_descriptions_for_llm(subtask_tools)

        selection_prompt = tool_selection_prompt(
            formulated_task=formulated_task,
            subtask=subtask,
            user_request=user_request,
            tool_desc_block=desc_block,
            previous_context=previous_context,
        )

        logger.info("📤 TOOL SELECTION PROMPT:")
        logger.info(selection_prompt.to_string())

        response = llm.invoke(selection_prompt).content

        logger.info("📥 TOOL SELECTION RESPONSE:")
        logger.info(response)

        parsed_response = {}
        selected_tools_calls: List[ReactToolCall] = []
        task_completed = False
        completion_reasoning = ""

        parsed, err = parse_json_strict(response, prefer_array=False)
        if err or not isinstance(parsed, dict):
            logger.error(f"❌ Selection JSON parse error: {err}")
            parsed_response = {}
        else:
            parsed_response = parsed

        if isinstance(parsed_response, dict):
            agent_thought = parsed_response.get("thought", None)
            agent_reasoning = parsed_response.get("reasoning", None)

            selected_list = parsed_response.get("selected_tools", []) or []
            for tool_data in selected_list:
                name = (tool_data or {}).get("tool", "")
                if (
                    not name
                    or name in selected_tools_history
                    or name not in subtask_tools
                ):
                    continue
                param = tool_data.get("param", {}) or {}
                reasoning = tool_data.get("reasoning", "") or ""
                selected_tools_calls.append(
                    ReactToolCall(tool=name, param=param, reasoning=reasoning)
                )
                selected_tools_history.add(name)

            task_completed = bool(parsed_response.get("task_completed", False))
            completion_reasoning = parsed_response.get("completion_reasoning", "") or ""
        else:
            agent_thought = None
            agent_reasoning = None

        if (
            not selected_tools_calls
            and ADAPTIVE_K_ENABLED
            and adaptive_k < ADAPTIVE_K_MAX
        ):
            adaptive_k = min(ADAPTIVE_K_MAX, adaptive_k + ADAPTIVE_K_STEP)
            logger.info(
                f"⬆️ Adaptive K increased to {adaptive_k} for next iteration due to empty selection."
            )

        if not selected_tools_calls and subtask_tools:
            for d in retrieved_docs:
                tn = d.metadata.get("tool_name")
                if tn and tn in subtask_tools and tn not in selected_tools_history:
                    logger.info(f"🛟 Fallback selects top candidate: {tn}")
                    selected_tools_calls.append(
                        ReactToolCall(
                            tool=tn,
                            param={},
                            reasoning="Fallback: top candidate from retrieval.",
                        )
                    )
                    selected_tools_history.add(tn)
                    break

        iteration = ReactIteration(
            iteration=iteration_num,
            formulated_task=formulated_task,
            selected_tools=selected_tools_calls,
            task_completed=task_completed,
            completion_reasoning=completion_reasoning,
            mapped_tools_count=len(subtask_tools),
            agent_thought=agent_thought,
            agent_reasoning=agent_reasoning,
        )

        iterations.append(iteration)

        logger.info("📊 ITERATION SUMMARY:")
        logger.info(f"  - Formulated Action: {formulated_task}")
        logger.info(f"  - Tools Retrieved from Vector DB: {len(subtask_tools)}")
        logger.info(f"  - Tools Selected: {len(selected_tools_calls)}")
        if selected_tools_calls:
            for tool in selected_tools_calls:
                logger.info(f"    • {tool.tool}: {tool.reasoning}")
        logger.info(f"  - Task Completed: {task_completed}")
        logger.info(f"  - Completion Reasoning: {completion_reasoning}")
        logger.info(f"  - Total Tools Selected So Far: {len(selected_tools_history)}")
        if selected_tools_history:
            logger.info(f"  - All Selected Tools: {list(selected_tools_history)}")

        if task_completed:
            logger.info("🎉 Breaking early - task marked completed by agent.")
            break

    final_tools = list(selected_tools_history)
    completed = any(iter.task_completed for iter in iterations)

    logger.info("\n" + "=" * 80)
    logger.info("🏁 REACT AGENT COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Subtask: {subtask}")
    logger.info(f"Total iterations: {len(iterations)}")
    logger.info(f"Final tools selected: {final_tools}")
    logger.info(f"Task completed: {completed}")

    return ReactSubtaskResult(
        subtask=subtask,
        iterations=iterations,
        final_tools=final_tools,
        completed=completed,
        total_iterations=len(iterations),
    )


def main() -> None:
    logger.info("🚀 Starting ReAct Agent Benchmark (improved)")
    logger.info("=" * 80)

    benchmark = load_benchmark(BENCHMARK_PATH)
    tools_schema = load_tools(TOOLS_PATH)
    if not benchmark or not tools_schema:
        logger.error("❌ Failed to load benchmark or tools JSON.")
        sys.exit("Failed to load benchmark or tools JSON.")

    logger.info(f"📊 Loaded {len(benchmark)} benchmark items")
    logger.info(f"🛠️ Loaded {len(tools_schema)} tools")

    logger.info("🤖 Initializing models and vector database...")

    st_model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = STEmbeddings(st_model, normalize=NORMALIZE_EMBEDDINGS)
    logger.info(
        f"✅ Sentence transformer model loaded (normalize={NORMALIZE_EMBEDDINGS})"
    )

    all_docs = build_docs(set(tools_schema), tools_schema)
    vectordb = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        collection_name="full_tools",
        persist_directory=tempfile.mkdtemp(prefix="full_tools_vdb_"),
    )
    logger.info(f"✅ Vector database created with {len(all_docs)} documents")

    llm = llm_init()
    logger.info(f"✅ LLM initialized: {OPENROUTER_MODEL}")

    if PROGRESS_FILE.exists():
        data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        completed: Set[int] = set(data.get("completed", []))
        detail_table = data.get("detail_table", [])
        logger.info(f"📂 Loaded progress: {len(completed)} completed items")
    else:
        completed = set()
        detail_table = []
        logger.info("📂 Starting fresh - no progress file found")

    logger.info(
        f"🎯 Processing {len(benchmark) - len(completed)} remaining benchmark items"
    )

    for idx, item in enumerate(benchmark):
        if idx in completed:
            continue

        logger.info("\n" + "=" * 100)
        logger.info(f"📋 BENCHMARK ITEM {idx + 1}/{len(benchmark)}")
        logger.info("=" * 100)

        user_request = item.question
        ref_tools = {r.tool for r in item.reference if r.tool in tools_schema}

        logger.info(f"❓ Question: {user_request}")
        logger.info(f"🎯 Reference tools: {sorted(ref_tools)}")

        retrieved_docs = vectordb.similarity_search(user_request, k=TOP_M)
        planner_tools = {
            doc.metadata["tool_name"]: tools_schema[doc.metadata["tool_name"]]
            for doc in retrieved_docs
            if doc.metadata["tool_name"] in tools_schema
        }
        tool_desc_block = format_tool_descriptions_for_llm(planner_tools)

        subtasks = invoke_planner(llm, user_request, tool_desc_block)

        logger.info(f"\n🔄 Starting ReAct process for {len(subtasks)} subtasks")
        subtask_results: List[ReactSubtaskResult] = []
        selected_tools: Set[str] = set()

        for subtask_idx, subtask in enumerate(subtasks, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"🎯 PROCESSING SUBTASK {subtask_idx}/{len(subtasks)}")
            logger.info(f"{'='*60}")

            subtask_result = invoke_react_agent(
                llm=llm,
                vectordb=vectordb,
                tools_schema=tools_schema,
                user_request=user_request,
                subtask=subtask,
                previous_selected_tools=selected_tools,
            )

            subtask_results.append(subtask_result)
            selected_tools.update(subtask_result.final_tools)

            logger.info(f"✅ Subtask {subtask_idx} completed:")
            logger.info(f"   - Tools selected: {subtask_result.final_tools}")
            logger.info(f"   - Iterations used: {subtask_result.total_iterations}")
            logger.info(f"   - Task completed: {subtask_result.completed}")

        precision, recall, f1 = compute_metrics(ref_tools, selected_tools)

        logger.info("\n📊 FINAL RESULTS:")
        logger.info("-" * 40)
        logger.info(f"Reference tools: {sorted(ref_tools)}")
        logger.info(f"Selected tools: {sorted(selected_tools)}")
        logger.info(f"Precision: {precision:.3f}")
        logger.info(f"Recall: {recall:.3f}")
        logger.info(f"F1: {f1:.3f}")

        react_result = ReactBenchmarkResult(
            idx=idx,
            question=user_request,
            subtasks=subtask_results,
            reference=sorted(ref_tools),
            selected=sorted(selected_tools),
            precision=precision,
            recall=recall,
            f1=f1,
        )

        detail_table.append(react_result.model_dump())
        completed.add(idx)

        PROGRESS_FILE.write_text(
            json.dumps(
                {"completed": sorted(completed), "detail_table": detail_table},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        total_iterations = sum(st.total_iterations for st in subtask_results)
        logger.info(
            f"✅ ITEM {idx + 1} COMPLETED: P={precision:.3f} R={recall:.3f} F1={f1:.3f} | "
            f"refs={len(ref_tools)} sel={len(selected_tools)} | "
            f"subtasks={len(subtask_results)} total_iters={total_iterations}"
        )

    if not detail_table:
        logger.warning("⚠️ No items processed.")
        return

    precisions = [r["precision"] for r in detail_table]
    recalls = [r["recall"] for r in detail_table]
    f1s = [r["f1"] for r in detail_table]

    logger.info("\n" + "=" * 80)
    logger.info("🏆 FINAL BENCHMARK RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total questions processed: {len(detail_table)}")
    logger.info(f"Average Precision: {statistics.mean(precisions):.3f}")
    logger.info(f"Average Recall: {statistics.mean(recalls):.3f}")
    logger.info(f"Average F1: {statistics.mean(f1s):.3f}")

    total_subtasks = sum(len(r["subtasks"]) for r in detail_table)
    total_iterations = sum(
        sum(st["total_iterations"] for st in r["subtasks"]) for r in detail_table
    )
    avg_iterations_per_subtask = (
        total_iterations / total_subtasks if total_subtasks > 0 else 0
    )

    logger.info(f"Total subtasks: {total_subtasks}")
    logger.info(f"Total ReAct iterations: {total_iterations}")
    logger.info(f"Average iterations per subtask: {avg_iterations_per_subtask:.2f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
