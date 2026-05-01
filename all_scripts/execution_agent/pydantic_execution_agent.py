from __future__ import annotations

import json
import logging
import os
import argparse
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

LOGGER = logging.getLogger(__name__)



class ToolCard(BaseModel):
    """Tool description passed into the agent (ToolBench-compatible card)."""

    name: str
    description: str = ""
    input_structure: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    results: dict[str, Any] = Field(default_factory=dict)


class ToolApiConfig(BaseModel):
    """HTTP execution metadata required by the tool backend."""

    category: str
    tool_name: str
    api_name: str


class ExecutionDeps(BaseModel):
    """Dependencies injected into pydantic-ai RunContext."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    executor: "ToolExecutor"


class ToolExecutor:
    """Resolves tool metadata and executes calls via ToolBench virtual endpoint."""

    def __init__(
        self,
        tool_index_path: str | Path,
        base_url: str = "http://a.dgx:61111/virtual",
        timeout: float = 45.0,
        stringify_tool_input: bool = False,
        log_calls: bool = True,
        response_preview_chars: int = 1200,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.stringify_tool_input = stringify_tool_input
        self.log_calls = log_calls
        self.response_preview_chars = response_preview_chars
        self._tool_index = self._load_tool_index(tool_index_path)

    def _load_tool_index(self, path: str | Path) -> dict[str, ToolApiConfig]:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))

        if isinstance(raw, dict):
            raw = raw.get("tools", raw)

        if not isinstance(raw, list):
            raise ValueError("Tool index must be a JSON list or {'tools': [...]} object")

        parsed: dict[str, ToolApiConfig] = {}
        for item in raw:
            name = item.get("name")
            if not name:
                continue

            payload = item
            if isinstance(item.get("call_config"), dict):
                payload = item["call_config"]
            elif isinstance(item.get("tool_call"), dict):
                payload = item["tool_call"]

            if (
                "Category" in payload
                and "normalized_name" in payload
                and "api_name" in payload
            ):
                payload = {
                    "category": payload["Category"],
                    "tool_name": payload["normalized_name"],
                    "api_name": payload["api_name"],
                }

            if all(k in payload for k in ("category", "tool_name", "api_name")):
                normalized_payload = {
                    "category": self._normalize_null_string(payload.get("category")),
                    "tool_name": self._normalize_null_string(payload.get("tool_name")),
                    "api_name": self._normalize_null_string(payload.get("api_name")),
                }
                parsed[name] = ToolApiConfig.model_validate(normalized_payload)

        if not parsed:
            raise ValueError("No executable tools found in tool index")
        return parsed

    @staticmethod
    def _normalize_null_string(value: Any) -> str:
        if value is None:
            return "Null"
        return str(value)

    def call_tool(self, name: str, args: dict[str, Any]) -> str:
        cfg = self._tool_index.get(name)
        if cfg is None:
            known = ", ".join(sorted(self._tool_index.keys())[:10])
            raise KeyError(f"Unknown tool '{name}'. Known sample: {known}")

        tool_input: dict[str, Any] | str = args
        if self.stringify_tool_input:
            tool_input = json.dumps(args, ensure_ascii=False)

        data = {
            "category": cfg.category,
            "tool_name": cfg.tool_name,
            "api_name": cfg.api_name,
            "tool_input": tool_input,
            "strip": "truncate",
            "toolbench_key": "",
        }
        if self.log_calls:
            LOGGER.info(
                "Tool call -> name=%s category=%s tool_name=%s api_name=%s args=%s",
                name,
                cfg.category,
                cfg.tool_name,
                cfg.api_name,
                json.dumps(args, ensure_ascii=False),
            )
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }

        response = requests.post(
            self.base_url,
            headers=headers,
            data=json.dumps(data, ensure_ascii=False),
            timeout=self.timeout,
        )
        if self.log_calls:
            preview = response.text[: self.response_preview_chars]
            LOGGER.info(
                "Tool response <- status=%s tool=%s body_preview=%s",
                response.status_code,
                name,
                preview,
            )
        response.raise_for_status()
        return response.text


SYSTEM_PROMPT = """
You are an execution agent for a dynamic toolset. You are answering all questions users asks you.

Your workflow:
1) Read users query and answer it IN FULL, considering all details user wants.
2) Read the user query and the provided `toolset`.
3) Pick the right tool(s) and prepare valid arguments.
4) Execute tools only through `run_tool(name, args)`.
5) If multiple steps are needed, call tools sequentially.
6) Never fabricate tool results; rely strictly on tool outputs.
7) Return the final answer and a short trace of tool calls.
8) If you cannot handle the request write names of tools in toolset that you were given and the reason why you cant execute them.
9) If you got some error of usage return what you tried to use with which arguments.
10) Check if you had provided all answers for all questions.

Rules:
- `name` must exactly match a `toolset[].name` value.
- `args` must be JSON-compatible.
- If no tool can solve the request, state that explicitly.
""".strip()


def build_model(
    *,
    model: str,
    use_openrouter: bool = False,
    openrouter_api_key: str | None = None,
) -> str | OpenAIChatModel:
    if not use_openrouter:
        return model

    api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required when use_openrouter=True")

    return OpenAIChatModel(
        model,
        provider=OpenRouterProvider(api_key=api_key),
    )


agent = Agent(
    "openai:gpt-4o-mini",
    deps_type=ExecutionDeps,
    system_prompt=SYSTEM_PROMPT,
)


@agent.tool
def run_tool(ctx: RunContext[ExecutionDeps], name: str, args: dict[str, Any]) -> str:
    """
    Execute an external tool by its name.

    Use this only when you have already identified the correct tool
    and prepared valid input arguments for it.

    Args:
        name: Exact name of the tool to execute.
        args: JSON-compatible arguments for the tool call.
    """
    return ctx.deps.executor.call_tool(name, args)


class ExecutionRequest(BaseModel):
    query: str
    toolset: list[ToolCard]

    def as_user_prompt(self) -> str:
        return (
            "Query:\n"
            f"{self.query}\n\n"
            "Toolset (JSON):\n"
            f"{json.dumps([t.model_dump() for t in self.toolset], ensure_ascii=False, indent=2)}"
        )


def run_execution_agent(
    request: ExecutionRequest,
    *,
    tool_index_path: str | Path,
    model: str = "openai/gpt-4o-mini",
    use_openrouter: bool = False,
    openrouter_base_url: str = "https://openrouter.ai/api/v1",
    openrouter_api_key: str | None = None,
    stringify_tool_input: bool = False,
    log_calls: bool = True,
) -> str:
    """Convenience entrypoint for quick experiments."""
    LOGGER.info(
        "Run execution agent: query=%s tool_count=%d model=%s use_openrouter=%s",
        request.query,
        len(request.toolset),
        model,
        use_openrouter,
    )
    LOGGER.debug("Tool names: %s", [tool.name for tool in request.toolset])
    runtime_agent = Agent(
        build_model(
            model=model,
            use_openrouter=use_openrouter,

            openrouter_api_key=openrouter_api_key,
        ),
        deps_type=ExecutionDeps,
        system_prompt=SYSTEM_PROMPT,
        tools=[run_tool],
    )
    deps = ExecutionDeps(
        executor=ToolExecutor(
            tool_index_path=tool_index_path,
            stringify_tool_input=stringify_tool_input,
            log_calls=log_calls,
        )
    )
    result = runtime_agent.run_sync(request.as_user_prompt(), deps=deps)
    return result.output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pydantic-ai execution agent")
    parser.add_argument("--query", required=True, help="User task")
    parser.add_argument("--toolset", required=True, help="Path to toolset JSON list")
    parser.add_argument(
        "--tool-index",
        required=True,
        help="Path to JSON with mapping: name -> {category, tool_name, api_name}",
    )
    parser.add_argument("--model", default="openai:gpt-4o-mini")
    parser.add_argument(
        "--use-openrouter",
        action="store_true",
        help="Use OpenRouter via OPENROUTER_API_KEY and --model as model id",
    )
    parser.add_argument(
        "--openrouter-base-url",
        default="https://openrouter.ai/api/v1",
        help="OpenRouter API base URL",
    )
    parser.add_argument(
        "--stringify-tool-input",
        action="store_true",
        help="Send tool_input as JSON string instead of object",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    toolset_data = json.loads(Path(args.toolset).read_text(encoding="utf-8"))
    request = ExecutionRequest(
        query=args.query,
        toolset=[ToolCard.model_validate(t) for t in toolset_data],
    )
    output = run_execution_agent(
        request,
        tool_index_path=args.tool_index,
        model=args.model,
        use_openrouter=args.use_openrouter,
        openrouter_base_url=args.openrouter_base_url,
        stringify_tool_input=args.stringify_tool_input,
        log_calls=True,
    )
    print(output)
