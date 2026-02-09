from typing import Any, Optional, Union

from pydantic import BaseModel, Field


AnyToolProperty = Union["SimpleToolProperty", "NestedObjectProperty"]


class BaseToolProperty(BaseModel):
    description: Optional[str] = None


class SimpleToolProperty(BaseToolProperty):
    type: str


class NestedObjectProperty(BaseToolProperty):
    type: str = "object"
    properties: dict[str, AnyToolProperty] = Field(default_factory=dict)
    required: Optional[list[str]] = None


class ToolIOSchema(BaseModel):
    type: str = "object"
    properties: dict[str, AnyToolProperty] = Field(default_factory=dict)
    required: Optional[list[str]] = None


class ToolSchema(BaseModel):
    """
    Unified tool schema used both for indexing (vector DB) and prompt-facing 'cards'.
    - description_expanded and synthetic_questions are optional; toggled by config.USE_AUGMENTED_DOCS
    """

    name: str
    description: str
    arguments: Optional[ToolIOSchema] = None
    results: Optional[ToolIOSchema] = None
    description_expanded: Optional[str] = None
    synthetic_questions: Optional[list[str]] = None


class BenchmarkReference(BaseModel):
    tool: str
    param: dict[str, Any] = Field(default_factory=dict)
    input_source: Optional[str] = None


class BenchmarkItem(BaseModel):
    id: Optional[str] = None
    question: str
    reference: list[BenchmarkReference]
    task_nodes: Optional[list[dict]] = None
    task_links: Optional[list[dict]] = None
    n_tools: Optional[int] = None
    type: Optional[str] = None


class ReactToolCall(BaseModel):
    tool: str
    param: dict[str, Any] = Field(default_factory=dict)
    reasoning: str


class ReactIteration(BaseModel):
    iteration: int
    formulated_task: str
    selected_tools: list[ReactToolCall]
    task_completed: bool
    completion_reasoning: str
    mapped_tools_count: int
    agent_thought: Optional[str] = None
    agent_reasoning: Optional[str] = None


class ReactSubtaskResult(BaseModel):
    subtask: str
    iterations: list[ReactIteration]
    final_tools: list[str]
    completed: bool
    total_iterations: int


class ReactBenchmarkResult(BaseModel):
    idx: int
    question: str
    subtasks: list[ReactSubtaskResult]
    reference: list[str]
    selected: list[str]
    precision: float
    recall: float
    f1: float
