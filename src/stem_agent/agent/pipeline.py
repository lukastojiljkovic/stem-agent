"""Pipeline = ordered list of (tool_name, params) executed against a Tool registry.

Supports:
  - validate(pipeline, registry, input_type, max_steps=5) -> (ok, message)
  - execute(pipeline, registry, initial_input) -> ExecutionResult
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..tools.base import Tool
from ..types import TypeName, is_compatible
from ..ui.console import log_tool, log_warn


class PipelineExecutionError(RuntimeError): ...


@dataclass
class PipelineStep:
    tool_name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Pipeline:
    steps: list[PipelineStep] = field(default_factory=list)

    def __len__(self) -> int: return len(self.steps)
    def to_dict(self) -> dict[str, Any]:
        return {"steps": [{"tool": s.tool_name, "params": s.params} for s in self.steps]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Pipeline":
        return cls(steps=[PipelineStep(s["tool"], dict(s.get("params") or {})) for s in d.get("steps", [])])


def validate(pipeline: Pipeline, registry: dict[str, Tool],
             input_type: TypeName, max_steps: int = 5) -> tuple[bool, str]:
    if not pipeline.steps:
        return False, "empty pipeline"
    if len(pipeline.steps) > max_steps:
        return False, f"pipeline length {len(pipeline.steps)} exceeds max {max_steps}"
    cur = input_type
    for i, step in enumerate(pipeline.steps):
        t = registry.get(step.tool_name)
        if t is None:
            return False, f"unknown tool {step.tool_name!r} at step {i}"
        if not is_compatible(cur, t.input_type):
            return False, (f"step {i} ({t.name}) expects {t.input_type.value} but "
                           f"previous output is {cur.value} (incompatible)")
        cur = t.output_type
    return True, "ok"


@dataclass
class StepRecord:
    index: int
    tool_name: str
    params: dict[str, Any]
    output: Any
    elapsed_s: float
    error: str | None = None


@dataclass
class ExecutionResult:
    success: bool
    step_outputs: list[StepRecord] = field(default_factory=list)
    final: Any = None
    error: str | None = None
    failed_step_index: int | None = None
    total_elapsed_s: float = 0.0


def _run_tool(tool: Tool, *, prev_output: Any, params: dict[str, Any]) -> Any:
    """Single source of truth for how the previous step's output is fed to a tool.

    Heuristic: pass the previous output positionally as the FIRST keyword in the
    tool function's signature whose name is one of {text, query, doc, document,
    documents, time_series, filing, data, output, project}. If none matches,
    pass under the first parameter name.
    """
    import inspect
    fn = tool.function
    sig = inspect.signature(fn)
    pnames = [p for p in sig.parameters.keys()]
    PREFER = ["text","query","doc","document","documents","time_series","filing",
              "data","output","project","series_dict","series_id","ctx","payload","a"]
    arg_name: str | None = None
    for p in PREFER:
        if p in pnames:
            arg_name = p; break
    if arg_name is None and pnames:
        arg_name = pnames[0]
    kwargs = dict(params)
    if arg_name is not None and arg_name not in kwargs:
        kwargs[arg_name] = prev_output
    return tool.run(**kwargs)


def execute(pipeline: Pipeline, registry: dict[str, Tool],
            initial_input: Any) -> ExecutionResult:
    import time
    res = ExecutionResult(success=True)
    cur = initial_input
    t0 = time.time()
    for i, step in enumerate(pipeline.steps):
        t = registry.get(step.tool_name)
        if t is None:
            res.success = False; res.error = f"unknown tool {step.tool_name}"; res.failed_step_index = i
            res.total_elapsed_s = time.time() - t0; return res
        s_start = time.time()
        try:
            log_tool(f"[{i+1}/{len(pipeline)}] {step.tool_name}({_short_params(step.params)})")
            cur = _run_tool(t, prev_output=cur, params=step.params)
            elapsed = time.time() - s_start
            res.step_outputs.append(StepRecord(i, step.tool_name, step.params, cur, elapsed))
        except Exception as e:
            elapsed = time.time() - s_start
            log_warn(f"step {i} ({step.tool_name}) failed: {e}")
            res.step_outputs.append(StepRecord(i, step.tool_name, step.params, None, elapsed, error=str(e)))
            res.success = False; res.error = str(e); res.failed_step_index = i
            break
    res.final = cur if res.success else None
    res.total_elapsed_s = time.time() - t0
    return res


def _short_params(p: dict[str, Any]) -> str:
    if not p: return ""
    items = []
    for k, v in p.items():
        sv = str(v)
        if len(sv) > 30: sv = sv[:27] + "..."
        items.append(f"{k}={sv}")
    return ", ".join(items)
