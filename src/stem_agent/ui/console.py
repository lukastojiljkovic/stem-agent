"""Single shared Rich console with project-specific log levels and theme.

Custom log levels:
    INFO=20, RETRIEVE=22, TOOL=25, EVAL=27, MUTATION=28, DECISION=29, REPORT=30
"""
from __future__ import annotations

import logging
import os
import sys

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme


RETRIEVE = 22
TOOL = 25
EVAL = 27
MUTATION = 28
DECISION = 29
REPORT = 30

logging.addLevelName(RETRIEVE, "RETRIEVE")
logging.addLevelName(TOOL, "TOOL")
logging.addLevelName(EVAL, "EVAL")
logging.addLevelName(MUTATION, "MUTATION")
logging.addLevelName(DECISION, "DECISION")
logging.addLevelName(REPORT, "REPORT")

_THEME = Theme({
    "logging.level.info": "blue",
    "logging.level.retrieve": "magenta",
    "logging.level.tool": "cyan",
    "logging.level.eval": "yellow",
    "logging.level.mutation": "green",
    "logging.level.decision": "bold white",
    "logging.level.report": "gold1",
    "logging.level.warning": "yellow",
    "logging.level.error": "red",
    "pipeline.tool": "cyan",
    "pipeline.param": "dim",
    "pipeline.score.good": "green",
    "pipeline.score.mid": "yellow",
    "pipeline.score.bad": "red",
})


def _force_windows_vt() -> None:
    if os.name == "nt":
        try:
            import colorama
            colorama.just_fix_windows_console()
        except ImportError:
            pass
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass


_force_windows_vt()
console = Console(theme=_THEME, log_path=False, log_time=True)


def get_logger(name: str = "stem_agent") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = RichHandler(
        console=console,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        omit_repeated_times=False,
    )
    handler.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


def _log_method(level: int):
    def fn(msg: str, *args, **kwargs):
        get_logger().log(level, msg, *args, **kwargs)
    return fn


log_info = _log_method(logging.INFO)
log_retrieve = _log_method(RETRIEVE)
log_tool = _log_method(TOOL)
log_eval = _log_method(EVAL)
log_mutation = _log_method(MUTATION)
log_decision = _log_method(DECISION)
log_report = _log_method(REPORT)
log_warn = _log_method(logging.WARNING)
log_error = _log_method(logging.ERROR)
