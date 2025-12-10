# logging_setup.py
from __future__ import annotations

import logging
from pathlib import Path

# Global variable to track the current workspace handler
_WORKSPACE_HANDLER: logging.Handler | None = None

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def init_global_logging(level: int = logging.INFO) -> None:
    """
    Configure global logging:
    - root logger set to 'level'
    - always logs to the terminal (stdout)
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicating handlers if called more than once
    has_stream = any(isinstance(h, logging.StreamHandler) for h in root.handlers)
    if not has_stream:
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)


def activate_workspace_logging(workspace_dir: str | Path,
                               filename: str = "ws.log",
                               level: int | None = None) -> Path:
    """
    Enable file logging for a workspace:
    - Create the workspace folder if it does not exist
    - Create a FileHandler pointing to workspace_dir/filename
    - Remove the previous workspace handler (if any)
    - Return the log file path
    """
    global _WORKSPACE_HANDLER

    root = logging.getLogger()

    # Remove previous handler, if present
    if _WORKSPACE_HANDLER is not None:
        root.removeHandler(_WORKSPACE_HANDLER)
        _WORKSPACE_HANDLER.close()
        _WORKSPACE_HANDLER = None

    workspace_dir = Path(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    log_path = workspace_dir / filename

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    file_handler.setFormatter(formatter)

    if level is not None:
        file_handler.setLevel(level)

    root.addHandler(file_handler)
    _WORKSPACE_HANDLER = file_handler

    root.info("Workspace logging activated: %s", log_path)
    return log_path


def deactivate_workspace_logging() -> None:
    """
    Remove the file handler associated with the current workspace (if any).
    Terminal logging stays active.
    """
    global _WORKSPACE_HANDLER

    if _WORKSPACE_HANDLER is None:
        return

    root = logging.getLogger()
    root.removeHandler(_WORKSPACE_HANDLER)
    _WORKSPACE_HANDLER.close()
    _WORKSPACE_HANDLER = None
    root.info("Workspace logging deactivated")
