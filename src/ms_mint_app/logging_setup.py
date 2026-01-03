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
    - root logger set to DEBUG (to allow file handlers to capture everything)
    - stream handler (console) filters to 'level'
    """
    root = logging.getLogger()
    # Root must be DEBUG so that file handlers can capture all levels
    root.setLevel(logging.DEBUG)

    # Avoid duplicating handlers if called more than once
    has_stream = any(isinstance(h, logging.StreamHandler) for h in root.handlers)
    if not has_stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)  # Console output filtered by --debug flag
        formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)


def activate_workspace_logging(workspace_dir: str | Path,
                               filename: str = "ws.log",
                               level: int | None = None,
                               workspace_name: str | None = None) -> Path:
    """
    Enable file logging for a workspace:
    - Create the workspace folder if it does not exist
    - Create a FileHandler pointing to workspace_dir/filename
    - Remove the previous workspace handler (if any)
    - Return the log file path
    """
    global _WORKSPACE_HANDLER

    root = logging.getLogger()
    workspace_dir = Path(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    log_path = workspace_dir / filename

    # Check if we are already logging to this file
    if _WORKSPACE_HANDLER is not None:
        try:
             # baseFilename is usually absolute string path
            if Path(_WORKSPACE_HANDLER.baseFilename).resolve() == log_path.resolve():
                return log_path
        except Exception:
             pass

        root.removeHandler(_WORKSPACE_HANDLER)
        _WORKSPACE_HANDLER.close()
        _WORKSPACE_HANDLER = None

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'-'*80}\n")
    except Exception:
        pass

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    file_handler.setFormatter(formatter)

    # Always log DEBUG level to the workspace file for full diagnostics
    # (Console output level is controlled separately by --debug flag)
    file_handler.setLevel(logging.DEBUG)

    root.addHandler(file_handler)
    _WORKSPACE_HANDLER = file_handler

    msg = f"Workspace logging activated: {log_path}"
    if workspace_name:
        msg += f" (Workspace: {workspace_name})"
    root.info(msg)
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
