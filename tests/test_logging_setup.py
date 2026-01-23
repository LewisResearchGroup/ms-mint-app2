import logging

from ms_mint_app.logging_setup import (
    init_global_logging,
    activate_workspace_logging,
    deactivate_workspace_logging,
)


def _count_handlers(handler_type):
    root = logging.getLogger()
    return sum(isinstance(h, handler_type) for h in root.handlers)


def test_init_global_logging_idempotent():
    before = _count_handlers(logging.StreamHandler)
    init_global_logging()
    init_global_logging()
    after = _count_handlers(logging.StreamHandler)

    assert after <= before + 1


def test_workspace_logging_activate_and_deactivate(tmp_path):
    root = logging.getLogger()
    before_files = _count_handlers(logging.FileHandler)

    log_path = activate_workspace_logging(tmp_path)
    assert log_path.exists()

    mid_files = _count_handlers(logging.FileHandler)
    assert mid_files == before_files + 1

    # Re-activate on same path should not add another handler
    log_path_2 = activate_workspace_logging(tmp_path)
    assert log_path_2 == log_path
    assert _count_handlers(logging.FileHandler) == mid_files

    deactivate_workspace_logging()
    assert _count_handlers(logging.FileHandler) == before_files
