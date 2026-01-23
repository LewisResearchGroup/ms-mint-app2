import os
import sys

import pytest

from ms_mint_app.instance import SingleInstance


def test_get_pid_file_default(tmp_path, monkeypatch):
    inst = SingleInstance(name="test_app", temp_dir=str(tmp_path), debug=True)
    assert str(inst.pid_file).endswith("test_app.pid")


def test_is_running_cleans_stale_pid(tmp_path, monkeypatch):
    inst = SingleInstance(name="test_app", temp_dir=str(tmp_path), debug=True)
    inst.pid_file.write_text("999999")

    monkeypatch.setattr("psutil.pid_exists", lambda pid: False)

    running, pid = inst._is_running()
    assert running is False
    assert pid is None
    assert not inst.pid_file.exists()


def test_ensure_single_exits_when_running(tmp_path, monkeypatch):
    inst = SingleInstance(name="test_app", temp_dir=str(tmp_path), debug=True)
    inst.debug = False

    monkeypatch.setattr(inst, "_is_running", lambda: (True, 1234))

    with pytest.raises(SystemExit):
        inst.ensure_single(force=False)


def test_ensure_single_creates_pid_file(tmp_path, monkeypatch):
    inst = SingleInstance(name="test_app", temp_dir=str(tmp_path), debug=True)
    inst.debug = False

    monkeypatch.setattr(inst, "_is_running", lambda: (False, None))

    inst.ensure_single(force=False)

    assert inst.pid_file.exists()
    assert inst.pid_file.read_text().strip() == str(os.getpid())
