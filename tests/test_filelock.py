import os
import pytest

from ms_mint_app.filelock import SoftFileLock, Timeout


def test_soft_file_lock_acquire_release(tmp_path):
    lock_path = tmp_path / "test.lock"
    lock = SoftFileLock(str(lock_path))

    with lock.acquire():
        assert lock.is_locked is True
        assert lock_path.exists()

    assert lock.is_locked is False
    assert not lock_path.exists()


def test_soft_file_lock_timeout(tmp_path):
    lock_path = tmp_path / "test.lock"
    lock1 = SoftFileLock(str(lock_path))
    lock2 = SoftFileLock(str(lock_path))

    with lock1.acquire():
        with pytest.raises(Timeout):
            lock2.acquire(timeout=0)
