import sqlite3
import sys
from unittest.mock import MagicMock

import pytest
from flask import Flask

# Dash v2 no longer ships dash_core_components/dash_html_components as separate packages.
# Provide lightweight stand-ins so auth.py can import.
sys.modules.setdefault("dash_core_components", MagicMock())
sys.modules.setdefault("dash_html_components", MagicMock())

from ms_mint_app.database import ConnectDB
import ms_mint_app.auth as auth


class _DummyApp:
    def __init__(self):
        self.server = Flask(__name__)
        self.engine = None
        self.registered = {}

    def callback(self, *args, **kwargs):
        def decorator(func):
            self.registered[func.__name__] = func
            return func
        return decorator


def test_connectdb_creates_users_table(tmp_path):
    app = _DummyApp()
    ConnectDB(str(tmp_path), app)

    db_path = tmp_path / "data.sqlite"
    assert db_path.exists()

    con = sqlite3.connect(db_path)
    tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    con.close()

    assert "users" in tables


def test_auth_update_output_invalid_credentials(monkeypatch):
    app = _DummyApp()
    class DummyUser:
        password = "hashed"

    class DummyQuery:
        def filter_by(self, **kwargs):
            return self

        def first(self):
            return DummyUser()

    class DummyUsers:
        query = DummyQuery()

    monkeypatch.setattr(auth, "Users", DummyUsers)
    monkeypatch.setattr(auth, "check_password_hash", lambda pw, inp: False)

    auth.callbacks(app)

    update_output = app.registered["update_output"]
    message = update_output(1, "user", "bad")

    assert "Incorrect username or password" in message


def test_auth_successful_login(monkeypatch):
    app = _DummyApp()
    class DummyUser:
        password = "hashed"

    class DummyQuery:
        def filter_by(self, **kwargs):
            return self

        def first(self):
            return DummyUser()

    class DummyUsers:
        query = DummyQuery()

    monkeypatch.setattr(auth, "Users", DummyUsers)
    monkeypatch.setattr(auth, "check_password_hash", lambda pw, inp: True)
    monkeypatch.setattr(auth, "login_user", lambda user: None)

    auth.callbacks(app)

    successful = app.registered["successful"]
    path = successful(1, "user", "pass")

    assert path == "/success"
