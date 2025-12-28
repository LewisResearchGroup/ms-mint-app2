import os
import json
import shutil
import pytest
import duckdb
from pathlib import Path
from unittest.mock import MagicMock, patch
from ms_mint_app.duckdb_manager import _create_workspace_tables, _create_tables
from ms_mint_app.plugins.workspaces import (
    _create_ws_input_validation,
    _create_workspace,
    _delete_workspace,
    _ws_activate,
    _save_ws_table_on_edit,
    _save_new_data_dir
)

@pytest.fixture
def temp_data_dir(tmp_path):
    data_dir = tmp_path / "MINT_DATA"
    data_dir.mkdir()
    # Create the global DB
    user_dir = data_dir / "Local"
    user_dir.mkdir(parents=True)
    con = duckdb.connect(os.path.join(str(user_dir), 'mint.db'))
    _create_workspace_tables(con)
    con.close()
    return str(user_dir)

@pytest.fixture
def db_con_mint(temp_data_dir):
    con = duckdb.connect(os.path.join(temp_data_dir, 'mint.db'))
    yield con
    con.close()

class TestWorkspaceValidation:
    def test_valid_name(self, temp_data_dir):
        status, help_text, okButtonProps = _create_ws_input_validation("valid_name", temp_data_dir)
        assert status == 'success'
        assert help_text is None
        assert okButtonProps['disabled'] is False

    def test_invalid_name_space(self, temp_data_dir):
        status, help_text, okButtonProps = _create_ws_input_validation("invalid name", temp_data_dir)
        assert status == 'error'
        assert "can only contain" in help_text
        assert okButtonProps['disabled'] is True

    def test_duplicate_name(self, temp_data_dir, db_con_mint):
        db_con_mint.execute("INSERT INTO workspaces (name) VALUES ('existing')")
        status, help_text, okButtonProps = _create_ws_input_validation("existing", temp_data_dir)
        assert status == 'error'
        assert "already exists" in help_text
        assert okButtonProps['disabled'] is True

    def test_missing_tmpdir(self):
        status, help_text, okButtonProps = _create_ws_input_validation("any", None)
        assert status == 'error'
        assert "path not available" in help_text
        assert okButtonProps['disabled'] is True

class TestWorkspaceLifecycle:
    def test_create_workspace_success(self, temp_data_dir, db_con_mint):
        # Initial check
        count = db_con_mint.execute("SELECT count(*) FROM workspaces").fetchone()[0]
        assert count == 0
        
        _create_workspace(1, temp_data_dir, "new_ws", "description")
        
        # Verify DB
        row = db_con_mint.execute("SELECT name, description, active FROM workspaces").fetchone()
        assert row[0] == "new_ws"
        assert row[1] == "description"
        assert row[2] is True
        
        # Verify filesystem
        ws_key = db_con_mint.execute("SELECT key FROM workspaces").fetchone()[0]
        ws_path = Path(temp_data_dir, 'workspaces', str(ws_key))
        assert ws_path.exists()
        assert ws_path.is_dir()

    def test_ws_activate(self, temp_data_dir, db_con_mint):
        # Create two workspaces
        _create_workspace(1, temp_data_dir, "ws1", "desc1")
        key1 = db_con_mint.execute("SELECT key FROM workspaces WHERE name='ws1'").fetchone()[0]
        
        _create_workspace(1, temp_data_dir, "ws2", "desc2")
        key2 = db_con_mint.execute("SELECT key FROM workspaces WHERE name='ws2'").fetchone()[0]
        
        # Initially ws2 is active
        assert db_con_mint.execute("SELECT active FROM workspaces WHERE key=?", (key2,)).fetchone()[0] is True
        
        # Activate ws1
        with patch('ms_mint_app.plugins.workspaces.activate_workspace_logging'):
            _ws_activate([str(key1)], temp_data_dir, None)
        
        assert db_con_mint.execute("SELECT active FROM workspaces WHERE key=?", (key1,)).fetchone()[0] is True
        assert db_con_mint.execute("SELECT active FROM workspaces WHERE key=?", (key2,)).fetchone()[0] is False

    def test_delete_workspace_success(self, temp_data_dir, db_con_mint):
        _create_workspace(1, temp_data_dir, "ws_to_delete", "desc")
        key = db_con_mint.execute("SELECT key FROM workspaces").fetchone()[0]
        ws_path = Path(temp_data_dir, 'workspaces', str(key))
        assert ws_path.exists()
        
        with patch('ms_mint_app.plugins.workspaces.deactivate_workspace_logging'):
            _delete_workspace(1, temp_data_dir, [str(key)])
            
        assert not ws_path.exists()
        assert db_con_mint.execute("SELECT count(*) FROM workspaces").fetchone()[0] == 0

class TestWorkspaceEdit:
    def test_edit_description_success(self, temp_data_dir, db_con_mint):
        _create_workspace(1, temp_data_dir, "ws1", "old desc")
        key = db_con_mint.execute("SELECT key FROM workspaces").fetchone()[0]
        
        row_edited = {'key': str(key), 'description': 'new desc'}
        _save_ws_table_on_edit(row_edited, 'description', temp_data_dir)
        
        desc = db_con_mint.execute("SELECT description FROM workspaces WHERE key=?", (key,)).fetchone()[0]
        assert desc == 'new desc'

    def test_edit_name_not_allowed(self, temp_data_dir):
        row_edited = {'key': '1', 'name': 'new name'}
        notification, action_store = _save_ws_table_on_edit(row_edited, 'name', temp_data_dir)
        assert "not allowed" in notification.message

class TestDataDirManagement:
    def test_save_new_data_dir_valid(self, tmp_path):
        new_dir = tmp_path / "NEW_MINT"
        config_path = tmp_path / ".mint_config.json"
        
        with patch.dict(os.environ, {"MINT_CONFIG_PATH": str(config_path)}):
            res_tmpdir, notification, visible = _save_new_data_dir(1, str(new_dir))
            
            assert res_tmpdir == str(new_dir / "Local")
            assert "updated successfully" in notification.message
            assert visible is False
            
            # Verify config file
            with open(config_path, "r") as f:
                cfg = json.load(f)
                assert cfg["data_dir"] == str(new_dir)

    def test_save_new_data_dir_relative(self, tmp_path):
        notification = _save_new_data_dir(1, "relative/path")[1]
        assert "must be absolute" in notification.message

    def test_save_new_data_dir_empty(self):
        notification = _save_new_data_dir(1, "")[1]
        assert "valid path" in notification.message
