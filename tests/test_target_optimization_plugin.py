import pytest
import duckdb
import json
import dash
from pathlib import Path
from unittest.mock import MagicMock, patch
from dash.exceptions import PreventUpdate

from ms_mint_app.plugins.target_optimization import (
    _update_sample_type_tree, _delete_target_logic, 
    _bookmark_target_logic, _compute_chromatograms_logic,
    _get_cpu_help_text, _get_ram_help_text
)
from ms_mint_app.duckdb_manager import duckdb_connection

@pytest.fixture
def temp_wdir(tmp_path):
    wdir = tmp_path / "workspace"
    wdir.mkdir(parents=True)
    # Register workspace and create tables via duckdb_connection
    with duckdb_connection(wdir) as conn:
        # Create necessary tables and types if they don't exist (duckdb_connection calls _create_tables)
        # Setup initial data for testing
        conn.execute("INSERT INTO samples (ms_file_label, sample_type, label, use_for_optimization, ms_type) VALUES ('File1', 'TypeA', 'Label1', TRUE, 'ms1')")
        conn.execute("INSERT INTO samples (ms_file_label, sample_type, label, use_for_optimization, ms_type) VALUES ('File2', 'TypeB', 'Label2', TRUE, 'ms1')")
        conn.execute("INSERT INTO targets (peak_label, mz_mean, rt, ms_type, bookmark) VALUES ('Target1', 100.0, 60.0, 'ms1', FALSE)")
    return str(wdir)

class TestTargetOptimizationLogic:
    
    def test_update_sample_type_tree_basic(self, temp_wdir):
        # Test initial tree load
        res = _update_sample_type_tree({'page': 'Optimization'}, None, None, None, 'all', temp_wdir, 'section-context')
        tree_data, checked_keys, expanded_keys, tree_style, empty_style = res
        
        assert len(tree_data) == 2
        assert any(item['title'] == 'TypeA' for item in tree_data)
        assert any(item['title'] == 'TypeB' for item in tree_data)
        assert tree_style['display'] == 'flex'
        assert empty_style['display'] == 'none'

    def test_update_sample_type_tree_ms_filter(self, temp_wdir):
        # Update one sample to ms2
        with duckdb_connection(temp_wdir) as conn:
            conn.execute("UPDATE samples SET ms_type = 'ms2' WHERE ms_file_label = 'File2'")
            
        # Filter for ms1
        res = _update_sample_type_tree({'page': 'Optimization'}, None, None, None, 'ms1', temp_wdir, 'chromatogram-preview-filter-ms-type')
        tree_data, _, _, _, _ = res
        assert len(tree_data) == 1
        assert tree_data[0]['title'] == 'TypeA'

    def test_update_sample_type_tree_mark_all(self, temp_wdir):
        res = _update_sample_type_tree({'page': 'Optimization'}, 1, None, None, 'all', temp_wdir, 'mark-tree-action')
        _, checked_keys, _, _, _ = res
        assert 'Label1' in checked_keys
        assert 'Label2' in checked_keys

    def test_update_sample_type_tree_expand_collapse(self, temp_wdir):
        # Expand
        res = _update_sample_type_tree({'page': 'Optimization'}, None, 1, None, 'all', temp_wdir, 'expand-tree-action')
        _, _, expanded_keys, _, _ = res
        assert 'TypeA' in expanded_keys
        assert 'TypeB' in expanded_keys
        
        # Collapse
        res = _update_sample_type_tree({'page': 'Optimization'}, None, None, 1, 'all', temp_wdir, 'collapse-tree-action')
        _, _, expanded_keys, _, _ = res
        assert expanded_keys == []

    def test_delete_target_logic_success(self, temp_wdir):
        # Insert chromatogram and result for Target1
        with duckdb_connection(temp_wdir) as conn:
            conn.execute("INSERT INTO chromatograms (peak_label, ms_file_label) VALUES ('Target1', 'File1')")
            conn.execute("INSERT INTO results (peak_label, ms_file_label) VALUES ('Target1', 'File1')")
            
        res = _delete_target_logic('Target1', temp_wdir)
        notification, drop_chrom, modal_vis, view_vis = res
        
        assert notification.type == 'success'
        assert drop_chrom is True
        assert modal_vis is False
        assert view_vis is False
        
        with duckdb_connection(temp_wdir) as conn:
            assert conn.execute("SELECT COUNT(*) FROM targets WHERE peak_label = 'Target1'").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM chromatograms WHERE peak_label = 'Target1'").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM results WHERE peak_label = 'Target1'").fetchone()[0] == 0

    def test_delete_target_database_error(self, temp_wdir):
        # Test with invalid wdir to trigger connection error (or simulate it)
        res = _delete_target_logic('Target1', "/non/existent/path")
        notification, drop_chrom, modal_vis, view_vis = res
        assert notification.type == 'error'
        assert "Database connection failed" in notification.message

    def test_bookmark_target_logic(self, temp_wdir):
        bookmarks = [True] 
        targets = ['Target1']
        trigger_id = 0
        res = _bookmark_target_logic(bookmarks, targets, trigger_id, temp_wdir)
        
        assert res.type == 'success'
        assert "Target1" in res.message
        
        with duckdb_connection(temp_wdir) as conn:
            val = conn.execute("SELECT bookmark FROM targets WHERE peak_label = 'Target1'").fetchone()[0]
            assert val is True

    @patch('ms_mint_app.plugins.target_optimization.compute_chromatograms_in_batches')
    def test_compute_chromatograms_logic(self, mock_compute, temp_wdir):
        set_progress = MagicMock()
        # Mock successful computation
        res = _compute_chromatograms_logic(set_progress, False, False, 1, 1, 1000, temp_wdir)
        
        assert res == (True, False)
        assert mock_compute.called
        # Check if first progress call was made
        set_progress.assert_any_call((0, "Chromatograms", "Preparing batches..."))

    def test_compute_chromatograms_logic_rt_adjustment(self, temp_wdir):
        # Setup target with rt_auto_adjusted = TRUE
        with duckdb_connection(temp_wdir) as conn:
            conn.execute("UPDATE targets SET rt_auto_adjusted = TRUE, rt_min = 50, rt_max = 70 WHERE peak_label = 'Target1'")
            # Insert dummy chromatogram data
            # scan_time: [55, 60, 65], intensity: [100, 500, 200] -> Max at 60
            conn.execute("INSERT INTO chromatograms (peak_label, ms_file_label, scan_time, intensity) VALUES ('Target1', 'File1', [55, 60, 65], [100, 500, 200])")

        # Mock compute_chromatograms_in_batches to do nothing
        with patch('ms_mint_app.plugins.target_optimization.compute_chromatograms_in_batches'):
            _compute_chromatograms_logic(None, False, False, 1, 1, 1000, temp_wdir)
            
        with duckdb_connection(temp_wdir) as conn:
            target = conn.execute("SELECT rt, rt_auto_adjusted FROM targets WHERE peak_label = 'Target1'").fetchone()
            assert target[0] == 60.0
            assert target[1] is False # Should be reset to FALSE

    def test_prevent_update_on_wrong_page(self, temp_wdir):
        with pytest.raises(PreventUpdate):
            _update_sample_type_tree({'page': 'Other'}, None, None, None, 'all', temp_wdir, 'section-context')

    def test_empty_df_handling(self, temp_wdir):
        # Delete all samples
        with duckdb_connection(temp_wdir) as conn:
            conn.execute("DELETE FROM samples")
        
        res = _update_sample_type_tree({'page': 'Optimization'}, None, None, None, 'all', temp_wdir, 'section-context')
        tree_data, checked_keys, expanded_keys, tree_style, empty_style = res
        assert tree_data == []
        assert tree_style['display'] == 'none'
        assert empty_style['display'] == 'block'

    def test_get_cpu_help_text(self):
        with patch('ms_mint_app.plugins.target_optimization.cpu_count', return_value=8):
            text = _get_cpu_help_text(4)
            assert text == "Selected 4 / 8 cpus"

    def test_get_ram_help_text(self):
        # Mock psutil.virtual_memory
        mock_memory = MagicMock()
        mock_memory.available = 16 * (1024 ** 3) # 16 GB available
        
        with patch('psutil.virtual_memory', return_value=mock_memory):
            text = _get_ram_help_text(8.5)
            # ram_max calculation: round(16 * 1024**3 / 1024**3, 1) = 16.0
            assert text == "Selected 8.5GB / 16.0GB available RAM"

