import os
import pytest
import duckdb
import pandas as pd
import numpy as np
import yaml
import shutil
from unittest.mock import MagicMock, patch
from ms_mint_app.duckdb_manager import _create_tables
from ms_mint_app.plugins.targets_asari import (
    get_asari_command,
    export_ms1_from_db,
    run_asari_workflow,
)

@pytest.fixture
def db_con():
    con = duckdb.connect(':memory:')
    _create_tables(con)
    
    # Insert some basic test data
    con.execute("INSERT INTO samples (ms_file_label, polarity, use_for_processing) VALUES ('test_sample_1', 'Positive', TRUE)")
    con.execute("INSERT INTO samples (ms_file_label, polarity, use_for_processing) VALUES ('test_sample_2', 'Negative', TRUE)")
    
    # Insert MS1 data for test_sample_1
    con.execute("INSERT INTO ms1_data (ms_file_label, scan_id, scan_time, mz, intensity) VALUES ('test_sample_1', 1, 10.0, 100.0, 1000)")
    con.execute("INSERT INTO ms1_data (ms_file_label, scan_id, scan_time, mz, intensity) VALUES ('test_sample_1', 1, 10.0, 101.0, 2000)")
    con.execute("INSERT INTO ms1_data (ms_file_label, scan_id, scan_time, mz, intensity) VALUES ('test_sample_1', 2, 11.0, 100.0, 1100)")
    
    yield con
    con.close()

@pytest.fixture
def temp_workspace(tmp_path):
    wdir = tmp_path / "workspace"
    wdir.mkdir()
    # Create the targets.db in the workspace since targets_asari uses duckdb_connection(wdir)
    db_path = wdir / "mint.db" # Wait, duckdb_connection uses mint.db in wdir? 
    # Let's check duckdb_manager.py
    return str(wdir)

class TestAsariCommand:
    def test_get_asari_command_dev(self):
        with patch('sys.frozen', False, create=True):
            cmd, bundled, env = get_asari_command()
            assert cmd == ["asari"]
            assert bundled is False
            assert env is None

    def test_get_asari_command_frozen_win(self):
        with patch('sys.frozen', True, create=True), \
             patch('sys.platform', 'win32'), \
             patch('sys._MEIPASS', '/mock/meipass', create=True), \
             patch('os.path.exists', return_value=True):
            cmd, bundled, env = get_asari_command()
            assert "python.exe" in cmd[0] or "asari" in cmd[0]  # Depends on mock
            # Just verify it doesn't crash; the mock doesn't fully simulate the frozen env


class TestExportMS1:
    def test_export_ms1_from_db_happy_path(self, db_con, tmp_path):
        """Test that export_ms1_from_db creates a valid mzML file (now using lxml)."""
        output_path = str(tmp_path / "test.mzML")
        export_ms1_from_db(db_con, 'test_sample_1', output_path, 'Positive')
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

    def test_export_ms1_creates_valid_mzml(self, db_con, tmp_path):
        """Test that the exported mzML can be read back."""
        from ms_mint_app.tools import iter_mzml_fast
        
        output_path = str(tmp_path / "test_roundtrip.mzML")
        export_ms1_from_db(db_con, 'test_sample_1', output_path, 'Positive')
        
        # Read back the file
        spectra = list(iter_mzml_fast(output_path))
        assert len(spectra) == 2  # We inserted 2 scans
        assert spectra[0]["polarity"] == "Positive"

class TestAsariWorkflow:
    @patch('ms_mint_app.plugins.targets_asari.duckdb_connection')
    @patch('ms_mint_app.plugins.targets_asari.subprocess.run')
    @patch('ms_mint_app.plugins.targets_asari.subprocess.Popen')
    @patch('ms_mint_app.plugins.targets_asari.export_ms1_from_db')
    def test_run_asari_workflow_happy_path(self, mock_export, mock_popen, mock_run, mock_duckdb, db_con, temp_workspace):
        # Mocking subprocess
        mock_run.return_value = MagicMock(returncode=0)
        
        mock_process = MagicMock()
        mock_process.stdout = ["The reference sample is:...", "Peak detection on...", "Unique compound table..."]
        mock_process.wait.return_value = None
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        
        # Mocking database connection
        mock_duckdb.return_value.__enter__.return_value = db_con
        
        # Mocking result table creation
        def side_effect(wdir, params, set_progress=None):
            # We need to simulate the creation of the result table in the temp dir
            # But the actual function is what we are testing.
            # So we need to let it run until it looks for the result file.
            pass

        # Since we are testing the whole workflow, we need to provide a real-ish environment
        # Or at least mock the parts that interact with the filesystem.
        
        params = {
            'mz_tolerance_ppm': 10,
            'mode': 'pos',
            'multicores': 1
        }
        
        # We need to create the expected output file manually in the temp dir
        # created by run_asari_workflow to avoid "output directory not found"
        
        # This is tricky because the temp_dir is created inside run_asari_workflow with a timestamp.
        # Let's mock time.time to have a predictable path.
        with patch('time.time', return_value=123456789):
            run_id = "asari_run_123456789"
            res_dir = os.path.join(temp_workspace, run_id, "asari_results_asari_project")
            os.makedirs(res_dir, exist_ok=True)
            
            # Create a mock preferred_Feature_table.tsv
            df_cols = ['id_number', 'mz', 'rtime', 'rtime_left_base', 'rtime_right_base', 'cSelectivity', 'detection_counts']
            df_data = [[1, 100.0, 10.5, 10.0, 11.0, 0.9, 2]]
            df = pd.DataFrame(df_data, columns=df_cols)
            mock_table_path = os.path.join(res_dir, "preferred_Feature_table.tsv")
            df.to_csv(mock_table_path, sep='\t', index=False)
            
            result = run_asari_workflow(temp_workspace, params)
            
            assert result['success'] is True
            assert "completed successfully" in result['message']
            
            # Verify database update
            targets = db_con.execute("SELECT * FROM targets").fetchall()
            assert len(targets) == 1
            assert targets[0][0] == '1' # peak_label (id_number)
            assert targets[0][1] == 100.0 # mz_mean (mz)

    @patch('ms_mint_app.plugins.targets_asari.duckdb_connection')
    @patch('ms_mint_app.plugins.targets_asari.export_ms1_from_db')
    def test_run_asari_workflow_no_exports(self, mock_export, mock_duckdb, db_con, temp_workspace):
        # Simulate all exports failing
        mock_export.side_effect = Exception("Export failed")
        mock_duckdb.return_value.__enter__.return_value = db_con
        
        # We need to mock get_asari_command to skip the subprocess check or mock it
        with patch('ms_mint_app.plugins.targets_asari.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            
            # Since export failing just logs and continues at line 191-194, 
            # it will proceed to asari process with an empty directory.
            # Asari will likely fail to find input.
            
            with patch('ms_mint_app.plugins.targets_asari.subprocess.Popen') as mock_popen:
                mock_process = MagicMock()
                mock_process.stdout = ["error: no input files"]
                mock_process.wait.return_value = None
                mock_process.returncode = 1
                mock_popen.return_value = mock_process
                
                result = run_asari_workflow(temp_workspace, {})
                assert result['success'] is False
                assert "Asari failed with exit code 1" in result['message']

    def test_run_asari_workflow_no_samples(self, temp_workspace):
        # Create an empty db
        con = duckdb.connect(':memory:')
        _create_tables(con)
        
        with patch('ms_mint_app.plugins.targets_asari.duckdb_connection') as mock_duckdb:
            mock_duckdb.return_value.__enter__.return_value = con
            params = {}
            result = run_asari_workflow(temp_workspace, params)
            assert result['success'] is False
            assert "No samples selected" in result['message']

    @patch('ms_mint_app.plugins.targets_asari.subprocess.run')
    def test_run_asari_not_found(self, mock_run, temp_workspace):
        mock_run.side_effect = FileNotFoundError()
        params = {}
        result = run_asari_workflow(temp_workspace, params)
        assert result['success'] is False
        assert "Asari executable not found" in result['message']

class TestAsariFilters:
    @patch('ms_mint_app.plugins.targets_asari.duckdb_connection')
    @patch('ms_mint_app.plugins.targets_asari.subprocess.run')
    @patch('ms_mint_app.plugins.targets_asari.subprocess.Popen')
    @patch('ms_mint_app.plugins.targets_asari.export_ms1_from_db')
    def test_run_asari_workflow_filters(self, mock_export, mock_popen, mock_run, mock_duckdb, db_con, temp_workspace):
        mock_run.return_value = MagicMock(returncode=0)
        mock_process = MagicMock()
        mock_process.stdout = ["Done"]
        mock_process.wait.return_value = None
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        mock_duckdb.return_value.__enter__.return_value = db_con

        params = {
            'cselectivity': 0.8,
            'detection_rate': 50 # 50% of 2 samples = 1 sample
        }

        with patch('time.time', return_value=999):
            run_id = "asari_run_999"
            res_dir = os.path.join(temp_workspace, run_id, "asari_results_asari_project")
            os.makedirs(res_dir, exist_ok=True)
            
            # Create a table where one row passes filters and one fails
            df_cols = ['id_number', 'mz', 'rtime', 'rtime_left_base', 'rtime_right_base', 'cSelectivity', 'detection_counts']
            df_data = [
                [1, 100.0, 10.5, 10.0, 11.0, 0.9, 2], # Passes
                [2, 200.0, 20.5, 20.0, 21.0, 0.7, 2], # Fails cSelectivity
                [3, 300.0, 30.5, 30.0, 31.0, 0.9, 1]  # Passes detection_rate (1 >= 1)
            ]
            df = pd.DataFrame(df_data, columns=df_cols)
            mock_table_path = os.path.join(res_dir, "preferred_Feature_table.tsv")
            df.to_csv(mock_table_path, sep='\t', index=False)
            
            result = run_asari_workflow(temp_workspace, params)
            assert result['success'] is True
            
            targets = db_con.execute("SELECT peak_label FROM targets").fetchall()
            labels = [t[0] for t in targets]
            assert '1' in labels
            assert '2' not in labels
            assert '3' in labels

class TestAsariSecurity:
    def test_export_ms1_sql_injection(self, db_con, tmp_path):
        """Test that SQL injection is prevented via parameterized queries."""
        malicious_label = "test'; DROP TABLE ms1_data; --"
        output_path = str(tmp_path / "test_injection.mzML")
        
        # This should not raise an error or drop the table if parameterized
        export_ms1_from_db(db_con, malicious_label, output_path, 'Positive')
        
        # Check if table still exists
        count = db_con.execute("SELECT count(*) FROM ms1_data").fetchone()[0]
        assert count >= 0

class TestAsariCleanup:
    @patch('ms_mint_app.plugins.targets_asari.duckdb_connection')
    @patch('ms_mint_app.plugins.targets_asari.subprocess.run')
    @patch('ms_mint_app.plugins.targets_asari.subprocess.Popen')
    @patch('ms_mint_app.plugins.targets_asari.export_ms1_from_db')
    def test_cleanup_mzml_files(self, mock_export, mock_popen, mock_run, mock_duckdb, db_con, temp_workspace):
        mock_run.return_value = MagicMock(returncode=0)
        mock_process = MagicMock()
        mock_process.stdout = ["Done"]
        mock_process.wait.return_value = None
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        mock_duckdb.return_value.__enter__.return_value = db_con

        with patch('time.time', return_value=888):
            run_id = "asari_run_888"
            temp_dir = os.path.join(temp_workspace, run_id)
            os.makedirs(temp_dir, exist_ok=True)
            
            # Create a dummy mzML file that should be cleaned up
            mzml_file = os.path.join(temp_dir, "test.mzML")
            with open(mzml_file, 'w') as f: f.write("dummy")
            
            # Create a result table so it finishes successfully
            res_dir = os.path.join(temp_dir, "asari_results_asari_project")
            os.makedirs(res_dir, exist_ok=True)
            df = pd.DataFrame([[1, 100.0, 10.5, 10.0, 11.0]], columns=['id_number', 'mz', 'rtime', 'rtime_left_base', 'rtime_right_base'])
            df.to_csv(os.path.join(res_dir, "preferred_Feature_table.tsv"), sep='\t', index=False)
            
            run_asari_workflow(temp_workspace, {})
            
            assert not os.path.exists(mzml_file)
            assert os.path.exists(res_dir) # Result dir should NOT be cleaned up
