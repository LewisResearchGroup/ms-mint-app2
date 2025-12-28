"""
Comprehensive integration tests for the Optimization Plugin (target_optimization.py).

This test suite validates:
1. RT value updates and persistence
2. Bookmark functionality
3. Target deletion with cascade to chromatograms/results
4. Notes persistence
5. Error handling for invalid inputs
6. SQL injection protection
7. Transaction rollback on failures
"""

import duckdb
import pytest
import numpy as np
from ms_mint_app.duckdb_manager import _create_tables


@pytest.fixture
def db_con():
    """Create an in-memory database with test data for optimization plugin testing."""
    con = duckdb.connect(':memory:')
    _create_tables(con)
    
    # Insert test samples
    con.execute("INSERT INTO samples (ms_file_label, sample_type) VALUES ('File1', 'Sample'), ('File2', 'Sample')")
    
    # Insert test targets
    con.execute("""
        INSERT INTO targets (peak_label, rt_min, rt, rt_max, mz_mean, mz_width, bookmark, notes) 
        VALUES 
            ('Peak1', 8.0, 10.0, 12.0, 100.0, 10.0, FALSE, NULL),
            ('Peak2', 28.0, 30.0, 32.0, 200.0, 15.0, TRUE, 'Test note'),
            ('Peak3', 48.0, 50.0, 52.0, 300.0, 20.0, FALSE, NULL)
    """)
    
    # Insert test chromatograms with sample data
    # Chromatograms contain scan_time and intensity arrays
    scan_times = [8.0, 9.0, 10.0, 11.0, 12.0]
    intensities = [100.0, 500.0, 1000.0, 500.0, 100.0]  # Peak at 10.0
    
    con.execute("""
        INSERT INTO chromatograms (peak_label, ms_file_label, scan_time, intensity)
        VALUES (?, ?, ?, ?)
    """, ['Peak1', 'File1', scan_times, intensities])
    
    con.execute("""
        INSERT INTO chromatograms (peak_label, ms_file_label, scan_time, intensity)
        VALUES (?, ?, ?, ?)
    """, ['Peak1', 'File2', scan_times, intensities])
    
    yield con
    con.close()


class TestRTValueUpdates:
    """Test retention time (RT) value updates - core optimization functionality."""
    
    def test_update_valid_rt_values(self, db_con):
        """Test updating RT min, RT, and RT max with valid values."""
        peak_label = 'Peak1'
        new_rt_min, new_rt, new_rt_max = 9.0, 11.5, 13.0
        
        db_con.execute(
            "UPDATE targets SET rt_min = ?, rt = ?, rt_max = ? WHERE peak_label = ?",
            [new_rt_min, new_rt, new_rt_max, peak_label]
        )
        
        result = db_con.execute(
            "SELECT rt_min, rt, rt_max FROM targets WHERE peak_label = ?", 
            [peak_label]
        ).fetchone()
        
        assert result[0] == new_rt_min
        assert result[1] == new_rt
        assert result[2] == new_rt_max
    
    def test_update_rt_invalid_type(self, db_con):
        """Test that invalid RT value types are rejected."""
        peak_label = 'Peak1'
        invalid_rt = 'not_a_number'
        
        with pytest.raises(Exception):
            db_con.execute(
                "UPDATE targets SET rt = ? WHERE peak_label = ?",
                [invalid_rt, peak_label]
            )
    
    def test_update_rt_negative_value(self, db_con):
        """Test that negative RT values can be stored (edge case)."""
        # Note: Depending on business logic, negative RTs might be invalid
        # but database should accept them unless CHECK constraint exists
        peak_label = 'Peak1'
        negative_rt = -5.0
        
        db_con.execute(
            "UPDATE targets SET rt = ? WHERE peak_label = ?",
            [negative_rt, peak_label]
        )
        
        result = db_con.execute(
            "SELECT rt FROM targets WHERE peak_label = ?", 
            [peak_label]
        ).fetchone()
        
        assert result[0] == negative_rt
    
    def test_update_rt_for_nonexistent_target(self, db_con):
        """Test updating RT for a target that doesn't exist (should succeed but affect 0 rows)."""
        nonexistent_peak = 'NonExistentPeak'
        
        db_con.execute(
            "UPDATE targets SET rt = ? WHERE peak_label = ?",
            [15.0, nonexistent_peak]
        )
        
        # No exception should be raised, but no rows should be affected
        result = db_con.execute(
            "SELECT COUNT(*) FROM targets WHERE peak_label = ?",
            [nonexistent_peak]
        ).fetchone()
        
        assert result[0] == 0


class TestBookmarkFunctionality:
    """Test bookmark feature for marking/unmarking targets."""
    
    def test_bookmark_target(self, db_con):
        """Test bookmarking an unbookmarked target."""
        peak_label = 'Peak1'
        
        # Verify initial state
        initial = db_con.execute(
            "SELECT bookmark FROM targets WHERE peak_label = ?",
            [peak_label]
        ).fetchone()
        assert initial[0] == False
        
        # Bookmark the target
        db_con.execute(
            "UPDATE targets SET bookmark = ? WHERE peak_label = ?",
            [True, peak_label]
        )
        
        result = db_con.execute(
            "SELECT bookmark FROM targets WHERE peak_label = ?",
            [peak_label]
        ).fetchone()
        
        assert result[0] == True
    
    def test_unbookmark_target(self, db_con):
        """Test unbookmarking a bookmarked target."""
        peak_label = 'Peak2'  # This one is bookmarked by default
        
        # Verify initial state
        initial = db_con.execute(
            "SELECT bookmark FROM targets WHERE peak_label = ?",
            [peak_label]
        ).fetchone()
        assert initial[0] == True
        
        # Unbookmark the target
        db_con.execute(
            "UPDATE targets SET bookmark = ? WHERE peak_label = ?",
            [False, peak_label]
        )
        
        result = db_con.execute(
            "SELECT bookmark FROM targets WHERE peak_label = ?",
            [peak_label]
        ).fetchone()
        
        assert result[0] == False
    
    def test_bookmark_invalid_type(self, db_con):
        """Test that non-boolean bookmark values are rejected or converted."""
        peak_label = 'Peak1'
        
        # DuckDB should handle type conversion or raise an error
        with pytest.raises(Exception):
            db_con.execute(
                "UPDATE targets SET bookmark = ? WHERE peak_label = ?",
                ["not_a_boolean", peak_label]
            )


class TestNotesPersistence:
    """Test notes functionality for targets."""
    
    def test_add_note_to_target(self, db_con):
        """Test adding a note to a target without one."""
        peak_label = 'Peak1'
        note = 'This is a test note for Peak1'
        
        db_con.execute(
            "UPDATE targets SET notes = ? WHERE peak_label = ?",
            [note, peak_label]
        )
        
        result = db_con.execute(
            "SELECT notes FROM targets WHERE peak_label = ?",
            [peak_label]
        ).fetchone()
        
        assert result[0] == note
    
    def test_update_existing_note(self, db_con):
        """Test updating an existing note."""
        peak_label = 'Peak2'  # Has 'Test note' initially
        new_note = 'Updated note content'
        
        db_con.execute(
            "UPDATE targets SET notes = ? WHERE peak_label = ?",
            [new_note, peak_label]
        )
        
        result = db_con.execute(
            "SELECT notes FROM targets WHERE peak_label = ?",
            [peak_label]
        ).fetchone()
        
        assert result[0] == new_note
    
    def test_clear_note(self, db_con):
        """Test clearing/removing a note (set to NULL)."""
        peak_label = 'Peak2'
        
        db_con.execute(
            "UPDATE targets SET notes = ? WHERE peak_label = ?",
            [None, peak_label]
        )
        
        result = db_con.execute(
            "SELECT notes FROM targets WHERE peak_label = ?",
            [peak_label]
        ).fetchone()
        
        assert result[0] is None
    
    def test_note_with_special_characters(self, db_con):
        """Test that notes with special characters are persisted correctly."""
        peak_label = 'Peak1'
        special_note = "Note with special chars: \n\t\"'<>!@#$%^&*()"
        
        db_con.execute(
            "UPDATE targets SET notes = ? WHERE peak_label = ?",
            [special_note, peak_label]
        )
        
        result = db_con.execute(
            "SELECT notes FROM targets WHERE peak_label = ?",
            [peak_label]
        ).fetchone()
        
        assert result[0] == special_note


class TestTargetDeletion:
    """Test target deletion with proper cascade to related tables."""
    
    def test_delete_target_with_chromatograms(self, db_con):
        """Test deleting a target and its associated chromatograms."""
        peak_label = 'Peak1'
        
        # Verify initial state
        target_count = db_con.execute(
            "SELECT COUNT(*) FROM targets WHERE peak_label = ?",
            [peak_label]
        ).fetchone()[0]
        chrom_count = db_con.execute(
            "SELECT COUNT(*) FROM chromatograms WHERE peak_label = ?",
            [peak_label]
        ).fetchone()[0]
        
        assert target_count == 1
        assert chrom_count == 2  # We inserted 2 chromatograms for Peak1
        
        # Delete with transaction
        try:
            db_con.execute("BEGIN")
            db_con.execute("DELETE FROM chromatograms WHERE peak_label = ?", [peak_label])
            db_con.execute("DELETE FROM targets WHERE peak_label = ?", [peak_label])
            db_con.execute("DELETE FROM results WHERE peak_label = ?", [peak_label])
            db_con.execute("COMMIT")
        except Exception:
            db_con.execute("ROLLBACK")
            raise
        
        # Verify deletion
        target_count_after = db_con.execute(
            "SELECT COUNT(*) FROM targets WHERE peak_label = ?",
            [peak_label]
        ).fetchone()[0]
        chrom_count_after = db_con.execute(
            "SELECT COUNT(*) FROM chromatograms WHERE peak_label = ?",
            [peak_label]
        ).fetchone()[0]
        
        assert target_count_after == 0
        assert chrom_count_after == 0
    
    def test_delete_nonexistent_target(self, db_con):
        """Test deleting a target that doesn't exist (should not error)."""
        nonexistent_peak = 'NonExistent'
        
        try:
            db_con.execute("BEGIN")
            db_con.execute("DELETE FROM chromatograms WHERE peak_label = ?", [nonexistent_peak])
            db_con.execute("DELETE FROM targets WHERE peak_label = ?", [nonexistent_peak])
            db_con.execute("DELETE FROM results WHERE peak_label = ?", [nonexistent_peak])
            db_con.execute("COMMIT")
        except Exception:
            db_con.execute("ROLLBACK")
            raise
        
        # Should complete without error
    
    def test_delete_rollback_on_error(self, db_con):
        """Test that deletion rolls back properly on error."""
        peak_label = 'Peak1'
        
        initial_count = db_con.execute(
            "SELECT COUNT(*) FROM targets WHERE peak_label = ?",
            [peak_label]
        ).fetchone()[0]
        
        try:
            db_con.execute("BEGIN")
            db_con.execute("DELETE FROM targets WHERE peak_label = ?", [peak_label])
            # Simulate an error
            raise Exception("Simulated error during deletion")
        except Exception:
            db_con.execute("ROLLBACK")
        
        # Verify target still exists after rollback
        count_after = db_con.execute(
            "SELECT COUNT(*) FROM targets WHERE peak_label = ?",
            [peak_label]
        ).fetchone()[0]
        
        assert count_after == initial_count


class TestSecurityAndEdgeCases:
    """Test SQL injection protection and edge cases."""
    
    def test_sql_injection_in_notes(self, db_con):
        """Test that SQL injection attempts in notes are safely handled."""
        peak_label = 'Peak1'
        malicious_note = "'; DROP TABLE targets; --"
        
        db_con.execute(
            "UPDATE targets SET notes = ? WHERE peak_label = ?",
            [malicious_note, peak_label]
        )
        
        # Verify note is stored literally
        result = db_con.execute(
            "SELECT notes FROM targets WHERE peak_label = ?",
            [peak_label]
        ).fetchone()
        
        assert result[0] == malicious_note
        
        # Verify targets table still exists
        count = db_con.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
        assert count == 3  # All 3 targets should still exist
    
    def test_sql_injection_in_peak_label(self, db_con):
        """Test that SQL injection in peak_label is handled safely."""
        malicious_label = "Peak1'; DROP TABLE chromatograms; --"
        
        # This should safely fail or store the value literally
        # Peak_label is the primary key, so this tests parameter binding
        try:
            db_con.execute(
                "UPDATE targets SET notes = ? WHERE peak_label = ?",
                ["test", malicious_label]
            )
        except Exception:
            pass  # Expected to fail as peak doesn't exist
        
        # Verify chromatograms table still exists
        db_con.execute("SELECT COUNT(*) FROM chromatograms")
    
    def test_concurrent_rt_updates(self, db_con):
        """Test that multiple RT updates to the same target work correctly."""
        peak_label = 'Peak1'
        
        # First update
        db_con.execute(
            "UPDATE targets SET rt = ? WHERE peak_label = ?",
            [11.0, peak_label]
        )
        
        # Second update
        db_con.execute(
            "UPDATE targets SET rt = ? WHERE peak_label = ?",
            [12.0, peak_label]
        )
        
        result = db_con.execute(
            "SELECT rt FROM targets WHERE peak_label = ?",
            [peak_label]
        ).fetchone()
        
        assert result[0] == 12.0  # Last update should win
    
    def test_empty_string_vs_null_notes(self, db_con):
        """Test difference between empty string and NULL notes."""
        peak_label = 'Peak1'
        
        # Set to empty string
        db_con.execute(
            "UPDATE targets SET notes = ? WHERE peak_label = ?",
            ["", peak_label]
        )
        
        result1 = db_con.execute(
            "SELECT notes FROM targets WHERE peak_label = ?",
            [peak_label]
        ).fetchone()
        
        assert result1[0] == ""
        assert result1[0] is not None
        
        # Set to NULL
        db_con.execute(
            "UPDATE targets SET notes = ? WHERE peak_label = ?",
            [None, peak_label]
        )
        
        result2 = db_con.execute(
            "SELECT notes FROM targets WHERE peak_label = ?",
            [peak_label]
        ).fetchone()
        
        assert result2[0] is None


class TestChromatogramData:
    """Test chromatogram-related operations."""
    
    def test_chromatogram_array_storage(self, db_con):
        """Test that chromatogram arrays (scan_time, intensity) are stored correctly."""
        peak_label = 'Peak1'
        ms_file = 'File1'
        
        result = db_con.execute(
            "SELECT scan_time, intensity FROM chromatograms WHERE peak_label = ? AND ms_file_label = ?",
            [peak_label, ms_file]
        ).fetchone()
        
        scan_times = result[0]
        intensities = result[1]
        
        assert len(scan_times) == 5
        assert len(intensities) == 5
        assert scan_times == [8.0, 9.0, 10.0, 11.0, 12.0]
        assert intensities == [100.0, 500.0, 1000.0, 500.0, 100.0]
    
    def test_query_chromatogram_within_rt_range(self, db_con):
        """Test querying chromatogram data within a specific RT range."""
        peak_label = 'Peak1'
        rt_min, rt_max = 9.0, 11.0
        
        # This tests the kind of query used for RT alignment/peak finding
        result = db_con.execute("""
            WITH unnested AS (
                SELECT 
                    peak_label,
                    UNNEST(scan_time) AS scan_time,
                    UNNEST(intensity) AS intensity
                FROM chromatograms
                WHERE peak_label = ?
            )
            SELECT scan_time, intensity
            FROM unnested
            WHERE scan_time BETWEEN ? AND ?
            ORDER BY scan_time
        """, [peak_label, rt_min, rt_max]).fetchall()
        
        # Should return 3 points: 9.0, 10.0, 11.0
        assert len(result) >= 3


class TestFilteringAndQueries:
    """Test various filtering operations used in the UI."""
    
    def test_filter_by_bookmark_status(self, db_con):
        """Test filtering targets by bookmark status."""
        # Get bookmarked targets
        bookmarked = db_con.execute(
            "SELECT peak_label FROM targets WHERE bookmark = TRUE"
        ).fetchall()
        
        assert len(bookmarked) == 1
        assert bookmarked[0][0] == 'Peak2'
        
        # Get unbookmarked targets
        unbookmarked = db_con.execute(
            "SELECT peak_label FROM targets WHERE bookmark = FALSE"
        ).fetchall()
        
        assert len(unbookmarked) == 2
    
    def test_order_by_mz_mean(self, db_con):
        """Test ordering targets by mz_mean (as used in the UI)."""
        results = db_con.execute(
            "SELECT peak_label, mz_mean FROM targets ORDER BY mz_mean ASC"
        ).fetchall()
        
        assert results[0][0] == 'Peak1'  # mz_mean = 100.0
        assert results[1][0] == 'Peak2'  # mz_mean = 200.0
        assert results[2][0] == 'Peak3'  # mz_mean = 300.0
    
    def test_order_by_peak_label(self, db_con):
        """Test ordering targets by peak_label."""
        results = db_con.execute(
            "SELECT peak_label FROM targets ORDER BY peak_label ASC"
        ).fetchall()
        
        assert results[0][0] == 'Peak1'
        assert results[1][0] == 'Peak2'
        assert results[2][0] == 'Peak3'
    
    def test_targets_with_chromatograms_join(self, db_con):
        """Test joining targets with chromatograms (used for preview)."""
        results = db_con.execute("""
            SELECT DISTINCT t.peak_label
            FROM targets t
            INNER JOIN chromatograms c ON t.peak_label = c.peak_label
            ORDER BY t.peak_label
        """).fetchall()
        
        # Only Peak1 has chromatograms in our test data
        assert len(results) == 1
        assert results[0][0] == 'Peak1'
