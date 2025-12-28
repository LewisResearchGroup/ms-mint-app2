
import duckdb
import pytest
from ms_mint_app.duckdb_manager import _create_tables

@pytest.fixture
def db_con():
    con = duckdb.connect(':memory:')
    _create_tables(con)
    # Insert dummy target
    con.execute("INSERT INTO targets (peak_label, rt, mz_mean, mz_width) VALUES ('Peak1', 10.0, 100.0, 10.0)")
    yield con
    con.close()

def test_update_valid_rt(db_con):
    column_edited = 'rt'
    new_value = 12.5
    peak_label = 'Peak1'
    query = f"UPDATE targets SET {column_edited} = ? WHERE peak_label = ?"
    db_con.execute(query, [new_value, peak_label])
    
    res = db_con.execute("SELECT rt FROM targets WHERE peak_label = 'Peak1'").fetchone()
    assert res[0] == 12.5

def test_update_invalid_rt_type(db_con):
    column_edited = 'rt'
    new_value = 'abc' # Invalid for DOUBLE
    peak_label = 'Peak1'
    query = f"UPDATE targets SET {column_edited} = ? WHERE peak_label = ?"
    
    # We expect DuckDB to raise an exception when trying to insert text into double
    # Use Exception generally to capture whatever DuckDB raises (ConversionException, BinderException etc)
    with pytest.raises(Exception) as excinfo:
        db_con.execute(query, [new_value, peak_label])
    
    print(f"Caught expected exception: {excinfo.value}")

def test_delete_logic(db_con):
    # Insert sample first because of foreign key constraints (if any? targets doesn't have FK to samples, but chromatograms might)
    # Chromatograms PK is (ms_file_label, peak_label). No explicit FK defined in schema but logic assumes it.
    db_con.execute("INSERT INTO samples (ms_file_label) VALUES ('File1')")
    db_con.execute("INSERT INTO chromatograms (peak_label, ms_file_label) VALUES ('Peak1', 'File1')")
    
    remove_targets = ['Peak1']
    
    # Logic from targets.py
    try:
        db_con.execute("BEGIN")
        # In actual code: conn.execute("DELETE FROM targets WHERE peak_label IN ?", (remove_targets,))
        # DuckDB Python client handles list params for IN clause?
        # Let's verify parameter binding for IN clause which is tricky
        
        # targets.py uses: conn.execute("DELETE FROM targets WHERE peak_label IN ?", (remove_targets,))
        # But remove_targets is a list ['Peak1']. 
        # A simplified version:
        placeholders = ','.join(['?'] * len(remove_targets))
        # Wait, targets.py does: conn.execute("DELETE FROM targets WHERE peak_label IN ?", (remove_targets,))
        # Does DuckDB python client support passing a list for a single ? in IN clause?
        # Let's test EXACTLY what the code does.
        
        # In targets.py: 
        # remove_targets = [row['peak_label'] for row in selectedRows]
        # conn.execute("DELETE FROM targets WHERE peak_label IN ?", (remove_targets,))
        
        # If this fails in the test, it means the app code is broken.
        db_con.execute("DELETE FROM targets WHERE peak_label IN ?", [remove_targets]) # Notice it passes [list] as params? 
             # Wait, (remove_targets,) is a tuple containing a list.
        
        db_con.execute("DELETE FROM chromatograms WHERE peak_label IN ?", [remove_targets])
        db_con.execute("DELETE FROM results WHERE peak_label IN ?", [remove_targets])
        db_con.execute("COMMIT")
    except Exception:
        db_con.execute("ROLLBACK")
        raise

    # Verify deletion
    res_t = db_con.execute("SELECT count(*) FROM targets WHERE peak_label = 'Peak1'").fetchone()[0]
    res_c = db_con.execute("SELECT count(*) FROM chromatograms WHERE peak_label = 'Peak1'").fetchone()[0]
    
    assert res_t == 0
    assert res_c == 0

def test_sql_injection_attempt(db_con):
    """
    Verify that SQL injection attempts in text fields are handled safely 
    (i.e., treated as literal strings, not executable SQL).
    """
    # Attempt to inject SQL into the 'notes' field
    malicious_input = "Note'; DROP TABLE chromatograms; --"
    peak_label = 'Peak1'
    
    # Insert the target first if not exists (Peak1 might be deleted by previous test order)
    # Re-insert to be safe
    try:
        db_con.execute("INSERT INTO targets (peak_label) VALUES (?)", [peak_label])
    except:
        pass # Already exists or constraint error
        
    query = "UPDATE targets SET notes = ? WHERE peak_label = ?"
    db_con.execute(query, [malicious_input, peak_label])
    
    # Verify the note is saved literally
    saved_note = db_con.execute("SELECT notes FROM targets WHERE peak_label = ?", [peak_label]).fetchone()[0]
    assert saved_note == malicious_input
    
    # Verify the table 'chromatograms' still exists (injection failed to drop it)
    # If table was dropped, this would raise a Catalog Error
    db_con.execute("SELECT count(*) FROM chromatograms")

def test_primary_key_constraint(db_con):
    """
    Verify that duplicate peak_labels cannot be inserted.
    """
    peak_label = 'UniquePeak'
    db_con.execute("INSERT INTO targets (peak_label) VALUES (?)", [peak_label])
    
    # Try inserting the same key again
    with pytest.raises(Exception):
         db_con.execute("INSERT INTO targets (peak_label) VALUES (?)", [peak_label])

