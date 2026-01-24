import duckdb
import pytest
import pandas as pd
import numpy as np
from ms_mint_app.duckdb_manager import validate_mint_database

def test_get_chromatogram_envelope_query():
    # Setup in-memory DuckDB
    con = duckdb.connect()
    
    # Create tables
    con.execute("""
        CREATE TABLE samples (
            ms_file_label VARCHAR,
            sample_type VARCHAR,
            color VARCHAR,
            use_for_optimization BOOLEAN
        );
        CREATE TABLE chromatograms (
            peak_label VARCHAR,
            ms_file_label VARCHAR,
            ms_type VARCHAR,
            scan_time_full_ds DOUBLE[],
            intensity_full_ds DOUBLE[],
            scan_time DOUBLE[],
            intensity DOUBLE[]
        );
        CREATE TABLE targets (
            peak_label VARCHAR,
            ms_type VARCHAR
        );
    """)
    
    # Insert dummy data
    # 5 files of type 'A', 5 files of type 'B'
    # Each file has a chromatogram with 100 points
    
    # Sample Type A: Intensity around 1000
    # Sample Type B: Intensity around 2000
    
    samples = []
    for i in range(5):
        samples.append((f"FileA_{i}", "TypeA", "red", True))
    for i in range(5):
        samples.append((f"FileB_{i}", "TypeB", "blue", True))
        
    con.executemany("INSERT INTO samples VALUES (?, ?, ?, ?)", samples)
    con.execute("INSERT INTO targets VALUES ('Target1', 'ms1')")

    chroms = []
    
    scan_time = [float(x) for x in range(100)]
    
    for i in range(5):
        # Type A: Random noise around 1000
        intensity = [1000.0 + (x % 10) * 10 for x in range(100)]
        chroms.append(('Target1', f"FileA_{i}", 'ms1', scan_time, intensity, scan_time, intensity))
        
    for i in range(5):
        # Type B: Random noise around 2000
        intensity = [2000.0 + (x % 10) * 20 for x in range(100)]
        chroms.append(('Target1', f"FileB_{i}", 'ms1', scan_time, intensity, scan_time, intensity))
        
    con.executemany("INSERT INTO chromatograms VALUES (?, ?, ?, ?, ?, ?, ?)", chroms)
    
    # Define query logic (similar to what will be in duckdb_manager.py)
    # Goal: Get Min, Max, Mean per sample_type, binned by time
    
    target_label = 'Target1'
    bins = 10
    
    # Actual implementation logic attempt
    # 1. Unnest arrays to get individual points
    # 2. Bin scan_time
    # 3. Group by sample_type, bin
    # 4. Aggregates
    
    query = """
    WITH picked_samples AS (
        SELECT ms_file_label, sample_type, color
        FROM samples
        WHERE use_for_optimization = TRUE
    ),
    raw_points AS (
        SELECT 
            s.sample_type,
            s.color,
            UNNEST(c.scan_time) as rt,
            UNNEST(c.intensity) as intens
        FROM chromatograms c
        JOIN picked_samples s ON c.ms_file_label = s.ms_file_label
        WHERE c.peak_label = ? AND c.ms_type = 'ms1'
    ),
    bounds AS (
        SELECT MIN(rt) as min_rt, MAX(rt) as max_rt FROM raw_points
    ),
    binned AS (
        SELECT
            p.sample_type,
            p.color,
            p.rt,
            p.intens,
            CAST(FLOOR((p.rt - b.min_rt) / (b.max_rt - b.min_rt + 1e-9) * ?) AS INTEGER) as bin_idx
        FROM raw_points p, bounds b
    ),
    aggregated AS (
        SELECT
            sample_type,
            color,
            bin_idx,
            AVG(rt) as rt,
            MIN(intens) as min_int,
            MAX(intens) as max_int,
            AVG(intens) as mean_int,
            COUNT(*) as count
        FROM binned
        GROUP BY sample_type, color, bin_idx
    )
    SELECT * FROM aggregated ORDER BY sample_type, bin_idx
    """
    
    df = con.execute(query, [target_label, bins]).df()
    print(df)
    
    assert len(df) <= 2 * bins  # 2 sample types * 10 bins
    assert 'TypeA' in df['sample_type'].values
    assert 'TypeB' in df['sample_type'].values
    
    # Check values for Type A
    type_a = df[df['sample_type'] == 'TypeA']
    assert type_a['min_int'].min() >= 1000
    
    # Check values for Type B
    type_b = df[df['sample_type'] == 'TypeB']
    assert type_b['min_int'].min() >= 2000

if __name__ == "__main__":
    test_get_chromatogram_envelope_query()
