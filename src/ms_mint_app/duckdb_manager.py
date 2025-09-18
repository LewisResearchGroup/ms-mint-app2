import duckdb
from pathlib import Path
from contextlib import contextmanager

@contextmanager
def duckdb_connection(workspace_path: str):
    """
    Provides a DuckDB connection as a context manager.

    The database file will be named 'mint.db' and will be located inside the workspace directory.

    :param workspace_path: The path to the MINT workspace directory.
    """
    if not workspace_path:
        yield None
        return

    db_file = Path(workspace_path, 'mint.db')
    print(f"Connecting to DuckDB at: {db_file}")
    con = None
    try:
        con = duckdb.connect(database=str(db_file), read_only=False)
        con.execute("PRAGMA enable_logging;")
        _create_tables(con)
        yield con
    except Exception as e:
        print(f"Error connecting to DuckDB: {e}")
        yield None
    finally:
        if con:
            con.close()


def _create_tables(conn: duckdb.DuckDBPyConnection):
    # Create tables if they don't exist
    conn.execute("""
                 CREATE TABLE IF NOT EXISTS samples_metadata
                 (
                     ms_file_label        VARCHAR PRIMARY KEY,
                     ms_level             INTEGER,
                     file_type            VARCHAR,
                     use_for_optimization BOOLEAN DEFAULT false,
                     use_for_analysis     BOOLEAN DEFAULT true,
                     polarity             VARCHAR,
                     color                VARCHAR,
                     label                VARCHAR,
                     sample_type          VARCHAR,
                     run_order            INTEGER,
                     plate                VARCHAR,
                     plate_row            VARCHAR,
                     plate_column         INTEGER
                 );
                 """)

    conn.execute("""
                 CREATE TABLE IF NOT EXISTS ms_data
                 (
                     ms_file_label      VARCHAR,  -- Label of the MS file, linking to samples_metadata
                     scan_id            INTEGER, -- Scan ID
                     mz                 DOUBLE,  -- Mass-to-charge ratio
                     intensity          DOUBLE,  -- Intensity
                     scan_time          DOUBLE,  -- Scan time
                     mz_precursor       DOUBLE,  -- Precursor m/z
                     filterLine         VARCHAR, -- Filter line from the raw file
                     filterLine_ELMAVEN VARCHAR -- Filter line formatted for El-Maven
                 );
                 """)

    conn.execute("""
                 CREATE TABLE IF NOT EXISTS targets
                 (
                     peak_label              VARCHAR PRIMARY KEY, -- Label for the peak
                     mz_mean                 DOUBLE,  -- Mean mass-to-charge ratio
                     mz_width                DOUBLE,  -- Width of the m/z window
                     mz                      DOUBLE,  -- Mass-to-charge ratio
                     rt                      DOUBLE,  -- Retention time
                     rt_min                  DOUBLE,  -- Minimum retention time
                     rt_max                  DOUBLE,  -- Maximum retention time
                     rt_unit                 VARCHAR, -- Unit of retention time
                     intensity_threshold     DOUBLE,  -- Intensity threshold
                     polarity                VARCHAR, -- Polarity of the target
                     filterLine              VARCHAR, -- Filter line from the raw file
                     ms_type                 VARCHAR, -- MS type (ms1 or ms2)
                     category                VARCHAR, -- Category of the target
                     score                   DOUBLE,  -- Score of the target
                     preselected_processing  BOOLEAN, -- Preselected target
                     bookmark                BOOLEAN, -- Bookmark the target
                     source                  VARCHAR -- Filename of the target list
                 );
                 """)
