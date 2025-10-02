import duckdb
from pathlib import Path
from contextlib import contextmanager

@contextmanager
def duckdb_connection(workspace_path: Path | str, register_activity=False):
    """
    Provides a DuckDB connection as a context manager.

    The database file will be named 'mint.db' and will be located inside the workspace directory.

    :param workspace_path: The path to the MINT workspace directory.
    """
    if not workspace_path:
        yield None
        return
    workspace_path = Path(workspace_path)
    db_file = Path(workspace_path, 'workspace_mint.db')
    print(f"Connecting to DuckDB at: {db_file}")
    con = None
    try:
        con = duckdb.connect(database=str(db_file), read_only=False)
        con.execute("PRAGMA enable_checkpoint_on_shutdown")
        _create_tables(con)
        yield con
    except Exception as e:
        print(f"Error connecting to DuckDB: {e}")
        yield None
    finally:
        if con:
            if register_activity:
                try:
                    with duckdb_connection_mint(workspace_path.parent.parent / 'mint.db') as mint_conn:
                        mint_conn.execute("UPDATE workspaces SET last_activity = NOW() WHERE name = ?", (Path(workspace_path).stem))
                except Exception as e:
                    print(f"Error updating workspace activity: {e}")
            con.close()

@contextmanager
def duckdb_connection_mint(mint_path: Path):
    if not mint_path:
        yield None
        return

    db_file = Path(mint_path, 'mint.db')
    con = None
    try:
        con = duckdb.connect(database=db_file, read_only=False)
        _create_workspace_tables(con)
        yield con
    except Exception as e:
        print(f"Error connecting to DuckDB: {e}")
        yield None
    finally:
        if con:
            con.close()

def _create_tables(conn: duckdb.DuckDBPyConnection):
    # Create tables if they don't exist
    conn.execute("CREATE TYPE IF NOT EXISTS ms_type_enum AS ENUM ('ms1', 'ms2');")
    conn.execute("CREATE TYPE IF NOT EXISTS polarity_enum AS ENUM ('Positive', 'Negative');")
    conn.execute("CREATE TYPE IF NOT EXISTS unit_type_enum AS ENUM ('s', 'min');")

    conn.execute("""
                 CREATE TABLE IF NOT EXISTS samples_metadata
                 (
                     ms_file_label        VARCHAR PRIMARY KEY,
                     ms_type              ms_type_enum,
                     use_for_optimization BOOLEAN DEFAULT false,
                     use_for_analysis     BOOLEAN DEFAULT true,
                     polarity             polarity_enum,
                     color                VARCHAR DEFAULT '#ffffff',
                     label                VARCHAR,
                     sample_type          VARCHAR,
                     run_order            INTEGER,
                     plate                VARCHAR,
                     plate_row            VARCHAR,
                     plate_column         TINYINT
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
                     rt_unit                 unit_type_enum, -- Unit of retention time
                     intensity_threshold     DOUBLE,  -- Intensity threshold
                     polarity                polarity_enum, -- Polarity of the target
                     filterLine              VARCHAR, -- Filter line from the raw file
                     ms_type                 ms_type_enum, -- MS type (ms1 or ms2)
                     category                VARCHAR, -- Category of the target
                     score                   DOUBLE,  -- Score of the target
                     preselected_processing  BOOLEAN, -- Preselected target
                     bookmark                BOOLEAN, -- Bookmark the target
                     source                  VARCHAR -- Filename of the target list
                 );
                 """)

    conn.execute("""
                 CREATE TABLE IF NOT EXISTS chromatograms
                 (
                     peak_label    VARCHAR,
                     ms_file_label VARCHAR,
                     scan_time     DOUBLE[],
                     intensity     DOUBLE[],
                     PRIMARY KEY (ms_file_label, peak_label)
                 );
                 """)

def _create_workspace_tables(conn: duckdb.DuckDBPyConnection):
    conn.execute("""
                 CREATE TABLE IF NOT EXISTS workspaces
                 (
                     key           UUID DEFAULT uuidv4() PRIMARY KEY,
                     name          VARCHAR UNIQUE,
                     description   VARCHAR,
                     active        BOOLEAN,
                     created_at    TIMESTAMP,
                     last_activity TIMESTAMP
                 )
                 """
                 )

def compute_and_insert_chromatograms_from_ms_data(con: duckdb.DuckDBPyConnection):
    """
    Computes chromatograms from raw MS data and inserts them into the 'chromatograms' table.

    This function performs the following steps:
    1. Filters targets that are used for optimization.
    2. Joins filtered targets with MS data based on the m/z range.
    3. Groups the data by peak label and MS file.
    4. Aggregates scan times and intensities into lists.
    5. Inserts the resulting chromatograms into the 'chromatograms' table, overwriting any existing entries.

    :param con: An active DuckDB connection.
    """
    query = """
    INSERT INTO chromatograms (peak_label, ms_file_label, scan_time, intensity)
    WITH precomputed_chromatograms AS (
        -- Step 1: Join targets and ms_data, filtering by mz range
        SELECT
            t.peak_label,
            m.ms_file_label,
            m.scan_time,
            m.intensity
        FROM targets AS t
        JOIN ms_data AS m
            ON (m.mz BETWEEN t.mz_mean - (t.mz_mean * t.mz_width / 1e6)
                       AND t.mz_mean + (t.mz_mean * t.mz_width / 1e6))
        JOIN samples_metadata AS s
            ON m.ms_file_label = s.ms_file_label
        WHERE s.use_for_optimization = TRUE
    ),
    aggregated_chromatograms AS (
        -- Step 2: Group by peak and file, and aggregate into lists
        SELECT
            peak_label,
            ms_file_label,
            list(scan_time ORDER BY scan_time) AS scan_time,
            list(intensity ORDER BY scan_time) AS intensity
        FROM precomputed_chromatograms
        GROUP BY peak_label, ms_file_label
    )
    -- Step 3: Select the final aggregated data to be inserted
    SELECT
        peak_label,
        ms_file_label,
        scan_time,
        intensity
    FROM aggregated_chromatograms
    ON CONFLICT (ms_file_label, peak_label) DO NOTHING    
    
    """
    # ON CONFLICT (ms_file_label, peak_label) DO UPDATE
    #     SET scan_time = excluded.scan_time,
    # intensity = excluded.intensity;
    con.execute(query)
    print("Chromatograms computed and inserted into DuckDB.")


def compute_and_insert_chromatograms_iteratively(con: duckdb.DuckDBPyConnection, set_progress=None):
    """
    Computes and inserts chromatograms iteratively by batching targets.

    :param con: An active DuckDB connection.
    :param set_progress: A callback function to update the progress bar.
    """
    targets_df = con.execute("SELECT peak_label, ms_type FROM targets").df()
    ms_files_count = con.execute("SELECT count(*) FROM samples_metadata WHERE use_for_optimization = TRUE").fetchone()[0]

    if ms_files_count == 0:
        if set_progress:
            set_progress(100)
        return

    ms1_targets = targets_df[targets_df['ms_type'] == 'ms1']['peak_label'].tolist()
    ms2_targets = targets_df[targets_df['ms_type'] == 'ms2']['peak_label'].tolist()

    n_total = len(targets_df)
    processed_count = 0

    def process_batch(targets_batch):
        nonlocal processed_count
        if not targets_batch:
            return

        placeholders = ', '.join(['?'] * len(targets_batch))
        query = f"""
        INSERT INTO chromatograms (peak_label, ms_file_label, scan_time, intensity)
        WITH precomputed AS (
            SELECT
                t.peak_label,
                m.ms_file_label,
                list(m.scan_time ORDER BY m.scan_time) AS scan_time,
                list(m.intensity ORDER BY m.scan_time) AS intensity
            FROM ms_data AS m
            JOIN targets AS t ON m.mz BETWEEN t.mz_mean - (t.mz_mean * t.mz_width / 1e6)
                                       AND t.mz_mean + (t.mz_mean * t.mz_width / 1e6)
            JOIN samples_metadata AS s ON m.ms_file_label = s.ms_file_label
            WHERE s.use_for_optimization = TRUE AND t.peak_label IN ({placeholders})
            GROUP BY t.peak_label, m.ms_file_label
        )
        SELECT peak_label, ms_file_label, scan_time, intensity
        FROM precomputed
        ON CONFLICT (ms_file_label, peak_label) DO UPDATE
            SET scan_time = excluded.scan_time,
                intensity = excluded.intensity;
        """
        con.execute(query, targets_batch)
        processed_count += len(targets_batch)
        if set_progress and n_total > 0:
            progress = round(processed_count / n_total * 100, 2)
            set_progress(progress)

    # Process MS2 targets in batches of 10
    for i in range(0, len(ms2_targets), 10):
        process_batch(ms2_targets[i:i + 10])

    # Determine batch size for MS1 targets
    ms1_batch_size = 1 if ms_files_count >= 50 else 5
    for i in range(0, len(ms1_targets), ms1_batch_size):
        process_batch(ms1_targets[i:i + ms1_batch_size])

    if set_progress and n_total > 0:
        set_progress(100)

    print("Iterative chromatogram computation complete.")