from contextlib import contextmanager
from pathlib import Path
from threading import Thread

import duckdb
import time
import logging

logger = logging.getLogger(__name__)

from .sample_metadata import GROUP_COLUMNS


def _send_progress(set_progress, percent, stage: str = "", detail: str = ""):
    """
    Safely call the provided set_progress callback.

    Supports custom stage/detail strings when the callback accepts them,
    and falls back to simple percent-only updates otherwise.
    """
    if not set_progress:
        return
    try:
        set_progress(percent, stage, detail)
    except TypeError:
        try:
            set_progress(percent)
        except Exception:
            pass
    except Exception:
        pass


def _write_progress_log(log_path: Path | str | None, message: str):
    """
    Append a progress line to the provided log file path.
    """
    if not log_path:
        return
    try:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        with path.open('a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
    except Exception:
        pass


def _center_lines(text: str, width: int = 100) -> str:
    """
    Center each line of a text block to the given width.
    Useful for nicer alignment inside the processing log file.
    """
    return "\n".join(line.center(width) for line in text.splitlines())

def _update_workspace_activity(mint_root: Path, workspace_id: str, retries: int = 3, delay_s: float = 0.05):
    for attempt in range(retries):
        try:
            with duckdb_connection_mint(mint_root) as mint_conn:
                if mint_conn:
                    mint_conn.execute(
                        "UPDATE workspaces SET last_activity = NOW() WHERE key = ?",
                        [workspace_id],
                    )
            return
        except Exception as e:
            message = str(e)
            if "TransactionContext Error: Conflict on update!" in message:
                time.sleep(delay_s * (attempt + 1))
                continue
            print(f"Error updating workspace activity: {e}")
            return


@contextmanager
def duckdb_connection(workspace_path: Path | str, register_activity=True, n_cpus=None, ram=None):
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
    # print(f"Connecting to DuckDB at: {db_file}")
    con = None
    try:
        con = duckdb.connect(database=str(db_file), read_only=False)
        con.execute("PRAGMA enable_checkpoint_on_shutdown")
        con.execute("SET enable_progress_bar = true")
        con.execute("SET enable_progress_bar_print = false")
        con.execute("SET progress_bar_time = 0")
        con.execute(f"SET temp_directory = '{workspace_path.as_posix()}';")
        if n_cpus:
            con.execute(f"SET threads = {n_cpus}", )
        if ram:
            con.execute(f"SET memory_limit = '{ram}GB'")
        _create_tables(con)
    except Exception as e:
        print(f"Error connecting to DuckDB: {e}")
        yield None
        return
    try:
        yield con
    finally:
        if con:
            if register_activity:
                try:
                    workspace_id = Path(workspace_path).name
                    mint_root = workspace_path.parent.parent
                    _update_workspace_activity(mint_root, workspace_id)
                except Exception as e:
                    print(f"Error updating workspace activity: {e}")
            if n_cpus:
                con.execute("RESET threads")
            if ram:
                con.execute("RESET memory_limit")
            con.close()


@contextmanager
def duckdb_connection_mint(mint_path: Path, workspace=None):
    if not mint_path:
        yield None
        return

    db_file = Path(mint_path, 'mint.db')
    con = None
    try:
        con = duckdb.connect(database=db_file, read_only=False)
        _create_workspace_tables(con)
    except Exception as e:
        print(f"Error connecting to DuckDB: {e}")
        yield None
        return
    try:
        yield con
    finally:
        if con:
            if workspace:
                try:
                    con.execute("UPDATE workspaces SET last_activity = NOW() WHERE key = ?", [workspace])
                except Exception:
                    pass
            con.close()


def _create_tables(conn: duckdb.DuckDBPyConnection):
    # Create tables if they don't exist
    conn.execute("CREATE TYPE IF NOT EXISTS ms_type_enum AS ENUM ('ms1', 'ms2');")
    conn.execute("CREATE TYPE IF NOT EXISTS polarity_enum AS ENUM ('Positive', 'Negative');")
    conn.execute("CREATE TYPE IF NOT EXISTS unit_type_enum AS ENUM ('s', 'min');")

    conn.execute("""
                 CREATE TABLE IF NOT EXISTS samples
                 (
                     ms_file_label        VARCHAR PRIMARY KEY,
                     ms_type              ms_type_enum,
                     file_type            VARCHAR,
                     use_for_optimization BOOLEAN DEFAULT true,
                     use_for_processing   BOOLEAN DEFAULT true,
                     use_for_analysis     BOOLEAN DEFAULT true,
                     polarity             polarity_enum,
                     color                VARCHAR DEFAULT '#bbbbbb',
                     label                VARCHAR,
                     sample_type          VARCHAR DEFAULT 'Unset',
                     group_1              VARCHAR,
                     group_2              VARCHAR,
                     group_3              VARCHAR,
                     group_4              VARCHAR,
                     group_5              VARCHAR
                 );
                 """)

    # Backfill new processing flag for existing DBs
    conn.execute("ALTER TABLE samples ADD COLUMN IF NOT EXISTS use_for_processing BOOLEAN DEFAULT true;")
    conn.execute("ALTER TABLE samples ADD COLUMN IF NOT EXISTS file_type VARCHAR;")
    for col in GROUP_COLUMNS:
        conn.execute(f"ALTER TABLE samples ADD COLUMN IF NOT EXISTS {col} VARCHAR;")
    try:
        conn.execute("""
                     UPDATE samples
                     SET use_for_processing = COALESCE(use_for_processing, use_for_analysis, TRUE)
                     """)
    except Exception:
        # Avoid failing during initialization if another write is in flight.
        pass

    conn.execute("""
                 CREATE TABLE IF NOT EXISTS ms1_data
                 (
                     ms_file_label      VARCHAR,  -- Label of the MS file, linking to samples
                     scan_id            INTEGER,  -- Scan ID
                     mz                 DOUBLE,   -- Mass-to-charge ratio
                     intensity          DOUBLE,   -- Intensity
                     scan_time          DOUBLE    -- Scan time
                 );
                 """)
    conn.execute("""
                 CREATE TABLE IF NOT EXISTS ms2_data
                 (
                     ms_file_label      VARCHAR,  -- Label of the MS file, linking to samples
                     scan_id            INTEGER,  -- Scan ID
                     mz                 DOUBLE,   -- Mass-to-charge ratio
                     intensity          DOUBLE,   -- Intensity
                     scan_time          DOUBLE,   -- Scan time
                     mz_precursor       DOUBLE,   -- Precursor m/z
                     filterLine         VARCHAR,  -- Filter line from the raw file
                     filterLine_ELMAVEN VARCHAR   -- Filter line formatted for El-Maven
                 );
                 """)

    conn.execute("""
                 CREATE TABLE IF NOT EXISTS targets
                 (
                     peak_label          VARCHAR PRIMARY KEY, -- Label for the peak
                     mz_mean             DOUBLE,              -- Mean mass-to-charge ratio
                     mz_width            DOUBLE,              -- Width of the m/z window
                     mz                  DOUBLE,              -- Mass-to-charge ratio
                     rt                  DOUBLE,              -- Retention time
                     rt_min              DOUBLE,              -- Minimum retention time
                     rt_max              DOUBLE,              -- Maximum retention time
                     rt_unit             unit_type_enum,      -- Unit of retention time
                     intensity_threshold DOUBLE,              -- Intensity threshold
                     polarity            polarity_enum,       -- Polarity of the target
                     filterLine          VARCHAR,             -- Filter line from the raw file
                     ms_type             ms_type_enum,        -- MS type (ms1 or ms2)
                     category            VARCHAR,             -- Category of the target
                     peak_selection      BOOLEAN,             -- Preselected target
                     score               DOUBLE,              -- Score of the target
                     bookmark            BOOLEAN,             -- Bookmark the target
                     source              VARCHAR,             -- Filename of the target list
                     notes               VARCHAR              -- Additional notes for the target
                 );
                 """)

    conn.execute("""
                 CREATE TABLE IF NOT EXISTS chromatograms
                 (
                     peak_label    VARCHAR,
                     ms_file_label VARCHAR,
                     scan_time     DOUBLE[],
                     intensity     DOUBLE[],
                     ms_type       ms_type_enum,
                     -- mz            DOUBLE[],
                     PRIMARY KEY (ms_file_label, peak_label)
                 );
                 """)

    conn.execute("""
                 CREATE TABLE IF NOT EXISTS results
                 (
                     peak_label        VARCHAR,
                     ms_file_label     VARCHAR,
                     total_intensity   DOUBLE,
                     peak_area         DOUBLE,
                     peak_area_top3    DOUBLE,
                     peak_max          DOUBLE,
                     peak_min          DOUBLE,
                     peak_mean         DOUBLE,
                     peak_rt_of_max    DOUBLE,
                     peak_median       DOUBLE,
                     peak_n_datapoints INT,

                     scan_time         DOUBLE[],
                     intensity         DOUBLE[],
                     PRIMARY KEY (ms_file_label, peak_label)
                 );
                 """)


def _create_workspace_tables(conn: duckdb.DuckDBPyConnection):
    conn.execute("""
                 CREATE TABLE IF NOT EXISTS workspaces
                 (
                     key           UUID DEFAULT uuidv4() PRIMARY KEY,
                     name          VARCHAR,
                     description   VARCHAR,
                     active        BOOLEAN,
                     created_at    TIMESTAMP,
                     last_activity TIMESTAMP
                 )
                 """
                 )


def build_where_and_params(filter_, filterOptions):
    where_sql, params = [], []

    if not isinstance(filter_, dict) or not filter_:
        return "", []

    for key, value in filter_.items():
        if not value:
            continue
        # keyword (ILIKE)
        if filterOptions[key].get('filterMode') == 'keyword':
            where_sql.append(f'"{key}" ILIKE ?')
            params.append(f"%{value[0]}%")
        # multiple selection (IN)
        else:
            ph = ",".join("?" for _ in value)
            where_sql.append(f'"{key}" IN ({ph})')
            params.extend(value)
    where_clause = f"WHERE {' AND '.join(where_sql)}" if where_sql else ""
    return where_clause, params


def build_order_by(
        sorter: dict | None,
        column_types: dict[str, str],
        *,
        tie: tuple[str, str] | None = None,  # e.g. ("id", "ASC"); used ONLY when a sorter is present
        nocase_text: bool = True
) -> str:
    """
    Returns 'ORDER BY ...' or '' if there is no valid sorter.
    - sorter: {'columns': [...], 'orders': ['ascend'|'descend', ...]}
    - column_types: map {col -> DUCKDB type} (from DESCRIBE)
    - tie: optional (col, dir); added ONLY when there is at least one sortable column in sorter
    """
    # 0) Normalize input
    cols_in = (sorter or {}).get("columns") or []
    ords_in = (sorter or {}).get("orders") or []
    if not cols_in:
        return ""  # no sorter => no ORDER BY

    # Fill missing order entries
    if len(ords_in) < len(cols_in):
        ords_in = ords_in + ["ascend"] * (len(cols_in) - len(ords_in))

    order_map = {"ascend": "ASC", "descend": "DESC"}
    parts: list[str] = []
    used_cols: set[str] = set()

    for col, ord_ in zip(cols_in, ords_in):
        if col not in column_types:
            continue  # ignore invalid columns
        direction = order_map.get(ord_, "ASC")
        nulls = "NULLS LAST" if direction == "ASC" else "NULLS FIRST"
        ctype = (column_types.get(col) or "").upper()
        is_text = any(t in ctype for t in ("CHAR", "VARCHAR", "TEXT", "STRING"))
        expr = f'"{col}" COLLATE NOCASE' if (nocase_text and is_text) else f'"{col}"'
        parts.append(f"{expr} {direction} {nulls}")
        used_cols.add(col)

    # If nothing valid remains, do not sort (and do not add tie)
    if not parts:
        return ""

    # Add tie ONLY if there is a valid sorter, tie was requested, and the column is not duplicated
    if tie:
        tie_col, tie_dir = tie[0], tie[1].upper()
        if tie_col not in used_cols and tie_col in column_types:
            parts.append(f'"{tie_col}" {tie_dir}')

    return f"ORDER BY {', '.join(parts)}"


def build_paginated_query_by_peak(
        conn,
        filter_: dict | None = None,
        filterOptions: dict | None = None,
        sorter: dict | None = None,
        limit: int = 10,
        offset: int = 0
) -> tuple[str, list]:
    """
    Build a paginated query grouped by peak_label.
    Uses build_where_and_params() and build_order_by() without modifying them.
    """

    # Get column types
    column_types = {
        row[0]: row[1]
        for row in conn.execute("DESCRIBE results").fetchall()
    }

    # 1. Build WHERE clause and params to filter rows
    where_sql, where_params = build_where_and_params(filter_, filterOptions or {})

    # 2. Build ORDER BY for individual rows
    order_by_sql = build_order_by(
        sorter,
        column_types,
        tie=("peak_label", "ASC"),
        nocase_text=True
    )

    # 3. Extract ordering columns to aggregate by peak_label
    agg_exprs = []
    order_exprs = []

    if order_by_sql:
        # Parse ORDER BY to extract columns
        order_part = order_by_sql.replace("ORDER BY", "").strip()
        for clause in order_part.split(","):
            clause = clause.strip()
            # Remove modifiers
            clean = clause.replace("COLLATE NOCASE", "").replace("NULLS LAST", "").replace("NULLS FIRST", "")
            parts = clean.split()
            if not parts:
                continue

            col = parts[0].strip('"')
            direction = parts[1] if len(parts) > 1 else "ASC"

            if col == "peak_label":
                agg_exprs.append(f"peak_label AS _ord_{col}")
                order_exprs.append(f"_ord_{col} {direction}")
            else:
                # For other columns, use MAX/MIN depending on direction
                ctype = (column_types.get(col) or "").upper()
                is_numeric = any(t in ctype for t in ("INT", "DOUBLE", "FLOAT", "DECIMAL", "NUMERIC", "REAL"))

                if is_numeric:
                    # For numeric: MAX if DESC, MIN if ASC (representative value)
                    agg_func = "MAX" if "DESC" in direction else "MIN"
                    agg_exprs.append(f'{agg_func}("{col}") AS _ord_{col}')
                    order_exprs.append(f"_ord_{col} {direction}")
                else:
                    # For text: MAX/MIN depending on direction
                    agg_func = "MAX" if "DESC" in direction else "MIN"
                    agg_exprs.append(f'{agg_func}("{col}") AS _ord_{col}')
                    order_exprs.append(f"_ord_{col} {direction}")

    # If there is no ordering, use peak_label by default
    if not agg_exprs:
        agg_exprs.append("peak_label AS _ord_peak_label")
        order_exprs.append("_ord_peak_label ASC")

    # 4. Build the query with CTEs
    sql = f"""
    WITH filtered AS (
      SELECT *
      FROM results
      {where_sql}
    ),
    peak_ordering AS (
      SELECT 
        peak_label,
        {', '.join(agg_exprs)},
        ROW_NUMBER() OVER(ORDER BY {', '.join(order_exprs)}) AS _rn
      FROM filtered
      GROUP BY peak_label
    ),
    total_peaks AS (
      SELECT COUNT(*) AS __total__
      FROM peak_ordering
    ),
    paged_peaks AS (
      SELECT peak_label, _rn
      FROM peak_ordering
      WHERE _rn > ? AND _rn <= ? + ?
    ),
    paged AS (
      SELECT 
        f.*,
        pp._rn AS __peak_order__,
        (SELECT __total__ FROM total_peaks) AS __total__
      FROM filtered f
      JOIN paged_peaks pp ON f.peak_label = pp.peak_label
      ORDER BY pp._rn, f.ms_file_label
    )
    SELECT * FROM paged;
    """

    # 5. Combine parameters: first WHERE params, then pagination
    all_params = where_params + [offset, offset, limit]

    return sql, all_params


def compute_and_insert_chromatograms_from_ms_data(con: duckdb.DuckDBPyConnection,
                                                  set_progress=None,
                                                  for_optimization=True,
                                                  recompute_ms1=False,
                                                  recompute_ms2=False):
    """
    Computes chromatograms from raw MS data and inserts them into the 'chromatograms' table.

    :param con: An active DuckDB connection.
    :param set_progress: Optional callback function to report progress (0-100).
    :param recompute_ms1: If True, deletes existing MS1 chromatograms before recomputing.
    :param recompute_ms2: If True, deletes existing MS2 chromatograms before recomputing.
    """

    info = con.execute("""
                       WITH samples_to_use AS (SELECT DISTINCT ms_file_label
                                               FROM samples
                                               WHERE use_for_optimization = TRUE
                                                  OR use_for_processing = TRUE),
                            ms1_targets AS (SELECT DISTINCT t.peak_label, s.ms_file_label
                                            FROM targets t
                                                     CROSS JOIN samples_to_use s
                                            WHERE t.mz_mean IS NOT NULL
                                                AND t.mz_width IS NOT NULL
                                                AND t.peak_selection IS TRUE
                                               OR NOT EXISTS (SELECT 1
                                                              FROM targets t1
                                                              WHERE t1.peak_selection IS TRUE)
                                                AND
                                                  EXISTS(SELECT 1 FROM ms1_data md WHERE md.ms_file_label = s.ms_file_label)),
                            ms2_targets AS (SELECT DISTINCT t.peak_label, s.ms_file_label
                                            FROM targets t
                                                     CROSS JOIN samples_to_use s
                                            WHERE t.filterLine IS NOT NULL -- ensures this is MS2
                                                AND t.peak_selection IS TRUE
                                               OR NOT EXISTS (SELECT 1
                                                              FROM targets t1
                                                              WHERE t1.peak_selection IS TRUE)
                                                AND
                                                  EXISTS(SELECT 1 FROM ms2_data md WHERE md.ms_file_label = s.ms_file_label)),
                            existing_chromatograms AS (SELECT DISTINCT peak_label, ms_file_label
                                                       FROM chromatograms)
                       SELECT
                           -- MS1 info
                           (SELECT COUNT(*) FROM ms1_targets)                          AS ms1_total_pairs,
                           (SELECT COUNT(*)
                            FROM ms1_targets mt
                                     JOIN existing_chromatograms ec
                                          ON ec.peak_label = mt.peak_label
                                              AND ec.ms_file_label = mt.ms_file_label) AS ms1_existing_pairs,
                           (SELECT COUNT(*)
                            FROM ms1_targets mt
                                     LEFT JOIN existing_chromatograms ec
                                               ON ec.peak_label = mt.peak_label
                                                   AND ec.ms_file_label = mt.ms_file_label
                            WHERE ec.peak_label IS NULL)                               AS ms1_missing_pairs,

                           -- MS2 info
                           (SELECT COUNT(*) FROM ms2_targets)                          AS ms2_total_pairs,
                           (SELECT COUNT(*)
                            FROM ms2_targets mt
                                     JOIN existing_chromatograms ec
                                          ON ec.peak_label = mt.peak_label
                                              AND ec.ms_file_label = mt.ms_file_label) AS ms2_existing_pairs,
                           (SELECT COUNT(*)
                            FROM ms2_targets mt
                                     LEFT JOIN existing_chromatograms ec
                                               ON ec.peak_label = mt.peak_label
                                                   AND ec.ms_file_label = mt.ms_file_label
                            WHERE ec.peak_label IS NULL)                               AS ms2_missing_pairs
                       """).fetchone()

    (ms1_total, ms1_existing, ms1_missing,
     ms2_total, ms2_existing, ms2_missing) = info

    # Decide what to process
    process_ms1 = ms1_total > 0 and (recompute_ms1 or ms1_missing > 0)
    process_ms2 = ms2_total > 0 and (recompute_ms2 or ms2_missing > 0)

    # Informational logging
    if ms1_total > 0:
        print(f"MS1: {ms1_existing} existing, {ms1_missing} missing (total: {ms1_total})")
    if ms2_total > 0:
        print(f"MS2: {ms2_existing} existing, {ms2_missing} missing (total: {ms2_total})")

    if not process_ms1 and not process_ms2:
        print("No chromatograms to process.")
        return

    # Remove existing chromatograms if recomputation is requested
    if recompute_ms1 and ms1_existing > 0:
        print(f"Deleting {ms1_existing} existing MS1 chromatograms for recalculation...")
        con.execute("""
                    DELETE
                    FROM chromatograms
                    WHERE EXISTS(SELECT 1
                                 FROM targets t
                                          CROSS JOIN samples s
                                 WHERE s.use_for_optimization = TRUE
                                    OR s.use_for_processing = TRUE
                                     AND t.mz_mean IS NOT NULL
                                     AND t.mz_width IS NOT NULL
                                     AND t.peak_selection IS TRUE
                                    OR NOT EXISTS (SELECT 1
                                                   FROM targets t1
                                                   WHERE t1.peak_selection IS TRUE)
                                     AND chromatograms.peak_label = t.peak_label
                                     AND chromatograms.ms_file_label = s.ms_file_label)
                    """)
        ms1_to_compute = ms1_total
    else:
        ms1_to_compute = ms1_missing

    if recompute_ms2 and ms2_existing > 0:
        print(f"Deleting {ms2_existing} existing MS2 chromatograms for recalculation...")
        con.execute("""
                    DELETE
                    FROM chromatograms
                    WHERE EXISTS(SELECT 1
                                 FROM targets t
                                          CROSS JOIN samples s
                                 WHERE s.use_for_optimization = TRUE
                                    OR s.use_for_processing = TRUE
                                     AND t.filterLine IS NOT NULL
                                     AND t.peak_selection IS TRUE
                                    OR NOT EXISTS (SELECT 1
                                                   FROM targets t1
                                                   WHERE t1.peak_selection IS TRUE)
                                     AND chromatograms.peak_label = t.peak_label
                                     AND chromatograms.ms_file_label = s.ms_file_label)
                    """)
        ms2_to_compute = ms2_total
    else:
        ms2_to_compute = ms2_missing

    # Compute weights for progress reporting
    total_to_compute = ms1_to_compute + ms2_to_compute
    ms1_weight = ms1_to_compute / total_to_compute if process_ms1 else 0
    ms2_weight = ms2_to_compute / total_to_compute if process_ms2 else 0

    print(f"Computing {ms1_to_compute} MS1 and {ms2_to_compute} MS2 chromatograms...")

    query_ms1 = """
                INSERT INTO chromatograms (peak_label, ms_file_label, scan_time, intensity, ms_type)
                WITH pairs_to_process AS (SELECT t.peak_label,
                                                 t.mz_mean,
                                                 t.mz_width,
                                                 s.ms_file_label
                                          FROM targets t
                                                   JOIN samples s
                                                        ON (CASE WHEN ? THEN s.use_for_optimization ELSE s.use_for_processing END) =
                                                           TRUE
                                          WHERE t.mz_mean IS NOT NULL
                                              AND t.mz_width IS NOT NULL
                                              AND t.peak_selection IS TRUE
                                             OR NOT EXISTS (SELECT 1
                                                            FROM targets t1
                                                            WHERE t1.peak_selection IS TRUE)
                                              AND (
                                                    ? -- recompute_ms1
                                                        OR NOT EXISTS (SELECT 1
                                                                       FROM chromatograms c
                                                                       WHERE c.peak_label = t.peak_label
                                                                         AND c.ms_file_label = s.ms_file_label)
                                                    )),
                     filtered AS (SELECT p.peak_label,
                                         p.ms_file_label,
                                         ROUND(ms1.scan_time, 3) AS scan_time,
                                         ROUND(ms1.intensity, 0) AS intensity
                                  FROM pairs_to_process p
                                           JOIN ms1_data ms1
                                                ON ms1.ms_file_label = p.ms_file_label
                                                    AND ms1.mz BETWEEN p.mz_mean - (p.mz_mean * p.mz_width / 1e6)
                                                       AND p.mz_mean + (p.mz_mean * p.mz_width / 1e6) QUALIFY
                                        ROW_NUMBER() OVER (
                                          PARTITION BY p.peak_label, p.ms_file_label, ROUND(ms1.scan_time, 2)
                                            ORDER BY ms1.intensity DESC
                                        ) = 1),
                     agg AS (SELECT peak_label,
                                    ms_file_label,
                                    LIST(scan_time ORDER BY scan_time) AS scan_time,
                                    LIST(intensity ORDER BY scan_time) AS intensity
                             FROM filtered
                             GROUP BY peak_label, ms_file_label)
                SELECT peak_label, ms_file_label, scan_time, intensity, 'ms1' AS ms_type
                FROM agg;
                """

    query_ms2 = """
                INSERT INTO chromatograms (peak_label, ms_file_label, scan_time, intensity, ms_type)
                WITH pairs_to_process AS (SELECT t.peak_label,
                                                 t.filterLine,
                                                 s.ms_file_label
                                          FROM targets AS t
                                                   JOIN samples s
                                                        ON (CASE WHEN ? THEN s.use_for_optimization ELSE s.use_for_processing END) =
                                                           TRUE
                                          WHERE t.filterLine IS NOT NULL
                                              AND t.peak_selection IS TRUE
                                             OR NOT EXISTS (SELECT 1
                                                            FROM targets t1
                                                            WHERE t1.peak_selection IS TRUE)
                                              AND (
                                                    ? -- recompute_ms2
                                                        OR NOT EXISTS (SELECT 1
                                                                       FROM chromatograms c
                                                                       WHERE c.peak_label = t.peak_label
                                                                         AND c.ms_file_label = s.ms_file_label)
                                                    )),
                     pre AS (SELECT p.peak_label,
                                    p.ms_file_label,
                                    ROUND(ms2.scan_time, 3) AS scan_time,
                                    ROUND(ms2.intensity, 0) AS intensity
                             -- ms2.mz
                             FROM pairs_to_process p
                                      JOIN ms2_data ms2
                                           ON ms2.ms_file_label = p.ms_file_label
                                               AND ms2.filterLine = p.filterLine),
                     grouped AS (SELECT peak_label,
                                        ms_file_label,
                                        scan_time,
                                        MAX(intensity) AS intensity -- max per time bin
                                 -- AVG(mz_mean)   AS mz_mean    -- stable within the bin
                                 FROM pre
                                 GROUP BY peak_label, ms_file_label, scan_time),
                     aggregated_chromatograms AS (SELECT peak_label,
                                                         ms_file_label,
                                                         LIST(scan_time ORDER BY scan_time) AS scan_time,
                                                         LIST(intensity ORDER BY scan_time) AS intensity
                                                  -- list(mz ORDER BY scan_time) AS mz
                                                  FROM grouped
                                                  GROUP BY peak_label, ms_file_label)
                SELECT peak_label, ms_file_label, scan_time, intensity, 'ms2' AS ms_type
                FROM aggregated_chromatograms;
                """

    # Shared variable to accumulate progress
    accumulated_progress = [0.0]
    stop_monitoring = [False]
    current_query_type = ['ms1']  # Track which query is running

    def monitor_progress():
        """Monitor progress of the current query"""
        while not stop_monitoring[0]:
            try:
                qp = con.query_progress()
                if qp != -1 and qp > 0:
                    # Total progress depends on which query is executing
                    if current_query_type[0] == 'ms1':
                        total_progress = qp * ms1_weight
                    else:  # ms2
                        total_progress = (ms1_weight * 100) + (qp * ms2_weight)

                    accumulated_progress[0] = total_progress
                    if set_progress:
                        set_progress(round(total_progress, 1))
                    print(f"Progress: {total_progress:.1f}%")

                time.sleep(0.05)

            except (duckdb.InvalidInputException, duckdb.ConnectionException):
                break
            except Exception as e:
                print(f"Progress monitoring error: {e}")
                break

    # Start monitoring
    if set_progress:
        progress_thread = Thread(target=monitor_progress, daemon=True)
        progress_thread.start()

    try:
        # Run MS1
        if process_ms1:
            print("Processing MS1 chromatograms...")
            current_query_type[0] = 'ms1'
            con.execute(query_ms1, [for_optimization, recompute_ms1])
            accumulated_progress[0] = ms1_weight * 100
            if set_progress:
                set_progress(round(accumulated_progress[0], 1))

        # Run MS2
        if process_ms2:
            print("Processing MS2 chromatograms...")
            current_query_type[0] = 'ms2'
            con.execute(query_ms2, [for_optimization, recompute_ms2])
            accumulated_progress[0] = 100.0
            if set_progress:
                set_progress(100.0)

        print("Chromatograms computed and inserted into DuckDB.")

    finally:
        stop_monitoring[0] = True
        if set_progress:
            progress_thread.join(timeout=0.5)

    print("Chromatograms computed and inserted into DuckDB.")




def compute_chromatograms_in_batches(wdir: str,
                                     # conn: duckdb.DuckDBPyConnection,
                                     use_for_optimization: bool,
                                     batch_size: int = 1000,
                                     checkpoint_every: int = 10,
                                     set_progress=None,
                                     recompute_ms1=False,
                                     recompute_ms2=False,
                                     n_cpus=None,
                                     ram=None,
                                     use_bookmarked: bool = False,
                                     ):
    # progress_log = Path(wdir) / "processing_progress.log" if wdir else None
    progress_log = None
    logger.info(f"Computing chromatograms in batches. wDir: {wdir}")
    QUERY_CREATE_SCAN_LOOKUP = """
                               CREATE TABLE IF NOT EXISTS ms_file_scans AS
                               SELECT DISTINCT ms_file_label,
                                      scan_id,
                                      scan_time,
                                      'ms1' AS ms_type
                               FROM ms1_data
                                   UNION ALL
                               SELECT DISTINCT ms_file_label,
                                      scan_id,
                                      scan_time,
                                      'ms2' AS ms_type
                               FROM ms2_data
                               ORDER BY ms_file_label, scan_id, ms_type;

                               CREATE INDEX IF NOT EXISTS idx_ms_file_scans_file
                                   ON ms_file_scans (ms_file_label);

                               CREATE INDEX IF NOT EXISTS idx_ms_file_scans_file_scan
                                   ON ms_file_scans (ms_file_label, scan_id, ms_type);
                               """

    QUERY_CREATE_PENDING_PAIRS = """
                                 CREATE TABLE IF NOT EXISTS pending_pairs AS
                                 WITH target_filter AS (SELECT peak_label,
                                                               ms_type,
                                                               mz_mean,
                                                               mz_width,
                                                               filterLine,
                                                               bookmark
                                                        FROM targets t
                                                        WHERE (
                                                            t.peak_selection IS TRUE
                                                                OR NOT EXISTS (SELECT 1
                                                                               FROM targets t1
                                                                               WHERE t1.peak_selection IS TRUE
                                                                                 AND t1.ms_type = t.ms_type)
                                                            )
                                                          AND (
                                                            CASE
                                                                WHEN ?
                                                                    THEN t.bookmark IS TRUE -- use_bookmarked = True → solo marcados
                                                                ELSE TRUE -- use_bookmarked = False → no filtra
                                                                END
                                                            )),
                                      sample_filter AS (SELECT ms_file_label
                                                        FROM samples
                                                        WHERE (CASE WHEN ? THEN use_for_optimization ELSE use_for_processing END) = TRUE),
                                      existing_pairs AS (SELECT DISTINCT peak_label,
                                                                         ms_file_label,
                                                                         ms_type
                                                         FROM chromatograms),
                                      all_possible_pairs AS (SELECT t.peak_label,
                                                                    s.ms_file_label,
                                                                    t.ms_type,
                                                                    t.mz_mean,
                                                                    t.mz_width,
                                                                    t.filterLine
                                                             FROM target_filter t
                                                                      CROSS JOIN sample_filter s),
                                      pending AS (SELECT a.peak_label,
                                                         a.ms_file_label,
                                                         a.ms_type,
                                                         a.mz_mean,
                                                         a.mz_width,
                                                         a.filterLine,
                                                         ROW_NUMBER() OVER () AS pair_id
                                                  FROM all_possible_pairs a
                                                           LEFT JOIN existing_pairs e
                                                                     ON a.peak_label = e.peak_label
                                                                         AND a.ms_file_label = e.ms_file_label
                                                                         AND a.ms_type = e.ms_type
                                                  WHERE e.peak_label IS NULL)
                                 SELECT pair_id,
                                        peak_label,
                                        ms_file_label,
                                        ms_type,
                                        mz_mean,
                                        mz_width,
                                        filterLine
                                 FROM pending
                                 ORDER BY pair_id;
                                 """

    QUERY_PROCESS_BATCH_MS1 = """
                              INSERT INTO chromatograms (peak_label, ms_file_label, scan_time, intensity, ms_type)
                              WITH batch_pairs AS (SELECT peak_label, ms_file_label, ms_type, mz_mean, mz_width
                                                   FROM pending_pairs
                                                   WHERE ms_type = 'ms1'
                                                     AND pair_id BETWEEN ? AND ?),
                                   -- Step 1: Find intensities (only rows with signal)
                                   matched_intensities AS (SELECT bp.peak_label,
                                                                  bp.ms_file_label,
                                                                  ms1.scan_id,
                                                                  MAX(ms1.intensity) AS intensity
                                                           FROM batch_pairs bp
                                                                    JOIN ms1_data ms1
                                                                         ON ms1.ms_file_label = bp.ms_file_label
                                                                             AND ms1.mz BETWEEN
                                                                                bp.mz_mean - (bp.mz_mean * bp.mz_width / 1e6)
                                                                                AND
                                                                                bp.mz_mean + (bp.mz_mean * bp.mz_width / 1e6)
                                                           GROUP BY bp.peak_label, bp.ms_file_label, ms1.scan_id),
                                   -- Step 2: Expand to all scans
                                   all_scans_needed AS (SELECT DISTINCT bp.peak_label,
                                                                        bp.ms_file_label,
                                                                        s.scan_id,
                                                                        s.scan_time
                                                        FROM batch_pairs bp
                                                                 JOIN ms_file_scans s ON s.ms_file_label = bp.ms_file_label),
                                   -- Step 3: LEFT JOIN (both tables are small)
                                   complete_data AS (SELECT a.peak_label,
                                                            a.ms_file_label,
                                                            a.scan_time,
                                                            a.scan_id,
                                                            a.scan_time,
                                                            COALESCE(ROUND(m.intensity, 0), 1) AS intensity
                                                     FROM all_scans_needed a
                                                              LEFT JOIN matched_intensities m
                                                                        ON a.peak_label = m.peak_label
                                                                            AND a.ms_file_label = m.ms_file_label
                                                                            AND a.scan_id = m.scan_id),
                                   agg AS (SELECT peak_label,
                                                  ms_file_label,
                                                  LIST(scan_time ORDER BY scan_time) AS scan_time,
                                                  LIST(intensity ORDER BY scan_time) AS intensity
                                           FROM complete_data
                                           GROUP BY peak_label, ms_file_label)
                              SELECT peak_label, ms_file_label, scan_time, intensity, 'ms1' AS ms_type
                              FROM agg;
                              """

    QUERY_PROCESS_BATCH_MS2 = """
                              INSERT INTO chromatograms (peak_label, ms_file_label, scan_time, intensity, ms_type)
                              WITH batch_pairs AS (SELECT peak_label, ms_file_label, ms_type, filterLine
                                                   FROM pending_pairs
                                                   WHERE ms_type = 'ms2'
                                                     AND pair_id BETWEEN ? AND ?),
                                   -- Step 1: Find intensities (only rows with signal)
                                   matched_filterline AS (SELECT bp.peak_label,
                                                      bp.ms_file_label,
                                                      ms2.scan_id,
                                                      ms2.intensity
                                               FROM batch_pairs bp
                                                        JOIN ms2_data ms2
                                                             ON ms2.ms_file_label = bp.ms_file_label
                                                                 AND ms2.filterLine = bp.filterLine),
                                  -- Step 2: Expand to all scans
                                   all_scans_needed AS (SELECT DISTINCT bp.peak_label,
                                                                        bp.ms_file_label,
                                                                        s.scan_id,
                                                                        s.scan_time
                                                        FROM batch_pairs bp
                                                                 JOIN ms_file_scans s ON s.ms_file_label = bp.ms_file_label),
                                  -- Step 3: LEFT JOIN (both tables are small)
                                   complete_data AS (SELECT a.peak_label,
                                                            a.ms_file_label,
                                                            a.scan_time,
                                                            a.scan_id,
                                                            a.scan_time,
                                                            COALESCE(ROUND(m.intensity, 0), 1) AS intensity
                                                     FROM all_scans_needed a
                                                              LEFT JOIN matched_filterline m
                                                                        ON a.peak_label = m.peak_label
                                                                            AND a.ms_file_label = m.ms_file_label
                                                                            AND a.scan_id = m.scan_id),
                                   -- Step 2: Aggregate after expanding scans
                                   agg AS (SELECT peak_label,
                                                  ms_file_label,
                                                  LIST(scan_time ORDER BY scan_time) AS scan_time,
                                                  LIST(intensity ORDER BY scan_time) AS intensity
                                           FROM complete_data
                                           GROUP BY peak_label, ms_file_label)
                              SELECT peak_label,
                                     ms_file_label,
                                     scan_time,
                                     intensity,
                                     'ms2' AS ms_type
                              FROM agg;
                              """

    if recompute_ms1:
        logger.info("Deleting existing MS1 chromatograms for recalculation...")
        with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as con:
            con.execute("DELETE FROM chromatograms WHERE ms_type = 'ms1'")
    if recompute_ms2:
        logger.info("Deleting existing MS2 chromatograms for recalculation...")
        with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as con:
            con.execute("DELETE FROM chromatograms WHERE ms_type = 'ms2'")

    with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as conn:
        conn.execute("DROP TABLE IF EXISTS pending_pairs")
        try:
            count = conn.execute("SELECT COUNT(*) FROM ms_file_scans").fetchone()[0]
            logger.info(f"Lookup table exists ({count:,} entries)")
        except:
            logger.warning("Lookup table does not exist. Creating...")

            start = time.perf_counter()
            conn.execute(QUERY_CREATE_SCAN_LOOKUP)
            elapsed = time.perf_counter() - start

            result = conn.execute("""
                                  SELECT COUNT(*)                      as total_entries,
                                         COUNT(DISTINCT ms_file_label) as total_files,
                                         AVG(scans_per_file)           as avg_scans
                                  FROM (SELECT ms_file_label, COUNT(*) as scans_per_file
                                        FROM ms_file_scans
                                        GROUP BY ms_file_label)
                                  """).fetchone()
            logger.info(f"  Total entries: {result[0]:,}")
            logger.info(f"  Total MS files: {result[1]:,}")
            logger.info(f"  Average scans per file: {result[2]:.0f}")
            logger.info(f"  Time elapsed: {elapsed:.2f}s")

    with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as conn:
        logger.info("Getting pending pairs...")
        start_time = time.time()
        conn.execute(QUERY_CREATE_PENDING_PAIRS, [use_bookmarked, use_for_optimization])
        rows = conn.execute("""
                            SELECT ms_type,
                                   COUNT(*)     AS total,
                                   MIN(pair_id) AS min_id,
                                   MAX(pair_id) AS max_id
                            FROM pending_pairs
                            GROUP BY ms_type
                            ORDER BY ms_type
                            """).fetchall()
        elapsed = time.time() - start_time

        if not rows:
            logger.info(f"No pending pairs ({elapsed:.2f}s)")
            conn.execute("DROP TABLE IF EXISTS pending_pairs")
            return {
                'total_pairs': 0,
                'processed': 0,
                'failed': 0,
                'batches': 0
            }

        global_total_pairs = sum(r[1] for r in rows)
        files_per_type = {}
        if set_progress:
            files_per_type = dict(conn.execute("""
                                               SELECT ms_type,
                                                      COUNT(DISTINCT ms_file_label) AS total_files
                                               FROM pending_pairs
                                               GROUP BY ms_type
                                               """).fetchall())

        _send_progress(
            set_progress,
            0,
            stage="Chromatograms",
            detail=f"Pending pairs: {global_total_pairs:,}",
        )



        logger.info(f"{global_total_pairs:,} pending pairs ({elapsed:.2f}s)")
        logger.info(f"Processing in batches of {batch_size}...")

        global_processed = 0  # accumulated counter
        global_stats: dict[str, dict] = {}

        for ms_type, total_pairs_type, min_id, max_id in rows:
            logger.info(f"--- Processing {ms_type} ---")
            logger.info(f"Pending pairs: {total_pairs_type:,} (pair_id {min_id}-{max_id})")

            if total_pairs_type == 0 or min_id is None or max_id is None:
                global_stats[ms_type] = {
                    'total_pairs': 0,
                    'processed': 0,
                    'failed': 0,
                    'batches': 0,
                }
                continue

            processed = 0
            failed = 0
            batches = 0
            processed_files: set[str] = set()

            current_id = min_id
            batch_num = 1
            total_batches = (total_pairs_type + batch_size - 1) // batch_size
            batches_since_checkpoint = 0
            total_files_type = files_per_type.get(ms_type, 0)

            with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as conn:
                # Process batches in a single connection; checkpoint periodically to avoid WAL stalls
                conn.execute("BEGIN TRANSACTION")

                while current_id <= max_id:
                    batch_count = 0
                    start_id = current_id
                    end_id = current_id + batch_size - 1

                    try:
                        batch_count = conn.execute("""
                                                   SELECT COUNT(*)
                                                   FROM pending_pairs
                                                   WHERE ms_type = ?
                                                     AND pair_id BETWEEN ? AND ?
                                                   """, [ms_type, start_id, end_id]).fetchone()[0]

                        if batch_count == 0:
                            current_id += batch_size
                            continue



                        batch_start = time.time()

                        if ms_type == 'ms1':
                            conn.execute(QUERY_PROCESS_BATCH_MS1, [start_id, end_id])
                        elif ms_type == 'ms2':
                            conn.execute(QUERY_PROCESS_BATCH_MS2, [start_id, end_id])

                        if set_progress:
                            batch_files = conn.execute("""
                                                       SELECT DISTINCT ms_file_label
                                                       FROM pending_pairs
                                                       WHERE ms_type = ?
                                                         AND pair_id BETWEEN ? AND ?
                                                       """, [ms_type, start_id, end_id]).fetchall()
                            processed_files.update(
                                row[0] for row in batch_files if row and row[0] is not None
                            )

                        batch_elapsed = time.time() - batch_start
                        processed += batch_count
                        batches += 1
                        batches_since_checkpoint += 1

                        logger.info(f"Batch {batch_num:>4}/{total_batches} | "
                                    f"IDs {start_id:>6}-{end_id:>6} | "
                                    f"{batch_count:>3} pairs | "
                                    f"Batch time: {batch_elapsed:>5.2f}s | "
                                    f"Progress {processed:>6,}/{total_pairs_type:,}")
                        log_line = (f"Batch {batch_num}/{total_batches} | "
                                    f"Progress {processed:,}/{total_pairs_type:,} | "
                                    f"Time/batch {batch_elapsed:0.2f}s")
                        _write_progress_log(progress_log, _center_lines(log_line))

                        if checkpoint_every and batches_since_checkpoint >= checkpoint_every:
                            conn.execute("COMMIT")
                            conn.execute("CHECKPOINT")
                            conn.execute("BEGIN TRANSACTION")
                            batches_since_checkpoint = 0

                        batch_num += 1

                    except Exception as e:
                        batch_elapsed = time.time() - batch_start if 'batch_start' in locals() else 0
                        failed += batch_count

                        failed += batch_count

                        logger.error(f"Error processing batch: {batch_elapsed:>5.2f}s | Error: {str(e)[:80]}")
                        error_line = (f"Batch {batch_num}/{total_batches} | "
                                      f"IDs {start_id}-{end_id} | "
                                      f"{batch_count} pairs | "
                                      f"Error: {str(e)}")
                        _write_progress_log(progress_log, _center_lines(error_line))

                        with open(f'failed_batches_{ms_type}.log', 'a') as f:
                            f.write(f"\n{'=' * 60}\n")
                            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write(f"ms_type: {ms_type}\n")
                            f.write(f"Batch {batch_num}/{total_batches}\n")
                            f.write(f"IDs: {start_id}-{end_id}\n")
                            f.write(f"Error: {str(e)}\n")
                            f.write(f"{'=' * 60}\n")

                        try:
                            conn.execute("ROLLBACK")
                            conn.execute("BEGIN TRANSACTION")
                            batches_since_checkpoint = 0
                        except Exception:
                            pass

                    finally:
                        # Update global progress even if the batch failed
                        if batch_count > 0:
                            global_processed += batch_count
                            progress_pct_global = (global_processed / global_total_pairs) * 100
                            files_done = len(processed_files) if set_progress else 0
                            detail_text = (
                                f"{log_line}"
                                if log_line else
                                f"{ms_type.upper()} batch {batch_num}/{total_batches} | "
                                f"Pairs {processed:,}/{total_pairs_type:,}"
                            )
                            _send_progress(
                                set_progress,
                                round(progress_pct_global, 1),
                                stage="Chromatograms",
                                detail=detail_text,
                            )

                    current_id += batch_size

                # Final checkpoint/commit for this ms_type
                conn.execute("COMMIT")
                conn.execute("CHECKPOINT")

    with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as conn:
        conn.execute("DROP TABLE IF EXISTS ms_file_scans")
        conn.execute("DROP TABLE IF EXISTS pending_pairs")


def compute_results_in_batches(wdir: str,
                               use_bookmarked: bool = False,
                               recompute: bool = False,
                               batch_size: int = 1000,
                               checkpoint_every: int = 20,
                               set_progress=None,
                               n_cpus=None,
                               ram=None):
    """
    Compute results with efficient macros.
    include_arrays=False: numeric metrics only (FAST)
    include_arrays=True: include scan_time and intensity arrays (SLOWER)
    """
    progress_log = Path(wdir) / "processing_progress.log" if wdir else None

    # INLINE macro - returns the table directly (AS TABLE)
    QUERY_CREATE_HELPERS = """
        CREATE OR REPLACE MACRO compute_chromatogram_metrics(scan_times, intensities, rt_min, rt_max) AS TABLE (
            WITH unnested_data AS (
                SELECT UNNEST(scan_times) AS scan_time,
                       UNNEST(intensities) AS intensity
            ),
            filtered_data AS (
                SELECT scan_time, intensity
                FROM unnested_data
                WHERE scan_time BETWEEN rt_min AND rt_max
            ),
            metrics AS (
                SELECT ROUND(SUM(intensity), 0) AS peak_area,
                       ROUND(MAX(intensity), 0) AS peak_max,
                       ROUND(MIN(intensity), 0) AS peak_min,
                       ROUND(AVG(intensity), 0) AS peak_mean,
                       ROUND(MEDIAN(intensity), 0) AS peak_median,
                       COUNT(*) AS peak_n_datapoints
                FROM filtered_data
            ),
            top3 AS (
                SELECT ROUND(AVG(intensity), 0) AS peak_area_top3
                FROM (
                    SELECT intensity
                    FROM filtered_data
                    ORDER BY intensity DESC
                    LIMIT 3
                )
            ),
            rt_of_max AS (
                SELECT scan_time AS peak_rt_of_max
                FROM filtered_data
                ORDER BY intensity DESC
                LIMIT 1
            ),
            arrays AS (
                SELECT LIST(scan_time ORDER BY scan_time) AS scan_time_list,
                       LIST(intensity ORDER BY scan_time) AS intensity_list
                FROM filtered_data
            )
            SELECT m.peak_area,
                   t3.peak_area_top3,
                   m.peak_max,
                   m.peak_min,
                   m.peak_mean,
                   rm.peak_rt_of_max,
                   m.peak_median,
                   m.peak_n_datapoints,
                   a.scan_time_list,
                   a.intensity_list
            FROM metrics m, top3 t3, rt_of_max rm, arrays a
        );
    """

    QUERY_CREATE_PENDING_PAIRS = """
                                 CREATE TABLE IF NOT EXISTS pending_result_pairs AS
                                 WITH pairs_to_process AS (
                                 SELECT c.peak_label,
                                                                  c.ms_file_label
                                                           FROM chromatograms c
                                                                    JOIN targets t ON c.peak_label = t.peak_label
                                                           WHERE CASE
                WHEN ? THEN c.peak_label IN (
                    SELECT peak_label FROM targets WHERE bookmark = TRUE
                )
                                                                     ELSE TRUE
                                                               END
                                                             AND (
                                                               ? OR NOT EXISTS (
                                                               SELECT 1 FROM results r
                                                                                WHERE r.peak_label = c.peak_label
                    AND r.ms_file_label = c.ms_file_label
                )
            )
        )
                                 SELECT peak_label,
                                        ms_file_label,
                                        ROW_NUMBER() OVER () AS pair_id
                                 FROM pairs_to_process
                                 ORDER BY peak_label, ms_file_label;
                                 """

    # Direct query - no intermediate CTE, streaming insert
    QUERY_PROCESS_BATCH = """
        INSERT INTO results (
            peak_label,
            ms_file_label,
            peak_area,
            peak_area_top3,
            peak_max,
            peak_min,
            peak_mean,
            peak_rt_of_max,
            peak_median,
            peak_n_datapoints,
            scan_time,
            intensity
        )
        WITH batch_pairs AS (
            SELECT peak_label, ms_file_label
            FROM pending_result_pairs
            WHERE pair_id BETWEEN ? AND ?
        )
        SELECT c.peak_label,
               c.ms_file_label,
               m.peak_area,
               m.peak_area_top3,
               m.peak_max,
               m.peak_min,
               m.peak_mean,
               m.peak_rt_of_max,
               m.peak_median,
               m.peak_n_datapoints,
               m.scan_time_list,
               m.intensity_list
                                            FROM chromatograms c
                                                     JOIN batch_pairs bp
                                                          ON c.peak_label = bp.peak_label
                                                              AND c.ms_file_label = bp.ms_file_label
        JOIN targets t ON c.peak_label = t.peak_label
        CROSS JOIN LATERAL compute_chromatogram_metrics(
            c.scan_time, 
            c.intensity, 
            t.rt_min, 
            t.rt_max
        ) AS m;
                          """

    if recompute:
        print("Deleting existing results for recalculation...")
        with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as con:
            con.execute("DELETE FROM results")

    # Create helper macro
    with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as conn:
        print("Creating helper macro...")
        conn.execute(QUERY_CREATE_HELPERS)

    # Create pending pairs table
    with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as conn:
        conn.execute("DROP TABLE IF EXISTS pending_result_pairs")

        print("Getting pending pairs...")
        start_time = time.time()
        conn.execute(QUERY_CREATE_PENDING_PAIRS, [use_bookmarked, recompute])

        total_pairs = conn.execute("""
            SELECT COUNT(*) AS total,
                                          MIN(pair_id) AS min_id,
                                          MAX(pair_id) AS max_id
                                   FROM pending_result_pairs
                                   """).fetchone()

        elapsed = time.time() - start_time

        if total_pairs[0] == 0 or total_pairs[1] is None:
            print(f"No pending pairs ({elapsed:.2f}s)")
            conn.execute("DROP TABLE IF EXISTS pending_result_pairs")
            return {'total_pairs': 0, 'processed': 0, 'failed': 0, 'batches': 0}

        total_count, min_id, max_id = total_pairs
        print(f"✓ {total_count:,} pending pairs ({elapsed:.2f}s)")
        print(f"Processing in batches of {batch_size}...\n")
        total_files = 0
        if set_progress:
            total_files = conn.execute("""
                                       SELECT COUNT(DISTINCT ms_file_label)
                                       FROM pending_result_pairs
                                       """).fetchone()[0]
        _send_progress(
            set_progress,
            0,
            stage="Results",
            detail=f"Pending pairs: {total_count:,}",
        )

    # Process in batches
    processed = 0
    failed = 0
    batches = 0
    processed_files: set[str] = set()

    current_id = min_id
    batch_num = 1
    total_batches = (total_count + batch_size - 1) // batch_size

    with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as conn:
        # Settings for bulk writes
        conn.execute("SET wal_autocheckpoint='1GB'")
        conn.execute("BEGIN TRANSACTION")

        batches_in_txn = 0

        while current_id <= max_id:
            start_id = current_id
            end_id = current_id + batch_size - 1

            try:
                batch_count = conn.execute("""
                                           SELECT COUNT(*)
                                           FROM pending_result_pairs
                                           WHERE pair_id BETWEEN ? AND ?
                                           """, [start_id, end_id]).fetchone()[0]

                if batch_count == 0:
                    current_id += batch_size
                    continue

                print(f"Batch {batch_num:>4}/{total_batches} | "
                      f"IDs {start_id:>6}-{end_id:>6} | "
                      f"{batch_count:>4} pairs | ", end='', flush=True)

                batch_start = time.time()

                conn.execute(QUERY_PROCESS_BATCH, [start_id, end_id])

                if set_progress:
                    batch_files = conn.execute("""
                                               SELECT DISTINCT ms_file_label
                                               FROM pending_result_pairs
                                               WHERE pair_id BETWEEN ? AND ?
                                               """, [start_id, end_id]).fetchall()
                    processed_files.update(
                        row[0] for row in batch_files if row and row[0] is not None
                    )

                batch_elapsed = time.time() - batch_start
                processed += batch_count
                batches += 1
                batches_in_txn += 1

                pairs_per_sec = batch_count / batch_elapsed
                print(f"✓ {batch_elapsed:>5.2f}s ({pairs_per_sec:>5.1f} pairs/s) | "
                      f"Progress {processed:>6,}/{total_count:,}")
                log_line = (f"Batch {batch_num}/{total_batches} | "
                            f"Progress {processed:,}/{total_count:,} | "
                            f"Time/batch {batch_elapsed:0.2f}s"
                            # f"Processing ({pairs_per_sec:0.1f} pairs/s)"
                            )
                _write_progress_log(progress_log, _center_lines(log_line))

                # Periodic checkpoint
                if batches_in_txn >= checkpoint_every:
                    print(f"  [Commit + Checkpoint]...", end='', flush=True)
                    flush_start = time.time()
                    conn.execute("COMMIT")
                    conn.execute("CHECKPOINT")
                    conn.execute("BEGIN TRANSACTION")
                    print(f" {time.time() - flush_start:.2f}s")
                    batches_in_txn = 0

                batch_num += 1

            except Exception as e:
                batch_elapsed = time.time() - batch_start if 'batch_start' in locals() else 0
                failed += batch_count if 'batch_count' in locals() else 0

                print(f"✗ {batch_elapsed:>5.2f}s | Error: {str(e)[:80]}")
                error_line = (f"RESULTS batch {batch_num}/{total_batches} | "
                              f"IDs {start_id}-{end_id} | "
                              f"{batch_count if 'batch_count' in locals() else 0} pairs | "
                              f"Error: {str(e)}")
                _write_progress_log(progress_log, _center_lines(error_line))

                with open('failed_batches_results.log', 'a') as f:
                    f.write(f"\n{'=' * 60}\n")
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Batch {batch_num}/{total_batches}\n")
                    f.write(f"IDs: {start_id}-{end_id}\n")
                    f.write(f"Error: {str(e)}\n")
                    f.write(f"{'=' * 60}\n")

            finally:
                if 'batch_count' in locals() and batch_count > 0:
                    progress_pct = (processed / total_count) * 100
                    files_done = len(processed_files) if set_progress else 0
                    detail_text = (
                        f"{log_line}"
                        if log_line else
                        f"Results batch {batch_num}/{total_batches} | "
                        f"Pairs {processed:,}/{total_count:,}"
                    )
                    _send_progress(
                        set_progress,
                        round(progress_pct, 1),
                        stage="Results",
                        detail=detail_text,
                    )

            current_id += batch_size

        # Commit final
        print("\n[Final commit + checkpoint]...", end='', flush=True)
        flush_start = time.time()
        conn.execute("COMMIT")
        conn.execute("CHECKPOINT")
        print(f" {time.time() - flush_start:.2f}s")

    # Clean up
    with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as conn:
        conn.execute("DROP TABLE IF EXISTS pending_result_pairs")

    print(f"\n{'=' * 60}")
    print(f"Summary:")
    print(f"  Total pairs: {total_count:,}")
    print(f"  Processed: {processed:,}")
    print(f"  Failed: {failed:,}")
    print(f"  Batches: {batches:,}")
    print(f"{'=' * 60}")

    return {
        'total_pairs': total_count,
        'processed': processed,
        'failed': failed,
        'batches': batches
    }

def compute_peak_properties(con: duckdb.DuckDBPyConnection,
                            set_progress=None,
                            recompute=False,
                            bookmarked=False
                            ):
    if recompute:
        print("Deleting existing results for recalculation...")
        con.execute("DELETE FROM results")

    query = """
            INSERT INTO results (peak_label,
                                 ms_file_label,
                                 total_intensity,
                                 peak_area,
                                 peak_area_top3,
                                 peak_max,
                                 peak_min,
                                 peak_mean,
                                 peak_rt_of_max,
                                 peak_median,
                                 peak_n_datapoints,
                                 scan_time,
                                 intensity)
            WITH pairs_to_process AS (SELECT c.peak_label,
                                             c.ms_file_label
                                      FROM chromatograms c
                                               JOIN targets t ON c.peak_label = t.peak_label
                                      WHERE c.ms_file_label IN
                                            (SELECT ms_file_label FROM samples WHERE use_for_processing = TRUE)
                                        AND t.rt_min IS NOT NULL
                                        AND t.rt_max IS NOT NULL
                                        AND CASE
                                                WHEN ? -- bookmarked
                                                    THEN c.peak_label IN (SELECT peak_label FROM targets WHERE bookmark = TRUE)
                                                ELSE TRUE
                                          END
                                        AND (
                                          ? -- recompute
                                              OR NOT EXISTS (SELECT 1
                                                             FROM results r
                                                             WHERE r.peak_label = c.peak_label
                                                               AND r.ms_file_label = c.ms_file_label)
                                          )),
                 unnested AS (SELECT c.peak_label,
                                     c.ms_file_label,
                                     UNNEST(c.scan_time) AS scan_time,
                                     UNNEST(c.intensity) AS intensity
                              FROM chromatograms c
                                       JOIN pairs_to_process p ON c.peak_label = p.peak_label
                                  AND c.ms_file_label = p.ms_file_label),
-- Compute total_intensity (without rt filter)
                 total_stats AS (SELECT peak_label,
                                        ms_file_label,
                                        SUM(intensity) AS total_intensity
                                 FROM unnested
                                 GROUP BY peak_label, ms_file_label),
-- Filter by rt_min - rt_max window
                 filtered_range AS (SELECT u.peak_label,
                                           u.ms_file_label,
                                           u.scan_time,
                                           u.intensity
                                    FROM unnested u
                                             JOIN targets t ON u.peak_label = t.peak_label
                                    WHERE u.scan_time BETWEEN t.rt_min AND t.rt_max),
-- Group the filtered data into lists
                 aggregated AS (SELECT peak_label,
                                       ms_file_label,
                                       LIST(scan_time ORDER BY scan_time) AS scan_time,
                                       LIST(intensity ORDER BY scan_time) AS intensity,
                                       ROUND(SUM(intensity), 0)           AS peak_area,
                                       ROUND(MAX(intensity), 0)           AS peak_max,
                                       ROUND(MIN(intensity), 0)           AS peak_min,
                                       ROUND(AVG(intensity), 0)           AS peak_mean,
                                       ROUND(MEDIAN(intensity), 0)        AS peak_median,
                                       COUNT(*)                           AS peak_n_datapoints
                                FROM filtered_range
                                GROUP BY peak_label, ms_file_label),
-- Compute peak_area_top3
                 top3_calc AS (SELECT peak_label,
                                      ms_file_label,
                                      ROUND(AVG(intensity), 0) AS peak_area_top3
                               FROM (SELECT peak_label,
                                            ms_file_label,
                                            intensity,
                                            ROW_NUMBER() OVER (PARTITION BY peak_label, ms_file_label ORDER BY intensity DESC) AS rn
                                     FROM filtered_range) sub
                               WHERE rn <= 3
                               GROUP BY peak_label, ms_file_label),
-- Find scan_time of peak_max
                 rt_of_max AS (SELECT peak_label,
                                      ms_file_label,
                                      scan_time AS peak_rt_of_max
                               FROM (SELECT peak_label,
                                            ms_file_label,
                                            scan_time,
                                            intensity,
                                            ROW_NUMBER() OVER (PARTITION BY peak_label, ms_file_label ORDER BY intensity DESC) AS rn
                                     FROM filtered_range) sub
                               WHERE rn = 1)
            SELECT a.peak_label,
                   a.ms_file_label,
                   ts.total_intensity,
                   a.peak_area,
                   t3.peak_area_top3,
                   a.peak_max,
                   a.peak_min,
                   a.peak_mean,
                   rm.peak_rt_of_max,
                   a.peak_median,
                   a.peak_n_datapoints,
                   a.scan_time,
                   a.intensity
            FROM aggregated a
                     JOIN total_stats ts ON a.peak_label = ts.peak_label AND a.ms_file_label = ts.ms_file_label
                     JOIN top3_calc t3 ON a.peak_label = t3.peak_label AND a.ms_file_label = t3.ms_file_label
                     JOIN rt_of_max rm ON a.peak_label = rm.peak_label AND a.ms_file_label = rm.ms_file_label
            ORDER BY a.peak_label, a.ms_file_label;
            """

    # Shared variable to accumulate progress
    accumulated_progress = [0.0]
    stop_monitoring = [False]

    def monitor_progress():
        """Monitor progress of the current query"""
        while not stop_monitoring[0]:
            try:
                qp = con.query_progress()
                if qp != -1 and qp > 0:
                    # Total progress depends on which query is executing
                    total_progress = qp

                    accumulated_progress[0] = total_progress
                    if set_progress:
                        set_progress(round(total_progress, 1))
                    print(f"Progress: {total_progress:.1f}%")
                time.sleep(0.05)

            except (duckdb.InvalidInputException, duckdb.ConnectionException):
                break
            except Exception as e:
                print(f"Progress monitoring error: {e}")
                break

    # Start monitoring
    if set_progress:
        progress_thread = Thread(target=monitor_progress, daemon=True)
        progress_thread.start()

    try:
        # Run MS1
        print("Processing MS1 chromatograms...")
        con.execute(query, [bookmarked, recompute])
        accumulated_progress[0] = 100
        if set_progress:
            set_progress(round(accumulated_progress[0], 1))

        print("Chromatograms computed and inserted into DuckDB.")

    finally:
        stop_monitoring[0] = True
        if set_progress:
            progress_thread.join(timeout=0.5)

    print("Peak properties computed and inserted into DuckDB.")


def create_pivot(conn, rows=None, cols=None, value='peak_area', table='results'):
    """
    Create pivot from DuckDB for unique per-pair data
    """

    ordered_pl = conn.execute("""
                              SELECT DISTINCT r.peak_label
                              FROM results r
                                       JOIN targets t ON r.peak_label = t.peak_label
                              ORDER BY t.ms_type""").df()

    group_cols_sql = ",\n                ".join([f"s.{col}" for col in GROUP_COLUMNS])

    query = f"""
        PIVOT (
            SELECT
                s.ms_type,
                s.sample_type,
                {group_cols_sql},
                r.ms_file_label,
                r.peak_label,
                r.{value}
            FROM results r
            JOIN samples s ON s.ms_file_label = r.ms_file_label
            WHERE s.use_for_analysis = TRUE
            ORDER BY s.ms_type, r.peak_label
        )
        ON peak_label
        USING FIRST({value})
        -- GROUP BY ms_type
        ORDER BY ms_type
    """
    df = conn.execute(query).df()
    meta_cols = ['ms_type', 'sample_type', *GROUP_COLUMNS, 'ms_file_label']
    keep_cols = [col for col in meta_cols if col in df.columns] + ordered_pl['peak_label'].to_list()
    return df[keep_cols]


def compute_and_insert_chromatograms_iteratively(con: duckdb.DuckDBPyConnection, set_progress=None):
    """
    Computes and inserts chromatograms iteratively by batching targets.

    :param con: An active DuckDB connection.
    :param set_progress: A callback function to update the progress bar.
    """
    targets_df = con.execute("SELECT peak_label, ms_type FROM targets").df()
    ms_files_count = con.execute("SELECT count(*) FROM samples WHERE use_for_optimization = TRUE").fetchone()[0]

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
            JOIN samples AS s ON m.ms_file_label = s.ms_file_label
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
