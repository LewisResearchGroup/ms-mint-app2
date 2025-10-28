import time
from threading import Thread

import duckdb
from pathlib import Path
from contextlib import contextmanager

@contextmanager
def duckdb_connection(workspace_path: Path | str, register_activity=True):
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
        con.execute("SET enable_progress_bar = true")
        con.execute("SET enable_progress_bar_print = false")
        con.execute("SET progress_bar_time = 0")
        _create_tables(con)
        yield con
    except Exception as e:
        print(f"Error connecting to DuckDB: {e}")
        yield None
    finally:
        if con:
            if register_activity:
                try:
                    with duckdb_connection_mint(workspace_path.parent.parent) as mint_conn:
                        mint_conn.execute("UPDATE workspaces SET last_activity = NOW() WHERE name = ?",
                                          [Path(workspace_path).stem])
                except Exception as e:
                    print(f"Error updating workspace activity: {e}")
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
        yield con
    except Exception as e:
        print(f"Error connecting to DuckDB: {e}")
        yield None
    finally:
        if con and workspace:
            con.execute("UPDATE workspaces SET last_activity = NOW() WHERE key = ?", [workspace])
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
                     use_for_optimization BOOLEAN DEFAULT false,
                     use_for_analysis     BOOLEAN DEFAULT true,
                     polarity             polarity_enum,
                     color                VARCHAR DEFAULT '#ffffff',
                     label                VARCHAR,
                     sample_type          VARCHAR DEFAULT 'Unset',
                     run_order            INTEGER,
                     plate                VARCHAR,
                     plate_row            VARCHAR,
                     plate_column         TINYINT
                 );
                 """)

    conn.execute("""
                 CREATE TABLE IF NOT EXISTS ms1_data
                 (
                     ms_file_label      VARCHAR,  -- Label of the MS file, linking to samples
                     scan_id            INTEGER, -- Scan ID
                     mz                 DOUBLE,  -- Mass-to-charge ratio
                     intensity          DOUBLE,  -- Intensity
                     scan_time          DOUBLE  -- Scan time
                 );
                 """)
    conn.execute("""
                 CREATE TABLE IF NOT EXISTS ms2_data
                 (
                     ms_file_label      VARCHAR,  -- Label of the MS file, linking to samples
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
        # múltiple selección (IN)
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
    tie: tuple[str, str] | None = None,   # p.ej. ("id", "ASC"); se usa SOLO si hay sorter
    nocase_text: bool = True
) -> str:
    """
    Devuelve 'ORDER BY ...' o '' si no hay sorter válido.
    - sorter: {'columns': [...], 'orders': ['ascend'|'descend', ...]}
    - column_types: mapa {col -> tipo DUCKDB} (de DESCRIBE)
    - tie: (col, dir) opcional; se agrega SOLO si hay al menos 1 columna ordenable en sorter
    """
    # 0) Normaliza entrada
    cols_in = (sorter or {}).get("columns") or []
    ords_in = (sorter or {}).get("orders") or []
    if not cols_in:
        return ""  # sin sorter => sin ORDER BY

    # Rellena órdenes faltantes
    if len(ords_in) < len(cols_in):
        ords_in = ords_in + ["ascend"] * (len(cols_in) - len(ords_in))

    order_map = {"ascend": "ASC", "descend": "DESC"}
    parts: list[str] = []
    used_cols: set[str] = set()

    for col, ord_ in zip(cols_in, ords_in):
        if col not in column_types:
            continue  # ignora columnas inválidas
        direction = order_map.get(ord_, "ASC")
        nulls = "NULLS LAST" if direction == "ASC" else "NULLS FIRST"
        ctype = (column_types.get(col) or "").upper()
        is_text = any(t in ctype for t in ("CHAR", "VARCHAR", "TEXT", "STRING"))
        expr = f'"{col}" COLLATE NOCASE' if (nocase_text and is_text) else f'"{col}"'
        parts.append(f"{expr} {direction} {nulls}")
        used_cols.add(col)

    # Si tras validar no quedó nada, no ordenes (y tampoco agregues tie)
    if not parts:
        return ""

    # Agrega tie SOLO si hay sorter válido y tie se pidió y no repite columna
    if tie:
        tie_col, tie_dir = tie[0], tie[1].upper()
        if tie_col not in used_cols and tie_col in column_types:
            parts.append(f'"{tie_col}" {tie_dir}')

    return f"ORDER BY {', '.join(parts)}"


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
                                                  OR use_for_analysis = TRUE),
                            ms1_targets AS (SELECT DISTINCT t.peak_label, s.ms_file_label
                                            FROM targets t
                                                     CROSS JOIN samples_to_use s
                                            WHERE t.mz_mean IS NOT NULL
                                              AND t.mz_width IS NOT NULL
                                              AND EXISTS(SELECT 1 FROM ms1_data md WHERE md.ms_file_label = s.ms_file_label)),
                            ms2_targets AS (SELECT DISTINCT t.peak_label, s.ms_file_label
                                            FROM targets t
                                                     CROSS JOIN samples_to_use s
                                            WHERE t.filterLine IS NOT NULL -- ESTO asegura que es MS2
                                              AND EXISTS(SELECT 1 FROM ms2_data md WHERE md.ms_file_label = s.ms_file_label)),
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

    # Determinar qué procesar
    process_ms1 = ms1_total > 0 and (recompute_ms1 or ms1_missing > 0)
    process_ms2 = ms2_total > 0 and (recompute_ms2 or ms2_missing > 0)

    # Logging informativo
    if ms1_total > 0:
        print(f"MS1: {ms1_existing} existing, {ms1_missing} missing (total: {ms1_total})")
    if ms2_total > 0:
        print(f"MS2: {ms2_existing} existing, {ms2_missing} missing (total: {ms2_total})")

    if not process_ms1 and not process_ms2:
        print("No chromatograms to process.")
        return

    # Eliminar chromatogramas existentes si se solicita recalcular
    if recompute_ms1 and ms1_existing > 0:
        print(f"Deleting {ms1_existing} existing MS1 chromatograms for recalculation...")
        con.execute("""
                    DELETE
                    FROM chromatograms
                    WHERE EXISTS(SELECT 1
                                 FROM targets t
                                          CROSS JOIN samples s
                                 WHERE s.use_for_optimization = TRUE OR s.use_for_analysis = TRUE
                                   AND t.mz_mean IS NOT NULL
                                   AND t.mz_width IS NOT NULL
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
                                 WHERE s.use_for_optimization = TRUE OR s.use_for_analysis = TRUE
                                   AND t.filterLine IS NOT NULL
                                   AND chromatograms.peak_label = t.peak_label
                                   AND chromatograms.ms_file_label = s.ms_file_label)
                    """)
        ms2_to_compute = ms2_total
    else:
        ms2_to_compute = ms2_missing

    # Calcular pesos para el progreso
    total_to_compute = ms1_to_compute + ms2_to_compute
    ms1_weight = ms1_to_compute / total_to_compute if process_ms1 else 0
    ms2_weight = ms2_to_compute / total_to_compute if process_ms2 else 0

    print(f"Computing {ms1_to_compute} MS1 and {ms2_to_compute} MS2 chromatograms...")

    query_ms1 = """
                INSERT INTO chromatograms (peak_label, ms_file_label, scan_time, intensity)
                WITH pairs_to_process AS (SELECT t.peak_label,
                                                 t.mz_mean,
                                                 t.mz_width,
                                                 s.ms_file_label
                                          FROM targets t
                                                   JOIN samples s
                                                        ON (CASE WHEN ? THEN s.use_for_optimization ELSE s.use_for_analysis END) =
                                                           TRUE
                                          WHERE t.mz_mean IS NOT NULL
                                            AND t.mz_width IS NOT NULL
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
                                                       AND p.mz_mean + (p.mz_mean * p.mz_width / 1e6)
                                      QUALIFY
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
                SELECT *
                FROM agg;
                """

    query_ms2 = """
                INSERT INTO chromatograms (peak_label, ms_file_label, scan_time, intensity)
                WITH pairs_to_process AS (SELECT t.peak_label,
                                                 t.filterLine,
                                                 s.ms_file_label
                                          FROM targets AS t
                                                   JOIN samples s
                                                        ON (CASE WHEN ? THEN s.use_for_optimization ELSE s.use_for_analysis END) =
                                                           TRUE
                                          WHERE t.filterLine IS NOT NULL
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
                                        MAX(intensity) AS intensity -- max por tiempo
                                 -- AVG(mz_mean)   AS mz_mean    -- estable dentro del bin
                                 FROM pre
                                 GROUP BY peak_label, ms_file_label, scan_time),
                     aggregated_chromatograms AS (SELECT peak_label,
                                                         ms_file_label,
                                                         LIST(scan_time ORDER BY scan_time) AS scan_time,
                                                         LIST(intensity ORDER BY scan_time) AS intensity
                                                  -- list(mz ORDER BY scan_time) AS mz
                                                  FROM grouped
                                                  GROUP BY peak_label, ms_file_label)
                SELECT *
                FROM aggregated_chromatograms;
                """

    # Variable compartida para acumular progreso
    accumulated_progress = [0.0]
    stop_monitoring = [False]
    current_query_type = ['ms1']  # Para saber qué estamos procesando

    def monitor_progress():
        """Monitorea el progreso de la query actual"""
        while not stop_monitoring[0]:
            try:
                qp = con.query_progress()
                if qp != -1 and qp > 0:
                    # Progreso total según qué query estamos ejecutando
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

    # Iniciar monitoreo
    if set_progress:
        progress_thread = Thread(target=monitor_progress, daemon=True)
        progress_thread.start()

    try:
        # Ejecutar MS1
        if process_ms1:
            print("Processing MS1 chromatograms...")
            current_query_type[0] = 'ms1'
            con.execute(query_ms1, [for_optimization, recompute_ms1])
            accumulated_progress[0] = ms1_weight * 100
            if set_progress:
                set_progress(round(accumulated_progress[0], 1))

        # Ejecutar MS2
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