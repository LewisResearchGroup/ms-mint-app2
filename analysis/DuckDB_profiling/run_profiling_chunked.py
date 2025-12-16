"""
Chunked ms1 profiling runner: processes a pair_id range in file-sized chunks to reduce ms1_data scans.
- Supports baseline or pruned ms1 query.
- Aggregates results into a temp table and can dump them for comparison.
- Measures total wall time across chunks (no DuckDB JSON profile per chunk).

Example:
  python run_profiling_chunked.py --db workspace_mint.db --start-id 1 --end-id 10000 \
    --chunk-size 20 --variant pruned --prepare-pending ms --use-for-optimization --include-existing-pairs \
    --dump-results ms1_1-10000_chunked.parquet
"""

import argparse
import os
import time
from pathlib import Path
from typing import List

import duckdb

MS1_QUERY_BASELINE = """
INSERT INTO {target_table} (peak_label, ms_file_label, scan_time, intensity, ms_type)
WITH batch_pairs AS (SELECT peak_label, ms_file_label, ms_type, mz_mean, mz_width
                     FROM pending_pairs
                     WHERE ms_type = 'ms1'
                       AND pair_id BETWEEN ? AND ?
                       AND ms_file_label IN (SELECT ms_file_label FROM chunk_files)),
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
     all_scans_needed AS (SELECT DISTINCT bp.peak_label,
                                          bp.ms_file_label,
                                          s.scan_id,
                                          s.scan_time
                          FROM batch_pairs bp
                                   JOIN ms_file_scans s ON s.ms_file_label = bp.ms_file_label),
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

MS1_QUERY_PRUNED = """
INSERT INTO {target_table} (peak_label, ms_file_label, scan_time, intensity, ms_type)
WITH batch_pairs AS (
    SELECT peak_label, ms_file_label, ms_type, mz_mean, mz_width
    FROM pending_pairs
    WHERE ms_type = 'ms1'
      AND pair_id BETWEEN ? AND ?
      AND ms_file_label IN (SELECT ms_file_label FROM chunk_files)
),
     batch_files AS (
         SELECT DISTINCT ms_file_label FROM batch_pairs
     ),
     ms1_slice AS (
         SELECT * FROM ms1_data WHERE ms_file_label IN (SELECT ms_file_label FROM batch_files)
     ),
     matched_intensities AS (
         SELECT bp.peak_label,
                bp.ms_file_label,
                ms1.scan_id,
                MAX(ms1.intensity) AS intensity
         FROM batch_pairs bp
         JOIN ms1_slice ms1
           ON ms1.ms_file_label = bp.ms_file_label
          AND ms1.mz BETWEEN bp.mz_mean - (bp.mz_mean * bp.mz_width / 1e6)
                          AND bp.mz_mean + (bp.mz_mean * bp.mz_width / 1e6)
         GROUP BY bp.peak_label, bp.ms_file_label, ms1.scan_id
     ),
     all_scans_needed AS (
         SELECT DISTINCT bp.peak_label,
                         bp.ms_file_label,
                         s.scan_id,
                         s.scan_time
         FROM batch_pairs bp
         JOIN ms_file_scans s ON s.ms_file_label = bp.ms_file_label
     ),
     complete_data AS (
         SELECT a.peak_label,
                a.ms_file_label,
                a.scan_time,
                a.scan_id,
                a.scan_time,
                COALESCE(ROUND(m.intensity, 0), 1) AS intensity
         FROM all_scans_needed a
         LEFT JOIN matched_intensities m
           ON a.peak_label = m.peak_label
          AND a.ms_file_label = m.ms_file_label
          AND a.scan_id = m.scan_id
     ),
     agg AS (
         SELECT peak_label,
                ms_file_label,
                LIST(scan_time ORDER BY scan_time) AS scan_time,
                LIST(intensity ORDER BY scan_time) AS intensity
         FROM complete_data
         GROUP BY peak_label, ms_file_label
     )
SELECT peak_label, ms_file_label, scan_time, intensity, 'ms1' AS ms_type
FROM agg;
"""

PENDING_PAIRS_QUERY = """
CREATE OR REPLACE TABLE pending_pairs AS
WITH target_filter AS (
    SELECT peak_label, ms_type, mz_mean, mz_width, filterLine, bookmark
    FROM targets t
    WHERE (
        t.peak_selection IS TRUE
        OR NOT EXISTS (
            SELECT 1 FROM targets t1
            WHERE t1.peak_selection IS TRUE AND t1.ms_type = t.ms_type
        )
    )
    AND (
        CASE WHEN ? THEN t.bookmark IS TRUE ELSE TRUE END
    )
),
 sample_filter AS (
    SELECT ms_file_label
    FROM samples
    WHERE (CASE WHEN ? THEN use_for_optimization ELSE use_for_analysis END) = TRUE
),
 existing_pairs AS (
    SELECT DISTINCT peak_label, ms_file_label, ms_type FROM chromatograms
),
 all_possible_pairs AS (
    SELECT t.peak_label, s.ms_file_label, t.ms_type, t.mz_mean, t.mz_width, t.filterLine
    FROM target_filter t
    CROSS JOIN sample_filter s
),
 pending AS (
    SELECT a.peak_label, a.ms_file_label, a.ms_type, a.mz_mean, a.mz_width, a.filterLine,
           ROW_NUMBER() OVER () AS pair_id
    FROM all_possible_pairs a
    LEFT JOIN existing_pairs e
      ON a.peak_label = e.peak_label
     AND a.ms_file_label = e.ms_file_label
     AND a.ms_type = e.ms_type
    WHERE (? OR e.peak_label IS NULL)
)
SELECT pair_id, peak_label, ms_file_label, ms_type, mz_mean, mz_width, filterLine
FROM pending
ORDER BY pair_id;
"""

def prepare_pending_pairs(con: duckdb.DuckDBPyConnection, use_bookmarked: bool, use_for_optimization: bool, include_existing: bool) -> int:
    con.execute(PENDING_PAIRS_QUERY, [use_bookmarked, use_for_optimization, include_existing])
    return con.execute("SELECT COUNT(*) FROM pending_pairs").fetchone()[0]

def get_batch_files(con: duckdb.DuckDBPyConnection, start_id: int, end_id: int) -> List[str]:
    rows = con.execute(
        "SELECT DISTINCT ms_file_label FROM pending_pairs WHERE ms_type='ms1' AND pair_id BETWEEN ? AND ? ORDER BY ms_file_label",
        [start_id, end_id],
    ).fetchall()
    return [r[0] for r in rows]

def parse_args():
    p = argparse.ArgumentParser(description="Chunked ms1 batch runner to reduce ms1_data scans")
    p.add_argument("--db", required=True, help="Path to DuckDB database file")
    p.add_argument("--start-id", type=int, required=True)
    p.add_argument("--end-id", type=int, required=True)
    p.add_argument("--chunk-size", type=int, default=20, help="Number of ms_file_labels per chunk")
    p.add_argument("--variant", choices=["baseline", "pruned"], default="pruned", help="Which ms1 query variant to use")
    p.add_argument("--prepare-pending", choices=["none", "ms"], default="none",
                   help="Optionally (re)create pending_pairs before profiling.")
    p.add_argument("--use-bookmarked", action="store_true")
    p.add_argument("--use-for-optimization", action="store_true")
    p.add_argument("--include-existing-pairs", action="store_true")
    p.add_argument("--temp-insert", action="store_true", help="Insert into temp table (recommended).")
    p.add_argument("--dump-results", default=None, help="Path to dump aggregated results (parquet/csv/json)")
    p.add_argument("--dump-format", choices=["parquet", "csv", "json"], default="csv")
    p.add_argument("--threads", type=int, default=min(8, os.cpu_count()), help="Set DuckDB threads (PRAGMA threads)")
    p.add_argument("--memory-limit", type=str, default="16GB", help="Set DuckDB memory_limit (e.g., 8GB)")
    return p.parse_args()

def chunked_insert(con: duckdb.DuckDBPyConnection, files: List[str], sql_template: str, start_id: int, end_id: int, target_table: str, chunk_size: int) -> float:
    # target_table exists and is empty temp table
    total_start = time.time()
    for i in range(0, len(files), chunk_size):
        chunk = files[i:i + chunk_size]
        con.execute("DROP TABLE IF EXISTS chunk_files")
        con.execute("CREATE TEMP TABLE chunk_files (ms_file_label VARCHAR)")
        con.executemany("INSERT INTO chunk_files VALUES (?)", [(f,) for f in chunk])
        sql = sql_template.format(target_table=target_table)
        con.execute(sql, [start_id, end_id])
    return time.time() - total_start

if __name__ == "__main__":
    args = parse_args()
    db_path = Path(args.db).expanduser()
    con = duckdb.connect(str(db_path))

    try:
        if args.threads is not None:
            con.execute(f"SET threads={args.threads}")
            print(f"Set threads={args.threads}")
        if args.memory_limit not in (None, ""):
            con.execute(f"SET memory_limit='{args.memory_limit}'")
            print(f"Set memory_limit={args.memory_limit}")

        if args.prepare_pending == "ms":
            cnt = prepare_pending_pairs(con, args.use_bookmarked, args.use_for_optimization, args.include_existing_pairs)
            print(f"Prepared pending_pairs with {cnt} rows")

        files = get_batch_files(con, args.start_id, args.end_id)
        print(f"Chunking {len(files)} ms_file_labels into size {args.chunk_size}")

        # Prepare target temp table
        target_base = "chromatograms"
        target_temp = f"temp_{target_base}"
        if args.temp_insert:
            con.execute(f"CREATE TEMP TABLE IF NOT EXISTS {target_temp} AS SELECT * FROM {target_base} WHERE 0")
            target_table = target_temp
        else:
            target_table = target_base

        sql_template = MS1_QUERY_PRUNED if args.variant == "pruned" else MS1_QUERY_BASELINE

        elapsed = chunked_insert(con, files, sql_template, args.start_id, args.end_id, target_table, args.chunk_size)

        if args.dump_results:
            fmt = args.dump_format.upper()
            con.execute(f"COPY (SELECT * FROM {target_table}) TO '{args.dump_results}' (FORMAT {fmt})")

    finally:
        con.close()

    print(f"Chunked run finished in {elapsed:.2f}s")
    if args.dump_results:
        print(f"Results dumped to {args.dump_results} (format={args.dump_format})")
