"""
Profiling runner that materializes a per-run ms1_data slice (by ms_file_label) to reduce scans.
Keeps other scripts intact. Supports baseline/pruned ms1 queries, profiling JSON, optional result dump.
"""

import argparse
import os
import time
from pathlib import Path
from typing import List

import duckdb

MS1_QUERY_TEMPLATE_BASELINE = """
INSERT INTO {target_table} (peak_label, ms_file_label, scan_time, intensity, ms_type)
WITH batch_pairs AS (SELECT peak_label, ms_file_label, ms_type, mz_mean, mz_width
                     FROM pending_pairs
                     WHERE ms_type = 'ms1'
                       AND pair_id BETWEEN ? AND ?),
     matched_intensities AS (SELECT bp.peak_label,
                                    bp.ms_file_label,
                                    ms1.scan_id,
                                    MAX(ms1.intensity) AS intensity
                             FROM batch_pairs bp
                                      JOIN {ms1_table} ms1
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

MS1_QUERY_TEMPLATE_PRUNED = """
INSERT INTO {target_table} (peak_label, ms_file_label, scan_time, intensity, ms_type)
WITH batch_pairs AS (
    SELECT peak_label, ms_file_label, ms_type, mz_mean, mz_width
    FROM pending_pairs
    WHERE ms_type = 'ms1'
      AND pair_id BETWEEN ? AND ?
),
     batch_files AS (
         SELECT DISTINCT ms_file_label FROM batch_pairs
     ),
     ms1_slice AS (
         SELECT * FROM {ms1_table} WHERE ms_file_label IN (SELECT ms_file_label FROM batch_files)
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

HELPER_MACRO = """
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

RESULTS_QUERY_TEMPLATE = """
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

MS2_QUERY = """
INSERT INTO chromatograms (peak_label, ms_file_label, scan_time, intensity, ms_type)
WITH batch_pairs AS (SELECT peak_label, ms_file_label, ms_type, filterLine
                     FROM pending_pairs
                     WHERE ms_type = 'ms2'
                       AND pair_id BETWEEN ? AND ?),
     matched_filterline AS (SELECT bp.peak_label,
                                   bp.ms_file_label,
                                   ms2.scan_id,
                                   ms2.intensity
                            FROM batch_pairs bp
                                     JOIN ms2_data ms2
                                          ON ms2.ms_file_label = bp.ms_file_label
                                              AND ms2.filterLine = bp.filterLine),
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
                                LEFT JOIN matched_filterline m
                                          ON a.peak_label = m.peak_label
                                              AND a.ms_file_label = m.ms_file_label
                                              AND a.scan_id = m.scan_id),
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

PENDING_RESULT_PAIRS_QUERY = """
CREATE OR REPLACE TABLE pending_result_pairs AS
WITH pairs_to_process AS (
    SELECT c.peak_label, c.ms_file_label
    FROM chromatograms c
    JOIN targets t ON c.peak_label = t.peak_label
    WHERE CASE
        WHEN ? THEN c.peak_label IN (SELECT peak_label FROM targets WHERE bookmark = TRUE)
        ELSE TRUE
    END
    AND (
        ? OR NOT EXISTS (
            SELECT 1
            FROM results r
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


def prepare_pending_pairs(con: duckdb.DuckDBPyConnection, use_bookmarked: bool, use_for_optimization: bool, include_existing: bool) -> int:
    con.execute(PENDING_PAIRS_QUERY, [use_bookmarked, use_for_optimization, include_existing])
    return con.execute("SELECT COUNT(*) FROM pending_pairs").fetchone()[0]


def prepare_pending_result_pairs(con: duckdb.DuckDBPyConnection, use_bookmarked: bool, recompute: bool) -> int:
    con.execute(PENDING_RESULT_PAIRS_QUERY, [use_bookmarked, recompute])
    return con.execute("SELECT COUNT(*) FROM pending_result_pairs").fetchone()[0]


def get_batch_files(con: duckdb.DuckDBPyConnection, start_id: int, end_id: int) -> List[str]:
    rows = con.execute(
        "SELECT DISTINCT ms_file_label FROM pending_pairs WHERE ms_type='ms1' AND pair_id BETWEEN ? AND ? ORDER BY ms_file_label",
        [start_id, end_id],
    ).fetchall()
    return [r[0] for r in rows]


def create_ms1_slice(con: duckdb.DuckDBPyConnection, files: List[str], cluster: bool) -> str:
    con.execute("DROP TABLE IF EXISTS ms1_needed")
    con.execute("CREATE TEMP TABLE ms1_needed AS SELECT * FROM ms1_data WHERE 0")
    if files:
        con.execute("CREATE TEMP TABLE tmp_files (ms_file_label VARCHAR)")
        con.executemany("INSERT INTO tmp_files VALUES (?)", [(f,) for f in files])
        order_clause = "ORDER BY ms_file_label, scan_id" if cluster else ""
        con.execute(
            f"INSERT INTO ms1_needed SELECT * FROM ms1_data WHERE ms_file_label IN (SELECT ms_file_label FROM tmp_files) {order_clause}"
        )
        con.execute("DROP TABLE tmp_files")
    return "ms1_needed"


def run_with_profile(con: duckdb.DuckDBPyConnection, sql: str, params: tuple[int, int], profile_path: Path,
                     do_rollback: bool, dump_path: Path | None, dump_format: str | None, dump_table: str | None) -> float:
    con.execute("SET enable_profiling = json")
    con.execute(f"SET profiling_output = '{profile_path}'")

    start = time.time()
    con.execute("BEGIN")
    try:
        con.execute(sql, params)
        if dump_path and dump_table:
            fmt = (dump_format or "parquet").upper()
            con.execute(f"COPY (SELECT * FROM {dump_table}) TO '{dump_path}' (FORMAT {fmt})")
    finally:
        if do_rollback:
            con.execute("ROLLBACK")
        else:
            con.execute("COMMIT")
    return time.time() - start


def ensure_helper_macro(con: duckdb.DuckDBPyConnection):
    con.execute(HELPER_MACRO)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile with ms1_data slice to reduce scans")
    p.add_argument("--db", required=True, help="Path to DuckDB database file")
    p.add_argument("--query", choices=["ms1", "ms2", "results"], required=True, help="Which query to profile")
    p.add_argument("--variant", choices=["baseline", "pruned"], default="baseline", help="ms1 query variant")
    p.add_argument("--start-id", type=int, required=True, help="Start pair_id for the batch")
    p.add_argument("--end-id", type=int, required=True, help="End pair_id for the batch")
    p.add_argument("--output", default=None, help="Profiling output filename (JSON). Defaults to <query>_<variant>_<start>-<end>_slice.json")
    p.add_argument("--commit", action="store_true", help="Commit instead of rolling back (mutates DB)")
    p.add_argument("--prepare-pending", choices=["none", "ms", "results", "all"], default="none",
                   help="Optionally (re)create pending tables")
    p.add_argument("--use-bookmarked", action="store_true")
    p.add_argument("--use-for-optimization", action="store_true")
    p.add_argument("--include-existing-pairs", action="store_true")
    p.add_argument("--recompute-results", action="store_true")
    p.add_argument("--temp-insert", action="store_true", help="Insert into temp table to avoid PK conflicts")
    p.add_argument("--dump-results", default=None, help="Path to dump results")
    p.add_argument("--dump-format", choices=["parquet", "csv", "json"], default="csv")
    p.add_argument("--threads", type=int, default=min(8, os.cpu_count()), help="PRAGMA threads")
    p.add_argument("--memory-limit", type=str, default="16GB", help="PRAGMA memory_limit, e.g., 16GB")
    p.add_argument("--cluster-slice", action="store_true", help="Order ms1 slice by ms_file_label, scan_id")
    return p.parse_args()


def main():
    args = parse_args()
    db_path = Path(args.db).expanduser()
    out_dir = Path(__file__).resolve().parent
    if args.output:
        profile_path = out_dir / args.output
    else:
        profile_path = out_dir / f"{args.query}_{args.variant}_{args.start_id}-{args.end_id}_slice.json"

    con = duckdb.connect(str(db_path))
    try:
        if args.threads is not None:
            con.execute(f"SET threads={args.threads}")
            print(f"Set threads={args.threads}")
        if args.memory_limit not in (None, ""):
            con.execute(f"SET memory_limit='{args.memory_limit}'")
            print(f"Set memory_limit={args.memory_limit}")

        if args.prepare_pending in ("ms", "all"):
            cnt = prepare_pending_pairs(con, args.use_bookmarked, args.use_for_optimization, args.include_existing_pairs)
            print(f"Prepared pending_pairs with {cnt} rows")
        if args.prepare_pending in ("results", "all"):
            cnt = prepare_pending_result_pairs(con, args.use_bookmarked, args.recompute_results)
            print(f"Prepared pending_result_pairs with {cnt} rows")

        if args.query == "results":
            ensure_helper_macro(con)

        # Build ms1 slice if needed
        ms1_table = "ms1_data"
        if args.query == "ms1":
            files = get_batch_files(con, args.start_id, args.end_id)
            print(f"Building ms1 slice for {len(files)} files")
            ms1_table = create_ms1_slice(con, files, cluster=args.cluster_slice)

        # Target table
        if args.temp_insert:
            target_base = "chromatograms" if args.query in ("ms1", "ms2") else "results"
            temp_table = f"temp_{target_base}"
            con.execute(f"CREATE TEMP TABLE IF NOT EXISTS {temp_table} AS SELECT * FROM {target_base} WHERE 0")
            target_table = temp_table
            print(f"Routing INSERT into temp table {temp_table}")
        else:
            target_table = "chromatograms" if args.query in ("ms1", "ms2") else "results"

        # Choose SQL
        if args.query == "ms1":
            tpl = MS1_QUERY_TEMPLATE_PRUNED if args.variant == "pruned" else MS1_QUERY_TEMPLATE_BASELINE
            sql = tpl.format(target_table=target_table, ms1_table=ms1_table)
        elif args.query == "ms2":
            sql = MS2_QUERY  # not defined in this file; use baseline from existing scripts if needed
        else:  # results
            sql = RESULTS_QUERY_TEMPLATE

        dump_table = target_table if args.dump_results else None

        elapsed = run_with_profile(
            con,
            sql,
            (args.start_id, args.end_id),
            profile_path,
            do_rollback=not args.commit,
            dump_path=Path(args.dump_results) if args.dump_results else None,
            dump_format=args.dump_format,
            dump_table=dump_table,
        )
    finally:
        con.close()

    print(f"Run finished in {elapsed:.2f}s -> {profile_path}")
    if args.dump_results:
        print(f"Results dumped to {args.dump_results} (format={args.dump_format})")
    if not args.commit:
        print("(Rolled back to avoid mutating the database)")


if __name__ == "__main__":
    main()
