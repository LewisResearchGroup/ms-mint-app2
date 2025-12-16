"""
Offline DuckDB profiling helpers for MINT heavy queries.

Usage examples (run in a throwaway copy of your DB):
    python run_profiling.py --db /path/to/mint.duckdb --query ms1 --start-id 1 --end-id 100
    python run_profiling.py --db /path/to/mint.duckdb --query ms2 --start-id 1 --end-id 100
    python run_profiling.py --db /path/to/mint.duckdb --query results --start-id 1 --end-id 100

By default this uses EXPLAIN ANALYZE inside a transaction and rolls back so the DB is not mutated.
Profiling output is written as JSON under ./analysis/DuckDB_profiling/.
"""

import argparse
import time
from pathlib import Path

import duckdb

MS1_QUERY = """
INSERT INTO chromatograms (peak_label, ms_file_label, scan_time, intensity, ms_type)
WITH batch_pairs AS (SELECT peak_label, ms_file_label, ms_type, mz_mean, mz_width
                     FROM pending_pairs
                     WHERE ms_type = 'ms1'
                       AND pair_id BETWEEN ? AND ?),
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

RESULTS_QUERY = """
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


def run_explain_analyze(con: duckdb.DuckDBPyConnection, sql: str, params: tuple[int, int], profile_path: Path, do_rollback: bool) -> float:
    con.execute("SET enable_profiling = json")
    con.execute(f"SET profiling_output = '{profile_path}'")

    start = time.time()
    con.execute("BEGIN")
    try:
        # Run the query normally so DuckDB writes JSON to profiling_output.
        con.execute(sql, params)
    finally:
        if do_rollback:
            con.execute("ROLLBACK")
        else:
            con.execute("COMMIT")
    return time.time() - start


def ensure_helper_macro(con: duckdb.DuckDBPyConnection):
    con.execute(HELPER_MACRO)


def prepare_pending_pairs(con: duckdb.DuckDBPyConnection, use_bookmarked: bool, use_for_optimization: bool, include_existing: bool) -> int:
    con.execute(PENDING_PAIRS_QUERY, [use_bookmarked, use_for_optimization, include_existing])
    return con.execute("SELECT COUNT(*) FROM pending_pairs").fetchone()[0]


def prepare_pending_result_pairs(con: duckdb.DuckDBPyConnection, use_bookmarked: bool, recompute: bool) -> int:
    con.execute(PENDING_RESULT_PAIRS_QUERY, [use_bookmarked, recompute])
    return con.execute("SELECT COUNT(*) FROM pending_result_pairs").fetchone()[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile heavy DuckDB queries (EXPLAIN ANALYZE).")
    parser.add_argument("--db", required=True, help="Path to DuckDB database file")
    parser.add_argument("--query", choices=["ms1", "ms2", "results"], required=True, help="Which query to profile")
    parser.add_argument("--start-id", type=int, required=True, help="Start pair_id for the batch")
    parser.add_argument("--end-id", type=int, required=True, help="End pair_id for the batch")
    parser.add_argument("--output", default=None, help="Profiling output filename (JSON). Defaults to <query>_<start>-<end>.json")
    parser.add_argument("--commit", action="store_true", help="Commit instead of rolling back (mutates DB)")
    parser.add_argument("--prepare-pending", choices=["none", "ms", "results", "all"], default="none",
                        help="Optionally (re)create pending_pairs/pending_result_pairs before profiling.")
    parser.add_argument("--use-bookmarked", action="store_true", help="When preparing pending tables, restrict to bookmarked targets.")
    parser.add_argument("--use-for-optimization", action="store_true",
                        help="When preparing pending_pairs, use use_for_optimization instead of use_for_analysis (default False = use_for_analysis).")
    parser.add_argument("--recompute-results", action="store_true",
                        help="When preparing pending_result_pairs, include pairs even if results already exist.")
    parser.add_argument("--include-existing-pairs", action="store_true",
                        help="When preparing pending_pairs, include pairs even if chromatograms already exist.")
    parser.add_argument("--temp-insert", action="store_true",
                        help="Route INSERT into a temporary table to avoid primary key conflicts when data already exists.")
    return parser.parse_args()


def main():
    args = parse_args()
    db_path = Path(args.db).expanduser()
    out_dir = Path(__file__).resolve().parent
    out_name = args.output or f"{args.query}_{args.start_id}-{args.end_id}.json"
    profile_path = out_dir / out_name

    sql = {"ms1": MS1_QUERY, "ms2": MS2_QUERY, "results": RESULTS_QUERY}[args.query]

    con = duckdb.connect(str(db_path))
    try:
        if args.prepare_pending in ("ms", "all"):
            count = prepare_pending_pairs(
                con,
                use_bookmarked=args.use_bookmarked,
                use_for_optimization=args.use_for_optimization,
                include_existing=args.include_existing_pairs,
            )
            print(f"Prepared pending_pairs with {count} rows")

        if args.prepare_pending in ("results", "all"):
            count = prepare_pending_result_pairs(con, use_bookmarked=args.use_bookmarked, recompute=args.recompute_results)
            print(f"Prepared pending_result_pairs with {count} rows")

        if args.query == "results":
            ensure_helper_macro(con)

        if args.temp_insert:
            target_table = "chromatograms" if args.query in ("ms1", "ms2") else "results"
            temp_table = f"temp_{target_table}"
            con.execute(f"CREATE TEMP TABLE IF NOT EXISTS {temp_table} AS SELECT * FROM {target_table} WHERE 0")
            sql = sql.replace(f"INSERT INTO {target_table}", f"INSERT INTO {temp_table}", 1)
            print(f"Routing INSERT into temporary table '{temp_table}' for profiling to avoid PK conflicts.")

        elapsed = run_explain_analyze(
            con,
            sql,
            (args.start_id, args.end_id),
            profile_path,
            do_rollback=not args.commit,
        )
    finally:
        con.close()

    print(f"EXPLAIN ANALYZE finished in {elapsed:.2f}s -> {profile_path}")
    if not args.commit:
        print("(Rolled back to avoid mutating the database)")


if __name__ == "__main__":
    main()
