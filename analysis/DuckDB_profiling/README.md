# DuckDB Profiling Workspace

This folder holds offline profiling for the heavy DuckDB workloads in `src/ms_mint_app/duckdb_manager.py`.

## Target queries
- Chromatogram batching: `QUERY_PROCESS_BATCH_MS1` and `QUERY_PROCESS_BATCH_MS2` inside `compute_chromatograms_in_batches`.
- Results batching: `QUERY_PROCESS_BATCH` (using the helper macro `compute_chromatogram_metrics`) inside `compute_results_in_batches`.

## What to measure
- `EXPLAIN ANALYZE` output for each batch query (ms1, ms2, results) to see where time is spent (joins vs LIST aggregation/order).
- End-to-end batch timings (per batch) to catch fast/slow cycles.
- WAL size growth during long runs (to correlate slow batches with checkpoint/commit stalls).

## Inputs needed
- A DuckDB database that already contains the same schema/data used by the app (`targets`, `samples`, `ms1_data`, `ms2_data`, `ms_file_scans`, `chromatograms`, `results`).
- Optional: a trimmed copy for faster profiling if the production DB is very large.

## Output convention
- Store profiling logs and artifacts here (e.g., `ms1_explain.json`, `ms2_explain.json`, `results_explain.json`, `batch_timings.csv`, `wal_watch.log`).

## How to run the profiling scripts
1) Prepare a copy of the DuckDB file you want to inspect (avoid the live DB).  
2) Ensure the relevant helper tables are populated in that copy:  
   - `pending_pairs` for `ms1`/`ms2` runs (the pair_id range you pass must exist).  
   - `pending_result_pairs` for `results` runs.  
   - `chromatograms`/`targets` are needed for the `results` query.  
   Use the app’s existing batch-creation steps to build these tables before profiling.
3) Run `python run_profiling.py --db /path/to/db.duckdb --query ms1 --start-id 1 --end-id 100` (adjust query type and id ranges).  
   - Default behavior wraps the query in a transaction and rolls back, so the DB is not mutated.  
   - Add `--commit` only if you intentionally want the INSERT to persist.
4) The script writes JSON profiling output to this folder (e.g., `ms1_1-100.json`). Use DuckDB’s JSON profiler viewer or any JSON tool to inspect timings/plan nodes.

### Auto-preparing pending tables from the script
If your DB copy lacks `pending_pairs` or `pending_result_pairs`, the script can create them for you using the same logic as the app:
- Add `--prepare-pending ms` to create `pending_pairs` (used by ms1/ms2 queries).  
- Add `--prepare-pending results` to create `pending_result_pairs` (used by the results query).  
- Add `--prepare-pending all` to create both.
Optional filters:  
- `--use-bookmarked` limits targets to bookmarked ones.  
- `--use-for-optimization` makes `pending_pairs` use `use_for_optimization` instead of `use_for_analysis` (default is analysis).  
- `--recompute-results` makes `pending_result_pairs` include pairs even if results already exist.  
- `--include-existing-pairs` makes `pending_pairs` include combinations even if chromatograms already exist (useful for profiling on a “full” DB).  
- `--temp-insert` routes the INSERT into a temporary table to avoid primary key conflicts when profiling against a DB that already has chromatograms/results.

## Watching WAL growth during a run
- In another shell: `watch -n 2 'du -h /path/to/db.duckdb.wal'` to see WAL size changes.  
- Correlate WAL spikes with slow batches to confirm whether checkpoint/commit phases are the bottleneck.  
- You can also log start/end times per batch in the app run and cross-reference with the WAL watch log.
