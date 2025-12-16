"""
Compare two result dumps (parquet/csv/json) to verify they are identical in content.
- Aligns columns and sorts rows to perform a stable comparison.
- Reports row/column counts and any differences.

Example:
    python compare_results.py baseline.parquet pruned.parquet
"""

import sys
from pathlib import Path
import duckdb


def load_table(con: duckdb.DuckDBPyConnection, path: Path, alias: str):
    suffix = path.suffix.lower()
    if suffix == '.parquet':
        con.execute(f"CREATE TEMP VIEW {alias} AS SELECT * FROM parquet_scan('{path}')")
    elif suffix == '.csv':
        con.execute(f"CREATE TEMP VIEW {alias} AS SELECT * FROM read_csv_auto('{path}')")
    elif suffix == '.json':
        con.execute(f"CREATE TEMP VIEW {alias} AS SELECT * FROM read_json_auto('{path}')")
    else:
        raise ValueError(f"Unsupported file extension for {path}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_results.py <file1> <file2>")
        sys.exit(1)

    f1 = Path(sys.argv[1]).expanduser()
    f2 = Path(sys.argv[2]).expanduser()

    con = duckdb.connect()

    load_table(con, f1, 't1')
    load_table(con, f2, 't2')

    # Column alignment: require same set of columns
    cols1 = con.execute("PRAGMA table_info('t1')").fetchall()
    cols2 = con.execute("PRAGMA table_info('t2')").fetchall()
    colnames1 = [c[1] for c in cols1]
    colnames2 = [c[1] for c in cols2]

    if set(colnames1) != set(colnames2):
        missing_in_1 = set(colnames2) - set(colnames1)
        missing_in_2 = set(colnames1) - set(colnames2)
        print("Column mismatch:")
        if missing_in_1:
            print(f"  Missing in file1: {sorted(missing_in_1)}")
        if missing_in_2:
            print(f"  Missing in file2: {sorted(missing_in_2)}")
        sys.exit(1)

    cols = ','.join(f'"{c}"' for c in sorted(colnames1))

    count1 = con.execute("SELECT COUNT(*) FROM t1").fetchone()[0]
    count2 = con.execute("SELECT COUNT(*) FROM t2").fetchone()[0]
    if count1 != count2:
        print(f"Row count differs: file1={count1}, file2={count2}")
        sys.exit(1)

    # Order-insensitive comparison by sorting all columns
    con.execute(f"CREATE TEMP VIEW t1_sorted AS SELECT {cols} FROM t1 ORDER BY {cols}")
    con.execute(f"CREATE TEMP VIEW t2_sorted AS SELECT {cols} FROM t2 ORDER BY {cols}")

    diff = con.execute(
        "SELECT * FROM t1_sorted EXCEPT ALL SELECT * FROM t2_sorted"
    ).fetchall()
    diff_rev = con.execute(
        "SELECT * FROM t2_sorted EXCEPT ALL SELECT * FROM t1_sorted"
    ).fetchall()

    if diff or diff_rev:
        print(f"Differences found. Rows only in file1: {len(diff)}, rows only in file2: {len(diff_rev)}")
        sys.exit(1)

    print(f"Files are identical. Rows={count1}, Columns={len(colnames1)}")


if __name__ == "__main__":
    main()
