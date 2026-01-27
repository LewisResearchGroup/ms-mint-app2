import contextlib
import json

import duckdb
import polars as pl

from ms_mint_app.duckdb_manager import _create_tables
from ms_mint_app.plugins import target_optimization as topt


@contextlib.contextmanager
def _conn_context(conn):
    yield conn


def test_save_target_state_does_not_strip_alignment_note_when_clearing_disallowed():
    conn = duckdb.connect(":memory:")
    _create_tables(conn)

    alignment_data = {
        "enabled": True,
        "target_label": "Peak1",
        "reference_rt": 101.0,
        "shifts_by_sample_type": {"TypeA": 0.2},
        "shifts_per_file": {"S1": 0.2},
        "rt_min": 100.0,
        "rt_max": 102.0,
    }
    auto_note = topt._format_rt_alignment_auto_note(alignment_data)
    seeded_note = f"user note\n\n{auto_note}"

    conn.execute(
        """
        INSERT INTO targets (
            peak_label, ms_type, notes,
            rt_align_enabled, rt_align_reference_rt, rt_align_shifts,
            rt_align_rt_min, rt_align_rt_max
        ) VALUES (?, 'ms1', ?, TRUE, ?, ?, ?, ?)
        """,
        [
            "Peak1",
            seeded_note,
            alignment_data["reference_rt"],
            json.dumps(alignment_data["shifts_per_file"]),
            alignment_data["rt_min"],
            alignment_data["rt_max"],
        ],
    )

    result = topt._save_target_state(
        conn,
        "Peak1",
        note_text="user note",
        save_rt_span=False,
        rt_align_toggle=False,
        rt_alignment_data=None,
        allow_clear_rt_alignment=False,
    )

    assert result["saved_notes"] is True

    stored_note = conn.execute(
        "SELECT notes FROM targets WHERE peak_label = 'Peak1'"
    ).fetchone()[0]
    assert topt._extract_rt_alignment_auto_note(stored_note)

    align_row = conn.execute(
        "SELECT rt_align_enabled, rt_align_shifts FROM targets WHERE peak_label = 'Peak1'"
    ).fetchone()
    assert align_row[0] is True
    assert align_row[1]


def test_build_rt_alignment_from_db_is_authoritative_without_store():
    chrom_df = pl.DataFrame(
        {
            "sample_type": ["TypeA", "TypeA", "TypeB"],
            "ms_file_label": ["S1", "S2", "S3"],
            "scan_time_sliced": [
                [99.0, 100.5, 101.0, 103.0],
                [99.5, 100.7, 101.2, 103.5],
                [98.0, 100.6, 101.1, 104.0],
            ],
            "intensity_sliced": [
                [1.0, 5.0, 10.0, 2.0],
                [1.0, 6.0, 9.0, 2.0],
                [1.0, 4.0, 11.0, 3.0],
            ],
        }
    )

    shifts_json = json.dumps({"S1": 0.3, "S2": 0.1, "S3": -0.2})

    active, alignment_data, shifts_per_file = topt._build_rt_alignment_from_db(
        chrom_df=chrom_df,
        target_label="Peak1",
        align_enabled=True,
        align_ref_rt=None,
        align_shifts_json=shifts_json,
        align_rt_min=100.0,
        align_rt_max=102.0,
        rt_min_fallback=100.0,
        rt_max_fallback=102.0,
    )

    assert active is True
    assert shifts_per_file == {"S1": 0.3, "S2": 0.1, "S3": -0.2}
    assert alignment_data["enabled"] is True
    assert alignment_data["target_label"] == "Peak1"
    assert alignment_data["reference_rt"] is not None

    note_block = topt._format_rt_alignment_auto_note(alignment_data)
    assert "RT alignment: enabled" in note_block
