import os
from datetime import datetime

import pandas as pd

from ms_mint_app.plugins.analysis.qc import _resolve_qc_x_axis
from ms_mint_app.tools import get_acquisition_datetime_with_source


def test_get_acquisition_datetime_with_source_falls_back_to_file_mtime(tmp_path):
    fn = tmp_path / "missing-header.mzML"
    fn.write_text("<mzML></mzML>", encoding="utf-8")
    ts = datetime(2026, 4, 14, 12, 30, 0).timestamp()
    os.utime(fn, (ts, ts))

    acq_datetime, source = get_acquisition_datetime_with_source(fn)

    assert source == "file_mtime"
    assert acq_datetime.startswith("2026-04-14T12:30:00")


def test_resolve_qc_x_axis_uses_derived_file_time_label():
    df = pd.DataFrame(
        {
            "acquisition_datetime": ["2026-04-14T12:30:00", "2026-04-14T12:45:00"],
            "acquisition_datetime_source": ["file_mtime", "raw_header"],
            "sample_order": [1, 2],
        }
    )

    x_col, x_title, note = _resolve_qc_x_axis(df)

    assert x_col == "acquisition_time_str"
    assert x_title == "File Time (derived from file timestamp)"
    assert "raw file did not contain an acquisition datetime" in note
