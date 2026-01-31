import base64
import zlib

import pandas as pd
import numpy as np

from ms_mint_app.tools import (
    normalize_column_names,
    sparsify_chrom,
    proportional_min1_selection,
    rt_to_seconds,
    get_acquisition_datetime,
    _encode_binary_mzml,
    _decode_binary_mzml,
    _decode_peaks_optimized,
    today,
)


def test_normalize_column_names_priority_and_rt_conversion():
    df = pd.DataFrame(
        {
            "compoundId": ["A"],
            "compound": ["B"],
            "medMz": [100.0],
            "medRt": [2.0],
            "parent": [999.0],
        }
    )

    out = normalize_column_names(df)

    assert "peak_label" in out.columns
    assert "mz_mean" in out.columns
    assert "rt" in out.columns
    assert out["mz_mean"].iloc[0] == 100.0
    assert out["peak_label"].iloc[0] == "B"


def test_read_el_maven_json_groups(tmp_path):
    path = tmp_path / "maven.json"
    path.write_text(
        """
{
  "groups": [
    {
      "groupId": 1,
      "meanMz": 250.0945892,
      "meanRt": 7.975498199,
      "rtmin": 7.860666752,
      "rtmax": 8.25313282,
      "compound": {
        "compoundId": "C00559",
        "compoundName": "deoxyadenosine",
        "expectedRt": 8,
        "expectedMz": 250.0945587,
        "adductName": "[M-H]-"
      }
    }
  ]
}
"""
    )
    from ms_mint_app.tools import read_tabular_file
    df = read_tabular_file(path)
    df = normalize_column_names(df)
    assert df["peak_label"].iloc[0] == "deoxyadenosine"
    assert df["mz_mean"].iloc[0] == 250.0945892
    assert df["rt"].iloc[0] == 7.975498199 * 60.0
    assert df["rt_min"].iloc[0] == 7.860666752 * 60.0
    assert df["rt_max"].iloc[0] == 8.25313282 * 60.0


def test_sparsify_chrom_filters_signal_and_neighbors():
    scan = np.arange(10)
    intensity = np.array([0, 0, 1, 5, 6, 1, 0, 0, 0, 0], dtype=float)

    kept_scan, kept_intensity = sparsify_chrom(scan, intensity, w=1, baseline=4.0, eps=0.0, min_peak_width=1)

    assert kept_scan.min() <= 2
    assert kept_scan.max() >= 5
    assert len(kept_scan) == len(kept_intensity)


def test_sparsify_chrom_no_signal():
    scan = np.arange(5)
    intensity = np.zeros(5)

    kept_scan, kept_intensity = sparsify_chrom(scan, intensity, baseline=1.0)

    assert len(kept_scan) == 0
    assert len(kept_intensity) == 0


def test_proportional_min1_selection():
    df = pd.DataFrame(
        {
            "group": ["A", "B"],
            "items": [[1, 2, 3, 4], [10, 11]],
        }
    )

    quotas, selected = proportional_min1_selection(df, "group", "items", total_select=3, seed=1)

    assert quotas["A"] + quotas["B"] >= 2
    assert len(selected) == sum(quotas.values())


def test_rt_to_seconds_parses_iso_duration():
    assert rt_to_seconds("PT1M30S") == 90.0
    assert rt_to_seconds("PT2H") == 7200.0
    assert rt_to_seconds("12.5") == 12.5


def test_get_acquisition_datetime_mzml(tmp_path):
    mzml = tmp_path / "test.mzML"
    mzml.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0">
  <run startTimeStamp="2024-01-01T10:00:00Z"></run>
</mzML>
"""
    )

    ts = get_acquisition_datetime(mzml)
    assert ts == "2024-01-01T10:00:00Z"


def test_get_acquisition_datetime_mzxml(tmp_path):
    mzxml = tmp_path / "test.mzXML"
    mzxml.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<mzXML xmlns="http://sashimi.sourceforge.net/schema_revision/mzXML_3.2">
  <msRun startTime="2024-01-02T12:00:00"></msRun>
</mzXML>
"""
    )

    ts = get_acquisition_datetime(mzxml)
    assert ts == "2024-01-02T12:00:00"


def test_encode_decode_binary_mzml_roundtrip():
    data = np.array([1.5, 2.5, 3.5], dtype=np.float64)

    encoded = _encode_binary_mzml(data, is_64bit=True, compress=True)
    decoded = _decode_binary_mzml(encoded, is_64bit=True, is_compressed=True)

    assert np.allclose(decoded, data)


def test_decode_peaks_optimized_roundtrip():
    mz = np.array([100.0, 200.0], dtype=np.float64)
    intensity = np.array([1000.0, 2000.0], dtype=np.float64)
    dtype = np.dtype([("mz", ">f8"), ("intensity", ">f8")])
    arr = np.empty(2, dtype=dtype)
    arr["mz"] = mz
    arr["intensity"] = intensity
    raw = arr.tobytes()
    encoded = base64.b64encode(zlib.compress(raw)).decode("ascii")

    attrs = {"precision": "64", "byteOrder": "network", "compressionType": "zlib"}
    out_mz, out_intensity = _decode_peaks_optimized(attrs, encoded)

    assert np.allclose(out_mz, mz)
    assert np.allclose(out_intensity, intensity)


def test_today_format():
    value = today()
    assert len(value.split("-")) == 3
