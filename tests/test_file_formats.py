import pandas as pd
import pytest

from ms_mint_app.tools import read_tabular_file


TEST_DF = pd.DataFrame(
    {
        "peak_label": ["Glucose", "Serine", "Alanine"],
        "mz_mean": [180.063, 106.050, 90.055],
        "rt": [120.5, 150.2, 95.0],
    }
)


@pytest.mark.parametrize(
    "suffix, writer",
    [
        (".csv", lambda df, path: df.to_csv(path, index=False)),
        (".tsv", lambda df, path: df.to_csv(path, index=False, sep="\t")),
        (".txt", lambda df, path: df.to_csv(path, index=False, sep="\t")),
        (".txt", lambda df, path: df.to_csv(path, index=False)),
        (".xlsx", lambda df, path: df.to_excel(path, index=False, engine="openpyxl")),
    ],
)
def test_read_tabular_file_supported_formats(tmp_path, suffix, writer):
    if suffix == ".xlsx":
        pytest.importorskip("openpyxl")

    file_path = tmp_path / f"targets{suffix}"
    writer(TEST_DF, file_path)

    df = read_tabular_file(file_path)
    assert len(df) == len(TEST_DF)
    assert list(df.columns) == list(TEST_DF.columns)


def test_read_tabular_file_unsupported_extension(tmp_path):
    file_path = tmp_path / "targets.dat"
    TEST_DF.to_csv(file_path, index=False)

    with pytest.raises(ValueError, match="Unsupported file format"):
        read_tabular_file(file_path)
