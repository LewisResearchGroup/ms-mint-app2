import numpy as np
import pandas as pd

from ms_mint_app.plugins.analysis import rocke_durbin, run_pca_samples_in_cols


def test_rocke_durbin_basic():
    df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["A", "B"], index=["s1", "s2"])

    out = rocke_durbin(df, c=1.0)

    assert out.shape == df.shape
    assert list(out.columns) == ["A", "B"]
    assert list(out.index) == ["s1", "s2"]
    assert np.isfinite(out.to_numpy()).all()


def test_run_pca_samples_in_cols_shapes():
    df = pd.DataFrame(
        {
            "feat1": [1.0, 2.0, 3.0],
            "feat2": [1.5, 2.5, 3.5],
        },
        index=["s1", "s2", "s3"],
    )

    result = run_pca_samples_in_cols(df, n_components=2, random_state=0)

    assert set(result.keys()) >= {
        "pca",
        "scores",
        "loadings",
        "explained_variance_ratio",
        "cumulative_variance_ratio",
    }
    assert result["scores"].shape == (3, 2)
    assert result["loadings"].shape == (2, 2)
    assert list(result["scores"].index) == ["s1", "s2", "s3"]
    assert list(result["loadings"].index) == ["feat1", "feat2"]
    assert np.isclose(result["explained_variance_ratio"].sum(), 1.0)
