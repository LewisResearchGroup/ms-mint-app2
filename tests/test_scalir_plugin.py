import pandas as pd
import pytest
from pathlib import Path
from ms_mint_app.plugins.scalir import (
    prepare_mint_results,
    prepare_standards,
    intersect_peaks,
    fit_estimator,
    build_concentration_table
)

@pytest.fixture
def sample_mint_csv(tmp_path):
    fn = tmp_path / "mint_results.csv"
    df = pd.DataFrame({
        'ms_file_label': ['file1', 'file1', 'file2', 'file2'],
        'peak_label': ['P1', 'P2', 'P1', 'P2'],
        'peak_area': [100, 200, 150, 250],
        'peak_max': [10, 20, 15, 25]
    })
    df.to_csv(fn, index=False)
    return fn

@pytest.fixture
def sample_standards_csv(tmp_path):
    fn = tmp_path / "standards.csv"
    df = pd.DataFrame({
        'peak_label': ['P1', 'P2', 'P3'],
        'file1': [1.0, 2.0, 3.0],
        'file2': [1.5, 2.5, 3.5],
        'unit': ['uM', 'uM', 'uM']
    })
    df.to_csv(fn, index=False)
    return fn

def test_prepare_mint_results(sample_mint_csv):
    df = prepare_mint_results(sample_mint_csv, 'peak_area')
    assert 'ms_file' in df.columns
    assert 'peak_label' in df.columns
    assert 'peak_area' in df.columns
    assert len(df) == 4

def test_prepare_standards(sample_standards_csv):
    df, units = prepare_standards(sample_standards_csv)
    assert 'unit' not in df.columns
    assert units is not None
    assert 'unit' in units.columns
    assert len(df) == 3

def test_intersect_peaks():
    mint_results = pd.DataFrame({
        'peak_label': ['P1', 'P2', 'P3'],
        'ms_file': ['f1', 'f1', 'f1'],
        'peak_area': [100, 200, 300]
    })
    standards = pd.DataFrame({
        'peak_label': ['P2', 'P3', 'P4'],
        'f1': [2.0, 3.0, 4.0]
    })
    units = pd.DataFrame({'peak_label': ['P2', 'P3', 'P4'], 'unit': ['uM', 'uM', 'uM']})
    
    m_f, s_f, u_f, common = intersect_peaks(mint_results, standards, units)
    
    assert sorted(common) == ['P2', 'P3']
    assert len(m_f) == 2
    assert len(s_f) == 2
    assert len(u_f) == 2

def test_fit_and_build_table(sample_mint_csv, sample_standards_csv):
    mint_results = prepare_mint_results(sample_mint_csv, 'peak_area')
    standards, units = prepare_standards(sample_standards_csv)
    m_f, s_f, u_f, common = intersect_peaks(mint_results, standards, units)
    
    estimator, std_results, x_train, y_train, params = fit_estimator(
        m_f, s_f, 'peak_area', 'fixed', (0.8, 1.2)
    )
    
    assert len(params) == 2 # P1 and P2
    
    conc_table = build_concentration_table(estimator, m_f, 'peak_area', u_f)
    assert 'pred_conc' in conc_table.columns
    assert 'unit' in conc_table.columns
    assert len(conc_table) == 4
