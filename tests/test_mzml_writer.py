"""
Test module for lxml-based mzML writer in ms-mint-app.

Verifies that the write_mzml_from_spectra function produces valid mzML files
that can be read back correctly by iter_mzml_fast.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ms_mint_app.tools import write_mzml_from_spectra, iter_mzml_fast


class TestMzmlWriter:
    """Tests for the lxml-based mzML writer."""

    def test_write_single_spectrum_roundtrip(self, tmp_path):
        """Test writing and reading back a single spectrum."""
        # Create test spectrum
        spectra = [{
            "scan_id": 1,
            "rt": 60.5,  # seconds
            "mz_array": np.array([100.0, 150.5, 200.123, 300.0], dtype=np.float64),
            "intensity_array": np.array([1000.0, 5000.0, 2500.5, 100.0], dtype=np.float64),
        }]
        
        output_path = tmp_path / "test_single.mzML"
        
        # Write mzML
        write_mzml_from_spectra(spectra, output_path, polarity="Positive")
        
        # Verify file was created
        assert output_path.exists(), "mzML file was not created"
        assert output_path.stat().st_size > 0, "mzML file is empty"
        
        # Read back
        read_spectra = list(iter_mzml_fast(output_path))
        
        assert len(read_spectra) == 1, f"Expected 1 spectrum, got {len(read_spectra)}"
        
        read_spec = read_spectra[0]
        
        # Verify scan ID
        assert read_spec["num"] == 1, f"Scan ID mismatch: expected 1, got {read_spec['num']}"
        
        # Verify retention time (within tolerance)
        assert abs(read_spec["retentionTime"] - 60.5) < 0.001, \
            f"RT mismatch: expected 60.5, got {read_spec['retentionTime']}"
        
        # Verify polarity
        assert read_spec["polarity"] == "Positive", \
            f"Polarity mismatch: expected 'Positive', got {read_spec['polarity']}"
        
        # Verify m/z values
        np.testing.assert_allclose(
            read_spec["m/z array"], spectra[0]["mz_array"],
            rtol=1e-10, atol=1e-10,
            err_msg="m/z array mismatch"
        )
        
        # Verify intensity values
        np.testing.assert_allclose(
            read_spec["intensity array"], spectra[0]["intensity_array"],
            rtol=1e-10, atol=1e-10,
            err_msg="Intensity array mismatch"
        )

    def test_write_multiple_spectra_roundtrip(self, tmp_path):
        """Test writing and reading back multiple spectra."""
        # Create test spectra
        spectra = [
            {
                "scan_id": 1,
                "rt": 10.0,
                "mz_array": np.array([100.0, 200.0], dtype=np.float64),
                "intensity_array": np.array([1000.0, 2000.0], dtype=np.float64),
            },
            {
                "scan_id": 2,
                "rt": 20.0,
                "mz_array": np.array([150.0, 250.0, 350.0], dtype=np.float64),
                "intensity_array": np.array([500.0, 1500.0, 2500.0], dtype=np.float64),
            },
            {
                "scan_id": 3,
                "rt": 30.0,
                "mz_array": np.array([175.5], dtype=np.float64),
                "intensity_array": np.array([999.9], dtype=np.float64),
            },
        ]
        
        output_path = tmp_path / "test_multiple.mzML"
        
        # Write mzML
        write_mzml_from_spectra(spectra, output_path, polarity="Negative")
        
        # Read back
        read_spectra = list(iter_mzml_fast(output_path))
        
        assert len(read_spectra) == 3, f"Expected 3 spectra, got {len(read_spectra)}"
        
        # Verify each spectrum
        for i, (orig, read) in enumerate(zip(spectra, read_spectra)):
            assert read["num"] == orig["scan_id"], f"Scan ID mismatch at spectrum {i}"
            assert abs(read["retentionTime"] - orig["rt"]) < 0.001, f"RT mismatch at spectrum {i}"
            assert read["polarity"] == "Negative", f"Polarity mismatch at spectrum {i}"
            
            np.testing.assert_allclose(
                read["m/z array"], orig["mz_array"],
                rtol=1e-10, atol=1e-10,
                err_msg=f"m/z mismatch at spectrum {i}"
            )
            np.testing.assert_allclose(
                read["intensity array"], orig["intensity_array"],
                rtol=1e-10, atol=1e-10,
                err_msg=f"Intensity mismatch at spectrum {i}"
            )

    def test_write_empty_spectrum(self, tmp_path):
        """Test writing a spectrum with empty arrays."""
        spectra = [{
            "scan_id": 1,
            "rt": 5.0,
            "mz_array": np.array([], dtype=np.float64),
            "intensity_array": np.array([], dtype=np.float64),
        }]
        
        output_path = tmp_path / "test_empty.mzML"
        
        # Should not raise
        write_mzml_from_spectra(spectra, output_path, polarity="Positive")
        
        # File should exist
        assert output_path.exists()
        
        # Should be readable
        read_spectra = list(iter_mzml_fast(output_path))
        assert len(read_spectra) == 1
        assert len(read_spectra[0]["m/z array"]) == 0
        assert len(read_spectra[0]["intensity array"]) == 0

    def test_write_large_spectrum(self, tmp_path):
        """Test writing a spectrum with many peaks."""
        n_peaks = 10000
        spectra = [{
            "scan_id": 1,
            "rt": 120.0,
            "mz_array": np.linspace(100, 1000, n_peaks),
            "intensity_array": np.random.random(n_peaks) * 10000,
        }]
        
        output_path = tmp_path / "test_large.mzML"
        
        # Write
        write_mzml_from_spectra(spectra, output_path, polarity="Positive")
        
        # Read back
        read_spectra = list(iter_mzml_fast(output_path))
        
        assert len(read_spectra) == 1
        assert len(read_spectra[0]["m/z array"]) == n_peaks
        
        np.testing.assert_allclose(
            read_spectra[0]["m/z array"], spectra[0]["mz_array"],
            rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            read_spectra[0]["intensity array"], spectra[0]["intensity_array"],
            rtol=1e-10, atol=1e-10
        )

    def test_polarity_positive(self, tmp_path):
        """Test positive polarity is written correctly."""
        spectra = [{
            "scan_id": 1,
            "rt": 1.0,
            "mz_array": np.array([100.0]),
            "intensity_array": np.array([1000.0]),
        }]
        
        output_path = tmp_path / "test_pos.mzML"
        write_mzml_from_spectra(spectra, output_path, polarity="Positive")
        
        read_spectra = list(iter_mzml_fast(output_path))
        assert read_spectra[0]["polarity"] == "Positive"

    def test_polarity_negative(self, tmp_path):
        """Test negative polarity is written correctly."""
        spectra = [{
            "scan_id": 1,
            "rt": 1.0,
            "mz_array": np.array([100.0]),
            "intensity_array": np.array([1000.0]),
        }]
        
        output_path = tmp_path / "test_neg.mzML"
        write_mzml_from_spectra(spectra, output_path, polarity="Negative")
        
        read_spectra = list(iter_mzml_fast(output_path))
        assert read_spectra[0]["polarity"] == "Negative"

    def test_ms_level_is_one(self, tmp_path):
        """Test that MS level is correctly set to 1."""
        spectra = [{
            "scan_id": 1,
            "rt": 1.0,
            "mz_array": np.array([100.0]),
            "intensity_array": np.array([1000.0]),
        }]
        
        output_path = tmp_path / "test_mslevel.mzML"
        write_mzml_from_spectra(spectra, output_path)
        
        read_spectra = list(iter_mzml_fast(output_path))
        assert read_spectra[0]["msLevel"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
