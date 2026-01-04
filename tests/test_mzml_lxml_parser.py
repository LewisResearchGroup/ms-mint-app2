"""
Test module for lxml-based mzML parser in ms-mint-app.

This module implements a fast lxml-based mzML parser (similar to iter_mzxml_fast)
and provides comparison tests against the existing mzXML parser to verify data consistency.

The files used for testing are the same raw file saved in both mzML and mzXML formats.
"""

import re
import base64
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator

import numpy as np
import pytest


# =============================================================================
# Test file paths - same file in different formats
# =============================================================================
TEST_MZML = "/media/mario/92bb1dc0-1f5c-4566-bfea-709a64de54f8/mario/Metabolights/RAW_FILES_01/mzML/2021_12_24RG_Col03_Conditioning11.mzML"
TEST_MZXML = "/media/mario/92bb1dc0-1f5c-4566-bfea-709a64de54f8/mario/Metabolights/RAW_FILES_01/2021_12_24RG_Col03_Conditioning11.mzXML"


# =============================================================================
# CV Param Accessions (controlled vocabulary for mzML)
# =============================================================================
CV_32BIT_FLOAT = "MS:1000521"
CV_64BIT_FLOAT = "MS:1000523"
CV_ZLIB_COMPRESSION = "MS:1000574"
CV_NO_COMPRESSION = "MS:1000576"
CV_MZ_ARRAY = "MS:1000514"
CV_INTENSITY_ARRAY = "MS:1000515"
CV_MS_LEVEL = "MS:1000511"
CV_POSITIVE_SCAN = "MS:1000130"
CV_NEGATIVE_SCAN = "MS:1000129"
CV_SCAN_START_TIME = "MS:1000016"


# =============================================================================
# Helper functions
# =============================================================================

def _decode_binary_mzml(
    binary_text: str, 
    is_64bit: bool = True, 
    is_compressed: bool = True
) -> np.ndarray:
    """
    Decode a mzML binary data array.
    
    Args:
        binary_text: Base64-encoded binary string
        is_64bit: True for 64-bit float, False for 32-bit
        is_compressed: True if zlib compressed
        
    Returns:
        numpy array of decoded values
    """
    if not binary_text or not binary_text.strip():
        return np.array([], dtype=np.float64 if is_64bit else np.float32)
    
    # Base64 decode
    raw = base64.b64decode(binary_text.strip())
    
    # Decompress if needed
    if is_compressed:
        raw = zlib.decompress(raw)
    
    # Convert to numpy array
    dtype = np.float64 if is_64bit else np.float32
    return np.frombuffer(raw, dtype=dtype)


def iter_mzml_fast(path: str | Path, *, decode_binary: bool = True) -> Iterator[Dict[str, Any]]:
    """
    Fast lxml-based iterator for mzML files.
    
    Similar to iter_mzxml_fast but handles mzML-specific structure:
    - Separate binaryDataArray elements for m/z and intensity
    - CV terms for metadata (compression, precision, polarity, etc.)
    - Always has namespace that must be stripped
    
    Args:
        path: Path to mzML file
        decode_binary: If True, decode binary arrays to numpy arrays
        
    Yields:
        Dictionary with spectrum data (same format as iter_mzxml_fast)
    """
    from lxml import etree
    
    path = Path(path)
    NS = "{http://psi.hupo.org/ms/mzml}"
    
    context = etree.iterparse(
        path.as_posix(),
        events=("start", "end"),
        remove_comments=True,
    )
    
    # Get root for memory cleanup
    _, root = next(context)
    
    current: Dict[str, Any] = {}
    binary_arrays: List[Dict[str, Any]] = []
    current_binary: Dict[str, Any] = {}
    
    for ev, elem in context:
        # Strip namespace
        tag = elem.tag
        if tag.startswith(NS):
            tag = tag[len(NS):]
        
        if ev == "start" and tag == "spectrum":
            # Start of new spectrum
            attribs = elem.attrib
            index = attribs.get("index", "0")
            spec_id = attribs.get("id", "")
            # Extract scan number from id like "controllerType=0 controllerNumber=1 scan=4"
            scan_match = re.search(r'scan=(\d+)', spec_id)
            scan_num = int(scan_match.group(1)) if scan_match else int(index) + 1
            
            current = {
                "num": scan_num,
                "msLevel": 1,  # Default, will be updated from cvParam
                "retentionTime": 0.0,
                "polarity": None,
                "filterLine": None,
            }
            binary_arrays = []
            
        elif ev == "end" and tag == "cvParam":
            # Parse CV parameters
            accession = elem.get("accession", "")
            value = elem.get("value", "")
            
            if accession == CV_MS_LEVEL:
                current["msLevel"] = int(value) if value else 1
            elif accession == CV_POSITIVE_SCAN:
                current["polarity"] = "Positive"
            elif accession == CV_NEGATIVE_SCAN:
                current["polarity"] = "Negative"
            elif accession == CV_SCAN_START_TIME:
                # Get scan time - check units
                unit_name = elem.get("unitName", "second")
                time_val = float(value) if value else 0.0
                if unit_name == "minute":
                    time_val *= 60.0
                current["retentionTime"] = time_val
            elif accession == CV_32BIT_FLOAT:
                current_binary["is_64bit"] = False
            elif accession == CV_64BIT_FLOAT:
                current_binary["is_64bit"] = True
            elif accession == CV_ZLIB_COMPRESSION:
                current_binary["is_compressed"] = True
            elif accession == CV_NO_COMPRESSION:
                current_binary["is_compressed"] = False
            elif accession == CV_MZ_ARRAY:
                current_binary["type"] = "mz"
            elif accession == CV_INTENSITY_ARRAY:
                current_binary["type"] = "intensity"
                
        elif ev == "start" and tag == "binaryDataArray":
            # Reset for new binary array
            current_binary = {
                "is_64bit": True,
                "is_compressed": False,
                "type": None,
                "data": None,
            }
            
        elif ev == "end" and tag == "binary":
            # Store binary text
            current_binary["data"] = elem.text
            
        elif ev == "end" and tag == "binaryDataArray":
            # Complete binary array - decode if requested
            if decode_binary and current_binary.get("data"):
                arr = _decode_binary_mzml(
                    current_binary["data"],
                    is_64bit=current_binary.get("is_64bit", True),
                    is_compressed=current_binary.get("is_compressed", False),
                )
                current_binary["array"] = arr
            binary_arrays.append(current_binary)
            current_binary = {}
            
        elif ev == "end" and tag == "spectrum":
            # End of spectrum - assemble result
            mz_array = np.array([], dtype=np.float64)
            intensity_array = np.array([], dtype=np.float64)
            
            for ba in binary_arrays:
                if ba.get("type") == "mz" and "array" in ba:
                    mz_array = ba["array"].astype(np.float64)
                elif ba.get("type") == "intensity" and "array" in ba:
                    intensity_array = ba["array"].astype(np.float64)
            
            current["m/z array"] = mz_array
            current["intensity array"] = intensity_array
            
            yield current
            root.clear()


# =============================================================================
# Import the existing mzXML parser from tools.py
# =============================================================================
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ms_mint_app.tools import iter_mzxml_fast


# =============================================================================
# TESTS
# =============================================================================

@pytest.fixture
def mzml_path():
    """Return path to test mzML file."""
    return Path(TEST_MZML)


@pytest.fixture
def mzxml_path():
    """Return path to test mzXML file."""
    return Path(TEST_MZXML)


def test_files_exist(mzml_path, mzxml_path):
    """Verify test files exist."""
    assert mzml_path.exists(), f"mzML file not found: {mzml_path}"
    assert mzxml_path.exists(), f"mzXML file not found: {mzxml_path}"


def test_iter_mzml_fast_returns_data(mzml_path):
    """Test that iter_mzml_fast returns spectra with expected fields."""
    spectra = list(iter_mzml_fast(mzml_path))
    
    assert len(spectra) > 0, "No spectra returned"
    
    first = spectra[0]
    assert "num" in first, "Missing scan number"
    assert "msLevel" in first, "Missing MS level"
    assert "retentionTime" in first, "Missing retention time"
    assert "polarity" in first, "Missing polarity"
    assert "m/z array" in first, "Missing m/z array"
    assert "intensity array" in first, "Missing intensity array"
    
    print(f"First spectrum: scan={first['num']}, "
          f"msLevel={first['msLevel']}, "
          f"RT={first['retentionTime']:.4f}s, "
          f"polarity={first['polarity']}, "
          f"peaks={len(first['m/z array'])}")


def test_compare_scan_counts(mzml_path, mzxml_path):
    """Verify both parsers return same number of scans."""
    mzml_spectra = list(iter_mzml_fast(mzml_path))
    mzxml_spectra = list(iter_mzxml_fast(mzxml_path))
    
    print(f"mzML scans: {len(mzml_spectra)}")
    print(f"mzXML scans: {len(mzxml_spectra)}")
    
    assert len(mzml_spectra) == len(mzxml_spectra), \
        f"Scan count mismatch: mzML={len(mzml_spectra)}, mzXML={len(mzxml_spectra)}"


def test_compare_polarity_detection(mzml_path, mzxml_path):
    """Verify polarity is detected consistently."""
    mzml_first = next(iter_mzml_fast(mzml_path))
    mzxml_first = next(iter_mzxml_fast(mzxml_path))
    
    # Normalize polarity representation
    mzml_pol = (mzml_first.get("polarity") or "").lower()
    mzxml_pol = (mzxml_first.get("polarity") or "").lower()
    
    print(f"mzML polarity: {mzml_pol}")
    print(f"mzXML polarity: {mzxml_pol}")
    
    # Both should indicate negative mode for this file
    assert "neg" in mzml_pol or mzml_pol == "-", f"Unexpected mzML polarity: {mzml_pol}"
    assert "neg" in mzxml_pol or mzxml_pol == "-", f"Unexpected mzXML polarity: {mzxml_pol}"


def test_compare_ms_level(mzml_path, mzxml_path):
    """Verify MS level is detected consistently."""
    mzml_first = next(iter_mzml_fast(mzml_path))
    mzxml_first = next(iter_mzxml_fast(mzxml_path))
    
    assert mzml_first["msLevel"] == mzxml_first["msLevel"], \
        f"MS level mismatch: mzML={mzml_first['msLevel']}, mzXML={mzxml_first['msLevel']}"


def test_compare_retention_times(mzml_path, mzxml_path):
    """Verify retention times match between formats."""
    mzml_spectra = list(iter_mzml_fast(mzml_path))
    mzxml_spectra = list(iter_mzxml_fast(mzxml_path))
    
    # Compare first few scans
    for i in range(min(10, len(mzml_spectra))):
        mzml_rt = mzml_spectra[i]["retentionTime"]
        mzxml_rt = mzxml_spectra[i]["retentionTime"]
        
        print(f"Scan {i+1}: mzML RT={mzml_rt:.4f}s, mzXML RT={mzxml_rt:.4f}s")
        
        # Allow small tolerance for floating point
        assert abs(mzml_rt - mzxml_rt) < 0.01, \
            f"RT mismatch at scan {i+1}: mzML={mzml_rt}, mzXML={mzxml_rt}"


def test_compare_mz_values(mzml_path, mzxml_path):
    """Verify m/z values match between formats."""
    mzml_spectra = list(iter_mzml_fast(mzml_path))
    mzxml_spectra = list(iter_mzxml_fast(mzxml_path))
    
    # Compare first few scans
    for i in range(min(5, len(mzml_spectra))):
        mzml_mz = mzml_spectra[i]["m/z array"]
        mzxml_mz = mzxml_spectra[i]["m/z array"]
        
        print(f"Scan {i+1}: mzML has {len(mzml_mz)} peaks, mzXML has {len(mzxml_mz)} peaks")
        
        assert len(mzml_mz) == len(mzxml_mz), \
            f"Peak count mismatch at scan {i+1}: mzML={len(mzml_mz)}, mzXML={len(mzxml_mz)}"
        
        if len(mzml_mz) > 0:
            # Check values are close (allow for precision differences)
            np.testing.assert_allclose(
                mzml_mz, mzxml_mz, rtol=1e-6, atol=1e-6,
                err_msg=f"m/z values mismatch at scan {i+1}"
            )


def test_compare_intensity_values(mzml_path, mzxml_path):
    """Verify intensity values match between formats."""
    mzml_spectra = list(iter_mzml_fast(mzml_path))
    mzxml_spectra = list(iter_mzxml_fast(mzxml_path))
    
    # Compare first few scans
    for i in range(min(5, len(mzml_spectra))):
        mzml_int = mzml_spectra[i]["intensity array"]
        mzxml_int = mzxml_spectra[i]["intensity array"]
        
        print(f"Scan {i+1}: mzML intensity range [{mzml_int.min():.1f}, {mzml_int.max():.1f}]")
        
        if len(mzml_int) > 0:
            # Check values are close (allow for precision differences)
            np.testing.assert_allclose(
                mzml_int, mzxml_int, rtol=1e-4, atol=1.0,
                err_msg=f"Intensity values mismatch at scan {i+1}"
            )


def test_performance_comparison(mzml_path, mzxml_path):
    """Compare parsing performance between lxml mzML vs lxml mzXML."""
    import time
    
    # Warm up
    _ = list(iter_mzml_fast(mzml_path))
    _ = list(iter_mzxml_fast(mzxml_path))
    
    # Time mzML parsing
    start = time.time()
    mzml_spectra = list(iter_mzml_fast(mzml_path))
    mzml_time = time.time() - start
    
    # Time mzXML parsing
    start = time.time()
    mzxml_spectra = list(iter_mzxml_fast(mzxml_path))
    mzxml_time = time.time() - start
    
    print(f"\nPerformance comparison:")
    print(f"  mzML (lxml): {mzml_time:.3f}s for {len(mzml_spectra)} scans")
    print(f"  mzXML (lxml): {mzxml_time:.3f}s for {len(mzxml_spectra)} scans")
    print(f"  Ratio (mzML/mzXML): {mzml_time/mzxml_time:.2f}x")
    
    # Just informational, no assertion on performance


if __name__ == "__main__":
    # Quick manual test
    print("Testing mzML parser...")
    spectra = list(iter_mzml_fast(TEST_MZML))
    print(f"Loaded {len(spectra)} spectra from mzML")
    
    if spectra:
        first = spectra[0]
        print(f"First spectrum: scan={first['num']}, "
              f"msLevel={first['msLevel']}, "
              f"RT={first['retentionTime']:.2f}s, "
              f"polarity={first['polarity']}, "
              f"peaks={len(first['m/z array'])}")
