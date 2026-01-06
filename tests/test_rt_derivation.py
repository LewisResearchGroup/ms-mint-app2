#!/usr/bin/env python
"""
Test script for smart RT derivation logic.
Tests all 4 scenarios:
1. RT only
2. RT_min and RT_max only  
3. RT + RT_min
4. RT + RT_max
"""

import pandas as pd
import tempfile
from pathlib import Path
import sys

# Add src to path so we can import ms_mint_app
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ms_mint_app.tools import get_targets_v2

def test_rt_derivation():
    """Test all RT derivation scenarios."""
    
    test_cases = [
        {
            "name": "Scenario 1: RT only",
           "data": pd.DataFrame({
                "peak_label": ["Test1"],
                "rt": [120.5],
            }),
            "expected": {
                "rt": 120.5,
                "rt_min": 119.5,  # rt - 1
                "rt_max": 121.5,  # rt + 1
            }
        },
        {
            "name": "Scenario 2: RT_min and RT_max only",
            "data": pd.DataFrame({
                "peak_label": ["Test2"],
                "rt_min": [119.5],
                "rt_max": [121.5],
            }),
            "expected": {
                "rt": 120.5,  # Average
                "rt_min": 119.5,
                "rt_max": 121.5,
            }
        },
        {
            "name": "Scenario 3a: RT + RT_min",
            "data": pd.DataFrame({
                "peak_label": ["Test3"],
                "rt": [120.0],
                "rt_min": [118.0],
            }),
            "expected": {
                "rt": 120.0,
                "rt_min": 118.0,
                "rt_max": 122.0,  # rt + (rt - rt_min) = 120 + 2
            }
        },
        {
            "name": "Scenario 3b: RT + RT_max",
            "data": pd.DataFrame({
                "peak_label": ["Test4"],
                "rt": [120.0],
                "rt_max": [123.0],
            }),
            "expected": {
                "rt": 120.0,
                "rt_min": 117.0,  # rt - (rt_max - rt) = 120 - 3
                "rt_max": 123.0,
            }
        },
        {
            "name": "Error: No RT values",
            "data": pd.DataFrame({
                "peak_label": ["Test5"],
            }),
            "expected": "error",  # Should raise ValueError
        },
        {
            "name": "Error: Negative RT_min",
            "data": pd.DataFrame({
                "peak_label": ["Test6"],
                "rt": [0.5],
                "rt_min": [-0.5],
            }),
            "expected": "error",
        },
        {
            "name": "Error: Inverted bounds",
            "data": pd.DataFrame({
               " peak_label": ["Test7"],
                "rt_min": [121.0],
                "rt_max": [119.0],
            }),
            "expected": "error",
        },
    ]
    
    print("=" * 70)
    print("SMART RT DERIVATION TESTS")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}")
        print("-" * 70)
        
        # Write test data to temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_case['data'].to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            # Run get_targets_v2
            targets_df, failed_files, failed_targets, stats = get_targets_v2([temp_file])
            
            if test_case['expected'] == "error":
                print(f"❌ FAILED: Expected error but got success")
                failed += 1
            else:
                # Check values
                row = targets_df.iloc[0]
                expected = test_case['expected']
                
                success = True
                for key, expected_val in expected.items():
                    actual_val = row[key]
                    if abs(actual_val - expected_val) > 0.01:  # Allow small float error
                        print(f"❌ {key}: expected {expected_val}, got {actual_val}")
                        success = False
                    else:
                        print(f"✓ {key}: {actual_val}")
                
                if success:
                    print(f"✅ PASSED")
                    passed += 1
                else:
                    failed += 1
                    
        except Exception as e:
            if test_case['expected'] == "error":
                print(f"✅ PASSED: Got expected error: {str(e)[:80]}")
                passed += 1
            else:
                print(f"❌ FAILED: Unexpected error: {e}")
                failed += 1
        finally:
            # Clean up temp file
            Path(temp_file).unlink(missing_ok=True)
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0

if __name__ == "__main__":
    success = test_rt_derivation()
    sys.exit(0 if success else 1)
