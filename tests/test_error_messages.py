#!/usr/bin/env python
"""Test improved error messages for unsupported RT combinations."""

import pandas as pd
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from ms_mint_app.tools import get_targets_v2

print("=" * 70)
print("Testing Error Messages for Unsupported RT Combinations")
print("=" * 70)

test_cases = [
    {
        "name": "Only rt_max (user's reported case)",
        "data": pd.DataFrame({
            "peak_label": ["Glucose"],
            "rt_max": [121.5],
        }),
    },
    {
        "name": "Only rt_min",
        "data": pd.DataFrame({
            "peak_label": ["Fructose"],
            "rt_min": [119.0],
        }),
    },
]

for test_case in test_cases:
    print(f"\n{test_case['name']}")
    print("-" * 70)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_case['data'].to_csv(f.name, index=False)
        temp_file = f.name
    
    try:
        targets_df, _, _, _ = get_targets_v2([temp_file])
        print("❌ ERROR: Should have raised ValueError but didn't!")
    except ValueError as e:
        error_msg = str(e)
        print(f"✅ Got expected error:")
        print(f"   {error_msg}")
        
        # Check if error message is informative
        is_informative = (
            "Cannot derive RT values" in error_msg and
            "You provided:" in error_msg and
            "Valid combinations:" in error_msg
        )
        
        if is_informative:
            print(f"✅ Error message is informative and actionable!")
        else:
            print(f"❌ Error message could be more informative")
            
    except Exception as e:
        print(f"❌ Unexpected error type: {type(e).__name__}: {e}")
    finally:
        Path(temp_file).unlink(missing_ok=True)

print("\n" + "=" * 70)
print("COMPARISON")
print("=" *70)
print("\nOLD ERROR (cryptic):")
print("  WARNING | Failed to process target at row 0: 'rt_min'")
print("\nNEW ERROR (informative):")
print("  WARNING | Failed to process target at row 0:")
print("  Target 'Glucose': Cannot derive RT values from the provided data.")
print("  You provided: rt_max=121.50.")
print("  Valid combinations: (1) 'rt' only, (2) 'rt_min' AND 'rt_max',")
print("  (3) 'rt' + 'rt_min', or (4) 'rt' + 'rt_max'.")
print("  Providing only 'rt_min' or only 'rt_max' is not supported.")
print("\n✓ Much clearer what went wrong and how to fix it!")
