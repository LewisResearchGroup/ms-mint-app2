#!/usr/bin/env python
"""
Pre-build matplotlib font cache for distribution.

This script generates the matplotlib font cache that can be bundled
with the app to eliminate first-time user startup delays.
"""

import sys
from pathlib import Path
import shutil
import tempfile

def build_matplotlib_cache(output_dir):
    """Build matplotlib font cache in specified directory."""
    import os
    
    print("=" * 70)
    print("Building Matplotlib Font Cache for Distribution")
    print("=" * 70)
    
    # Set matplotlib to use our custom cache directory
    cache_dir = Path(output_dir) / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    os.environ['MPLCONFIGDIR'] = str(cache_dir)
    
    print(f"\n📁 Cache directory: {cache_dir}")
    print(f"✓ Created cache directory")
    
    # Force matplotlib to build font cache
    print(f"\n⏳ Building font cache (this takes ~5 seconds)...")
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.font_manager
    
    # This triggers font cache building
    _ = matplotlib.font_manager.fontManager.ttflist
    
    print(f"✓ Font cache built!")
    
    # Show what was created
    cache_files = list(cache_dir.rglob("*"))
    print(f"\n📊 Generated {len(cache_files)} cache files:")
    total_size = 0
    for f in cache_files:
        if f.is_file():
            size = f.stat().st_size
            total_size += size
            print(f"   {f.name:40s} {size:>10,} bytes")
    
    print(f"\n💾 Total cache size: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
    
    # Get matplotlib version for documentation
    print(f"\n📌 Matplotlib version: {matplotlib.__version__}")
    print(f"   Python version: {sys.version.split()[0]}")
    
    return cache_dir

def create_integration_instructions(cache_dir):
    """Create instructions for integrating the cache into the app."""
    
    instructions = f"""
╔══════════════════════════════════════════════════════════════════════╗
║              MATPLOTLIB CACHE INTEGRATION INSTRUCTIONS               ║
╚══════════════════════════════════════════════════════════════════════╝

The matplotlib font cache has been built at:
  {cache_dir}

────────────────────────────────────────────────────────────────────────
OPTION 1: Bundle with app distribution (RECOMMENDED)
────────────────────────────────────────────────────────────────────────

1. Copy the cache to your app's asset directory:
   
   cp -r {cache_dir} ./assets/matplotlib_cache/

2. In your Mint.py startup code, set the cache location BEFORE importing
   matplotlib anywhere:

   ```python
   import os
   from pathlib import Path
   
   # Set matplotlib cache to bundled version
   if hasattr(sys, '_MEIPASS'):  # PyInstaller frozen app
       cache_path = Path(sys._MEIPASS) / 'assets' / 'matplotlib_cache'
   else:  # Development mode
       cache_path = Path(__file__).parent / 'assets' / 'matplotlib_cache'
   
   os.environ['MPLCONFIGDIR'] = str(cache_path)
   ```

3. For PyInstaller, add to your .spec file:

   ```python
   datas=[
       ('assets/matplotlib_cache', 'assets/matplotlib_cache'),
   ]
   ```

────────────────────────────────────────────────────────────────────────
OPTION 2: Auto-deploy on first run
────────────────────────────────────────────────────────────────────────

1. Bundle the cache as above

2. On first app run, copy bundled cache to user's home directory:

   ```python
   user_cache = Path.home() / '.cache' / 'matplotlib'
   if not user_cache.exists():
       shutil.copytree(bundled_cache, user_cache)
       print("✓ Matplotlib cache initialized")
   ```

────────────────────────────────────────────────────────────────────────
IMPORTANT NOTES
────────────────────────────────────────────────────────────────────────

⚠️  Platform-specific: Font cache may differ between:
    - Operating systems (Linux, macOS, Windows)
    - System fonts installed
    
    Solution: Build separate caches for each platform

⚠️  Version-specific: Cache is tied to matplotlib version
    - Rebuild when updating matplotlib
    - Include version check in startup code

⚠️  Cache size: ~1-2 MB per platform
    - Consider build-time generation in CI/CD
    - Or bundle pre-built caches for all platforms

────────────────────────────────────────────────────────────────────────
ALREADY IMPLEMENTED IN YOUR APP
────────────────────────────────────────────────────────────────────────

Looking at your Mint.py (lines 153-164), you already have:

```python
if is_frozen:
    try:
        mpl_cfg = P(DATADIR) / ".cache" / "matplotlib"
        mpl_cfg.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cfg))
```

✓ You're already setting a custom cache directory!

TO INTEGRATE:
1. Build cache during your PyInstaller build process
2. Bundle it in your frozen app
3. Copy bundled cache to DATADIR/.cache/matplotlib on first run
4. Result: Instant matplotlib startup for all users!

"""
    
    print(instructions)

if __name__ == "__main__":
    # Build cache in current directory
    output_dir = Path("./prebuild_matplotlib_cache")
    
    if output_dir.exists():
        print(f"Removing existing cache: {output_dir}")
        shutil.rmtree(output_dir)
    
    cache_dir = build_matplotlib_cache(output_dir)
    create_integration_instructions(cache_dir)
    
    print("\n" + "=" * 70)
    print("✅ DONE! Cache ready for bundling with your app.")
    print("=" * 70)
