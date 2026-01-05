#!/usr/bin/env python
"""
Integration code for bundling pre-built matplotlib cache with frozen app.
Add this to your Mint.py startup sequence.
"""

def setup_bundled_matplotlib_cache():
    """
    Copy bundled matplotlib cache to user directory on first run.
    This eliminates the 5-second font cache building delay for first-time users.
    """
    import sys
    import shutil
    import platform
    from pathlib import Path
    import logging
    
    is_frozen = hasattr(sys, '_MEIPASS')
    
    if is_frozen:
        # Detect platform
        system = platform.system().lower()
        
        # Path to bundled cache in frozen app (platform-specific)
        bundled_cache = Path(sys._MEIPASS) / 'assets' / f'matplotlib_cache_{system}'
        
        # Fallback: try without platform suffix for backward compatibility
        if not bundled_cache.exists():
            bundled_cache = Path(sys._MEIPASS) / 'assets' / 'matplotlib_cache'
        
        # User's matplotlib cache directory
        import os
        DATADIR = os.environ.get('MINT_DATA_DIR', str(Path.home() / 'MINT'))
        user_cache = Path(DATADIR) / '.cache' / 'matplotlib'
        
        # Check if cache already exists
        cache_file = user_cache / 'fontlist-v390.json'
        
        if not cache_file.exists() and bundled_cache.exists():
            try:
                logging.info(f"Copying bundled matplotlib cache ({system}) to {user_cache}")
                user_cache.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy the entire cache directory
                if user_cache.exists():
                    shutil.rmtree(user_cache)
                shutil.copytree(bundled_cache, user_cache)
                
                logging.info("✓ Matplotlib cache initialized from bundle")
                logging.info("  First-time startup will be ~5 seconds faster!")
                
            except Exception as e:
                logging.warning(f"Failed to copy matplotlib cache: {e}")
                logging.warning("  Font cache will be built on first matplotlib use")
        
        # Set matplotlib to use our cache directory
        os.environ['MPLCONFIGDIR'] = str(user_cache)


# ============================================================================
# INTEGRATION INSTRUCTIONS FOR Mint.py
# ============================================================================

INTEGRATION_CODE = """
# In your Mint.py, add this BEFORE any matplotlib imports
# (around line 150-160, in the frozen app setup section):

if is_frozen:
    try:
        from ms_mint_app.prebuild_cache import setup_bundled_matplotlib_cache
        setup_bundled_matplotlib_cache()
        
        # Your existing matplotlib config can be simplified:
        # The setup function already handles MPLCONFIGDIR
        import logging as _logging
        _logging.getLogger("matplotlib").setLevel(_logging.WARNING)
        _logging.getLogger("matplotlib.font_manager").setLevel(_logging.WARNING)
    except Exception:
        pass
"""

# ============================================================================
# BUILD PROCESS INTEGRATION
# ============================================================================

BUILD_INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════════╗
║              BUILD PROCESS INTEGRATION STEPS                         ║
╚══════════════════════════════════════════════════════════════════════╝

STEP 1: Create assets directory if it doesn't exist
───────────────────────────────────────────────────────────────────────
  mkdir -p assets

STEP 2: Copy pre-built cache
───────────────────────────────────────────────────────────────────────
  cp -r prebuild_matplotlib_cache/matplotlib assets/matplotlib_cache

STEP 3: Add to PyInstaller spec file
───────────────────────────────────────────────────────────────────────
  In your .spec file, update the datas section:
  
  datas=[
      # ... existing data files ...
      ('assets/matplotlib_cache', 'assets/matplotlib_cache'),
  ]

STEP 4: Add integration code to your codebase
───────────────────────────────────────────────────────────────────────
  Save this file as: src/ms_mint_app/prebuild_cache.py
  
  Then in Mint.py (around line 153), add:
  
  if is_frozen:
      try:
          from ms_mint_app.prebuild_cache import setup_bundled_matplotlib_cache
          setup_bundled_matplotlib_cache()
          
          # matplotlib logging config
          mpl_cfg = P(DATADIR) / ".cache" / "matplotlib"
          os.environ.setdefault("MPLCONFIGDIR", str(mpl_cfg))
          
          import logging as _logging
          _logging.getLogger("matplotlib").setLevel(_logging.WARNING)
          _logging.getLogger("matplotlib.font_manager").setLevel(_logging.WARNING)
      except Exception:
          pass

STEP 5: Rebuild cache when matplotlib is updated
───────────────────────────────────────────────────────────────────────
  When you update matplotlib version:
  
  1. python prebuild_matplotlib_cache.py
  2. cp -r prebuild_matplotlib_cache/matplotlib assets/matplotlib_cache
  3. Rebuild your frozen app

────────────────────────────────────────────────────────────────────────
RESULT
────────────────────────────────────────────────────────────────────────

✅ First-time users: No 5-second matplotlib cache delay!
✅ App bundle size: +158 KB (negligible)
✅ Platform-specific: Build separate caches for Windows/macOS/Linux
✅ Maintenance: Regenerate cache when matplotlib updates

────────────────────────────────────────────────────────────────────────
PLATFORM-SPECIFIC BUILDS
────────────────────────────────────────────────────────────────────────

If building for multiple platforms, create platform-specific caches:

  # On Linux build machine
  python prebuild_matplotlib_cache.py
  mv prebuild_matplotlib_cache/matplotlib assets/matplotlib_cache_linux
  
  # On macOS build machine  
  python prebuild_matplotlib_cache.py
  mv prebuild_matplotlib_cache/matplotlib assets/matplotlib_cache_macos
  
  # On Windows build machine
  python prebuild_matplotlib_cache.py
  mv prebuild_matplotlib_cache/matplotlib assets/matplotlib_cache_windows
  
Then in setup_bundled_matplotlib_cache(), detect platform:
  
  import platform
  system = platform.system().lower()
  bundled_cache = Path(sys._MEIPASS) / 'assets' / f'matplotlib_cache_{system}'

"""

if __name__ == "__main__":
    print(BUILD_INSTRUCTIONS)
    print("\n✅ This file can be saved as: src/ms_mint_app/prebuild_cache.py")
