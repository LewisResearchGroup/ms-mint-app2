# Complete PyInstaller Build Guide

This guide covers the **complete process** for building MS-MINT frozen executables with optimized startup performance.

## Prerequisites

Before building, ensure you have:
- Python 3.10+ installed
- PyInstaller installed: `pip install pyinstaller`
- All project dependencies installed: `pip install -e .`
- Matplotlib 3.10.8+: `pip install 'matplotlib>=3.10.8'`

---

## Build Process (Step-by-Step)

### Step 1: Create Asari Environment

The Asari environment is a bundled Python virtual environment that allows Asari to run independently.

```bash
cd pyinstaller
python create_asari_env.py
```

**What this does:**
- Creates `pyinstaller/asari_env/` directory
- Installs `asari-metabolomics` package in isolated environment
- Verifies installation
- Reports environment size

**Expected output:**
```
✓ Asari installed successfully!
Environment size: ~XXX MB
Environment location: .../pyinstaller/asari_env
```

**Platform notes:**
- **Windows**: Creates `asari_env/Scripts/asari.exe`
- **Linux/macOS**: Creates `asari_env/bin/asari`

---

### Step 2: Generate Matplotlib Font Cache

Pre-build the matplotlib font cache to eliminate 5-second first-time user delay.

```bash
# Still in pyinstaller/ directory
python prebuild_matplotlib_cache.py
```

**What this does:**
- Builds matplotlib font cache for your platform
- Creates `pyinstaller/prebuild_matplotlib_cache/matplotlib/` directory
- Generates `fontlist-v390.json` (158 KB)

**Expected output:**
```
✓ Font cache built!
Total cache size: 158,332 bytes (0.15 MB)
Matplotlib version: 3.10.8
```

**Platform-specific caches:**
The cache is platform-specific due to different system fonts:
- **Windows**: Different from Linux/macOS
- **Linux**: Different from Windows/macOS  
- **macOS**: Different from Windows/Linux

Build on each target platform to generate its cache.

---

### Step 3: Build the Frozen Application

Now build the complete frozen application with PyInstaller.

```bash
# Still in pyinstaller/ directory
pyinstaller Mint.spec
```

**What this does:**
- Analyzes dependencies and creates executables
- Bundles Asari environment from `asari_env/`
- Bundles matplotlib cache from `prebuild_matplotlib_cache/matplotlib/`
- Includes all application assets and data files
- Creates platform-specific executable

**Expected output:**
```
✓ Bundling matplotlib cache for linux (154.6 KB)
Building EXE...
Building COLLECT...
Done!
```

**Output location:**
- **Frozen app**: `dist/Mint/`
- **Executable**: `dist/Mint/Mint` (or `Mint.exe` on Windows)

---

## Platform-Specific Instructions

### Linux
```bash
cd pyinstaller
python create_asari_env.py          # Step 1
python prebuild_matplotlib_cache.py # Step 2
pyinstaller Mint.spec               # Step 3

# Result: dist/Mint/ with matplotlib_cache_linux/
```

### Windows
```bash
cd pyinstaller
python create_asari_env.py          # Step 1
python prebuild_matplotlib_cache.py # Step 2
pyinstaller Mint.spec               # Step 3

# Result: dist\Mint\ with matplotlib_cache_windows\
```

### macOS
```bash
cd pyinstaller
python create_asari_env.py          # Step 1
python prebuild_matplotlib_cache.py # Step 2
pyinstaller Mint.spec               # Step 3

# Result: dist/Mint/ with matplotlib_cache_darwin/
```

---

## Verification

After building, verify everything is bundled correctly:

### 1. Check Asari Environment
```bash
# Linux/macOS
ls -R dist/Mint/asari_env/bin/asari

# Windows
dir dist\Mint\asari_env\Scripts\asari.exe
```

### 2. Check Matplotlib Cache
```bash
# Linux
ls dist/Mint/assets/matplotlib_cache_linux/fontlist-v390.json

# Windows
dir dist\Mint\assets\matplotlib_cache_windows\fontlist-v390.json

# macOS
ls dist/Mint/assets/matplotlib_cache_darwin/fontlist-v390.json
```

### 3. Test Startup Performance
```bash
# Linux/macOS
time ./dist/Mint/Mint --no-browser

# Windows
# Use Task Manager or time command to measure

# Expected: ~2 seconds startup time
```

---

## When to Rebuild

### Rebuild Asari Environment When:
- Asari package is updated
- Python version changes
- Building for new platform

**Command:**
```bash
cd pyinstaller
python create_asari_env.py
```

### Rebuild Matplotlib Cache When:
- Matplotlib version is updated (e.g., 3.10.8 → 3.11.0)
- Building for new platform
- System fonts significantly changed (rare)

**Command:**
```bash
cd pyinstaller
python prebuild_matplotlib_cache.py
```

### Rebuild Application When:
- Source code changes
- Dependencies updated
- After rebuilding Asari env or matplotlib cache

**Command:**
```bash
cd pyinstaller
pyinstaller Mint.spec
```

---

## Build Artifacts

After successful build, you'll have:

```
pyinstaller/
├── asari_env/                      # Bundled Asari environment (~XXX MB)
├── prebuild_matplotlib_cache/      # Pre-built font cache
│   └── matplotlib/
│       └── fontlist-v390.json      # (158 KB)
├── dist/
│   └── Mint/                       # Frozen application
│       ├── Mint                    # Main executable
│       ├── asari_env/              # Bundled Asari
│       ├── assets/
│       │   └── matplotlib_cache_linux/  # (or _windows/_darwin)
│       ├── ms_mint_app/
│       └── ... (dependencies)
└── build/                          # Build artifacts (can be deleted)
```

---

## Expected Performance

### Startup Time

| User Type | Without Optimization | With Optimization | Improvement |
|-----------|---------------------|-------------------|-------------|
| **First-time** | ~21 seconds | **~2 seconds** | **90% faster** ⚡ |
| **Returning** | ~12 seconds | **~2 seconds** | **83% faster** ⚡ |

### Bundle Size Impact

| Component | Size |
|-----------|------|
| Base application | ~XXX MB |
| Asari environment | ~XXX MB |
| Matplotlib cache | **+158 KB** |
| **Total increase** | **<0.1%** |

---

## Troubleshooting

### "Asari environment not found"
**Solution**: Run `python create_asari_env.py` before building

### "Matplotlib cache not found"  
**Solution**: Run `python prebuild_matplotlib_cache.py` before building

### Slow startup on Windows
**Cause**: Missing Windows-specific cache (built on Linux)  
**Solution**: Build on Windows or include all platform caches

### "Permission denied" on Linux
**Solution**: Make executable: `chmod +x dist/Mint/Mint`

---

## Distribution

### Single Platform
Distribute `dist/Mint/` directory with:
- Platform-specific executable
- Platform-specific matplotlib cache
- Bundled Asari environment

### Multi-Platform
Build separately on each platform:
1. Build on Linux → `dist_linux/`
2. Build on Windows → `dist_windows/`
3. Build on macOS → `dist_macos/`

Each distribution is self-contained and optimized for its platform.

---

## Summary Checklist

Before building, ensure:
- [ ] In `pyinstaller/` directory
- [ ] Asari environment created (`asari_env/` exists)
- [ ] Matplotlib cache generated (`prebuild_matplotlib_cache/matplotlib/` exists)
- [ ] Dependencies installed (`pip install -e .`)
- [ ] Matplotlib 3.10.8+ installed

Then run:
```bash
pyinstaller Mint.spec
```

**Result**: Optimized frozen app in `dist/Mint/` with <2 second startup! 🚀

---

## Related Documentation

- **Asari Environment**: `create_asari_env.py` - Script documentation
- **Cache Details**: `CROSS_PLATFORM_CACHE_GUIDE.md` - Technical details
- **Spec File**: `Mint.spec` - PyInstaller configuration

For detailed explanation of the optimization strategy, see `CROSS_PLATFORM_CACHE_GUIDE.md`.
