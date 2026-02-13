# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_submodules, copy_metadata

src_dir = os.path.abspath(os.path.join(SPECPATH, os.pardir))
hooks_dir = os.path.join(src_dir, 'pyinstaller', 'hooks')
package_root = os.path.join(src_dir, 'src')
script = os.path.join(package_root, 'ms_mint_app', 'scripts', 'Mint.py')


def _safe_collect_submodules(module_name: str):
    try:
        return collect_submodules(module_name)
    except Exception:
        return []


all_hidden_imports = (
    _safe_collect_submodules('sklearn')
    + _safe_collect_submodules('bs4')
    + _safe_collect_submodules('scipy')
    + _safe_collect_submodules('pyarrow')
    + _safe_collect_submodules('ms_mint_app')
    + _safe_collect_submodules('packaging')
    + _safe_collect_submodules('brotli')
    + _safe_collect_submodules('waitress')
    + _safe_collect_submodules('dash')
    + _safe_collect_submodules('dash_extensions')
    + _safe_collect_submodules('feffery_antd_components')
    + _safe_collect_submodules('feffery_utils_components')
    + _safe_collect_submodules('numpy')
    + _safe_collect_submodules('webview')
    + _safe_collect_submodules('diskcache')
    + _safe_collect_submodules('flask_caching')
    + _safe_collect_submodules('duckdb')
    + _safe_collect_submodules('polars')
    + _safe_collect_submodules('fastcluster')
)


import sys

# Path to the bundled Asari environment (created by create_asari_env.py)
asari_env_dir = os.path.join(SPECPATH, 'asari_env')

# Platform-specific configurations
binaries_list = []
datas_list = [
    (os.path.join(package_root, 'ms_mint_app', 'assets'), os.path.join('ms_mint_app', 'assets')),
]

# Bundle package metadata so importlib.metadata.version(...) works in frozen builds.
for pkg in ("ms-mint-app2", "ms_mint_app2"):
    try:
        datas_list += copy_metadata(pkg)
        break
    except Exception:
        pass

static_dir = os.path.join(package_root, 'ms_mint_app', 'static')
if os.path.isdir(static_dir):
    datas_list.append((static_dir, os.path.join('ms_mint_app', 'static')))

# Add Asari environment if it exists
if os.path.isdir(asari_env_dir):
    datas_list.append((asari_env_dir, 'asari_env'))

# ============================================================================
# INCLUDE SOURCE FILES FOR DASH BACKGROUND CALLBACKS
# ============================================================================
# Include .py source files so inspect.getsource() works in frozen executable.
# This is needed for Dash's DiskcacheManager to build cache keys for
# background callbacks.
# ============================================================================
import glob

def collect_source_files(package_path, package_name):
    """Collect all .py files from a package for inclusion in frozen app."""
    py_files = []
    for py_file in glob.glob(os.path.join(package_path, '**', '*.py'), recursive=True):
        # Get relative path from package root
        rel_path = os.path.relpath(py_file, package_path)
        dest_dir = os.path.join(package_name, os.path.dirname(rel_path))
        py_files.append((py_file, dest_dir))
    return py_files

# Add ms_mint_app source files
ms_mint_app_src = os.path.join(package_root, 'ms_mint_app')
source_files = collect_source_files(ms_mint_app_src, 'ms_mint_app')
datas_list.extend(source_files)
print(f"[OK] Bundling {len(source_files)} source files for background callbacks")

# ============================================================================
# MATPLOTLIB FONT CACHE - Startup Optimization
# ============================================================================
# Bundle pre-built matplotlib font cache to eliminate 5-second first-time delay.
# The cache is platform-specific due to different system fonts.
# 
# To rebuild cache (when matplotlib is updated):
#   cd pyinstaller
#   python prebuild_matplotlib_cache.py
#
# This copies the prebuild_matplotlib_cache/matplotlib/ directory into assets/
# during the build process.
# ============================================================================

import platform
matplotlib_cache_src = os.path.join(SPECPATH, 'prebuild_matplotlib_cache', 'matplotlib')

# Check if pre-built cache exists
if os.path.isdir(matplotlib_cache_src):
    # Bundle with platform identifier for cross-platform builds
    system = platform.system().lower()
    datas_list.append((matplotlib_cache_src, os.path.join('assets', f'matplotlib_cache_{system}')))
    print(f"[OK] Bundling matplotlib cache for {system} ({os.path.getsize(os.path.join(matplotlib_cache_src, 'fontlist-v390.json'))/1024:.1f} KB)")
else:
    print("[WARNING] Matplotlib cache not found. Run 'python prebuild_matplotlib_cache.py' in pyinstaller/ to generate it.")

# ============================================================================
# PyInstaller Configuration
# ============================================================================

a = Analysis(
    [script],
    pathex=[],
    binaries=binaries_list,
    datas=datas_list,
    hiddenimports=all_hidden_imports,
    hookspath=[hooks_dir],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

# Prefer a known-good libstdc++ from the active conda env to satisfy newer CXXABI symbols.
a.binaries = [b for b in a.binaries if os.path.basename(b[0]) != "libstdc++.so.6"]
#
# Also force OpenSSL libraries from the same Python/conda prefix as _ssl to avoid
# ABI mismatches like _ssl requiring OPENSSL_3.3.0 while a system libcrypto 3.0 is bundled.
a.binaries = [
    b for b in a.binaries
    if os.path.basename(b[0]) not in {"libssl.so.3", "libcrypto.so.3"}
]
conda_prefix = os.environ.get("CONDA_PREFIX")
if not conda_prefix:
    conda_prefix = os.path.dirname(os.path.dirname(sys.executable))
conda_lib = os.path.join(conda_prefix, "lib")
conda_libstdcpp = os.path.join(conda_lib, "libstdc++.so.6")
conda_libgcc = os.path.join(conda_lib, "libgcc_s.so.1")
conda_libssl = os.path.join(conda_lib, "libssl.so.3")
conda_libcrypto = os.path.join(conda_lib, "libcrypto.so.3")
if os.path.isfile(conda_libstdcpp):
    a.binaries.append((os.path.basename(conda_libstdcpp), conda_libstdcpp, "BINARY"))
if os.path.isfile(conda_libgcc):
    a.binaries.append((os.path.basename(conda_libgcc), conda_libgcc, "BINARY"))
if os.path.isfile(conda_libssl):
    a.binaries.append((os.path.basename(conda_libssl), conda_libssl, "BINARY"))
if os.path.isfile(conda_libcrypto):
    a.binaries.append((os.path.basename(conda_libcrypto), conda_libcrypto, "BINARY"))

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Mint',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Mint',
)
