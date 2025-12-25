# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_submodules

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

# Path to PyOpenMS share directory (contains OpenMS data files)
pyopenms_share = os.path.join(
    sys.prefix, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}',
    'site-packages', 'pyopenms', 'share'
)

a = Analysis(
    [script],
    pathex=[src_dir, package_root],
    hookspath=[hooks_dir],
    hiddenimports=all_hidden_imports,
    module_collection_mode={
        'ms_mint_app': 'pyz+py',
    },
    binaries=[
        # OpenSSL libraries - use conda env versions to avoid version conflicts
        (os.path.join(sys.prefix, 'lib', 'libssl.so.3'), '.'),
        (os.path.join(sys.prefix, 'lib', 'libcrypto.so.3'), '.'),
        # libstdc++ - use conda env version which has CXXABI_1.3.15
        (os.path.join(sys.prefix, 'lib', 'libstdc++.so.6'), '.'),
    ],
    datas=[
        (os.path.join(package_root, 'ms_mint_app', 'assets'), os.path.join('ms_mint_app', 'assets')),
        (os.path.join(package_root, 'ms_mint_app', 'static'), os.path.join('ms_mint_app', 'static')),
        # Bundle the Asari Python environment
        (asari_env_dir, 'asari_env'),
        # Bundle PyOpenMS share directory for OPENMS_DATA_PATH
        (pyopenms_share, 'pyopenms_share'),
    ],
    excludes=['PySide6'],  # Exclude to avoid Qt bindings conflict with PyQt5
    runtime_hooks=[os.path.join(hooks_dir, 'rthook_openms.py')],
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name='Mint',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='Mint',
)
