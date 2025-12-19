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
    + _safe_collect_submodules('webview')
)


a = Analysis(
    [script],
    pathex=[src_dir, package_root],
    hookspath=[hooks_dir],
    hiddenimports=all_hidden_imports,
    module_collection_mode={
        'ms_mint_app': 'pyz+py',
    },
    datas=[
        (os.path.join(package_root, 'ms_mint_app', 'assets'), os.path.join('ms_mint_app', 'assets')),
        (os.path.join(package_root, 'ms_mint_app', 'static'), os.path.join('ms_mint_app', 'static')),
    ],
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
