#!/usr/bin/env python3
"""
Script to create a minimal Python virtual environment with Asari installed.
This environment is bundled with the PyInstaller app to allow Asari to run
via subprocess without relying on the user's system Python.

Usage:
    python pyinstaller/create_asari_env.py

This will create: pyinstaller/asari_env/
"""

import subprocess
import sys
import os
import shutil
import sysconfig
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
ENV_DIR = SCRIPT_DIR / "asari_env"

# Packages to install in the asari environment
ASARI_VERSION = "1.13.1"
PACKAGES = [
    f"asari-metabolomics=={ASARI_VERSION}",  # Known-good version for the bundled env
]

PY_MAJOR_MINOR = f"{sys.version_info.major}.{sys.version_info.minor}"


def copy_stdlib_into_venv():
    """
    Make the bundled venv relocatable by copying the Python stdlib into it.

    On GitHub runners, venv points at hostedtoolcache paths that do not exist
    on end-user machines. A self-contained stdlib avoids missing core modules
    such as encodings/random at runtime.
    """
    src_stdlib = Path(sysconfig.get_path("stdlib"))
    if not src_stdlib.exists():
        raise FileNotFoundError(f"Could not find stdlib at {src_stdlib}")

    if sys.platform == "win32":
        dst_stdlib = ENV_DIR / "Lib"
        src_dynload = Path(sys.base_prefix) / "DLLs"
        dst_dynload = ENV_DIR / "DLLs"
    else:
        dst_stdlib = ENV_DIR / "lib" / f"python{PY_MAJOR_MINOR}"
        src_dynload = src_stdlib / "lib-dynload"
        dst_dynload = dst_stdlib / "lib-dynload"

    print(f"Copying stdlib from {src_stdlib} to {dst_stdlib}")
    dst_stdlib.mkdir(parents=True, exist_ok=True)

    for item in src_stdlib.iterdir():
        if item.name in {"site-packages", "__pycache__"}:
            continue
        if item.is_dir():
            shutil.copytree(item, dst_stdlib / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst_stdlib / item.name)

    if src_dynload.exists():
        print(f"Copying dynload modules from {src_dynload} to {dst_dynload}")
        shutil.copytree(src_dynload, dst_dynload, dirs_exist_ok=True)


def make_windows_interpreter_portable():
    """
    Replace venv redirector executables with real interpreter binaries.

    The default venv launcher on Windows points to the build machine's base
    interpreter (e.g. hostedtoolcache). After distribution this path is gone,
    causing "No Python at ..." errors.
    """
    if sys.platform != "win32":
        return

    scripts_dir = ENV_DIR / "Scripts"
    base_prefix = Path(sys.base_prefix)
    base_exec_prefix = Path(sys.exec_prefix)

    # Copy actual interpreter launchers over venv redirectors.
    for exe_name in ("python.exe", "pythonw.exe"):
        src = base_prefix / exe_name
        dst = scripts_dir / exe_name
        if src.exists():
            print(f"Copying {src} -> {dst}")
            shutil.copy2(src, dst)

    # Copy runtime DLLs required by python.exe into Scripts for local loading.
    dll_candidates = (
        "python3.dll",
        f"python{sys.version_info.major}{sys.version_info.minor}.dll",
        "vcruntime140.dll",
        "vcruntime140_1.dll",
        "ucrtbase.dll",
    )
    for dll_name in dll_candidates:
        for prefix in (base_prefix, base_exec_prefix):
            src = prefix / dll_name
            if src.exists():
                dst = scripts_dir / dll_name
                print(f"Copying {src} -> {dst}")
                shutil.copy2(src, dst)
                break

    # Make pyvenv.cfg portable (no hostedtoolcache absolute paths).
    pyvenv_cfg = ENV_DIR / "pyvenv.cfg"
    pyvenv_cfg.write_text(
        "\n".join(
            [
                f"home = {ENV_DIR}",
                "include-system-site-packages = false",
                f"version = {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main():
    print(f"Creating Asari environment at: {ENV_DIR}")
    
    # Remove existing env if present
    if ENV_DIR.exists():
        print(f"Removing existing environment...")
        shutil.rmtree(ENV_DIR)
    
    # Create virtual environment
    print("Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", str(ENV_DIR)], check=True)
    
    # Determine python/pip paths
    if sys.platform == "win32":
        python_path = ENV_DIR / "Scripts" / "python.exe"
        pip_path = ENV_DIR / "Scripts" / "pip.exe"
    else:
        python_path = ENV_DIR / "bin" / "python"
        pip_path = ENV_DIR / "bin" / "pip"
    
    # Upgrade pip
    print("Upgrading pip...")
    subprocess.run([str(python_path), "-m", "pip", "install", "--upgrade", "pip"], check=True)

    # Ensure stdlib is physically present in the venv for portability.
    copy_stdlib_into_venv()
    make_windows_interpreter_portable()
    
    # Install packages
    print(f"Installing packages: {PACKAGES}")
    subprocess.run([str(pip_path), "install"] + PACKAGES, check=True)
    
    # Verify installation
    print("\nVerifying Asari installation...")
    if sys.platform == "win32":
        asari_path = ENV_DIR / "Scripts" / "asari.exe"
    else:
        asari_path = ENV_DIR / "bin" / "asari"
    
    result = subprocess.run(
        [str(asari_path), "--help"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("[OK] Asari installed successfully!")
    else:
        print("[ERROR] Asari installation verification failed:")
        print(result.stderr)
        sys.exit(1)
    
    # Print size info
    total_size = sum(f.stat().st_size for f in ENV_DIR.rglob('*') if f.is_file())
    print(f"\nEnvironment size: {total_size / (1024*1024):.1f} MB")
    print(f"Environment location: {ENV_DIR}")
    print("\nDone! The environment is ready to be bundled with PyInstaller.")


if __name__ == "__main__":
    main()
