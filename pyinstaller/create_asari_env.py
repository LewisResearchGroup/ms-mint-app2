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
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
ENV_DIR = SCRIPT_DIR / "asari_env"

# Packages to install in the asari environment
PACKAGES = [
    "asari-metabolomics",  # Main package
]


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
        print("✓ Asari installed successfully!")
    else:
        print("✗ Asari installation verification failed:")
        print(result.stderr)
        sys.exit(1)
    
    # Print size info
    total_size = sum(f.stat().st_size for f in ENV_DIR.rglob('*') if f.is_file())
    print(f"\nEnvironment size: {total_size / (1024*1024):.1f} MB")
    print(f"Environment location: {ENV_DIR}")
    print("\nDone! The environment is ready to be bundled with PyInstaller.")


if __name__ == "__main__":
    main()
