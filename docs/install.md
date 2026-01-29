   
# Installation notes
## Executables
Builds are provided with all dependencies integrated for Windows, Linux, and macOS on the GitHub releases page:  
https://github.com/LewisResearchGroup/ms-mint-app2/releases/latest

After double-clicking the executable, a terminal window will open, followed by the app in your default web browser. If the browser does not open automatically, navigate to `http://localhost:9999` (or the port printed in the terminal log).

> **Note**: If port 9999 is already in use, MINT automatically selects a free port and prints a warning with the new URL.

## Installation with `pip` (Recommended for Users)
The latest stable release can be installed in a Python 3.12+ environment using `pip`. We recommend using [conda](https://docs.anaconda.com/free/miniconda/) or [mamba](https://conda-forge.org/miniforge/) to create a virtual environment.

```bash
# Create conda environment
conda create -n ms-mint-app2 python==3.12
conda activate ms-mint-app2

# Install the package normally
pip install ms-mint-app2

```

This will download and install all necessary dependencies and MINT.  
Start the app via:

```
Mint
```

## Installation from source
If you want to contribute to MINT or use the latest unreleased features, install from source.

```bash
# Create conda environment
conda create -n ms-mint-app2 python==3.12
conda activate ms-mint-app2

# Get the code
git clone https://github.com/LewisResearchGroup/ms-mint-app2
cd ms-mint-app2

# Install the package normally
pip install .

# Or, install the package in development mode
pip install -e .
```

## Options
After installation, MINT can be started by running `Mint`.

```console
Mint --help
usage: Mint [-h] [--no-browser] [--version] [--data-dir DATA_DIR] [--debug] [--port PORT] [--host HOST] [--ncpu NCPU] [--local] [--config CONFIG] [--repo-path REPO_PATH] [--fallback-repo-path FALLBACK_REPO_PATH] [--skip-update] [--serve-path SERVE_PATH]

MINT frontend.

optional arguments:
  -h, --help            show this help message and exit
  --no-browser          do not start the browser
  --version             print current version
  --data-dir            target directory for MINT data
  --debug               start MINT server in debug mode
  --port                change the port (default: 9999)
  --host                host binding address (default: 127.0.0.1)
  --ncpu                number of CPUs to use for processing
  --local               run locally and use the File System Access API (best for desktop usage)
  --config              path to JSON config (auto-created if missing)
  --repo-path           repo path or VCS URL for auto-update on launch
  --fallback-repo-path  fallback repo path if the primary update fails
  --skip-update         skip auto-update before launching
  --serve-path          (deprecated) serve app at a different path, e.g. '/mint/'
```

If the browser does not open automatically, open it manually and navigate to `http://localhost:9999` (or the port shown in the terminal). The app's frontend is built using [Plotly-Dash](https://plot.ly/dash/) and runs locally in a browser. The GUI is under active development and may be optimized in the future.

### Configuration file
On first run, MINT creates a JSON config file (default: `~/.mint_config.json`) that can store:
- `repo_path` and `fallback_repo_path` (for auto-update on launch)
- `data_dir` (default MINT data directory)
