import os
import subprocess
import logging
import yaml
import shutil
import numpy as np
import time
from pathlib import Path

# Try importing pyopenms
try:
    from pyopenms import MSExperiment, MSSpectrum, MzMLFile, IonSource
    PYOPENMS_AVAILABLE = True
except ImportError:
    PYOPENMS_AVAILABLE = False

from ..duckdb_manager import duckdb_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_ms1_from_db(conn, ms_file_label, output_path, polarity_str):
    """
    Exports MS1 data for a given file label from DuckDB to an mzML file
    using pyopenms.
    """
    if not PYOPENMS_AVAILABLE:
        raise ImportError("pyopenms is required for mzML export.")

    # Fetch data grouped by scan_id
    query = """
        SELECT scan_id, MAX(scan_time) as rt, LIST(mz) as mzs, LIST(intensity) as intensities
        FROM ms1_data
        WHERE ms_file_label = ?
        GROUP BY scan_id
        ORDER BY scan_id
    """
    scans = conn.execute(query, [ms_file_label]).fetchall()
    
    is_positive = polarity_str.lower().startswith('pos') or polarity_str == '+'
    
    exp = MSExperiment()
    
    for scan_id, rt, mzs, intensities in scans:
        if not mzs:
            continue
            
        spec = MSSpectrum()
        spec.setRT(float(rt)) # Seconds
        spec.setMSLevel(1)
        
        # Set peaks
        mz_array = np.array(mzs, dtype=np.float64)
        int_array = np.array(intensities, dtype=np.float32) # or float64
        spec.set_peaks((mz_array, int_array))
        
        # Set Polarity
        settings = spec.getInstrumentSettings()
        if is_positive:
            settings.setPolarity(IonSource.Polarity.POSITIVE)
        else:
            settings.setPolarity(IonSource.Polarity.NEGATIVE)
        spec.setInstrumentSettings(settings)
        
        # Add to experiment
        exp.addSpectrum(spec)

    MzMLFile().store(output_path, exp)


def run_asari_workflow(wdir, params, set_progress=None):
    """
    Orchestrates the Asari workflow using data from DuckDB:
    1. Create temp directory in WORKSPACE (PERSISTENT for debugging).
    2. Export MS1 data from DuckDB to mzML in temp dir.
    3. Run asari process on temp dir.
    4. Process results and return.
    """
    
    def report_progress(percent, message, detail=""):
        if set_progress:
            set_progress((percent, message, detail))
            
    report_progress(0, "Initializing", "Checking requirements...")
    
    # 1. Check Asari availability
    try:
        subprocess.run(["asari", "--help"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        return {"success": False, "message": "Asari executable not found. Please install it using 'pip install asari'."}
    except Exception as e:
         return {"success": False, "message": f"Error checking asari: {e}"}

    if not PYOPENMS_AVAILABLE:
        return {"success": False, "message": "pyopenms not found. Please install it to export data."}

    # 2. Export Files from DB to Temp Dir
    report_progress(5, "Preparing Data", "Connecting to database...")
    
    # Create temp dir in the WORKSPACE as requested
    run_id = f"asari_run_{int(time.time())}"
    temp_dir = os.path.join(wdir, run_id)
    os.makedirs(temp_dir, exist_ok=True)
    
    logger.info(f"==========================================")
    logger.info(f"DEBUG MODE: Asari temp directory: {temp_dir}")
    logger.info(f"==========================================")

    try:
        with duckdb_connection(wdir) as conn:
            if conn is None:
                return {"success": False, "message": "Could not connect to workspace database."}
            
            # Get list of files marked for processing
            samples = conn.execute("""
                SELECT ms_file_label, polarity 
                FROM samples 
                WHERE use_for_processing = TRUE OR use_for_optimization = TRUE
            """).fetchall()
            
            if not samples:
                return {"success": False, "message": "No samples selected for processing (check 'For Processing' or 'For Optimization' flags)."}
            
            total_files = len(samples)
            logger.info(f"Found {total_files} samples to process.")
            
            for i, (label, polarity) in enumerate(samples):
                report_progress(
                    10 + int(30 * (i / total_files)), 
                    "Exporting Data", 
                    f"Exporting {label} ({i+1}/{total_files})..."
                )
                
                # Synthesize filename
                file_path = os.path.join(temp_dir, f"{label}.mzML")
                try:
                    export_ms1_from_db(conn, label, file_path, polarity or "Positive")
                except Exception as ex:
                    logger.error(f"Failed to export {label}: {ex}")
                    pass
                
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "message": f"Error exporting data from database: {e}"}

    # 3. Generate Parameter File
    report_progress(45, "Configuration", "Generating parameter file...")
    param_file_path = os.path.join(temp_dir, "asari_parameters.yaml")
    
    asari_params = {
        'project_name': 'asari_project',
        'mz_tolerance_ppm': int(params.get('mz_tolerance_ppm', 5)),
        'mode': params.get('mode', 'pos'),
        'multicores': int(params.get('multicores', 4)),
        'outdir': "asari_results", 
        'signal_noise_ratio': int(params.get('signal_noise_ratio', 5)),
        'min_peak_height': int(params.get('min_peak_height', 10000)),
        'min_timepoints': int(params.get('min_timepoints', 6)),
        'database_mode': 'auto',
        'rt_align_method': 'lowess'
    }
    
    try:
        with open(param_file_path, 'w') as f:
            yaml.dump(asari_params, f, default_flow_style=False)
            logger.info(f"Parameters written to {param_file_path}")
    except Exception as e:
            return {"success": False, "message": f"Failed to write parameter file: {e}"}

    # 4. Run Asari
    report_progress(50, "Running Asari", "Starting Asari process...")
    
    # FIXED: Added --input argument pointing to the directory containing mzML files
    cmd = ["asari", "process", "--parameters", "asari_parameters.yaml", "--input", temp_dir]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    logger.info(f"Working directory: {temp_dir}")
    
    try:
        process = subprocess.Popen(
            cmd, 
            cwd=temp_dir, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, # Merge stderr for simpler streaming
            text=True, 
            bufsize=1
        )
        
        # Monitor stdout
        for line in process.stdout:
            line = line.strip()
            if line:
                # Log to terminal
                print(f"[ASARI] {line}") 
                if "processing" in line.lower():
                    report_progress(60, "Running Asari", "Processing samples...")
                elif "correspondence" in line.lower():
                    report_progress(80, "Running Asari", " Correspondence analysis...")
                elif "annotation" in line.lower():
                    report_progress(90, "Running Asari", "Annotating features...")

        process.wait()

        if process.returncode != 0:
             logger.error(f"Asari failed with exit code {process.returncode}")
             return {"success": False, "message": f"Asari failed with exit code {process.returncode}. Check terminal for logs."}

    except Exception as e:
        logger.error(f"Exception running asari: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "message": f"Error running Asari: {e}"}
        
    # 5. Process Results
    report_progress(95, "Finalizing", "Processing results...")
    
    results_dir = os.path.join(temp_dir, "asari_results", "export")
    if not os.path.exists(results_dir):
        results_dir = os.path.join(temp_dir, "asari_results")
        
    feature_tables = list(Path(results_dir).rglob("*full_feature_table.tsv"))
    if not feature_tables:
        feature_tables = list(Path(results_dir).rglob("*preferred_feature_table.tsv"))
        
    if not feature_tables:
         logger.warning(f"No feature tables found in {results_dir}")
         # List content of temp dir for debugging
         for root, dirs, files in os.walk(temp_dir):
             for f in files:
                 logger.info(f"Found file: {os.path.join(root, f)}")
                 
         return {"success": False, "message": "Asari completed but no feature table found."}
         
    target_table_path = str(feature_tables[0])
    logger.info(f"Found result table: {target_table_path}")
    
    dest_path = os.path.join(wdir, "asari_results_latest.tsv")
    shutil.copy(target_table_path, dest_path)
    logger.info(f"Copied result to {dest_path}")
    
    report_progress(100, "Done", "Workflow completed.")
    
    return {
        "success": True, 
        "message": f"Asari analysis completed successfully. Results saved to {dest_path}",
        "result_path": dest_path
    }
