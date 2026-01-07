import os
import subprocess
import logging
import yaml
import shutil
import numpy as np
import time
import sys
import platform
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


def get_asari_command():
    """
    Returns the command to run Asari.
    
    In a frozen PyInstaller app, uses the bundled Python environment.
    Otherwise, uses the system 'asari' command.
    
    Returns:
        tuple: (command_list, is_bundled) where command_list is the base command
               to run asari and is_bundled indicates if using bundled env.
    """
    if getattr(sys, 'frozen', False):
        # Running in PyInstaller frozen app
        # The bundled env is at: <_MEIPASS>/asari_env/ (inside _internal)
        meipass = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
        asari_env_path = os.path.join(meipass, 'asari_env')
        
        # Use Python interpreter + module instead of asari.exe directly
        # This avoids hardcoded paths in pip-installed scripts from build machine
        if sys.platform == 'win32':
            python_path = os.path.join(asari_env_path, 'Scripts', 'python.exe')
            scripts_dir = os.path.join(asari_env_path, 'Scripts')
        else:
            python_path = os.path.join(asari_env_path, 'bin', 'python')
            scripts_dir = os.path.join(asari_env_path, 'bin')
        
        # Patch pyvenv.cfg with correct paths for this machine
        # The file contains hardcoded paths from the build machine that break portability
        pyvenv_cfg = os.path.join(asari_env_path, 'pyvenv.cfg')
        try:
            # Create a minimal pyvenv.cfg pointing to the current location
            cfg_content = f"""home = {scripts_dir}
include-system-site-packages = false
version = 3.12.0
"""
            with open(pyvenv_cfg, 'w') as f:
                f.write(cfg_content)
            logger.info(f"Patched pyvenv.cfg for portability: {pyvenv_cfg}")
        except Exception as e:
            logger.warning(f"Could not patch pyvenv.cfg: {e}")
        
        if os.path.exists(python_path):
            logger.info(f"Using bundled Python for Asari at: {python_path}")
            # Call asari as a module: python -m asari
            return ([python_path, "-m", "asari"], True)
        else:
            logger.warning(f"Bundled Python not found at: {python_path}")
            # Fall back to system asari
            return (["asari"], False)
    else:
        # Running in development mode - use system asari
        return (["asari"], False)


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
    
    
    logs = []

    def report_progress(percent, message, detail="", current_logs=None):
        if set_progress:
            # If current_logs is provided, use it, otherwise join the accumulated logs
            log_text = current_logs if current_logs is not None else "\n".join(logs)
            set_progress((percent, message, detail, log_text))
            
    report_progress(0, "Initializing", "Checking requirements...")
    
    # 1. Check Asari availability
    asari_base_cmd, is_bundled = get_asari_command()
    try:
        subprocess.run(asari_base_cmd + ["--help"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if is_bundled:
            logger.info("Using bundled Asari environment")
        else:
            logger.info("Using system Asari")
    except FileNotFoundError:
        if is_bundled:
            return {"success": False, "message": "Bundled Asari environment not found. Please reinstall the application."}
        else:
            return {"success": False, "message": "Asari executable not found. Please install it using 'pip install asari'."}
    except Exception as e:
         return {"success": False, "message": f"Error checking asari: {e}"}

    if not PYOPENMS_AVAILABLE:
        return {"success": False, "message": "pyopenms not found. Please install it to export data."}

    # 2. Export Files from DB to Temp Dir
    report_progress(5, "Preparing Data", "Connecting to database...")
    
    # Check for existing asari_run directories with mzML files we can reuse
    existing_mzml_dir = None
    existing_mzml_files = set()
    for item in os.listdir(wdir):
        item_path = os.path.join(wdir, item)
        if item.startswith("asari_run_") and os.path.isdir(item_path):
            mzml_files = [f for f in os.listdir(item_path) if f.lower().endswith(".mzml")]
            if mzml_files:
                existing_mzml_dir = item_path
                existing_mzml_files = {os.path.splitext(f)[0] for f in mzml_files}
                logger.info(f"Found {len(mzml_files)} existing mzML files in {item_path}, will reuse them")
                break
    
    # Use existing dir if found, otherwise create new one
    if existing_mzml_dir:
        temp_dir = existing_mzml_dir
        logs.append(f"Reusing existing mzML files from: {temp_dir}")
    else:
        run_id = f"asari_run_{int(time.time())}"
        temp_dir = os.path.join(wdir, run_id)
        os.makedirs(temp_dir, exist_ok=True)
    
    logger.info(f"==========================================")
    logger.info(f"Asari working directory: {temp_dir}")
    logger.info(f"==========================================")
    
    logs.append(f"Working directory: {temp_dir}")

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
            logs.append(f"Found {total_files} samples to process.")
            
            skipped_count = 0
            exported_count = 0
            for i, (label, polarity) in enumerate(samples):
                report_progress(
                    10 + int(30 * (i / total_files)), 
                    "Exporting Data", 
                    f"Processing {label} ({i+1}/{total_files})..."
                )
                
                # Check if mzML already exists (from previous run)
                file_path = os.path.join(temp_dir, f"{label}.mzML")
                if label in existing_mzml_files and os.path.exists(file_path):
                    skipped_count += 1
                    continue  # Skip export, reuse existing file
                
                try:
                    export_ms1_from_db(conn, label, file_path, polarity or "Positive")
                    exported_count += 1
                except Exception as ex:
                    logger.error(f"Failed to export {label}: {ex}")
                    logs.append(f"Failed to export {label}: {ex}")
                    pass
            
            if skipped_count > 0:
                logs.append(f"Reused {skipped_count} existing mzML files, exported {exported_count} new files")
                logger.info(f"Reused {skipped_count} existing mzML files, exported {exported_count} new files")
                
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
    
    if 'gaussian_shape' in params and params['gaussian_shape'] is not None:
         asari_params['gaussian_shape'] = float(params['gaussian_shape'])
    
    try:
        with open(param_file_path, 'w') as f:
            yaml.dump(asari_params, f, default_flow_style=False)
            logger.info(f"Parameters written to {param_file_path}")
            logs.append(f"Parameters written to {param_file_path}")
    except Exception as e:
            return {"success": False, "message": f"Failed to write parameter file: {e}"}

    # 4. Run Asari
    report_progress(50, "Running Asari", "Starting Asari process...")
    
    # Build command using the detected Asari (bundled or system)
    cmd = asari_base_cmd + ["process", "--parameters", "asari_parameters.yaml", "--input", temp_dir]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    logs.append(f"Running command: {' '.join(cmd)}")
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
        current_percent = 50
        current_message = "Running Asari"
        current_detail = "Starting Asari process..."
        
        for line in process.stdout:
            line = line.strip()
            if line:
                # Log to terminal and workspace log
                logger.info(f"[ASARI] {line}") 
                logs.append(line)
                
                # Update progress based on keywords
                if "The reference sample is:" in line:
                    current_percent = 55
                    current_detail = "Identified reference sample..."
                elif "mapped pairs" in line:
                    current_percent = 65
                    current_detail = "Mapping pairs..."
                elif "Peak detection on" in line:
                    current_percent = 70
                    current_detail = "Detecting peaks..."
                elif "Khipu search grid:" in line:
                    current_percent = 80
                    current_detail = "Searching Khipu grid..."
                elif "Unique compound table" in line:
                    current_percent = 95
                    current_detail = "Finalizing results..."
                
                # Report progress with new log line
                report_progress(current_percent, current_message, current_detail)




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
    
    # Asari appends project name and timestamp to outdir
    # e.g., asari_results_asari_project_1223142749
    # We need to find the actual directory.
    possible_result_dirs = [d for d in os.listdir(temp_dir) if d.startswith("asari_results") and os.path.isdir(os.path.join(temp_dir, d))]
    
    if not possible_result_dirs:
         return {"success": False, "message": "Asari completed but output directory not found."}
         
    # Take the most recent one if multiple (though unlikely in fresh temp dir)
    actual_results_dir = sorted(possible_result_dirs)[-1]
    results_dir_path = os.path.join(temp_dir, actual_results_dir)
    export_dir = os.path.join(results_dir_path, "export")
    
    # Check export dir first, then main dir
    search_dir = export_dir if os.path.exists(export_dir) else results_dir_path
        
    # Note: Asari uses 'Feature' with capital F in filenames (full_Feature_table.tsv)
    # User requested to use preferred_Feature_table.tsv (in main dir)
    feature_tables = list(Path(results_dir_path).rglob("*preferred_Feature_table.tsv"))
    if not feature_tables:
        # Fallback to full table in export dir or elsewhere
        feature_tables = list(Path(search_dir).rglob("*full_Feature_table.tsv"))
        
    if not feature_tables:
         logger.warning(f"No feature tables found in {search_dir}")
         # List content of temp dir for debugging
         for root, dirs, files in os.walk(temp_dir):
             for f in files:
                 logger.info(f"Found file: {os.path.join(root, f)}")
                 
         return {"success": False, "message": "Asari completed but no feature table found."}
         
    target_table_path = str(feature_tables[0])
    logger.info(f"Found result table: {target_table_path}")
    
    # NEW: Parse and Convert to MINT Targets
    import pandas as pd
    try:
        report_progress(98, "Finalizing", "Converting to MINT targets...")
        
        df = pd.read_csv(target_table_path, sep='\t')
        
        # --- Post-Processing Filters ---
        # User requested filtering based on cSelectivity and detection_counts
        initial_count = len(df)
        
        # cSelectivity Filter (if column exists and parameter provided)
        cselectivity_threshold = params.get('cselectivity')
        if cselectivity_threshold is not None:
            if 'cSelectivity' in df.columns:
                 df = df[df['cSelectivity'] >= float(cselectivity_threshold)]
                 logger.info(f"Filtered by cSelectivity >= {cselectivity_threshold}. Count: {initial_count} -> {len(df)}")
            else:
                 logger.warning("cSelectivity parameter provided but column not found in Asari output.")

        # Detection Rate Filter (if column exists and parameter provided)
        # Note: Column name might be 'detection_counts' or 'detection counts', checking both
        detection_rate_pct = params.get('detection_rate')
        if detection_rate_pct is not None:
             det_col = None
             if 'detection_counts' in df.columns:
                 det_col = 'detection_counts'
             elif 'detection counts' in df.columns:
                 det_col = 'detection counts'
             
             if det_col:
                 # Calculate required count based on percentage of total files
                 # total_files is available from outer scope
                 min_detections = int(np.ceil((float(detection_rate_pct) / 100.0) * total_files))
                 
                 df = df[df[det_col] >= min_detections]
                 logger.info(f"Filtered by {det_col} >= {min_detections} ({detection_rate_pct}% of {total_files} samples). Count: {initial_count} -> {len(df)}")
             else:
                 logger.warning("detection_rate parameter provided but detection count column not found in Asari output.")
        # -------------------------------

        
        # Column Mapping
        # id_number -> peak_label
        # mz -> mz_mean
        # rtime -> rt
        # rtime_left_base -> rt_min
        # rtime_right_base -> rt_max
        
        rename_map = {
            'id_number': 'peak_label',
            'mz': 'mz_mean',
            'rtime': 'rt',
            'rtime_left_base': 'rt_min',
            'rtime_right_base': 'rt_max'
        }
        
        # Check if columns exist
        missing_cols = [c for c in rename_map.keys() if c not in df.columns]
        if missing_cols:
             return {"success": False, "message": f"Missing expected columns in Asari output: {missing_cols}"}
             
        df = df.rename(columns=rename_map)
        
        # Add mz_width from params
        mz_tol = float(params.get('mz_tolerance_ppm', 5))
        df['mz_width'] = mz_tol
        df['intensity_threshold'] = 0
        df['target_filename'] = 'unknown' # Optional standard fields
        
        # Select only relevant columns
        final_cols = ['peak_label', 'mz_mean', 'mz_width', 'rt', 'rt_min', 'rt_max', 'intensity_threshold']
        df = df[final_cols]
        
        # Check if no features passed filters
        if df.empty:
            suggestions = []
            
            current_detection_rate = params.get('detection_rate', 90)
            if current_detection_rate is not None and float(current_detection_rate) > 50:
                suggestions.append(f"  - Lower Detection Rate from {current_detection_rate}% to 50% or less")
            
            current_snr = params.get('signal_noise_ratio', 20)
            if current_snr is not None and int(current_snr) > 5:
                suggestions.append(f"  - Lower Signal/Noise Ratio from {current_snr} to 5-10")
            
            current_min_height = params.get('min_peak_height', 100000)
            if current_min_height is not None and int(current_min_height) > 10000:
                suggestions.append(f"  - Lower Min Peak Height from {int(current_min_height):,} to 10,000-50,000")
            
            current_cselectivity = params.get('cselectivity', 1.0)
            if current_cselectivity is not None and float(current_cselectivity) > 0.5:
                suggestions.append(f"  - Lower cSelectivity from {current_cselectivity} to 0.5-0.7")
            
            msg = (
                "Asari completed but no features passed the current filter thresholds.\n\n"
                "Suggestions:\n" + "\n".join(suggestions) + "\n\n"
                "This often happens when blanks or low-quality samples are included. "
                "Your mzML files have been kept so the next run will be faster."
            )
            return {"success": False, "message": msg, "no_features": True}
        
        # Save to Workspace data/targets.csv
        # Backup existing targets.csv if present
        data_dir = os.path.join(wdir, "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            
        # Save directly to targets_backup.csv (matching user upload behavior)
        backup_path = os.path.join(data_dir, "targets_backup.csv")
        df.to_csv(backup_path, index=False)
        logger.info(f"Saved MINT targets to {backup_path}")
        
        # INSERT INTO DATABASE
        report_progress(99, "Finalizing", "Updating database...")
        with duckdb_connection(wdir) as conn:
            conn.execute("DELETE FROM targets")
            
            # Prepare dataframe for DB insertion
            # Add missing columns with defaults
            df['mz'] = None
            df['rt_unit'] = 's'
            
            mode_val = params.get('mode', 'pos')
            polarity_map = {'pos': 'Positive', 'neg': 'Negative'}
            df['polarity'] = polarity_map.get(mode_val, 'Positive')
            
            df['filterLine'] = None
            df['ms_type'] = 'ms1'
            df['category'] = None
            df['peak_selection'] = True
            df['score'] = None
            df['bookmark'] = False
            df['source'] = 'Asari'
            df['notes'] = f"Generated by Asari on {time.strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Register df as table for easy insertion
            conn.register('temp_targets_load', df)
            
            # Insert by name to be safe
            insert_query = """
                INSERT INTO targets (
                    peak_label, mz_mean, mz_width, rt, rt_min, rt_max, intensity_threshold, 
                    mz, rt_unit, polarity, filterLine, ms_type, category, 
                    peak_selection, score, bookmark, source, notes
                )
                SELECT 
                    peak_label, mz_mean, mz_width, rt, rt_min, rt_max, intensity_threshold, 
                    mz, CAST(rt_unit AS unit_type_enum), CAST(polarity AS polarity_enum), filterLine, CAST(ms_type AS ms_type_enum), category, 
                    peak_selection, score, bookmark, source, notes
                FROM temp_targets_load
            """
            conn.execute(insert_query)
            conn.unregister('temp_targets_load')
            logger.info("Successfully populated targets table in database.")
        
    except Exception as e:
        logger.error(f"Error converting targets: {e}")
        return {"success": False, "message": f"Error converting Asari results to MINT targets: {e}"}

    report_progress(100, "Done", "Workflow completed.")
    
    # Only cleanup mzML files on FULL SUCCESS (targets detected)
    # This allows reuse if user needs to retry with different parameters
    try:
        if os.path.exists(temp_dir):
            mzml_files = [f for f in os.listdir(temp_dir) if f.lower().endswith(".mzml")]
            for f in mzml_files:
                os.remove(os.path.join(temp_dir, f))
            if mzml_files:
                logger.info(f"Cleaned up {len(mzml_files)} intermediate .mzML files (success with targets).")
    except Exception as cleanup_ex:
        logger.warning(f"Failed to cleanup intermediate files: {cleanup_ex}")
    
    return {
        "success": True, 
        "message": f"Asari analysis completed successfully. {len(df)} targets saved to {backup_path}",
        "result_path": backup_path
    }
