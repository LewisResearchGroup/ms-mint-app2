import base64
import io
import logging
import math
import os
import random
import tempfile
from pathlib import Path
from glob import glob
from typing import Union

import numpy as np
import pandas as pd
from dash.exceptions import PreventUpdate
from scipy.ndimage import binary_opening
from tqdm import tqdm

from ms_mint.io import ms_file_to_df
from ms_mint.standards import TARGETS_COLUMNS
from ms_mint.targets import standardize_targets, read_targets
from .duckdb_manager import duckdb_connection
from .filelock import FileLock
from .plugins.ms_files import generate_colors


def list_to_options(x):
    return [{"label": e, "value": e} for e in x]


def lock(fn):
    return FileLock(f"{fn}.lock", timeout=1)


def get_targets_from_upload(file_path: str, ms_mode=None):
    """
    Read a target CSV file and return a DataFrame.

    Args:
        file_path: Path to the target CSV file.

    Returns:
        DataFrame containing the target data.
    """
    failed = False
    try:
        df = pd.read_csv(file_path)
        df = standardize_targets(df, ms_mode=ms_mode, filename=os.path.basename(file_path))
        df = df[TARGETS_COLUMNS]
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        df = pd.DataFrame()
        failed = True
    return df, failed


class Chromatograms:
    def __init__(self, wdir, targets, ms_files, progress_callback=None):
        self.wdir = wdir
        self.targets = targets
        self.ms_files = ms_files
        self.n_peaks = len(targets)
        self.n_files = len(ms_files)
        self.progress_callback = progress_callback

    def create_all(self):
        for fn in tqdm(self.ms_files):
            self.create_all_for_ms_file(fn)
        return self

    def create_all_for_ms_file(self, ms_file: str):
        fn = ms_file
        df = ms_file_to_df(fn)
        for ndx, row in self.targets.iterrows():
            mz_mean, mz_width = row[["mz_mean", "mz_width"]]
            fn_chro = get_chromatogram_fn(fn, mz_mean, mz_width, self.wdir)
            if os.path.isfile(fn_chro):
                continue
            dirname = os.path.dirname(fn_chro)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            dmz = mz_mean * 1e-6 * mz_width
            chrom = df[(df["mz"] - mz_mean).abs() <= dmz]
            chrom["scan_time"] = chrom["scan_time"].round(3)
            chrom = chrom.groupby("scan_time").max().reset_index()
            chrom[["scan_time", "intensity"]].to_feather(fn_chro)

    def get_single(self, mz_mean, mz_width, ms_file):
        return self._get_chromatogram(ms_file, mz_mean, mz_width, self.wdir)

    def _get_chromatogram(self, ms_file, mz_mean, mz_width, wdir):
        fn = self._get_chromatogram_fn(ms_file, mz_mean, mz_width, wdir)
        if not os.path.isfile(fn):
            chrom = self._create_chromatogram(ms_file, mz_mean, mz_width, fn)
        else:
            try:
                chrom = pd.read_feather(fn)
            except:
                os.remove(fn)
                logging.warning(f"Cound not read {fn}.")
                return None

        chrom = chrom.rename(
            columns={
                "retentionTime": "scan_time",
                "intensity array": "intensity",
                "m/z array": "mz",
            }
        )

        return chrom

    @staticmethod
    def _create_chromatogram(
            ms_file: Union[str, Path],
            mz_mean: float,
            mz_width: float,
            fn_out: Union[str, Path],
            time_step: float = 0.25
    ) -> pd.DataFrame:
        """
        Create a chromatogram from mass spectrometry data.

        Args:
            ms_file: Path to the mass spectrometry file
            mz_mean: Mean m/z value for filtering
            mz_width: Width of m/z window in ppm
            fn_out: Output file path for the Feather file
            time_step: Time step for equidistant time points (default: 0.25)

        Returns:
            pd.DataFrame: Processed chromatogram data with equidistant time points
                         Returns empty DataFrame if no data is found
        """

        # Convert MS file to DataFrame
        df = ms_file_to_df(ms_file)
        # Create output directory if not exists
        dirname: str = os.path.dirname(str(fn_out))
        if not os.path.isdir(dirname):
            os.makedirs(dirname, exist_ok=True)

        if mz_width:
            # Calculate m/z tolerance
            dmz = mz_mean * 1e-6 * mz_width
            # Filter DataFrame to specific m/z range
            chrom = df[(df["mz"] - mz_mean).abs() <= dmz]
        else:
            chrom = df

        # If no data found, return empty DataFrame
        if chrom.empty:
            empty_result = pd.DataFrame(columns=["scan_time", "intensity"])

            # Save empty DataFrame to Feather file
            with lock(fn_out):
                empty_result.to_feather(fn_out)
            return empty_result
        # Group by scan time and get max intensity
        cols_to_drop_list = ["polarity", 'filterLine', 'filterLine_to_ELMAVEN']
        if cols_to_drop := [col for col in cols_to_drop_list if col in df.columns]:
            chrom = chrom.drop(columns=cols_to_drop)
        chrom = chrom.groupby("scan_time").max().reset_index()

        # Determine start and end times
        start_time: float = chrom['scan_time'].min()
        end_time: float = chrom['scan_time'].max()

        # Check if start_time or end_time is NaN
        if np.isnan(start_time) or np.isnan(end_time):
            empty_result: pd.DataFrame = pd.DataFrame(columns=["scan_time", "intensity"])

            # Save empty DataFrame to Feather file
            with lock(fn_out):
                empty_result.to_feather(fn_out)

            return empty_result

        # Create equidistant time points
        time_points: np.ndarray = np.arange(start_time, end_time + time_step, time_step)

        # Interpolate intensities
        interpolated_intensities: np.ndarray = np.interp(
            time_points,
            chrom['scan_time'],
            chrom['intensity']
        )

        # Create new equidistant DataFrame
        equidistant_chrom: pd.DataFrame = pd.DataFrame({
            'scan_time': time_points,
            'intensity': interpolated_intensities
        })

        # Round scan time to 3 decimal places
        equidistant_chrom['scan_time'] = equidistant_chrom['scan_time'].round(3)

        # Save to Feather file
        with lock(fn_out):
            equidistant_chrom[["scan_time", "intensity"]].to_feather(fn_out)

        return equidistant_chrom

    @staticmethod
    def _get_chromatogram_fn(ms_file, mz_mean, mz_width, wdir):
        ms_file = os.path.basename(ms_file)
        base, _ = os.path.splitext(ms_file)
        fn = (
                os.path.join(wdir, "chromato", f"{mz_mean}-{mz_width}".replace(".", "_"), base)
                + ".feather"
        )
        return fn


def get_targets_fn(wdir):
    return os.path.join(wdir, "targets", "targets.csv")


def get_targets(wdir):
    fn = get_targets_fn(wdir)
    if os.path.isfile(fn):
        targets = read_targets(fn)
    else:
        targets = pd.DataFrame(columns=TARGETS_COLUMNS)
    return targets


def update_targets(wdir, peak_label, rt_min=None, rt_max=None, rt=None):
    targets = get_targets(wdir)

    if isinstance(peak_label, int):
        targets = targets.reset_index()

    if rt_min is not None and not np.isnan(rt_min):
        targets.loc[targets['peak_label'] == peak_label, "rt_min"] = rt_min
    if rt_max is not None and not np.isnan(rt_max):
        targets.loc[targets['peak_label'] == peak_label, "rt_max"] = rt_max
    if rt is not None and not np.isnan(rt):
        targets.loc[targets['peak_label'] == peak_label, "rt"] = rt

    # if isinstance(peak_label, int):
    #     targets = targets.set_index("peak_label")

    fn = get_targets_fn(wdir)
    with lock(fn):
        targets.to_csv(fn)


def get_results_fn(wdir):
    return os.path.join(wdir, "results", "results.csv")


def get_results(wdir):
    fn = get_results_fn(wdir)
    df = pd.read_csv(fn)
    df["ms_file_label"] = [filename_to_label(fn) for fn in df["ms_file"]]
    return df


def get_metadata(files_path):
    ref_cols = {
        "ms_file_label": 'string',
        "color": 'string',
        "use_for_optimization": 'boolean',
        "use_for_analysis": 'boolean',
        "label": 'string',
        "sample_type": 'string',
        "run_order": 'Int32',
        "plate": 'string',
        "plate_row": 'string',
        "plate_column": 'Int32'
    }
    required = {"ms_file_label"}

    failed_files = {}
    dfs = []
    for file_path in files_path:
        try:
            df = pd.read_csv(file_path, dtype=ref_cols)
            df['use_for_optimization'] = df['use_for_optimization'].fillna(True)

            if missing_required := required - set(df.columns):
                raise ValueError(f"Missing required columns: {missing_required}")
            dfs.append(df)
        except Exception as e:
            failed_files[file_path] = str(e)

    metadata_df = pd.concat(dfs).drop_duplicates(subset="ms_file_label")

    ref_names = list(ref_cols.keys())
    if missing := set(ref_names) - set(metadata_df.columns):
        for col in missing:
            metadata_df[col] = np.nan
    return metadata_df[ref_names], failed_files


def get_targets_v2(files_path):
    ref_cols = {
        "peak_label": 'string',
        "mz_mean": float,
        "mz_width": float,
        "mz": float,
        "rt": float,
        "rt_min": float,
        "rt_max": float,
        "rt_unit": 'string',
        "intensity_threshold": float,
        "polarity": 'string',
        "filterLine": 'string',
        "ms_type": 'string',
        "category": 'string',
        "score": float,
        "peak_selection": 'boolean',
        "bookmark": 'boolean',
        "source": 'string',
        "notes": 'string',
    }
    required_cols = {"peak_label", "rt_min", "rt_max"}

    failed_files = {}
    failed_targets = []
    valid_targets = []

    total_files = len(files_path)
    files_processed = 0
    files_failed = 0
    targets_processed = 0
    targets_failed = 0

    for file_path in files_path:
        file_name = Path(file_path).name

        try:
            df = pd.read_csv(file_path, dtype=ref_cols)

            if missing_required := required_cols - set(df.columns):
                raise ValueError(f"Missing required columns: {missing_required}")

            for idx, row in df.iterrows():
                targets_processed += 1

                try:
                    target = row.to_dict()
                    target['source'] = file_name

                    if pd.isna(target.get('peak_label')) or target.get('peak_label') == '':
                        raise ValueError("peak_label is empty or null")

                    has_rt_min = not pd.isna(target.get('rt_min'))
                    has_rt_max = not pd.isna(target.get('rt_max'))

                    if not has_rt_min or not has_rt_max:
                        raise ValueError(
                            f"Target '{target['peak_label']}' must have both 'rt_min' and 'rt_max' defined"
                        )

                    if pd.isna(target.get('rt')):
                        target['rt'] = (target['rt_min'] + target['rt_max']) / 2

                    # check if RT-unit is seconds or minutes
                    if 'rt_unit' in target:
                        if target['rt_unit'] in ['minutes', 'min', 'm']:
                            target['rt'] = target['rt'] * 60
                            target['rt_max'] = target['rt_max'] * 60
                            target['rt_min'] = target['rt_min'] * 60
                        elif target['rt_unit'] in ['seconds', 'sec', 's']:
                            target['rt'] = target['rt']
                            target['rt_max'] = target['rt_max']
                            target['rt_min'] = target['rt_min']
                        else:
                            raise ValueError(f"Invalid RT-unit: {target['rt_unit']}")

                    target['rt_unit'] = 's'


                    pol = target.get('polarity')
                    if pd.isna(pol):
                        target['polarity'] = 'Positive'
                    else:
                        pol_str = str(pol)
                        target['polarity'] = (
                            pol_str
                            .replace('+', 'Positive')
                            .replace('positive', 'Positive')
                            .replace('-', 'Negative')
                            .replace('negative', 'Negative')
                        )
                    if 'filterLine' in target:
                        target['ms_type'] = 'ms2' if target['filterLine'] else 'ms1'
                    else:
                        target['ms_type'] = 'ms1'

                    valid_targets.append(target)

                except Exception as e:
                    targets_failed += 1
                    failed_targets.append({
                        'file': file_name,
                        'row': idx,
                        'peak_label': row.get('peak_label', 'UNKNOWN'),
                        'error': str(e)
                    })
                    logging.warning(f"Failed to process target at row {idx} in {file_name}: {str(e)}")

            files_processed += 1

        except Exception as e:
            files_failed += 1
            failed_files[file_path] = str(e)
            logging.error(f"Failed to process file {file_path}: {str(e)}")

            try:
                df_count = pd.read_csv(file_path, usecols=[0])  # Read only first column to count rows
                file_target_count = len(df_count)
                targets_failed += file_target_count
                targets_processed += file_target_count
            except:
                logging.warning(f"Could not count targets in failed file {file_path}")

    # Check if there are valid targets
    if not valid_targets:
        error_msg = "No valid targets found in any file."
        error_msg += f"\n  Total files: {total_files}"
        error_msg += f"\n  Files processed: {files_processed}"
        error_msg += f"\n  Files failed: {files_failed}"
        error_msg += f"\n  Targets processed: {targets_processed}"
        error_msg += f"\n  Targets failed: {targets_failed}"
        raise ValueError(error_msg)

    # Build DataFrame with valid targets
    targets_df = pd.DataFrame(valid_targets)

    # Eliminar duplicados basados en peak_label
    targets_df = targets_df.drop_duplicates(subset="peak_label")

    # Ensure all reference columns exist
    ref_names = list(ref_cols.keys())
    for col in ref_names:
        if col not in targets_df.columns:
            targets_df[col] = np.nan

    # Apply the correct data types
    for col, dtype in ref_cols.items():
        if col in targets_df.columns:
            if dtype == 'string':
                targets_df[col] = targets_df[col].astype('string')
            elif dtype == 'boolean':
                targets_df[col] = targets_df[col].fillna(False).astype(bool)
            elif dtype == float:
                targets_df[col] = pd.to_numeric(targets_df[col], errors='coerce')

    # Fill default values
    targets_df['peak_selection'] = targets_df['peak_selection'].fillna(False)
    targets_df['bookmark'] = targets_df['bookmark'].fillna(False)

    # Summary log
    logging.info("Processing summary:")
    logging.info(f"  Total files: {total_files}")
    logging.info(f"  Files processed: {files_processed}")
    logging.info(f"  Files failed: {files_failed}")
    logging.info(f"  Targets processed: {targets_processed}")
    logging.info(f"  Targets failed: {targets_failed}")
    logging.info(f"  Unique valid targets: {len(targets_df)}")

    # Prepare stats dictionary
    stats = {
        'total_files': total_files,
        'files_processed': files_processed,
        'files_failed': files_failed,
        'targets_processed': targets_processed,
        'targets_failed': targets_failed,
        'unique_valid_targets': len(targets_df)
    }

    return targets_df[ref_names], failed_files, failed_targets, stats


def _insert_ms_data(wdir, ms_type, batch_ms, batch_ms_data):
    failed_files = []

    if not batch_ms[ms_type]:
        return 0, failed_files

    pldf = pd.DataFrame(batch_ms[ms_type], columns=['ms_file_label', 'label', 'ms_type', 'polarity'])

    with duckdb_connection(wdir) as conn:
        if conn is None:
            raise PreventUpdate
        try:
            conn.execute(
                "INSERT INTO samples(ms_file_label, label, ms_type, polarity) "
                "SELECT ms_file_label, label, ms_type, polarity FROM pldf"
            )
            ms_data_table = f'{ms_type}_data'
            extra_columns = ['mz_precursor', 'filterLine', 'filterLine_ELMAVEN']
            select_columns = ['ms_file_label', 'scan_id', 'mz', 'intensity', 'scan_time']
            if ms_type == 'ms2':
                select_columns.extend(extra_columns)
            conn.execute(f"""
                         INSERT INTO {ms_data_table} ({', '.join(select_columns)})
                         SELECT {', '.join(select_columns)}
                         FROM read_parquet(?)
                         """,
                         [batch_ms_data[ms_type]])
        except Exception as e:
            logging.error(f"DB error: {e}")
            failed_files.extend([{file_path: str(e)} for file_path in batch_ms_data[ms_type]])
    return len(batch_ms_data[ms_type]), failed_files


# IMPORTANT: We've defined these functions here temporarily, but it should be moved to the backend.
def process_ms_files(wdir, set_progress, selected_files, n_cpus):
    file_list = sorted([file for folder in selected_files.values() for file in folder])
    n_total = len(file_list)
    failed_files = []
    total_processed = 0
    import concurrent.futures
    from ms_mint.io import convert_mzxml_to_parquet_fast_batches
    import multiprocessing

    # set progress to 1 to the user feedback
    set_progress(1)

    # get the ms_file_label data as df to avoid multiple queries
    with duckdb_connection(wdir) as conn:
        if conn is None:
            raise PreventUpdate
        data = conn.execute("SELECT ms_file_label FROM samples").df()

    files_name = {Path(file_path).stem: Path(file_path) for file_path in file_list}

    mask = data["ms_file_label"].isin(files_name)  # dict como conjunto de claves
    duplicates = data.loc[mask, "ms_file_label"]  # Serie con solo las etiquetas duplicadas
    if not duplicates.empty:
        set_progress(round(duplicates.shape[0] / n_total * 100, 1))
        failed_files.append({files_name[label]: "duplicate" for label in duplicates})
        logging.info("Found %d duplicates: %s", duplicates.shape[0], duplicates.tolist())

    if len(files_name) - len(duplicates) > 0:
        with tempfile.TemporaryDirectory() as tmpdir:
            futures_name = {}
            # ctx = multiprocessing.get_context('fork')
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_cpus, mp_context=None) as executor:
                futures_name.update(
                    {
                        executor.submit(
                            convert_mzxml_to_parquet_fast_batches, file_path, tmp_dir=tmpdir
                        ): file_path
                        for file_name, file_path in files_name.items()
                        if file_name not in duplicates.tolist()
                    }
                )
                batch_ms = {'ms1': [], 'ms2': []}
                batch_ms_data = {'ms1': [], 'ms2': []}
                batch_size = n_cpus

                for future in concurrent.futures.as_completed(futures_name.keys()):
                    try:
                        result = future.result()
                    except Exception as e:
                        failed_files.append({futures_name[future]: str(e)})
                        total_processed += 1
                        set_progress(round((total_processed + len(failed_files)) / n_total * 100, 1))
                        continue

                    _file_path, _ms_file_label, _ms_level, _polarity, _parquet_df = result
                    batch_ms[f'ms{_ms_level}'].append((_ms_file_label, _ms_file_label, f'ms{_ms_level}', _polarity))
                    batch_ms_data[f'ms{_ms_level}'].append(_parquet_df)

                    if len(batch_ms['ms1']) == batch_size:
                        b_processed, b_failed = _insert_ms_data(wdir, 'ms1', batch_ms, batch_ms_data)
                        total_processed += b_processed
                        failed_files.extend(b_failed)

                        set_progress(round((total_processed + len(failed_files)) / n_total * 100, 1))
                        batch_ms['ms1'] = []
                        batch_ms_data['ms1'] = []

                    elif len(batch_ms['ms2']) == batch_size:
                        b_processed, b_failed = _insert_ms_data(wdir, 'ms2', batch_ms, batch_ms_data)
                        total_processed += b_processed
                        failed_files.extend(b_failed)

                        set_progress(round((total_processed + len(failed_files)) / n_total * 100, 1))
                        batch_ms['ms2'] = []
                        batch_ms_data['ms2'] = []

                if len(batch_ms['ms1']):
                    b_processed, b_failed = _insert_ms_data(wdir, 'ms1', batch_ms, batch_ms_data)
                    total_processed += b_processed
                    failed_files.extend(b_failed)

                    set_progress(round((total_processed + len(failed_files)) / n_total * 100, 1))
                    batch_ms['ms1'] = []
                    batch_ms_data['ms1'] = []

                elif len(batch_ms['ms2']):
                    b_processed, b_failed = _insert_ms_data(wdir, 'ms2', batch_ms, batch_ms_data)
                    total_processed += b_processed
                    failed_files.extend(b_failed)

                    set_progress(round((total_processed + len(failed_files)) / n_total * 100, 1))
                    batch_ms['ms2'] = []
                    batch_ms_data['ms2'] = []

    set_progress(round(100, 1))
    return total_processed, failed_files


def process_metadata(wdir, set_progress, selected_files):
    file_list = [file for folder in selected_files.values() for file in folder]
    n_total = len(file_list)
    set_progress(10)
    metadata_df, failed_files = get_metadata(file_list)

    with duckdb_connection(wdir) as conn:
        if conn is None:
            raise PreventUpdate
        columns_to_update = {"use_for_optimization": "BOOLEAN", "use_for_analysis": "BOOLEAN", "color": "VARCHAR",
                             "label": "VARCHAR", "sample_type": "VARCHAR", "run_order": "INTEGER",
                             "plate": "VARCHAR", "plate_row": "VARCHAR", "plate_column": "INTEGER"}
        set_clauses = []
        for col, cast in columns_to_update.items():
            set_clauses.append(f"""
                            {col} = CASE 
                                WHEN metadata_df.{col} IS NOT NULL 
                                THEN CAST(metadata_df.{col} AS {cast}) 
                                ELSE samples.{col} 
                            END
                        """)
        set_clause = ", ".join(set_clauses)
        stmt = f"""
            UPDATE samples 
            SET {set_clause}
            FROM metadata_df 
            WHERE samples.ms_file_label = metadata_df.ms_file_label
        """
        conn.execute(stmt)

        generate_colors(wdir)

    set_progress(100)
    return len(metadata_df), failed_files


def process_targets(wdir, set_progress, selected_files):
    file_list = [file for folder in selected_files.values() for file in folder]
    n_total = len(file_list)
    set_progress(10)
    try:
        targets_df, failed_files, failed_targets, stats = get_targets_v2(file_list)
    except Exception as e:
        logging.error(f"Error processing targets: {e}")
        failed_files = {file: str(e) for file in file_list}
        return 0, failed_files, [], {}

    with duckdb_connection(wdir) as conn:
        if conn is None:
            raise PreventUpdate
        conn.execute(
            "INSERT INTO targets(peak_label, mz_mean, mz_width, mz, rt, rt_min, rt_max, rt_unit, intensity_threshold, "
            "polarity, filterLine, ms_type, category, score, peak_selection, bookmark, source, notes) "
            "SELECT peak_label, mz_mean, mz_width, mz, rt, rt_min, rt_max, rt_unit, intensity_threshold, polarity, "
            "filterLine, ms_type, category, score, peak_selection, bookmark, source, notes "
            "FROM targets_df ORDER BY peak_label"
        )

    set_progress(100)
    return len(targets_df), failed_files, failed_targets, stats


# TODO: check if we need to use the intensity_threshold as baseline
def sparsify_chrom(
    scan,
    intensity,
    w=1,
    baseline=1.0,
    eps=0.0,
    min_peak_width=3,
    ):
    """
    Keep all points above the baseline (or baseline+eps)
    plus ±w neighboring points.
    """
    scan = np.asarray(scan)
    intensity = np.asarray(intensity)

    signal = intensity > (baseline + eps)
    if not np.any(signal):
        return scan[:0], intensity[:0]

    # Remove small islands (shorter than min_peak_width)
    structure = np.ones(min_peak_width, dtype=bool)
    cleaned_signal = binary_opening(signal, structure=structure)
    if not np.any(cleaned_signal):
        return scan[:0], intensity[:0]

    kernel = np.ones(2 * w + 1, dtype=np.int8)
    keep = np.convolve(cleaned_signal.astype(np.int8), kernel, mode="same") > 0

    return scan[keep], intensity[keep]

def proportional_min1_selection(df, group_col, list_col, total_select, seed=None):
    """
    Regla:
      - Si total de elementos < total_select -> incluir todos.
      - Si no -> cuota por grupo = max(1, floor(total_select * p_grupo)).
    Retorna: (quotas, lista_plana_seleccionada)
    """
    rng = random.Random(seed) if seed is not None else random

    # Normalize list column into plain lists
    groups = []
    for lst in df[list_col]:
        if isinstance(lst, list):
            groups.append(lst)
        elif lst is None:
            groups.append([])
        else:
            try:
                groups.append(list(lst))
            except TypeError:
                groups.append([lst])

    # Conteo total
    sizes = [len(lst) for lst in groups]
    total = sum(sizes)

    # CASO 1: total menor que total_select → incluir TODO
    if total <= total_select:
        # Simplemente concatenamos todos los elementos
        full_list = []
        for lst in groups:
            full_list.extend(lst)
        return {g: s for g, s in zip(df[group_col], sizes)}, full_list

    # CASO 2: aplicar cuota proporcional
    quotas = {}
    for g, n in zip(df[group_col], sizes):
        p = n / total
        q = max(1, math.floor(total_select * p))
        quotas[g] = q

    # Muestreo por grupo
    selected = []
    for (g, lst), q in zip(zip(df[group_col], groups), quotas.values()):
        seq = list(lst)
        k = min(q, len(seq))
        selected.extend(rng.sample(seq, k))

    return quotas, selected

def write_metadata(meta, wdir):
    fn = get_metadata_fn(wdir)
    with lock(fn):
        meta.to_csv(fn, index=False)


def get_metadata_fn(wdir):
    fn = os.path.join(wdir, "metadata", "metadata.csv")
    return fn


def get_ms_dirname(wdir):
    return os.path.join(wdir, "ms_files")


def get_ms_fns(wdir, abs_path=True):
    path = get_ms_dirname(wdir)
    fns = glob(os.path.join(path, "**", "*.*"), recursive=True)
    fns = [fn for fn in fns if is_ms_file(fn)]
    if not abs_path:
        fns = [os.path.basename(fn) for fn in fns]
    return fns


def is_ms_file(fn: str):
    if (
            fn.lower().endswith(".mzxml")
            or fn.lower().endswith(".mzml")
            or fn.lower().endswith(".feather")
    ):
        return True
    return False


def get_complete_results(
        wdir,
        include_labels=None,
        exclude_labels=None,
        file_types=None,
        include_excluded=False,
):
    meta = get_metadata(wdir)
    resu = get_results(wdir)

    if not include_excluded:
        meta = meta[meta["in_analysis"]]
    df = pd.merge(meta, resu, on=["ms_file_label"])
    if include_labels is not None and len(include_labels) > 0:
        df = df[df.peak_label.isin(include_labels)]
    if exclude_labels is not None and len(exclude_labels) > 0:
        df = df[~df.peak_label.isin(exclude_labels)]
    if file_types is not None and file_types != []:
        df = df[df.sample_type.isin(file_types)]
    df["log(peak_max+1)"] = df.peak_max.apply(np.log1p)
    if "index" in df.columns:
        df = df.drop("index", axis=1)
    return df


def gen_tabulator_columns(
        col_names=None,
        add_ms_file_col=False,
        add_color_col=False,
        add_peakopt_col=False,
        add_ms_file_active_col=False,
        col_width="12px",
        editor="input",
):
    if col_names is None:
        col_names = []
    col_names = list(col_names)

    standard_columns = [
        "ms_file_label",
        "in_analysis",
        "color",
        "index",
        "use_for_optimization",
    ]

    for col in standard_columns:
        if col in col_names:
            col_names.remove(col)

    columns = [
        {
            "formatter": "rowSelection",
            "titleFormatter": "rowSelection",
            "titleFormatterParams": {
                "rowRange": "active"  # only toggle the values of the active filtered rows
            },
            "hozAlign": "center",
            "headerSort": False,
            "width": "1px",
            "frozen": True,
        }
    ]

    if add_ms_file_col:
        columns.append(
            {
                "title": "ms_file_label",
                "field": "ms_file_label",
                "headerFilter": True,
                "headerSort": True,
                "editor": "input",
                "sorter": "string",
                "frozen": True,
            }
        )

    if add_color_col:
        columns.append(
            {
                "title": "color",
                "field": "color",
                "headerFilter": False,
                "editor": "input",
                "formatter": "color",
                "width": "3px",
                "headerSort": False,
            }
        )

    if add_peakopt_col:
        columns.append(
            {
                "title": "use_for_optimization",
                "field": "use_for_optimization",
                "headerFilter": False,
                "formatter": "tickCross",
                "width": "6px",
                "headerSort": True,
                "hozAlign": "center",
                "editor": True,
            }
        )

    if add_ms_file_active_col:
        columns.append(
            {
                "title": "in_analysis",
                "field": "in_analysis",
                "headerFilter": True,
                "formatter": "tickCross",
                "width": "6px",
                "headerSort": True,
                "hozAlign": "center",
                "editor": True,
            }
        )

    for col in col_names:
        content = {
            "title": col,
            "field": col,
            "headerFilter": True,
            # "width": col_width,
            "editor": editor,
        }

        columns.append(content)
    return columns


def fig_to_src(fig, dpi=100):
    out_img = io.BytesIO()
    fig.savefig(out_img, format="jpeg", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)


def merge_metadata(old_df: pd.DataFrame, new_df: pd.DataFrame, index_col='ms_file_label') -> pd.DataFrame:
    """
    This function updates one existing dataframe 
    with information from a second dataframe.
    If a column of the new dataframe does not 
    exist it will be created.

    Parameters:
    old (pd.DataFrame): The DataFrame to merge new data into.
    new (pd.DataFrame): The DataFrame containing the new data to merge.

    Returns:
    pd.DataFrame: The merged DataFrame.

    """
    old_df = old_df.set_index(index_col)
    new_df = new_df.groupby(index_col).first().replace("null", None)
    if 'file_type' in new_df.columns:
        new_df = new_df.drop(columns=['file_type'])

    if len(old_df.columns.intersection(new_df.columns).tolist()) > 1:
        old_df.update(new_df)  # actualiza solo donde hay match
    else:
        old_df = old_df.join(new_df, on=index_col)
    return old_df.reset_index()


def get_figure_fn(kind, wdir, label, format):
    path = os.path.join(wdir, "figures", kind)
    clean_label = clean_string(label)
    fn = f"{kind}__{clean_label}.{format}"
    fn = os.path.join(path, fn)
    return path, fn


def clean_string(fn: str):
    for x in ['"', "'", "(", ")", "[", "]", " ", "\\", "/", "{", "}"]:
        fn = fn.replace(x, "_")
    return fn


def write_targets(targets, wdir):
    fn = get_targets_fn(wdir)
    if "peak_label" in targets.columns:
        targets = targets.set_index("peak_label")
    with lock(fn):
        targets.to_csv(fn)


def filename_to_label(fn: str):
    if is_ms_file(fn):
        fn = os.path.splitext(fn)[0][:-4]
    return os.path.basename(fn)


def df_to_in_memory_excel_file(df):
    def to_xlsx(bytes_io):
        xslx_writer = pd.ExcelWriter(bytes_io, engine="xlsxwriter")
        df.to_excel(xslx_writer, index=True, sheet_name="sheet1")
        xslx_writer.close()

    return to_xlsx


def fix_first_emtpy_line_after_upload_workaround(file_path):
    logging.warning(f'Check if first line is empty in {file_path}.')

    with open(file_path, 'r') as file:
        lines = file.readlines()

    if not lines:
        return

    # Check if the first line is an empty line (contains only newline character)
    if lines[0] == "\n":
        logging.warning(f'Empty first line detected in {file_path}. Removing it.')
        lines.pop(0)

        with open(file_path, 'w') as file:
            file.writelines(lines)


def describe_transformation(var_name, apply, groupby, scaler):
    # Only apply the function if it's provided
    if apply is not None:
        apply_desc = transformations[apply]['description']
        apply_desc = apply_desc.replace('x', var_name)
    else:
        apply_desc = var_name

    # If scaler or groupby is None, no scaling applied so return just the transformed variable
    if groupby is None or scaler is None:
        return apply_desc

    if not scaler:
        return apply_desc

    # Define human-readable names for known scalers
    scaler_mapping = {"standard": "Standard scaling", "robust": "Robust scaling", '': ''}  # expand as needed
    if isinstance(scaler, str):
        scaler_description = scaler_mapping.get(scaler.lower(), scaler)
    else:
        scaler_description = scaler.__name__

    # Groupby can be a list or a string, so make sure it's a list for consistent handling
    if isinstance(groupby, str):
        groupby = [groupby]

    groupby_description = ", ".join(groupby)

    # Scaling was applied, so return the description in < >
    return f"<{apply_desc}> ({scaler_description}, grouped by {groupby_description})"


log2p1 = lambda x: np.log2(1 + x)
log1p = np.log1p

transformations = {
    "log1p": {'function': log1p, 'description': 'log10(x + 1)'},
    "log2p1": {'function': log2p1, 'description': 'log2(x + 1)'}
}
