import base64
import io
import logging
import math
import os
import random
import re
import tempfile
import zlib
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Iterator, Any, List

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq
from dash.exceptions import PreventUpdate
from lxml.etree import XMLSyntaxError
from scipy.ndimage import binary_opening

from .duckdb_manager import duckdb_connection
from .sample_metadata import GROUP_COLUMNS

logger = logging.getLogger(__name__)

_RT_SECONDS = re.compile(
    r"^P(?:T(?:(?P<h>\d+(?:\.\d+)?)H)?(?:(?P<m>\d+(?:\.\d+)?)M)?(?:(?P<s>\d+(?:\.\d+)?)S)?)$",
    re.I
)


def today() -> str:
    """Return today's date in ISO format (YYYY-MM-DD). Used for filenames."""
    return date.today().isoformat()


def rt_to_seconds(val) -> float:
    """Converts retentionTime to seconds (if it comes as PT...); if it is already numeric, returns it as is."""
    if isinstance(val, (int, float)):
        return float(val)
    s = (val or "").strip()
    # If it is like "0.12345", treat as seconds
    try:
        return float(s)
    except ValueError:
        pass
    m = _RT_SECONDS.match(s)
    if not m:
        return 0.0
    h = float(m.group("h") or 0.0)
    mi = float(m.group("m") or 0.0)
    se = float(m.group("s") or 0.0)
    return h * 3600.0 + mi * 60.0 + se


def _decode_peaks_optimized(attrs: Dict[str, str], text: Optional[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    KEY OPTIMIZATION: Decodes mz and intensity in A SINGLE operation
    using structured arrays (like pyteomics).

    This is ~2-3x faster than decoding separately.
    """
    if not text:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    # Determine dtype based on precision
    dt = np.float32 if attrs.get("precision") == "32" else np.float64

    # KEY: Create structured dtype for both arrays (mz, intensity)
    # byteorder '>' = big-endian (network byte order)
    endian = ">" if attrs.get("byteOrder") in ("network", "big") else "<"
    dtype = np.dtype([("mz", dt), ("intensity", dt)]).newbyteorder(endian)

    # Decode base64
    raw = base64.b64decode(text)

    # Decompress if necessary
    if attrs.get("compressionType") == "zlib":
        raw = zlib.decompress(raw)

    # SINGLE conversion from bytes to arrays
    arr = np.frombuffer(raw, dtype=dtype)

    # Extract fields from structured array (no copy, only views)
    return arr["mz"], arr["intensity"]


def iter_mzxml_fast(path: str | Path, *, decode_binary: bool = True) -> Iterator[Dict[str, Any]]:
    from lxml import etree  # CRITICAL IMPORT

    path = Path(path)

    # KEY CHANGE: lxml.etree with remove_comments=True
    context = etree.iterparse(
        path.as_posix(),
        events=("start", "end"),
        remove_comments=True,  # Speeds up parsing
        huge_tree=False,  # Security (default)
    )

    # Get root to clear memory
    _, root = next(context)

    current: Dict[str, Any] = {}
    have_peaks = False

    for ev, elem in context:
        # lxml uses .tag directly (without namespace by default in mzXML)
        tag = elem.tag
        if '}' in tag:  # Only if there is a namespace
            tag = tag.rsplit("}", 1)[-1]

        if ev == "start" and tag == "scan":
            a = elem.attrib
            current = {
                "num": int(a.get("num", "0")),
                "msLevel": int(a.get("msLevel", "0")),
                "retentionTime": rt_to_seconds(a.get("retentionTime", "0")),
                "polarity": (
                    "Positive" if a.get("polarity") == "+"
                    else ("Negative" if a.get("polarity") == "-" else None)
                ),
                "filterLine": a.get("filterLine"),
            }
            have_peaks = False

        elif ev == "end" and tag == "precursorMz":
            txt = (elem.text or "").strip()
            if txt:
                try:
                    current["precursorMz"] = float(txt)
                except ValueError:
                    pass

        elif ev == "end" and tag == "peaks":
            if decode_binary:
                # KEY OPTIMIZATION: Uses the optimized version
                mz, it = _decode_peaks_optimized(elem.attrib, (elem.text or "").strip() or None)
                current["m/z array"] = mz
                current["intensity array"] = it
                have_peaks = True
            else:
                current["peaks"] = {"attrs": dict(elem.attrib), "text": elem.text}

        elif ev == "end" and tag == "scan":
            # ELMAVEN-like extra (optional)
            if current.get("msLevel") == 2 and have_peaks:
                pol_str = current.get("polarity") or ""
                prec = current.get("precursorMz")
                mz_arr = current.get("m/z array", [])
                mz0 = float(mz_arr[0]) if len(mz_arr) else None
                if prec is not None and mz0 is not None:
                    current["filterLine_ELMAVEN"] = f"{pol_str} {prec:.3f} [{mz0:.3f}]"

            yield current
            root.clear()  # frees memory

BATCH_SIZE_POINTS = 500_000


def _build_table_from_lists(lists_dict: Dict[str, List], ms_level: int) -> pa.Table:
    """Helper to convert the current lists into a pa.Table."""
    counts = lists_dict.get('counts') or []
    if not counts:
        return pa.Table.from_pydict({})

    counts_arr = np.asarray(counts, dtype=np.int64)
    labels = np.asarray(lists_dict['labels'], dtype=object)
    scan_ids = np.asarray(lists_dict['scan_ids'], dtype=np.int32)
    scan_times = np.asarray(lists_dict['scan_times'], dtype=np.float64)

    labels_repeated = np.repeat(labels, counts_arr)
    scan_ids_repeated = np.repeat(scan_ids, counts_arr)
    scan_times_repeated = np.repeat(scan_times, counts_arr)

    mz_arrays = [np.asarray(arr, dtype=np.float64) for arr in lists_dict['mzs']]
    inten_arrays = [np.asarray(arr, dtype=np.float64) for arr in lists_dict['intensities']]
    mz_concat = np.concatenate(mz_arrays) if mz_arrays else np.array([], dtype=np.float64)
    inten_concat = np.concatenate(inten_arrays) if inten_arrays else np.array([], dtype=np.float64)

    arrays_dict = {
        'ms_file_label': pa.array(labels_repeated, type=pa.string()),
        'scan_id': pa.array(scan_ids_repeated, type=pa.int32()),
        'mz': pa.array(mz_concat, type=pa.float64()),
        'intensity': pa.array(inten_concat, type=pa.float64()),
        'scan_time': pa.array(scan_times_repeated, type=pa.float64()),
    }

    if ms_level == 2:
        mz_precursors = np.asarray(lists_dict['mz_precursors'], dtype=np.float64)
        filter_lines = np.asarray(lists_dict['filterLines'], dtype=object)
        filter_lines_elm = np.asarray(lists_dict['filterLines_ELMAVEN'], dtype=object)

        mz_prec_values = np.repeat(mz_precursors, counts_arr)
        filter_line_values = np.repeat(filter_lines, counts_arr)
        filter_line_elm_values = np.repeat(filter_lines_elm, counts_arr)

        arrays_dict.update({
            'mz_precursor': pa.array(mz_prec_values, type=pa.float64()),
            'filterLine': pa.array(filter_line_values, type=pa.string()),
            'filterLine_ELMAVEN': pa.array(filter_line_elm_values, type=pa.string()),
        })

    return pa.Table.from_pydict(arrays_dict)


def _init_lists() -> Dict[str, List]:
    """Helper to init/reset lists."""
    return {
        'labels': [],
        'scan_ids': [],
        'scan_times': [],
        'counts': [],
        'mzs': [],
        'intensities': [],
        'mz_precursors': [],
        'filterLines': [],
        'filterLines_ELMAVEN': []
    }


def _clear_lists(lists_dict: Dict[str, List]) -> None:
    for values in lists_dict.values():
        values.clear()


def convert_mzxml_to_parquet_fast_batches(
        file_path: str,
        time_unit: str = "s",
        remove_original: bool = False,
        tmp_dir: Optional[str] = None,
        batch_size_points: int = BATCH_SIZE_POINTS,
):
    file_path = Path(file_path)
    time_factor = 60.0 if time_unit == "min" else 1.0
    file_stem = file_path.stem

    current_lists = _init_lists()
    parquet_writer: Optional[pq.ParquetWriter] = None
    wrote_data = False
    if not tmp_dir:
        tmp_dir = tempfile.mkdtemp()
    tmp_fn = Path(tmp_dir, f"{file_stem}.parquet")

    ms_level = None
    polarity = None
    first_scan = True

    total_points = 0  # Counter within current batch

    def flush_batch() -> None:
        nonlocal current_lists, parquet_writer, total_points, wrote_data
        if not current_lists['mzs']:
            return
        table = _build_table_from_lists(current_lists, ms_level or 1)
        # Skip empty tables to avoid creating parquet files with no columns
        if table.num_rows == 0 or table.num_columns == 0:
            _clear_lists(current_lists)
            total_points = 0
            return
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(
                tmp_fn,
                table.schema,
                compression="snappy",
            )
        parquet_writer.write_table(table)
        wrote_data = True
        _clear_lists(current_lists)
        total_points = 0

    try:
        for data in iter_mzxml_fast(file_path.as_posix(), decode_binary=True):
            mz_arr = data.get("m/z array")
            if mz_arr is None or len(mz_arr) == 0:
                continue

            inten_arr = data.get("intensity array")
            n_points = len(mz_arr)
            total_points += n_points

            if first_scan:
                ms_level = int(data.get("msLevel", 0))
                polarity = "Positive" if data.get("polarity") == "+" else "Negative"
                first_scan = False

            scan_id = int(data.get("num", 0))
            scan_time = float(data.get("retentionTime", 0.0)) * time_factor

            current_lists['labels'].append(file_stem)
            current_lists['scan_ids'].append(scan_id)
            current_lists['scan_times'].append(scan_time)
            current_lists['counts'].append(n_points)
            current_lists['mzs'].append(np.asarray(mz_arr, dtype=np.float64))
            current_lists['intensities'].append(np.asarray(inten_arr, dtype=np.float64))

            if ms_level == 2:
                mz_prec = None
                fline = data.get("filterLine")
                fline_elm = None
                try:
                    mz_prec = float(data["precursorMz"][0]["precursorMz"])
                    if mz_prec is not None and n_points > 0:
                        fline_elm = f"{polarity} {mz_prec:.3f} [{mz_arr[0]:.3f}]"
                except (KeyError, IndexError, TypeError):
                    pass
                current_lists['mz_precursors'].append(mz_prec if mz_prec is not None else np.nan)
                current_lists['filterLines'].append(fline)
                current_lists['filterLines_ELMAVEN'].append(fline_elm)

            # --- START BATCH LOGIC ---
            if total_points >= batch_size_points:
                flush_batch()
            # --- END BATCH LOGIC ---
    except XMLSyntaxError as e:
        raise ValueError(f"Invalid XML in {file_path}: {str(e)}") from e
    except Exception as e:
        raise ValueError(f"Invalid XML in {file_path}: {e}") from e
    else:
        flush_batch()
    finally:
        if parquet_writer is not None:
            parquet_writer.close()

    if not wrote_data:
        logger.warning(f"No valid data found in {file_path}")
        return 0, file_path, file_stem, 1, "Unknown", None

    if ms_level is None:
        ms_level = 1
    if polarity is None:
        polarity = "Unknown"

    if remove_original:
        try:
            os.remove(file_path)
        except OSError:
            pass

    return file_path, file_stem, ms_level, polarity, tmp_fn.as_posix()


def _scan_id_from_mzml(spectrum: Dict[str, Any]) -> int:
    scan_id = spectrum.get("id", "")
    if isinstance(scan_id, str) and "scan=" in scan_id:
        try:
            return int(scan_id.split("scan=")[-1])
        except ValueError:
            pass
    return int(spectrum.get("index", 0))


def iter_mzml_pyteomics(path: str | Path) -> Iterator[Dict[str, Any]]:
    from pyteomics import mzml

    with mzml.read(str(path)) as reader:
        for spectrum in reader:
            mz = spectrum.get("m/z array")
            intensity = spectrum.get("intensity array")
            if mz is None or intensity is None or len(mz) == 0:
                continue

            scan = spectrum.get("scanList", {}).get("scan", [{}])[0]
            rt = scan.get("scan start time")
            scan_time = float(rt) if rt is not None else 0.0
            unit = getattr(rt, "unit_info", None)
            if unit == "minute":
                scan_time *= 60.0

            polarity = None
            if "positive scan" in spectrum:
                polarity = "Positive"
            elif "negative scan" in spectrum:
                polarity = "Negative"

            ms_level = spectrum.get("ms level")

            precursor_mz = None
            if ms_level == 2:
                precursors = spectrum.get("precursorList", {}).get("precursor", [])
                if precursors:
                    sel_ions = precursors[0].get("selectedIonList", {}).get("selectedIon", [])
                    if sel_ions:
                        precursor_mz = sel_ions[0].get("selected ion m/z")

            yield {
                "num": _scan_id_from_mzml(spectrum),
                "msLevel": int(ms_level or 0),
                "retentionTime": scan_time,
                "polarity": polarity,
                "filterLine": spectrum.get("filter string"),
                "precursorMz": precursor_mz,
                "m/z array": np.asarray(mz, dtype=np.float64),
                "intensity array": np.asarray(intensity, dtype=np.float64),
            }


def convert_mzml_to_parquet_fast_batches(
        file_path: str,
        time_unit: str = "s",
        remove_original: bool = False,
        tmp_dir: Optional[str] = None,
        batch_size_points: int = BATCH_SIZE_POINTS,
):
    file_path = Path(file_path)
    time_factor = 60.0 if time_unit == "min" else 1.0
    file_stem = file_path.stem

    current_lists = _init_lists()
    parquet_writer: Optional[pq.ParquetWriter] = None
    wrote_data = False
    if not tmp_dir:
        tmp_dir = tempfile.mkdtemp()
    tmp_fn = Path(tmp_dir, f"{file_stem}.parquet")

    ms_level = None
    polarity = None
    first_scan = True

    total_points = 0  # Counter within current batch

    def flush_batch() -> None:
        nonlocal current_lists, parquet_writer, total_points, wrote_data
        if not current_lists['mzs']:
            return
        table = _build_table_from_lists(current_lists, ms_level or 1)
        # Skip empty tables to avoid creating parquet files with no columns
        if table.num_rows == 0 or table.num_columns == 0:
            _clear_lists(current_lists)
            total_points = 0
            return
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(
                tmp_fn,
                table.schema,
                compression="snappy",
            )
        parquet_writer.write_table(table)
        wrote_data = True
        _clear_lists(current_lists)
        total_points = 0

    try:
        for data in iter_mzml_pyteomics(file_path.as_posix()):
            mz_arr = data.get("m/z array")
            if mz_arr is None or len(mz_arr) == 0:
                continue

            inten_arr = data.get("intensity array")
            n_points = len(mz_arr)
            total_points += n_points

            if first_scan:
                ms_level = int(data.get("msLevel", 0))
                polarity = data.get("polarity")
                first_scan = False

            scan_id = int(data.get("num", 0))
            scan_time = float(data.get("retentionTime", 0.0)) * time_factor

            current_lists['labels'].append(file_stem)
            current_lists['scan_ids'].append(scan_id)
            current_lists['scan_times'].append(scan_time)
            current_lists['counts'].append(n_points)
            current_lists['mzs'].append(np.asarray(mz_arr, dtype=np.float64))
            current_lists['intensities'].append(np.asarray(inten_arr, dtype=np.float64))

            if ms_level == 2:
                mz_prec = data.get("precursorMz")
                fline = data.get("filterLine")
                fline_elm = None
                if mz_prec is not None and n_points > 0:
                    fline_elm = f"{polarity} {float(mz_prec):.3f} [{mz_arr[0]:.3f}]"
                current_lists['mz_precursors'].append(mz_prec if mz_prec is not None else np.nan)
                current_lists['filterLines'].append(fline)
                current_lists['filterLines_ELMAVEN'].append(fline_elm)

            # --- START BATCH LOGIC ---
            if total_points >= batch_size_points:
                flush_batch()
            # --- END BATCH LOGIC ---
    except Exception as e:
        raise ValueError(f"Invalid mzML in {file_path}: {e}") from e
    else:
        flush_batch()
    finally:
        if parquet_writer is not None:
            parquet_writer.close()

    if not wrote_data:
        logger.warning(f"No valid data found in {file_path}")
        return 0, file_path, file_stem, 1, "Unknown", None

    if ms_level is None:
        ms_level = 1
    if polarity is None:
        polarity = "Unknown"

    if remove_original:
        try:
            os.remove(file_path)
        except OSError:
            pass

    return file_path, file_stem, ms_level, polarity, tmp_fn.as_posix()



def convert_ms_file_to_parquet_fast_batches(
        file_path: str,
        time_unit: str = "s",
        remove_original: bool = False,
        tmp_dir: Optional[str] = None,
):
    suffix = Path(file_path).suffix.lower()
    if suffix == ".mzxml":
        return convert_mzxml_to_parquet_fast_batches(
            file_path=file_path,
            time_unit=time_unit,
            remove_original=remove_original,
            tmp_dir=tmp_dir,
        )
    if suffix == ".mzml":
        return convert_mzml_to_parquet_fast_batches(
            file_path=file_path,
            time_unit=time_unit,
            remove_original=remove_original,
            tmp_dir=tmp_dir,
        )
    raise ValueError(f"Unsupported MS file type: {suffix}")


def get_metadata(files_path):
    ref_cols = {
        "ms_file_label": 'string',
        "label": 'string',
        "color": 'string',
        "use_for_optimization": 'boolean',
        "use_for_processing": 'boolean',
        "use_for_analysis": 'boolean',
        "sample_type": 'string',
    }
    ref_cols.update({col: 'string' for col in GROUP_COLUMNS})
    required = {"ms_file_label"}

    failed_files = {}
    dfs = []
    for file_path in files_path:
        try:
            preview = pd.read_csv(file_path, nrows=0)
            dtype_map = {col: ref_cols[col] for col in preview.columns if col in ref_cols}
            df = pd.read_csv(file_path, dtype=dtype_map)
            if 'use_for_optimization' in df.columns:
                df['use_for_optimization'] = df['use_for_optimization'].fillna(True)
            if 'use_for_processing' in df.columns:
                df['use_for_processing'] = df['use_for_processing'].fillna(True)

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
        "rt_auto_adjusted": 'boolean',
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
    rt_adjusted_labels = []  # Track targets with RT outside span that were adjusted

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
                    
                    # Validate RT is within [rt_min, rt_max] span, otherwise set to midpoint
                    rt_val = target.get('rt')
                    rt_min_val = target.get('rt_min')
                    rt_max_val = target.get('rt_max')
                    if rt_val is not None and rt_min_val is not None and rt_max_val is not None:
                        if rt_val < rt_min_val or rt_val > rt_max_val:
                            old_rt = rt_val
                            target['rt'] = (rt_min_val + rt_max_val) / 2
                            target['rt_auto_adjusted'] = True  # Mark for later update to max intensity
                            rt_adjusted_labels.append(target['peak_label'])
                            logging.warning(
                                f"Target '{target['peak_label']}': RT {old_rt:.1f}s was outside span "
                                f"[{rt_min_val:.1f}, {rt_max_val:.1f}], adjusted to midpoint {target['rt']:.1f}s"
                            )


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

    # Eliminar duplicados basados en peak_label, pero registrar los duplicados encontrados
    duplicate_labels = (
        targets_df[targets_df.duplicated(subset="peak_label", keep=False)]
        .get("peak_label", pd.Series([], dtype="string"))
        .dropna()
        .unique()
        .tolist()
    )
    if duplicate_labels:
        logging.warning(
            "Found duplicate target labels; keeping first occurrence: %s", duplicate_labels
        )
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
                if col != 'peak_selection':  # handle peak_selection defaults separately below
                    targets_df[col] = targets_df[col].astype('boolean').fillna(False)
            elif dtype == float:
                targets_df[col] = pd.to_numeric(targets_df[col], errors='coerce')

    # Fill default values
    # Default to selected when the column is missing/blank so uploaded targets are included by default.
    targets_df['peak_selection'] = targets_df['peak_selection'].fillna(True).astype(bool)
    targets_df['bookmark'] = targets_df['bookmark'].fillna(False).astype(bool)

    logging.info("Processing summary:")
    logging.info(f"  Total files: {total_files}")
    logging.info(f"  Files processed: {files_processed}")
    logging.info(f"  Files failed: {files_failed}")
    logging.info(f"  Targets processed: {targets_processed}")
    logging.info(f"  Targets failed: {targets_failed}")
    logging.info(f"  Unique valid targets: {len(targets_df)}")
    if rt_adjusted_labels:
        logging.info(f"  RT values adjusted (outside span): {len(rt_adjusted_labels)}")

    # Prepare stats dictionary
    stats = {
        'total_files': total_files,
        'files_processed': files_processed,
        'files_failed': files_failed,
        'targets_processed': targets_processed,
        'targets_failed': targets_failed,
        'unique_valid_targets': len(targets_df),
        'duplicate_peak_labels': len(duplicate_labels),
        'duplicate_peak_labels_list': duplicate_labels,
        'rt_adjusted_count': len(rt_adjusted_labels),
        'rt_adjusted_labels': rt_adjusted_labels,
    }

    return targets_df[ref_names], failed_files, failed_targets, stats


def _insert_ms_data(wdir, ms_type, batch_ms, batch_ms_data):
    failed_files = []

    if not batch_ms[ms_type]:
        return 0, failed_files

    pldf = pd.DataFrame(
        batch_ms[ms_type],
        columns=['ms_file_label', 'label', 'ms_type', 'polarity', 'file_type'],
    )

    parquet_files_to_delete = batch_ms_data[ms_type].copy()
    insert_success = False
    
    with duckdb_connection(wdir) as conn:
        if conn is None:
            raise PreventUpdate
        try:
            conn.execute(
                "INSERT INTO samples(ms_file_label, label, ms_type, polarity, file_type) "
                "SELECT ms_file_label, label, ms_type, polarity, file_type FROM pldf"
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
            insert_success = True
                    
        except Exception as e:
            logging.error(f"DB error: {e}")
            failed_files.extend([{file_path: str(e)} for file_path in batch_ms_data[ms_type]])
    
    # Delete parquet files AFTER connection closes (files are released)
    if insert_success:
        for parquet_file in parquet_files_to_delete:
            try:
                os.remove(parquet_file)
            except OSError:
                pass  # File may already be deleted or locked
                
    return len(batch_ms_data[ms_type]), failed_files


# IMPORTANT: We've defined these functions here temporarily, but it should be moved to the backend.
def process_ms_files(wdir, set_progress, selected_files, n_cpus):
    def _send_progress(percent, detail=""):
        if not set_progress:
            return
        try:
            set_progress(percent, detail)
        except TypeError:
            try:
                set_progress(percent)
            except Exception:
                pass

    file_list = sorted([file for folder in selected_files.values() for file in folder])
    n_total = len(file_list)
    failed_files = []
    duplicates_count = 0
    total_processed = 0
    import concurrent.futures
    import time

    start_time = time.perf_counter()
    logger.info(f"Starting MS file processing for {n_total} file(s).")

    # set progress to 1 to the user feedback
    _send_progress(1)

    # get the ms_file_label data as df to avoid multiple queries
    with duckdb_connection(wdir) as conn:
        if conn is None:
            raise PreventUpdate
        data = conn.execute("SELECT ms_file_label FROM samples").df()

    files_name = {Path(file_path).stem: Path(file_path) for file_path in file_list}

    mask = data["ms_file_label"].isin(files_name)  # dict as set of keys
    duplicates = data.loc[mask, "ms_file_label"]  # Series with only the duplicate labels
    if not duplicates.empty:
        duplicates_count = int(duplicates.shape[0])
        _send_progress(round(duplicates.shape[0] / n_total * 100, 1))
        logging.info("Found %d duplicates: %s", duplicates.shape[0], duplicates.tolist())
        logger.info(f"Skipped {duplicates.shape[0]} duplicate file(s): {', '.join(duplicates.tolist())}")

    if len(files_name) - len(duplicates) > 0:
        total_to_process = len(files_name) - len(duplicates)
        # Use workspace's data/temp folder for intermediate parquet files
        workspace_temp = Path(wdir) / "data" / "temp"
        workspace_temp.mkdir(parents=True, exist_ok=True)
        
        # Filter out duplicates and create list of files to process
        files_to_process = [
            file_path for file_name, file_path in files_name.items()
            if file_name not in duplicates.tolist()
        ]
        
        # Chunked processing: submit files in chunks to limit temp folder size
        chunk_size = 64  # Max files in-flight at once
        batch_ms = {'ms1': [], 'ms2': []}
        batch_ms_data = {'ms1': [], 'ms2': []}
        batch_size = n_cpus
        total_batches = (total_to_process + batch_size - 1) // batch_size
        batch_num = 0
        batch_start = time.time()
        
        with tempfile.TemporaryDirectory(dir=workspace_temp) as tmpdir:
            # Process files in chunks
            for chunk_start in range(0, len(files_to_process), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(files_to_process))
                chunk_files = files_to_process[chunk_start:chunk_end]
                
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=min(n_cpus, len(chunk_files)), 
                    mp_context=None
                ) as executor:
                    futures_name = {
                        executor.submit(
                            convert_ms_file_to_parquet_fast_batches, file_path, tmp_dir=tmpdir
                        ): file_path
                        for file_path in chunk_files
                    }
                    
                    for future in concurrent.futures.as_completed(futures_name.keys()):
                        try:
                            result = future.result()
                        except Exception as e:
                            failed_files.append({futures_name[future]: str(e)})
                            total_processed += 1
                            done_count = total_processed + len(failed_files) + duplicates_count
                            _send_progress(round(done_count / n_total * 100, 1))
                            logger.error(f"Failed: {Path(futures_name[future]).name} ({e})")
                            continue

                        _file_path, _ms_file_label, _ms_level, _polarity, _parquet_df = result
                        
                        if _parquet_df is None:
                            failed_files.append({_file_path: "No valid data found or conversion failed"})
                            total_processed += 1
                            done_count = total_processed + len(failed_files) + duplicates_count
                            _send_progress(round(done_count / n_total * 100, 1))
                            logger.warning(f"Failed (no data): {Path(_file_path).name}")
                            continue

                        suffix = Path(_file_path).suffix.lower()
                        if suffix == ".mzxml":
                            file_type = "mzXML"
                        elif suffix == ".mzml":
                            file_type = "mzML"
                        else:
                            file_type = suffix.lstrip(".")
                        batch_ms[f'ms{_ms_level}'].append(
                            (_ms_file_label, _ms_file_label, f'ms{_ms_level}', _polarity, file_type)
                        )
                        batch_ms_data[f'ms{_ms_level}'].append(_parquet_df)

                        if len(batch_ms['ms1']) == batch_size:
                            b_processed, b_failed = _insert_ms_data(wdir, 'ms1', batch_ms, batch_ms_data)
                            batch_elapsed = time.time() - batch_start
                            total_processed += b_processed
                            failed_files.extend(b_failed)

                            batch_num += 1
                            detail = (
                                f"Batch {batch_num}/{total_batches} | "
                                f"Progress {total_processed:,}/{total_to_process:,} | "
                                f"Time/batch {batch_elapsed:0.2f}s"
                            )
                            done_count = total_processed + len(failed_files) + duplicates_count
                            _send_progress(round(done_count / n_total * 100, 1), detail)
                            logger.info(detail)
                            batch_ms['ms1'] = []
                            batch_ms_data['ms1'] = []
                            batch_start = time.time()
                            
                            # Periodic CHECKPOINT every 20 batches
                            if batch_num % 20 == 0:
                                with duckdb_connection(wdir) as conn:
                                    if conn is not None:
                                        conn.execute("CHECKPOINT")
                                        logger.info(f"Checkpoint after batch {batch_num}")

                        elif len(batch_ms['ms2']) == batch_size:
                            b_processed, b_failed = _insert_ms_data(wdir, 'ms2', batch_ms, batch_ms_data)
                            batch_elapsed = time.time() - batch_start
                            total_processed += b_processed
                            failed_files.extend(b_failed)

                            batch_num += 1
                            detail = (
                                f"Batch {batch_num}/{total_batches} | "
                                f"Progress {total_processed:,}/{total_to_process:,} | "
                                f"Time/batch {batch_elapsed:0.2f}s"
                            )
                            done_count = total_processed + len(failed_files) + duplicates_count
                            _send_progress(round(done_count / n_total * 100, 1), detail)
                            logger.info(detail)
                            batch_ms['ms2'] = []
                            batch_ms_data['ms2'] = []
                            batch_start = time.time()
                            
                            # Periodic CHECKPOINT every 20 batches
                            if batch_num % 20 == 0:
                                with duckdb_connection(wdir) as conn:
                                    if conn is not None:
                                        conn.execute("CHECKPOINT")
                                        logger.info(f"Checkpoint after batch {batch_num}")
            
            # Process remaining items in batches
            if len(batch_ms['ms1']):
                b_processed, b_failed = _insert_ms_data(wdir, 'ms1', batch_ms, batch_ms_data)
                batch_elapsed = time.time() - batch_start
                total_processed += b_processed
                failed_files.extend(b_failed)

                batch_num += 1
                detail = (
                    f"Batch {batch_num}/{total_batches} | "
                    f"Progress {total_processed:,}/{total_to_process:,} | "
                    f"Time/batch {batch_elapsed:0.2f}s"
                )
                done_count = total_processed + len(failed_files) + duplicates_count
                _send_progress(round(done_count / n_total * 100, 1), detail)
                logger.info(detail)
                batch_ms['ms1'] = []
                batch_ms_data['ms1'] = []

            if len(batch_ms['ms2']):
                b_processed, b_failed = _insert_ms_data(wdir, 'ms2', batch_ms, batch_ms_data)
                batch_elapsed = time.time() - batch_start
                total_processed += b_processed
                failed_files.extend(b_failed)

                batch_num += 1
                detail = (
                    f"Batch {batch_num}/{total_batches} | "
                    f"Progress {total_processed:,}/{total_to_process:,} | "
                    f"Time/batch {batch_elapsed:0.2f}s"
                )
                done_count = total_processed + len(failed_files) + duplicates_count
                _send_progress(round(done_count / n_total * 100, 1), detail)
                logger.info(detail)
                batch_ms['ms2'] = []
                batch_ms_data['ms2'] = []

    _send_progress(round(100, 1))
    elapsed = time.perf_counter() - start_time
    logger.info(
        f"Completed MS file processing. Success: {total_processed}, "
        f"Failed: {len(failed_files)}. Total time: {elapsed:0.2f}s."
    )
    
    # Clean up the workspace's temp folder
    workspace_temp = Path(wdir) / "data" / "temp"
    if workspace_temp.exists():
        try:
            import shutil
            shutil.rmtree(workspace_temp, ignore_errors=True)
            logger.info(f"Cleaned up temp folder: {workspace_temp}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp folder: {e}")
    
    return total_processed, failed_files, duplicates_count


def process_metadata(wdir, set_progress, selected_files):
    file_list = [file for folder in selected_files.values() for file in folder]
    n_total = len(file_list)
    set_progress(10)
    metadata_df, failed_files = get_metadata(file_list)

    with duckdb_connection(wdir) as conn:
        if conn is None:
            raise PreventUpdate
        columns_to_update = {
            "use_for_optimization": "BOOLEAN",
            "use_for_processing": "BOOLEAN",
            "use_for_analysis": "BOOLEAN",
            "color": "VARCHAR",
            "label": "VARCHAR",
            "sample_type": "VARCHAR",
            **{col: "VARCHAR" for col in GROUP_COLUMNS},
        }
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

        # Import lazily to avoid circular dependency at module import time.
        from .plugins.ms_files import generate_colors
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
            "INSERT OR REPLACE INTO targets(peak_label, mz_mean, mz_width, mz, rt, rt_min, rt_max, rt_unit, "
            "intensity_threshold, polarity, filterLine, ms_type, category, score, peak_selection, bookmark, source, notes, rt_auto_adjusted) "
            "SELECT peak_label, mz_mean, mz_width, mz, rt, rt_min, rt_max, rt_unit, intensity_threshold, polarity, "
            "filterLine, ms_type, category, score, peak_selection, bookmark, source, notes, rt_auto_adjusted "
            "FROM targets_df ORDER BY mz_mean, peak_label"
        )
        
        # Create initial backup CSV for recovery
        try:
            from pathlib import Path
            data_dir = Path(wdir) / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            backup_path = data_dir / "targets_backup.csv"
            conn.execute(
                "COPY (SELECT * FROM targets) TO ? (HEADER, DELIMITER ',')",
                (str(backup_path),)
            )
            logging.info(f"Created initial targets backup: {backup_path}")
        except Exception as e:
            logging.warning(f"Failed to create initial targets backup: {e}")

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
    Rule:
      - If total elements < total_select -> include all.
      - If not -> quota per group = max(1, floor(total_select * p_group)).
    Returns: (quotas, flat_selected_list)
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

    # Total count
    sizes = [len(lst) for lst in groups]
    total = sum(sizes)

    # CASE 1: total less than total_select -> include ALL
    if total <= total_select:
        # Simply concatenate all elements
        full_list = []
        for lst in groups:
            full_list.extend(lst)
        return {g: s for g, s in zip(df[group_col], sizes)}, full_list

    # CASE 2: apply proportional quota
    quotas = {}
    for g, n in zip(df[group_col], sizes):
        p = n / total
        q = max(1, math.floor(total_select * p))
        quotas[g] = q

    # Sampling by group
    selected = []
    for (g, lst), q in zip(zip(df[group_col], groups), quotas.values()):
        seq = list(lst)
        k = min(q, len(seq))
        selected.extend(rng.sample(seq, k))

    return quotas, selected


def fig_to_src(fig, dpi=100):
    out_img = io.BytesIO()
    fig.savefig(out_img, format="jpeg", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)


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
