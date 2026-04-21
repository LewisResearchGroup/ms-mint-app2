"""
RT Span Optimizer for MS-MINT App

This module provides adaptive RT span detection using chromatogram data
to automatically determine optimal rt_min and rt_max values.

Algorithm: Peak boundary detection at a threshold percentage of peak height,
which naturally handles asymmetric/tailed peaks without requiring parameter estimation.
"""

import logging
import time
from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)
DEFAULT_RT_OPTIMIZATION_WINDOW = 30.0


def _fallback_rt_span(expected_rt: float, min_width: float) -> Tuple[float, float, float]:
    half_width = min_width / 2
    return expected_rt - half_width, expected_rt + half_width, expected_rt


def _adaptive_bin_width(scan_time: np.ndarray) -> float:
    diffs = np.diff(np.unique(scan_time))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 1.0
    median_gap = float(np.median(diffs))
    return float(np.clip(median_gap * 3.0, 0.5, 5.0))


def _safe_savgol(y: np.ndarray) -> np.ndarray:
    if len(y) < 5:
        return np.clip(y, a_min=0, a_max=None)

    window_length = min(len(y) if len(y) % 2 == 1 else len(y) - 1, 11)
    if window_length < 5:
        return np.clip(y, a_min=0, a_max=None)

    polyorder = min(3, window_length - 2)
    return np.clip(
        savgol_filter(y, window_length=window_length, polyorder=polyorder),
        a_min=0,
        a_max=None,
    )


def _max_bin_signal(
    scan_time: np.ndarray,
    intensity: np.ndarray,
    bin_width: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin a sparse signal using per-bin maxima.

    This is a lighter-weight replacement for scipy.stats.binned_statistic(..., "max")
    tailored to the RT optimizer hot path.
    """
    if scan_time.size == 0 or intensity.size == 0 or bin_width <= 0:
        return np.array([]), np.array([])

    rt_min = float(np.min(scan_time))
    rt_max = float(np.max(scan_time))
    if not np.isfinite(rt_min) or not np.isfinite(rt_max) or rt_max <= rt_min:
        return np.array([]), np.array([])

    n_bins = max(1, int(np.ceil((rt_max - rt_min) / bin_width)))
    bin_idx = np.floor((scan_time - rt_min) / bin_width).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    binned_y = np.full(n_bins, -np.inf, dtype=np.float64)
    np.maximum.at(binned_y, bin_idx, intensity)
    binned_y[~np.isfinite(binned_y)] = 0.0

    binned_x = rt_min + (np.arange(n_bins, dtype=np.float64) + 0.5) * bin_width
    return binned_x, binned_y


def _crop_to_expected_rt_window(
    scan_time: np.ndarray,
    intensity: np.ndarray,
    expected_rt: float,
    half_window: float = DEFAULT_RT_OPTIMIZATION_WINDOW,
) -> Tuple[np.ndarray, np.ndarray]:
    """Restrict RT optimization to the default chromatogram preview neighborhood."""
    mask = np.abs(scan_time - expected_rt) <= half_window
    if np.count_nonzero(mask) >= 5:
        return scan_time[mask], intensity[mask]
    return scan_time, intensity


def _crop_chromatograms_to_expected_rt_window(
    chromatograms: list,
    expected_rt: float,
    half_window: float = DEFAULT_RT_OPTIMIZATION_WINDOW,
) -> list:
    """Crop each chromatogram before combination to avoid merging irrelevant RT ranges."""
    cropped = []
    for chrom in chromatograms:
        scan_time = np.asarray(chrom["scan_time"], dtype=np.float64)
        intensity = np.asarray(chrom["intensity"], dtype=np.float64)
        cropped_time, cropped_intensity = _crop_to_expected_rt_window(
            scan_time,
            intensity,
            expected_rt,
            half_window=half_window,
        )
        if len(cropped_time) >= 3:
            cropped.append({"scan_time": cropped_time, "intensity": cropped_intensity})
    return cropped or chromatograms


def optimize_rt_span(
    scan_time: np.ndarray,
    intensity: np.ndarray,
    expected_rt: float,
    min_width: float = 5.0,
    max_width: float = 120.0,
    threshold_pct: float = 0.10,
    sigma_smooth: float = 2.0,
    apex_search_window: float = 15.0,
) -> Tuple[float, float, float]:
    """
    Find optimal rt_min, rt_max from chromatogram data using adaptive peak detection.

    Algorithm:
    1. Smooth the signal with a Gaussian filter to reduce noise
    2. Find the peak apex NEAR the expected RT (within apex_search_window)
    3. Determine peak boundaries at threshold_pct of peak height (default 10%)
    4. Apply min/max width constraints

    Args:
        scan_time: Array of retention times (seconds)
        intensity: Array of intensity values
        expected_rt: Expected retention time (target RT) - this is TRUSTED
        min_width: Minimum allowed peak width in seconds (default 5s)
        max_width: Maximum allowed peak width in seconds (default 120s)
        threshold_pct: Fraction of peak height for boundary detection (default 0.10 = 10%)
        sigma_smooth: Gaussian smoothing sigma in data points (default 2.0)
        apex_search_window: How far from expected_rt to search for apex (default ±15s)

    Returns:
        Tuple of (rt_min, rt_max, apex_rt)
    """
    if len(scan_time) < 3 or len(intensity) < 3:
        # Not enough data points, return expected RT with min_width
        half_width = min_width / 2
        return expected_rt - half_width, expected_rt + half_width, expected_rt

    # Ensure arrays are numpy and sorted by time
    scan_time = np.asarray(scan_time, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    sort_idx = np.argsort(scan_time)
    scan_time = scan_time[sort_idx]
    intensity = intensity[sort_idx]

    # Apply Gaussian smoothing to reduce noise
    if len(intensity) > 5:
        intensity_smooth = gaussian_filter1d(intensity, sigma=sigma_smooth)
    else:
        intensity_smooth = intensity.copy()

    # Find the apex: maximum intensity within a NARROW window around expected_rt
    # This respects the user's RT hint instead of finding global max
    search_mask = np.abs(scan_time - expected_rt) <= apex_search_window
    if not np.any(search_mask):
        # No data in search window, use full data but warn
        logger.warning(f"No data within ±{apex_search_window}s of expected_rt={expected_rt:.1f}s")
        search_mask = np.ones(len(scan_time), dtype=bool)

    # Find local maximum near expected_rt
    search_intensities = intensity_smooth.copy()
    search_intensities[~search_mask] = -np.inf  # Exclude points outside window

    apex_idx = np.argmax(search_intensities)
    apex_rt = scan_time[apex_idx]
    apex_intensity = intensity_smooth[apex_idx]

    if apex_intensity <= 0:
        # No valid peak, return expected RT with min_width
        half_width = min_width / 2
        return expected_rt - half_width, expected_rt + half_width, expected_rt

    # Calculate threshold intensity
    # Use baseline as the minimum in the search window
    baseline = np.min(intensity_smooth[search_mask])
    peak_height = apex_intensity - baseline
    threshold_intensity = baseline + peak_height * threshold_pct

    # Find left boundary: walk left from apex until below threshold
    rt_min = scan_time[0]  # Default to start
    for i in range(apex_idx - 1, -1, -1):
        if intensity_smooth[i] < threshold_intensity:
            rt_min = scan_time[i]
            break

    # Find right boundary: walk right from apex until below threshold
    rt_max = scan_time[-1]  # Default to end
    for i in range(apex_idx + 1, len(scan_time)):
        if intensity_smooth[i] < threshold_intensity:
            rt_max = scan_time[i]
            break

    # Apply width constraints
    current_width = rt_max - rt_min
    if current_width < min_width:
        # Expand symmetrically
        expand = (min_width - current_width) / 2
        rt_min -= expand
        rt_max += expand
    elif current_width > max_width:
        # Contract symmetrically around apex
        rt_min = apex_rt - max_width / 2
        rt_max = apex_rt + max_width / 2

    # Ensure boundaries don't exceed data range
    rt_min = max(rt_min, scan_time[0])
    rt_max = min(rt_max, scan_time[-1])

    logger.debug(
        f"RT span optimized: apex={apex_rt:.1f}s, "
        f"span=[{rt_min:.1f}, {rt_max:.1f}]s, width={rt_max - rt_min:.1f}s"
    )

    return rt_min, rt_max, apex_rt


def optimize_rt_span_ms2(
    scan_time: np.ndarray,
    intensity: np.ndarray,
    expected_rt: float,
    min_width: float = 5.0,
    max_width: float = 120.0,
    threshold_pct: float = 0.05,
    apex_search_window: float = 15.0,
) -> Tuple[float, float, float]:
    """
    Optimize MS2 RT span using a sparse-signal envelope pipeline.

    Steps:
    1. Max-bin the raw trace in RT to suppress gaps between sparse scans.
    2. Smooth the binned signal with Savitzky-Golay.
    3. Interpolate a dense non-negative envelope with PCHIP.
    4. Find the envelope apex near expected_rt, then snap it to the true raw apex.
    5. Derive rt_min/rt_max from the dense envelope at a low fractional threshold.
    """
    t0 = time.perf_counter()
    timings = {}

    scan_time = np.asarray(scan_time, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)

    valid = np.isfinite(scan_time) & np.isfinite(intensity)
    scan_time = scan_time[valid]
    intensity = intensity[valid]
    timings["coerce_arrays"] = time.perf_counter() - t0

    if len(scan_time) < 5 or len(intensity) < 5:
        return _fallback_rt_span(expected_rt, min_width)

    t1 = time.perf_counter()
    scan_time, intensity = _crop_to_expected_rt_window(
        scan_time,
        intensity,
        expected_rt,
    )
    if len(scan_time) < 5 or len(intensity) < 5:
        return _fallback_rt_span(expected_rt, min_width)
    timings["sort_and_crop"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    bin_width = _adaptive_bin_width(scan_time)
    binned_x, binned_y = _max_bin_signal(scan_time, intensity, bin_width)
    if len(binned_x) < 3:
        return optimize_rt_span(
            scan_time,
            intensity,
            expected_rt,
            min_width=min_width,
            max_width=max_width,
            threshold_pct=threshold_pct,
        )
    timings["binning"] = time.perf_counter() - t2

    if len(binned_x) < 4:
        return optimize_rt_span(
            scan_time,
            intensity,
            expected_rt,
            min_width=min_width,
            max_width=max_width,
            threshold_pct=threshold_pct,
        )

    t3 = time.perf_counter()
    smoothed_binned_y = _safe_savgol(binned_y)
    timings["savgol"] = time.perf_counter() - t3

    t4 = time.perf_counter()
    search_mask = np.abs(binned_x - expected_rt) <= apex_search_window
    if not np.any(search_mask):
        search_mask = np.ones(len(binned_x), dtype=bool)

    coarse_search = smoothed_binned_y.copy()
    coarse_search[~search_mask] = -np.inf
    coarse_apex_idx = int(np.argmax(coarse_search))
    coarse_apex_rt = float(binned_x[coarse_apex_idx])
    coarse_apex_intensity = float(smoothed_binned_y[coarse_apex_idx])

    if coarse_apex_intensity <= 0:
        return _fallback_rt_span(expected_rt, min_width)
    timings["coarse_apex"] = time.perf_counter() - t4

    # Interpolate only a narrow local window for apex refinement.
    t5 = time.perf_counter()
    local_half_window = max(bin_width * 2.0, 2.0)
    local_mask = np.abs(binned_x - coarse_apex_rt) <= local_half_window
    local_x = binned_x[local_mask]
    local_y = smoothed_binned_y[local_mask]

    if len(local_x) >= 2:
        interpolator = PchipInterpolator(local_x, local_y, extrapolate=False)
        dense_points = int(np.clip(len(local_x) * 8, 50, 200))
        x_dense = np.linspace(local_x.min(), local_x.max(), dense_points)
        y_dense = np.nan_to_num(interpolator(x_dense), nan=0.0)
        y_dense = np.clip(y_dense, a_min=0, a_max=None)
        dense_apex_idx = int(np.argmax(y_dense))
        dense_apex_rt = float(x_dense[dense_apex_idx])
    else:
        dense_apex_rt = coarse_apex_rt
    timings["local_pchip_apex_refine"] = time.perf_counter() - t5

    t6 = time.perf_counter()
    raw_search_mask = (
        (scan_time >= dense_apex_rt - bin_width)
        & (scan_time <= dense_apex_rt + bin_width)
    )
    if np.any(raw_search_mask):
        local_raw_time = scan_time[raw_search_mask]
        local_raw_intensity = intensity[raw_search_mask]
        raw_apex_idx = int(np.argmax(local_raw_intensity))
        apex_rt = float(local_raw_time[raw_apex_idx])
    else:
        apex_rt = dense_apex_rt
    timings["raw_apex_snap"] = time.perf_counter() - t6

    t7 = time.perf_counter()
    envelope_threshold = max(coarse_apex_intensity * threshold_pct, 1.0)
    apex_binned_idx = int(np.argmin(np.abs(binned_x - apex_rt)))

    left_idx = 0
    for i in range(apex_binned_idx, -1, -1):
        if smoothed_binned_y[i] < envelope_threshold:
            left_idx = i
            break

    right_idx = len(binned_x) - 1
    for i in range(apex_binned_idx, len(binned_x)):
        if smoothed_binned_y[i] < envelope_threshold:
            right_idx = i
            break

    rt_min = float(binned_x[left_idx])
    rt_max = float(binned_x[right_idx])

    current_width = rt_max - rt_min
    if current_width < min_width:
        expand = (min_width - current_width) / 2
        rt_min -= expand
        rt_max += expand
    elif current_width > max_width:
        rt_min = apex_rt - max_width / 2
        rt_max = apex_rt + max_width / 2

    data_rt_min = float(np.min(scan_time))
    data_rt_max = float(np.max(scan_time))
    rt_min = max(rt_min, data_rt_min)
    rt_max = min(rt_max, data_rt_max)
    timings["boundary_search_and_clamp"] = time.perf_counter() - t7
    timings["total_ms2_optimize_rt_span"] = time.perf_counter() - t0

    logger.debug(
        f"MS2 RT span optimized: apex={apex_rt:.1f}s, "
        f"span=[{rt_min:.1f}, {rt_max:.1f}]s, width={rt_max - rt_min:.1f}s"
    )
    logger.debug(
        "MS2 RT span timings: " +
        ", ".join(f"{k}={v:.4f}s" for k, v in timings.items())
    )

    return rt_min, rt_max, apex_rt


def combine_chromatograms(
    chromatograms: list,
    method: str = "max"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine multiple chromatograms into one representative signal.

    Args:
        chromatograms: List of dicts with 'scan_time' and 'intensity' arrays
        method: Combination method - 'max', 'mean', or 'median'

    Returns:
        Tuple of (combined_scan_time, combined_intensity)
    """
    if not chromatograms:
        return np.array([]), np.array([])

    if len(chromatograms) == 1:
        return (
            np.asarray(chromatograms[0]['scan_time']),
            np.asarray(chromatograms[0]['intensity'])
        )

    # Collect all unique time points
    all_times = set()
    for chrom in chromatograms:
        all_times.update(chrom['scan_time'])

    combined_time = np.array(sorted(all_times))

    # Interpolate each chromatogram to the common time grid
    interpolated = []
    for chrom in chromatograms:
        t = np.asarray(chrom['scan_time'])
        i = np.asarray(chrom['intensity'])
        if len(t) > 1:
            interp_i = np.interp(combined_time, t, i, left=0, right=0)
            interpolated.append(interp_i)

    if not interpolated:
        return np.array([]), np.array([])

    stacked = np.vstack(interpolated)

    if method == "max":
        combined_intensity = np.max(stacked, axis=0)
    elif method == "mean":
        combined_intensity = np.mean(stacked, axis=0)
    elif method == "median":
        combined_intensity = np.median(stacked, axis=0)
    else:
        combined_intensity = np.max(stacked, axis=0)

    return combined_time, combined_intensity


def _combine_ms2_chromatograms_sparse(chromatograms: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine MS2 chromatograms without interpolation.

    The MS2 optimizer bins the sparse signal with a max statistic anyway, so
    concatenating all raw points across files is substantially cheaper than
    interpolating each file onto a shared time grid first.
    """
    if not chromatograms:
        return np.array([]), np.array([])

    times = []
    intensities = []
    for chrom in chromatograms:
        scan_time = np.asarray(chrom["scan_time"], dtype=np.float64)
        intensity = np.asarray(chrom["intensity"], dtype=np.float64)
        if scan_time.size == 0 or intensity.size == 0:
            continue
        times.append(scan_time)
        intensities.append(intensity)

    if not times:
        return np.array([]), np.array([])

    if len(times) == 1:
        return times[0], intensities[0]

    return np.concatenate(times), np.concatenate(intensities)


def _prefetch_chromatograms_for_rt_optimization(conn, target_info: dict, progress_callback=None) -> dict:
    """
    Prefetch chromatograms for RT optimization.

    For MS2 targets, crop array payloads in DuckDB before returning them to Python.
    This avoids deserializing full chromatogram arrays when the optimizer only needs
    the local expected-RT neighborhood.
    """
    if not target_info:
        return {}

    half_window = float(DEFAULT_RT_OPTIMIZATION_WINDOW)
    chrom_by_target = {}

    ms1_labels = [
        peak_label
        for peak_label, info in target_info.items()
        if info.get("ms_type") != "ms2"
    ]
    ms2_targets = [
        (peak_label, info.get("expected_rt"))
        for peak_label, info in target_info.items()
        if info.get("ms_type") == "ms2"
    ]

    if ms1_labels:
        if progress_callback:
            progress_callback(0, len(target_info), f"Loading MS1 chromatograms ({len(ms1_labels)} targets)...")
        ms1_placeholders = ",".join(["?"] * len(ms1_labels))
        ms1_rows = conn.execute(
            f"""
                SELECT peak_label, scan_time, intensity
                FROM chromatograms
                WHERE peak_label IN ({ms1_placeholders})
            """,
            ms1_labels,
        ).fetchall()
        for peak_label, scan_time, intensity in ms1_rows:
            chrom_by_target.setdefault(peak_label, []).append(
                {"scan_time": scan_time, "intensity": intensity}
            )

    if not ms2_targets:
        return chrom_by_target

    if progress_callback:
        progress_callback(0, len(target_info), f"Loading MS2 chromatograms ({len(ms2_targets)} targets)...")

    values_sql = ",".join(["(?, ?)"] * len(ms2_targets))
    query = f"""
        WITH target_info AS (
            SELECT
                peak_label,
                expected_rt
            FROM (
                VALUES {values_sql}
            ) AS v(peak_label, expected_rt)
        ),
        base AS (
            SELECT
                c.peak_label,
                c.scan_time,
                c.intensity,
                t.expected_rt
            FROM chromatograms c
            JOIN target_info t USING (peak_label)
        ),
        zipped AS (
            SELECT
                peak_label,
                expected_rt,
                scan_time,
                intensity,
                list_transform(
                    range(1, len(scan_time) + 1),
                    i -> struct_pack(
                        t := list_extract(scan_time, i),
                        i := list_extract(intensity, i)
                    )
                ) AS pairs
            FROM base
        ),
        filtered AS (
            SELECT
                peak_label,
                scan_time,
                intensity,
                CASE
                    WHEN expected_rt IS NOT NULL THEN list_filter(pairs, p -> abs(p.t - expected_rt) <= ?)
                    ELSE pairs
                END AS pairs_in
            FROM zipped
        )
        SELECT
            peak_label,
            CASE
                WHEN len(pairs_in) >= 5 THEN list_transform(pairs_in, p -> p.t)
                ELSE scan_time
            END AS scan_time,
            CASE
                WHEN len(pairs_in) >= 5 THEN list_transform(pairs_in, p -> p.i)
                ELSE intensity
            END AS intensity
        FROM filtered
    """

    params = []
    for peak_label, expected_rt in ms2_targets:
        params.extend([peak_label, expected_rt])
    params.append(half_window)

    rows = conn.execute(query, params).fetchall()
    ms2_parts = {}
    for peak_label, scan_time, intensity in rows:
        scan_time_arr = np.asarray(scan_time, dtype=np.float64)
        intensity_arr = np.asarray(intensity, dtype=np.float64)
        if scan_time_arr.size == 0 or intensity_arr.size == 0:
            continue
        parts = ms2_parts.setdefault(peak_label, {"scan_time": [], "intensity": []})
        parts["scan_time"].append(scan_time_arr)
        parts["intensity"].append(intensity_arr)

    for peak_label, parts in ms2_parts.items():
        chrom_by_target[peak_label] = [{
            "scan_time": parts["scan_time"][0] if len(parts["scan_time"]) == 1 else np.concatenate(parts["scan_time"]),
            "intensity": parts["intensity"][0] if len(parts["intensity"]) == 1 else np.concatenate(parts["intensity"]),
        }]
    return chrom_by_target


def _chunk_targets(targets: list, chunk_size: int):
    if chunk_size <= 0:
        chunk_size = len(targets) or 1
    for start in range(0, len(targets), chunk_size):
        yield targets[start:start + chunk_size]


def optimize_rt_spans_batch(
    conn,
    threshold_pct: float = 0.10,
    min_width: float = 5.0,
    max_width: float = 120.0,
    progress_callback=None,
    ms2_prefetch_chunk_size: int = 16,
) -> int:
    """
    Optimize RT spans for all targets that were auto-adjusted.

    This function:
    1. Finds all targets with rt_auto_adjusted = TRUE
    2. For each target, combines chromatograms across all files
    3. Detects peak boundaries using adaptive method
    4. Updates rt_min, rt_max, and rt in the database

    Args:
        conn: Active DuckDB connection
        threshold_pct: Fraction of peak height for boundary detection
        min_width: Minimum allowed peak width in seconds
        max_width: Maximum allowed peak width in seconds

    Returns:
        Number of targets updated
    """
    # Get all targets that need optimization
    targets_to_optimize = conn.execute("""
        SELECT peak_label, rt, ms_type
        FROM targets
        WHERE rt_auto_adjusted = TRUE
    """).fetchall()

    if not targets_to_optimize:
        logger.info("No targets require RT span optimization")
        return 0

    total_targets = len(targets_to_optimize)
    if progress_callback:
        progress_callback(0, total_targets, f"Finding ROI targets ({total_targets})...")

    target_info = {
        peak_label: {"expected_rt": expected_rt, "ms_type": ms_type}
        for peak_label, expected_rt, ms_type in targets_to_optimize
    }

    updated_count = 0
    processed_targets = 0

    def process_target(peak_label, expected_rt, ms_type, chrom_data):
        target_t0 = time.perf_counter()
        fetch_elapsed = 0.0
        if not chrom_data:
            logger.warning(f"No chromatograms found for target '{peak_label}'")
            return 0

        prep_t0 = time.perf_counter()
        chromatograms = chrom_data
        prep_elapsed = time.perf_counter() - prep_t0

        # Combine chromatograms
        combine_t0 = time.perf_counter()
        if ms_type == "ms2":
            combined_time, combined_intensity = _combine_ms2_chromatograms_sparse(
                chromatograms
            )
        else:
            combined_time, combined_intensity = combine_chromatograms(
                chromatograms, method="max"
            )
        combine_elapsed = time.perf_counter() - combine_t0

        if len(combined_time) < 3:
            logger.warning(f"Insufficient data for target '{peak_label}'")
            return 0

        # Optimize RT span
        try:
            optimizer = optimize_rt_span_ms2 if ms_type == "ms2" else optimize_rt_span
            effective_threshold = min(threshold_pct, 0.05) if ms_type == "ms2" else threshold_pct

            optimize_t0 = time.perf_counter()
            rt_min, rt_max, apex_rt = optimizer(
                combined_time,
                combined_intensity,
                expected_rt or np.median(combined_time),
                min_width=min_width,
                max_width=max_width,
                threshold_pct=effective_threshold,
            )
            optimize_elapsed = time.perf_counter() - optimize_t0

            # Update database
            update_t0 = time.perf_counter()
            conn.execute("""
                UPDATE targets
                SET rt_min = ?,
                    rt_max = ?,
                    rt = ?,
                    rt_auto_adjusted = FALSE
                WHERE peak_label = ?
            """, [rt_min, rt_max, apex_rt, peak_label])
            update_elapsed = time.perf_counter() - update_t0

            logger.debug(
                f"Optimized RT span for '{peak_label}': "
                f"rt={apex_rt:.1f}s, span=[{rt_min:.1f}, {rt_max:.1f}]s"
            )
            logger.debug(
                f"RT span batch timings for '{peak_label}' ({ms_type}): "
                f"fetch={fetch_elapsed:.4f}s, "
                f"prepare={prep_elapsed:.4f}s, "
                f"combine={combine_elapsed:.4f}s, "
                f"optimize={optimize_elapsed:.4f}s, "
                f"update={update_elapsed:.4f}s, "
                f"total={time.perf_counter() - target_t0:.4f}s"
            )
            return 1

        except Exception as e:
            logger.error(f"Failed to optimize RT span for '{peak_label}': {e}")
            return 0

    ms1_targets = [t for t in targets_to_optimize if t[2] != "ms2"]
    ms2_targets = [t for t in targets_to_optimize if t[2] == "ms2"]

    if ms1_targets:
        ms1_target_info = {
            peak_label: target_info[peak_label]
            for peak_label, _expected_rt, _ms_type in ms1_targets
        }
        if progress_callback:
            progress_callback(
                processed_targets,
                total_targets,
                f"Loading MS1 chromatograms ({len(ms1_targets)})...",
            )
        fetch_t0 = time.perf_counter()
        ms1_chrom_by_target = _prefetch_chromatograms_for_rt_optimization(
            conn,
            ms1_target_info,
            progress_callback=None,
        )
        fetch_elapsed = time.perf_counter() - fetch_t0
        logger.debug(
            f"RT span batch prefetch (MS1): targets={len(ms1_target_info)}, "
            f"chromatograms={sum(len(v) for v in ms1_chrom_by_target.values())}, "
            f"total_fetch={fetch_elapsed:.4f}s"
        )
        for peak_label, expected_rt, ms_type in ms1_targets:
            if progress_callback:
                progress_callback(
                    processed_targets,
                    total_targets,
                    f"Optimizing ROI bounds ({processed_targets + 1}/{total_targets})...",
                )
            updated_count += process_target(
                peak_label,
                expected_rt,
                ms_type,
                ms1_chrom_by_target.get(peak_label, []),
            )
            processed_targets += 1
            if progress_callback:
                progress_callback(
                    processed_targets,
                    total_targets,
                    f"Optimized ROI bounds ({processed_targets}/{total_targets})...",
                )

    for chunk_idx, chunk in enumerate(_chunk_targets(ms2_targets, ms2_prefetch_chunk_size), start=1):
        chunk_start = processed_targets + 1
        chunk_end = processed_targets + len(chunk)
        chunk_target_info = {
            peak_label: target_info[peak_label]
            for peak_label, _expected_rt, _ms_type in chunk
        }
        if progress_callback:
            progress_callback(
                processed_targets,
                total_targets,
                f"Loading MS2 chromatograms ({chunk_start}-{chunk_end}/{total_targets})...",
            )
        fetch_t0 = time.perf_counter()
        ms2_chrom_by_target = _prefetch_chromatograms_for_rt_optimization(
            conn,
            chunk_target_info,
            progress_callback=None,
        )
        fetch_elapsed = time.perf_counter() - fetch_t0
        logger.debug(
            f"RT span batch prefetch (MS2 chunk {chunk_idx}): targets={len(chunk_target_info)}, "
            f"chromatograms={sum(len(v) for v in ms2_chrom_by_target.values())}, "
            f"total_fetch={fetch_elapsed:.4f}s"
        )
        if progress_callback:
            progress_callback(
                processed_targets,
                total_targets,
                f"ROI traces ready ({chunk_start}-{chunk_end}/{total_targets})...",
            )
        for peak_label, expected_rt, ms_type in chunk:
            if progress_callback:
                progress_callback(
                    processed_targets,
                    total_targets,
                    f"Optimizing ROI bounds ({processed_targets + 1}/{total_targets})...",
                )
            updated_count += process_target(
                peak_label,
                expected_rt,
                ms_type,
                ms2_chrom_by_target.get(peak_label, []),
            )
            processed_targets += 1
            if progress_callback:
                progress_callback(
                    processed_targets,
                    total_targets,
                    f"Optimized ROI bounds ({processed_targets}/{total_targets})...",
                )

    logger.info(f"Optimized RT spans for {updated_count} targets")
    return updated_count
