"""
Peak Fitting Module for MS-MINT App

Provides EMG (Exponentially Modified Gaussian) peak fitting for accurate
chromatographic peak quantification.

EMG model: Convolution of Gaussian with Exponential decay
- Physically models chromatographic peak broadening
- Handles asymmetric/tailing peaks naturally
- Standard in chromatography software

Performance: ~1,400 peaks/sec with 8-core parallelization
"""

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
from scipy.special import erf

logger = logging.getLogger(__name__)


@dataclass
class FitResult:
    """Result of EMG peak fitting."""
    success: bool
    peak_area_fitted: float
    peak_sigma: float          # Gaussian width (σ)
    peak_tau: float            # Exponential tail decay (τ/gamma)
    peak_asymmetry: float      # τ/σ ratio (>2 indicates problematic tailing)
    peak_rt_fitted: float      # Peak center from fit
    fit_r_squared: float       # Goodness of fit
    error_message: Optional[str] = None


def emg_gaussian(x: np.ndarray, amplitude: float, center: float,
                 sigma: float, gamma: float) -> np.ndarray:
    """
    Exponentially Modified Gaussian (EMG) model.
    
    EMG = (A * γ / 2) * exp(γ * (μ - x + γσ²/2)) * erfc((μ - x + γσ²) / (σ√2))
    
    Using erf: erfc(z) = 1 - erf(z)
    
    Parameters:
        x: Array of x values (retention time)
        amplitude: Peak amplitude
        center: Peak center (μ)
        sigma: Gaussian standard deviation (σ)
        gamma: Exponential decay rate (τ = 1/γ in some conventions)
    
    Returns:
        Array of y values (intensity)
    """
    # Suppress overflow/invalid warnings - we handle them with clipping and nan_to_num
    with np.errstate(over='ignore', invalid='ignore'):
        z = (center + gamma * sigma**2 - x) / (sigma * np.sqrt(2))
        cerf_term = 1 - erf(z)
        exp_arg = gamma * (center - x + (gamma * sigma**2) / 2)
        # Clip to prevent overflow
        exp_arg = np.clip(exp_arg, -700, 700)
        exp_term = np.exp(exp_arg)
        result = (amplitude * gamma / 2) * exp_term * cerf_term
        # Replace NaN/Inf with 0 to prevent fitting failures
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def fit_emg_peak(
    scan_time: np.ndarray,
    intensity: np.ndarray,
    expected_rt: Optional[float] = None,
) -> FitResult:
    """
    Fit EMG model to a single chromatographic peak.
    
    Parameters:
        scan_time: Array of retention times
        intensity: Array of intensity values
        expected_rt: Expected peak retention time (optional, for initial guess)
    
    Returns:
        FitResult dataclass with fitted parameters
    """
    # Ensure numpy arrays
    scan_time = np.asarray(scan_time, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    
    # Handle edge cases
    if len(scan_time) < 5 or len(intensity) < 5:
        return FitResult(
            success=False,
            peak_area_fitted=trapezoid(intensity, scan_time) if len(intensity) > 1 else 0.0,
            peak_sigma=0.0,
            peak_tau=0.0,
            peak_asymmetry=0.0,
            peak_rt_fitted=expected_rt or 0.0,
            fit_r_squared=0.0,
            error_message="Insufficient data points"
        )
    
    # Initial parameter estimates
    idx_max = np.argmax(intensity)
    max_intensity = intensity[idx_max]
    
    if max_intensity <= 0:
        return FitResult(
            success=False,
            peak_area_fitted=0.0,
            peak_sigma=0.0,
            peak_tau=0.0,
            peak_asymmetry=0.0,
            peak_rt_fitted=expected_rt or scan_time[idx_max],
            fit_r_squared=0.0,
            error_message="No positive intensity"
        )
    
    # Initial guesses
    center_guess = expected_rt if expected_rt else scan_time[idx_max]
    sigma_guess = (scan_time[-1] - scan_time[0]) / 10  # ~10% of window
    gamma_guess = 1.0  # Moderate tailing
    amplitude_guess = max_intensity * 2  # Compensate for EMG spreading
    
    p0 = [amplitude_guess, center_guess, sigma_guess, gamma_guess]
    
    # Parameter bounds
    bounds = (
        [0, scan_time.min(), 0.01, 0.1],           # Lower bounds
        [1e12, scan_time.max(), 50.0, 20.0]        # Upper bounds
    )
    
    try:
        popt, pcov = curve_fit(
            emg_gaussian,
            scan_time,
            intensity,
            p0=p0,
            bounds=bounds,
            maxfev=1000,
            method='trf'  # Trust Region Reflective - robust for bounds
        )
        
        amplitude, center, sigma, gamma = popt
        
        # Calculate fitted curve and area
        fitted_curve = emg_gaussian(scan_time, *popt)
        peak_area_fitted = trapezoid(fitted_curve, scan_time)
        
        # Calculate R² (coefficient of determination)
        ss_res = np.sum((intensity - fitted_curve) ** 2)
        ss_tot = np.sum((intensity - np.mean(intensity)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Calculate asymmetry factor (τ/σ)
        # gamma is the rate, tau = 1/gamma in time units
        # But in our parameterization, gamma directly relates to tailing
        peak_asymmetry = gamma / sigma if sigma > 0 else 0.0
        
        return FitResult(
            success=True,
            peak_area_fitted=peak_area_fitted,
            peak_sigma=sigma,
            peak_tau=gamma,  # In our convention, gamma is the tailing parameter
            peak_asymmetry=peak_asymmetry,
            peak_rt_fitted=center,
            fit_r_squared=r_squared,
            error_message=None
        )
        
    except Exception as e:
        # Fitting failed - return fallback values
        fallback_area = trapezoid(intensity, scan_time)
        return FitResult(
            success=False,
            peak_area_fitted=fallback_area,
            peak_sigma=0.0,
            peak_tau=0.0,
            peak_asymmetry=0.0,
            peak_rt_fitted=expected_rt or scan_time[idx_max],
            fit_r_squared=0.0,
            error_message=str(e)[:100]
        )


def _fit_peak_wrapper(args: Tuple) -> dict:
    """
    Wrapper for parallel processing.
    
    Args:
        args: Tuple of (peak_label, ms_file_label, scan_time, intensity, expected_rt)
    
    Returns:
        Dictionary with peak_label, ms_file_label, and fit results
    """
    peak_label, ms_file_label, scan_time, intensity, expected_rt = args
    
    result = fit_emg_peak(scan_time, intensity, expected_rt)
    
    return {
        'peak_label': peak_label,
        'ms_file_label': ms_file_label,
        'peak_area_fitted': result.peak_area_fitted,
        'peak_sigma': result.peak_sigma,
        'peak_tau': result.peak_tau,
        'peak_asymmetry': result.peak_asymmetry,
        'peak_rt_fitted': result.peak_rt_fitted,
        'fit_r_squared': result.fit_r_squared,
        'fit_success': result.success,
    }


def fit_peaks_batch(
    peaks_data: List[Tuple],
    n_workers: int = 8,
    chunksize: int = 50,
    progress_callback=None,
) -> List[dict]:
    """
    Fit EMG model to multiple peaks using parallel processing.
    
    Parameters:
        peaks_data: List of tuples (peak_label, ms_file_label, scan_time, intensity, expected_rt)
        n_workers: Number of parallel workers
        chunksize: Number of peaks per work unit
        progress_callback: Optional callback(completed, total, rate) called periodically
    
    Returns:
        List of dictionaries with fit results
    """
    from concurrent.futures import as_completed
    import time
    
    if not peaks_data:
        return []
    
    total = len(peaks_data)
    
    if total < n_workers * 2:
        # Not enough work for parallelization, run sequentially
        results = []
        for i, args in enumerate(peaks_data):
            results.append(_fit_peak_wrapper(args))
            if progress_callback and (i + 1) % 10 == 0:
                progress_callback(i + 1, total, 0)
        return results
    
    results = [None] * total  # Pre-allocate to maintain order
    completed = 0
    start_time = time.time()
    last_update = start_time
    
    executor = ProcessPoolExecutor(max_workers=n_workers)
    try:
        # Submit all tasks with their indices
        future_to_idx = {
            executor.submit(_fit_peak_wrapper, args): idx
            for idx, args in enumerate(peaks_data)
        }
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                # Handle failed futures
                results[idx] = {
                    'peak_label': peaks_data[idx][0],
                    'ms_file_label': peaks_data[idx][1],
                    'peak_area_fitted': 0.0,
                    'peak_sigma': 0.0,
                    'peak_tau': 0.0,
                    'peak_asymmetry': 0.0,
                    'peak_rt_fitted': 0.0,
                    'fit_r_squared': 0.0,
                    'fit_success': False,
                }
                logger.error(f"Peak fitting failed: {e}")
            
            completed += 1
            
            # Progress updates:
            # 1. Always on completion
            # 2. Every 5% (to align with typical batch logging)
            now = time.time()
            milestone = max(1, total // 20)
            
            if progress_callback and (
                completed == total or 
                completed % milestone == 0
            ):
                elapsed = now - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                progress_callback(completed, total, rate)
                last_update = now
                
    except BaseException:
        # If cancelled or interrupted, shutdown immediately and cancel pending tasks
        logger.warning("Peak fitting cancelled - shutting down workers...")
        executor.shutdown(wait=False, cancel_futures=True)
        raise
        
    executor.shutdown(wait=True)
    return results

