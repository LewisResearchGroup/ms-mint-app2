# Core Concepts

Understanding the underlying concepts of MINT helps in optimizing workflows and interpreting results.

## Sparsification

Chromatogram data can be extremely dense, especially in high-resolution MS1 measurements. MINT uses **Sparsification** to reduce the number of data points while preserving the essential shape of the peak.

- **How it works**: Instead of keeping every recorded intensity, MINT identifies transition points (e.g., scan times where the signal starts to rise, reaches a peak, or returns to baseline).
- **Benefits**: 
    - Dramatically reduces database size.
    - Speeds up plot rendering in the browser.
    - Minimizes memory usage during analysis.

## LTTB Downsampling

For real-time visualization of "full-range" chromatograms (before RT slicing), MINT employs the **Largest Triangle Three Buckets (LTTB)** algorithm.

- **Purpose**: LTTB effectively downsamples the data to a fixed number of points (e.g., 1000) for visual representation.
- **Visual Integrity**: Unlike simple decimation, LTTB preserves visual features like local minima and maxima, ensuring that the "Shadow Plot" accurately reflects the raw data shape.

## RT Alignment Logic

Retention Time (RT) alignment in MINT is designed for simplicity and robustness.

- **Median Apex Strategy**: MINT identifies the peak apex within a user-defined RT span across multiple samples. It then calculates the median of these apex positions to use as a stable "Reference RT".
- **Authoritative Shifts**: For each sample, the "Shift" is defined as `Reference_RT - Sample_Apex_RT`. These shifts are applied to the chromatogram traces in the Optimization view, effectively "snapping" them together.
- **Persistence**: Unlike transient visualization-only shifts, MINT alignment state is authoritative and persisted in the database, ensuring that downstream processing uses the corrected windows.

## Progressive Loading (Shadow Plots)

MINT prioritizes UI responsiveness through progressive loading:

1. **Step 1**: Load a pre-calculated "Shadow Plot" (Envelope) using binned/downsampled data. This provides an immediate overview of the peak shape across all samples.
2. **Step 2**: Load detailed, sparsified chromatogram data in the background.
3. **Step 3**: Re-render with full interaction (manual optimization, legend filtering) once the detailed data is available.
