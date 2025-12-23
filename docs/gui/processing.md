## Processing {: #processing }

The `Processing` tab is where the core data extraction takes place. MINT extracts the peak areas and other features for all defined targets across your loaded MS-files.

![Processing](../image/processing_v1.1.1.png "Processing")

> **Tip**: Click the help icon (small "i" symbol) next to the "Processing" title to take a guided tour of this section.

### Running the Analysis {: #running-the-analysis }

1.  **Run MINT**: Click the `RUN MINT` button at the top left. A progress bar will indicate the status of the extraction process.
2.  **Select Targets**: Use the `Targets` dropdown menu to filter which metabolites are displayed in the results table.
3.  **Review Results**: The main table displays the extraction results. Key columns include:
    *   **peak_area**: The total area under the peak (primary quantitative metric).
    *   **peak_max**: The maximum intensity value of the peak.
    *   **peak_rt_of_max**: The retention time at which the maximum intensity occurs.
    *   **peak_mean / peak_median**: Statistical properties of the peak intensity.

### Exporting Data {: #exporting-data }

Click the `DOWNLOAD RESULTS` button to open the export options:

*   **All Results**: Download the complete results table in "tidy" (long) format, suitable for advanced analysis in Python or R. You can customize which columns to include.
*   **Dense Matrix**: Generate a pivot table (wide format) which is often easier to use with spreadsheet software or for PCA/Heatmaps.
    *   **Rows/Columns**: Typically, files are rows and targets are columns (or vice versa).
    *   **Values**: Select the metric to fill the cells (default is `peak_area_top3` or `peak_area`).
    *   **Transpose**: Check this box to swap rows and columns.

### Managing Results {: #managing-results }

Use the `Options` menu (top right) to:

*   **Delete selected results**: Remove specific rows from the current view.
*   **Clear results**: Delete all processed results to start over (useful if you need to rerun with different parameters).
