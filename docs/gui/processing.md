## Processing {: #processing }

The `Processing` tab is where the core data extraction takes place. MINT extracts the peak areas and other features for all defined targets across your loaded MS-files.

![Processing](../image/processing_v1.1.1.png "Processing")

> **Tip**: Click the help icon (small "i" symbol) next to the "Processing" title to take a guided tour of this section.

### Running the Analysis {: #running-the-analysis }

1.  **Run MINT**: Click the `RUN MINT` button. This opens a configuration modal:
    *   **Bookmarked Targets Only**: Check this to process *only* the targets you bookmarked in the Optimization tab.
    *   **Recompute results**: Force MINT to re-process files even if results already exist.
    *   **Resources**: Adjust **CPU** cores, **RAM** limit, and **Batch Size**. For most users, the defaults are sufficient.
    *   Click **Run** to start.
2.  **Select Targets**: Use the `Targets` dropdown menu to filter which metabolites are displayed in the results table.
3.  **Review Results**: The main table displays the extraction results (peak area, height, RT, etc.). You can sort and filter columns directly in the table.

### Exporting Data {: #exporting-data }

Click the `DOWNLOAD RESULTS` button to open the export options:

*   **All Results**: Download the complete, raw results table in "tidy" (long) format. Ideal for downstream analysis in Python/R.
*   **Dense Matrix**: Create a pivoted table (wide format). This is best for heatmaps or spreadsheet analysis.
    *   **Rows**: Select the field for rows (usually `ms_file_label` or `sample_type`).
    *   **Columns**: Select the field for columns (usually `peak_label`).
    *   **Values**: Select the metric to fill the cells (e.g., `peak_area_top3`, `peak_max` or `peak_area`).
    *   Click **Download** to generate the CSV.

### Managing Results {: #managing-results }

Use the `Options` menu (top right) to:

*   **Delete selected results**: Remove specific rows from the current view.
*   **Clear results**: Delete all processed results to start over (useful if you need to rerun with different parameters).
