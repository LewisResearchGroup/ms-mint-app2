## MS-Files {: #ms-files }

The `MS-Files` tab is the entry point for your analysis. Here you organize the raw mass spectrometry data that constitutes your workspace. MINT currently supports `.mzXML` and `.mzML` file formats.

![MS-files](../image/ms_files_v1.1.1.png "MS-files")

> **Tip**: Click the help icon (small "i" symbol) next to the "MS-Files" title to take a guided tour of this section.

### Loading Files and Metadata {: #loading-files }
To add data to your workspace, click the `Load MS-Files` button. This opens a dedicated file browser modal where you can:

1.  **Browse Server Files**: Use the left pane to navigate the directory structure of the computer running MINT.
2.  **Select Files**: Check the boxes next to the files you wish to import. You can filter by file extension (e.g., `.mzXML`, `.mzML`) using the tags below the file list.
3.  **Review Selection**: The right pane shows your currently selected files. You can remove specific files or clear the entire selection before processing.
4.  **Process**: Click `Process Files` to import them into your workspace. MINT will automatically extract the information to a DuckDB database.
    *   **CPUs**: You can specify the number of CPU cores to use for parallel processing to speed up the import of large datasets.

Same can be done for metadata files using the `Load Metadata` button. You can use the `DOWNLOAD TEMPLATE` button to download a template file that you can fill out and import using the `Load Metadata` button. This file contains important information about your samples. Only `ms_file_label` and `sample_type`are essential; the remaining columns are optional but useful for grouping and plotting. If any of the columns `use_for_optimization`, `use_for_processing`, `use_for_analysis` are left blank they will be assumed to be `TRUE`.

| Column Name             | Description                                                       |
|-------------------------|-------------------------------------------------------------------|
|`ms_file_label`          | Unique file name; must match the MS file on disk                  |
|`label`                  | Friendly label to display in plots and reports                    |
|`color`                  | Hex color for visualizations (auto-generated if blank)            |
|`use_for_optimization`   | True to include in optimization steps (COMPUTE CHROMATOGRAMS)     |
|`use_for_processing`     | True to include in processing (RUN MINT)                          |
|`use_for_analysis`       | True to include in analysis outputs                               |
|`sample_type`            | Sample category (e.g.; Sample; QC; Blank; Standard)               |
|`group_1`                | User-defined grouping field 1 for analysis/grouping (free text)   |
|`group_2`                | User-defined grouping field 2 for analysis/grouping (free text)   |
|`group_3`                | User-defined grouping field 3 for analysis/grouping (free text)   |
|`group_4`                | User-defined grouping field 4 for analysis/grouping (free text)   |
|`group_5`                | User-defined grouping field 5 for analysis/grouping (free text)   |
|`polarity`               | Polarity (Positive or Negative)                                   |
|`ms_type`                | Acquisition type (ms1 or ms2)                                     |

### The Main Table {: #the-main-table }
The main table displays an overview of all imported files with several interactive columns:

*   **Checkbox**: Select multiple files for batch actions (like deletion).
*   **Color**: This color will be used in plots to identify these samples. You can change the color by clicking the color rectangle, by using the `Generate Color` function under the `Options` menu, or importing a metadata file.
*   **For Optimization / Processing / Analysis**: Toggle switches to include or exclude specific files from different stages of the workflow. This is useful if you want to exclude certain files from optimization or analysis. For example, if you have a large dataset and you want to exclude certain files from optimization, you can use this feature to exclude them. Likewise, if you want to exclude certain files from analysis like blanks or standards, you can use this feature to exclude them. This process can be done individually or by importing a metadata file.
*   **Metadata Columns**: Columns like `Label`, `Sample Type`, and `Groups` allow you to organize your data. These are typically populated by importing a metadata file using the `Load Metadata` button.

### Options Menu {: #options-menu }
The `Options` dropdown (top right) provides quick actions such as:

*   **Delete selected files**: Removes currently checked files from the workspace.
*   **Clear table**: Removes all MS-files from the workspace.
