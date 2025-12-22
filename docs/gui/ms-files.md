## MS-Files {: #ms-files }

The `MS-Files` tab is the entry point for your analysis. Here you organize the raw mass spectrometry data that constitutes your workspace. MINT currently supports `.mzXML` and `.mzML` file formats.

![MS-files](../image/ms_files_v1.1.1.png "MS-files")

> **Tip**: Click the help icon (small "i" symbol) next to the "MS-Files" title to take a guided tour of this section.

### Loading Files {: #loading-files }
To add data to your workspace, click the `Load MS-Files` button. This opens a dedicated file browser modal where you can:

1.  **Browse Server Files**: Use the left pane to navigate the directory structure of the computer running MINT.
2.  **Select Files**: Check the boxes next to the files you wish to import. You can filter by file extension (e.g., `.mzXML`, `.mzML`) using the tags below the file list.
3.  **Review Selection**: The right pane shows your currently selected files. You can remove specific files or clear the entire selection before processing.
4.  **Process**: Click `Process Files` to import them into your workspace. MINT will automatically extract the information to a DuckDB database.
    *   **CPUs**: You can specify the number of CPU cores to use for parallel processing to speed up the import of large datasets.

### The Main Table {: #the-main-table }
The main table displays an overview of all imported files with several interactive columns:

*   **Checkbox**: Select multiple files for batch actions (like deletion).
*   **Color**: This color will be used in plots to identify these samples. You can change the color by clicking the color rectangle, by using the `Generate Color` function under the `Options` menu, or importing a metadata file.
*   **For Optimization / Processing / Analysis**: Toggle switches to include or exclude specific files from different stages of the workflow. This is useful if you want to exclude certain files from optimization or analysis. For example, if you have a large dataset and you want to exclude certain files from optimization, you can use this feature to exclude them. Likewise, if you want to exclude certain files from analysis like blanks or standards, you can use this feature to exclude them. THis process can be done individually or by importing a metadata file.
*   **Metadata Columns**: Columns like `Label`, `Sample Type`, and `Groups` allow you to organize your data. These are typically populated by importing a metadata file using the `Load Metadata` button.

**Options Menu**: The `Options` dropdown (top right) provides quick actions such as:

*   **Delete selected files**: Removes currently checked files from the workspace.
*   **Reset filters/columns**: Restores the default table view.
