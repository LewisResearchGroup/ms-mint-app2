## Targets {: #targets }

Target lists are collections of peak definitions used to extract MS intensities for specific metabolites. The `Targets` tab allows you to manage, review, and refine these definitions before processing.

> **Tip**: Click the help icon (small "i" symbol) next to the "Targets" title to take a guided tour of this section.

![Targets](../image/targets_v1.1.1.png "Targets")

### Loading and Managing Targets {: #loading-and-managing-targets }

To import a target list, click the `Load Targets` button. This opens a file browser where you can navigate your filesystem and select one or more CSV files containing your peak definitions.

*   **Template**: For best results, use the standard format. Click the `DOWNLOAD TEMPLATE` button to download a pre-formatted CSV file with the required columns (`peak_label`, `mz_mean`, `mz_width`, `rt`, `rt_min`, `rt_max`).

### The Targets Table {: #the-targets-table }

Once loaded, your targets are displayed in an interactive table with the following key columns:

*   **Selection & Bookmark**: Selection is meant for selection of targets for processing. Bookmarking is meant for bookmarking specific targets for later use.
*   **Peak Label**: The unique identifier for the metabolite or feature.
*   **MZ Data**: `mz_mean` and `mz_width` define the mass-to-charge ratio window for extraction.
*   **Retention Time**: `rt_min` and `rt_max` define the expected time window for the peak. The `rt` column typically represents the expected retention time of the peak.
*   **Filtering**: Each column header includes a filter icon, allowing you to search for specific compounds or filter by values.

### Options Menu {: #options-menu }

The **Options** dropdown (top right) provides bulk actions for list management:

*   **Delete selected**: Remove currently checked rows from the workspace.
*   **Clear table**: Remove all targets.

Target lists can be provided as CSV files. For more details on the file format, see [Target Lists](../targets.md).
