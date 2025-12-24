## Workspaces {: #workspaces }

Workspaces allow you to organize and separate different projects. Each workspace is a self-contained environment for your project files.

> **Tip**: Click the help icon (small "i" symbol) next to the "Workspaces" title to take a guided tour of this section.

![Workspaces](../image/workspaces_v1.1.1.png)

### Storage Location {: #storage-location }

All files for a workspace are stored in a subdirectory within your MINT data folder (`--data-dir`).

-   **Default Location**: `~/MINT` (Linux/macOS) or `C:/Users/<username>/MINT` (Windows)
-   **Active Workspace**: The active workspace is displayed in the sidebar under "Workspace:".

#### Changing the Data Directory {: #changing-data-directory }
You can change the location where MINT stores all workspaces.

1.  Click the `Change Location` button next to the "Current Data Directory" display.
2.  Enter the **absolute path** for the new folder.
3.  Click `Save & Reload`.

![Data Directory Modal](../image/workspaces_data_dir_v1.1.1.png "Changing Global Data Directory")

> **Note**: Changing this path will reload the application. Workspaces from the old directory are not moved; you will start with an empty list in the new location until you create new workspaces or move your valid workspace folders manually.

### Managing Workspaces {: #managing-workspaces }

You can manage your workspaces using the controls in the Workspaces tab:

-   **Create a Workspace**: Click the `+ Create Workspace` button (bottom left). Enter a name in the popup window and click `Create`.
-   **Activate a Workspace**: Click on the selection circle in the list. The active workspace is indicated by a text notification and update in the sidebar.
-   **Delete a Workspace**: Select a workspace and click the `- Delete Workspace` button (bottom right). Confirm the action in the popup window.

    > **Warning**: Deleting a workspace will permanently remove the workspace folder and all its contents from your hard drive. This action cannot be undone.

### Workspace Details {: #workspace-details }
Click the `+` icon next to a workspace name to expand the row. This shows the absolute path to the workspace and a summary of the data it contains (_e.g._, number of MS-files, targets, and results).

### Logging and Troubleshooting {: #logging-and-troubleshooting }
MINT automatically tracks actions and errors to help with debugging.

-   **Log File**: A file named `ws.log` is created inside each workspace folder. It contains a detailed history of operations performed within that workspace.
-   **Terminal Output**: If you run MINT from the command line, logs are also printed to the terminal in real-time.

If you encounter an issue, please check the `ws.log` file in your workspace directory for error messages before reporting a bug.
