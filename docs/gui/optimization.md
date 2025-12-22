## Optimization {: #optimization }

The **Optimization** tab is designed to refine your peak integration windows (Retention Time) for specific targets. By optimizing the Retention Time (RT) ranges, you ensure that MINT extracts the maximum intensity for the correct peak, improving data quality.

![Optimization](../image/optimization_v1.1.1.png "Optimization")

> **Tip**: Click the help icon (small "i" symbol) next to the "Optimization" title to take a guided tour of this section.

### Peak Optimization Workflow {: #peak-optimization-workflow }

1.  **Define Scope**: Use the sidebar to select the specific `Samples` (by type, batch, etc.) and `Targets` you want to optimize.
2.  **Compute Chromatograms**: Click the `COMPUTE CHROMATOGRAMS` button. MINT will extract and display the Ion Chromatograms (EIC) for the selected targets across your samples.
3.  **Review Optimization Cards**: The results are displayed as "Optimization Cards". Each card represents a target and shows:
    *   **Chromatogram Plot**: The shape of the peak across samples.
    *   **Current RT Range**: The vertical dashed lines indicate the start and end of the integration window.

### Interactive Manual Optimization {: #interactive-manual-optimization }
Click on any `Optimization Card` or the `Graph` icon to open the detailed `Manual Optimization` modal. In this view, you can fine-tune the peak selection:

*   **Interactive Tuning**: Click and drag on the plot to manually set the new Retention Time (RT) start and end points for the target.
*   **Intensity Scale**: Toggle between `Linear` and `Log` scales to better visualize low-intensity peaks.
*   **Legend Behavior**: Switch between `Group` (aggregated) or `Single` (individual file) legend displays.
*   **Edit RT-span**: Enable `Edit` mode to adjust ranges, or `Lock` to prevent accidental changes.
*   **Save/Reset**: Save your adjusted RT bounds to the target list or reset them to the original attributes.
