## Quality Control {: #quality-control }
Analytical visualizations to display a few quality metrics and comparisons. The `m/z drift` compares the observed m/z values with the ones set in the target list. This value will always be lower than the `mz_width` set in the target list for each target. It is one way of evaluating how well the machine is calibrated. Generally speaking, values between [-5, 5] are acceptible, but it depends on the specific assay and experiment.

The graphs are categorized by `sample_type` set in the Metadata tab. You should have some quality control, or calibration samples with known metabolite composition, to be able to make judgements about the quality.

The second plot breaks down the `m/z drift` by target, to see how the calibration varies between targets.

The PCA (Principal Components Analysis) plot shows a PCA using `peak_area_top3`. You can compare different groups of samples as set in the `sample_types` column in the Metadata tab.

The final plot displays peak shapes of a random sample of files for all targets. To change the sample, you can refresh this page.
