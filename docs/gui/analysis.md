## Analysis {: #analysis }
After running MINT the results can be downloaed or analysed using the provided tools.
For quality control purposes histograms and boxplots can be generated in the
quality control tab. The interactive heatmap tool can be used to explore the results data after `RUN MINT`
has been exectuted. The tool allows to explore the generated data in from of heatmaps.


## Selections and transformations {: #selections-and-transformations }
  - Include/exclude file types (based on `Type` column in metadata)
  - Include/exclude peak labels for analysis
  - Set file sorting (e.g. by name, by batch etc.)
  - Select group-by column for coloring and statistics

![Selections](../image/general-selection-elements.png "Selections")

- `Types of files to include`: Uses the `sample_types` column in the Metadata tab to select files. If nothing is selected, all files are included.
- `Include peak_labels`: Targets to include. If nothing is selected all targets are included.
- `Exclude peak_labels`: Targets to exclude. If nothing is selected no target is excluded.
- `Variable to plot`: This determines which column to run the analysis on. For example, you can set this to `peak_mass_diff_50pc` to analyse the instrument calibration. The default is `peak_area_top3`.
- `MS-file sorting`: Before plotting sets the order of the MS-files in the underlying dataframe. This will change the order of files in some plots.
- `Color by`: PCA and `Plotting` tool can use a categoric or numeric column to color code samples. Some plots (e.g. Hierarchical clustering tool are unaffected).
- `Transformation`: The values can be log transformed before subjected to normalization. If nothing is selected, the raw values are used.
- `Scaling group(s)`: Column or selection of columns to group the data and apply the normalization function in the dropdown menu for each group. If you want to z-scores for each target, you need to select `peak_label` here, and in the dropdown menu 'Standard scaling`.
- `Scaling technique`: You can choose between standard scaling, min-max scaling, or robust scaling, or no scaling (if nothing is selected).

### Scaling Techniques

#### 1. Standard Scaling

**Standard scaling** (also known as z-score normalization) transforms the data such that the mean of each feature becomes 0 and the standard deviation becomes 1. This is useful when the features have different units or magnitudes, as it ensures they are on the same scale.

The formula for standard scaling is:

    z = (x - mean) / standard_deviation

Where:
- `x` is the original value.
- `mean` is the mean of the feature.
- `standard_deviation` is the standard deviation of the feature.

#### 2. Robust Scaling

**Robust scaling** is used to scale features using statistics that are robust to outliers. This scaling technique uses the median and the interquartile range (IQR) instead of the mean and standard deviation, making it more suitable for datasets with outliers.

The formula for robust scaling is:

    x_scaled = (x - median) / IQR

Where:
- `x` is the original value.
- `median` is the median of the feature.
- `IQR` is the interquartile range of the feature (IQR = Q3 - Q1).

#### 3. Min-Max Scaling

**Min-max scaling** (also known as normalization) transforms the data to fit within a the range [0, 1]. This scaling techique is useful when you want to preserve the relationships within the data, but want to adjust the scale.

The formula for min-max scaling is:

    x_scaled = (x - x_min) / (x_max - x_min)

Where:
- `x` is the original value.
- `x_min` is the minimum value of the feature.
- `x_max` is the maximum value of the feature.


## Heatmap {: #heatmap }
![Heatmap](../image/heatmap.png "Heatmap")

The first dropdown menu allows to include certain file types e.g. biological samples rather than quality control samples.
The second dropdown menu distinguishes the how the heatmap is generated.

- `Cluster`: Cluster rows with hierachical clustering.
- `Dendrogram`: Plots a dendrogram instead of row labels (only in combination with `Cluster`).
- `Transpose`: Switch columns and rows.
- `Correlation`: Calculate pearson correlation between columns.
- `Show in new tab`: The figure will be generated in a new independent tab. That way multiple heatmaps can be generated at the same time. This may only work when you serve MINT locally, since the plot is served on a different port. If the app becomes unresponsive to changes, reload the tab.

### Example: Plot correlation between metabolites using scaled peak_area_top3 values

![Heatmap](../image/heatmap-correlation.png "Correlation")


## Distributions {: #distributions }

  - Plot histograms
  - Density distributions
  - Boxplots

### Example: Box-plot of scaled peak_area_top3 values by metabolite

![Quality Control](../image/distributions.png "Quality Control")

The MS-files can be grouped based on the values in the metadata table. If nothing
is selected the data will not be grouped in order to plot the overall distribution.
The second dropdown menu allows to select one or multple kinds of graphs that
to generate. The third dropdown menu allows to include certain file types.
For example, the analysis can be limited to only the biological samples if such a
type has been defined in the type column of the metadata table.

The checkbox can be used to create a dense view. If the box is unchecked the output will be visually grouped into an individual section for each metabolite.

The plots are interactive. You can switch off labels, zoom in on particular areas of interest, or hover the mouse cursor over a datapoint to get more information about underlying sample and/or target.

## Principal Component Analysis (PCA) {: #principal-component-analysis-pca }

 - Perform Principal Component Analysis (PCA)
 - Plot projections to first N principal components
 - Contributions of original variables to each component.

Principal Component Analysis (PCA) is a statistical technique used to reduce the dimensionality of a dataset while preserving as much variability (information) as possible. It transforms the original data into a new coordinate system where the greatest variances by any projection of the data come to lie on the first coordinates called principal components.

**Principal Components**

  - **Definition**: Principal components are the new set of axes in the transformed feature space. They are linear combinations of the original features.
  - **Purpose**: These components are ordered by the amount of variance they explain from the data. The first principal component explains the most variance, the second the second most, and so on.

**Cumulative Explained Variance**

  - **Definition**: The cumulative explained variance is the sum of the explained variances of the principal components up to a given component. It indicates the proportion of the total variance in the dataset that is accounted for by the principal components.
  - **Purpose**: This helps in deciding how many principal components to keep by showing how much of the total variance is captured as you include more components.

**PCA Loadings**

  - **Definition**: PCA loadings represent the coefficients of the linear combination of the original variables that define each principal component. They indicate the contribution of each original feature to the principal components.
  - **Purpose**: Loadings help in understanding the importance of each feature in the principal components and how they contribute to the variance explained by each component.

### Example: PCA colored by sample label (i.e. biological organism) using z-scores
![PCA](../image/pca.png "Principal Components Analysis")


## Hierarchical clustering {: #hierarchical-clustering }
Hierarchical clustering is a technique for cluster analysis that seeks to build a hierarchy of clusters. It can be divided into two main types: **agglomerative** and **divisive**.  MINT uses agglomerative hierarchical clustering, also known as bottom-up clustering, starts with each data point as a separate cluster and iteratively merges the closest clusters until all points are in a single cluster or a stopping criterion is met.

### Steps for Agglomerative Clustering
1. **Initialization**: Start with each data point as its own cluster.
2. **Distance Calculation**: Compute the pairwise distance between all clusters.
3. **Merge Closest Clusters**: Find the two closest clusters and merge them into a single cluster.
4. **Update Distances**: Recalculate the distances between the new cluster and all other clusters.
5. **Repeat**: Repeat steps 3 and 4 until all data points are in a single cluster or the desired number of clusters is achieved.

### Dendrogram

The output of hierarchical clustering is often visualized using a dendrogram, which is a tree-like diagram that shows the arrangement of clusters and their hierarchical relationships. Each branch of the dendrogram represents a merge or split, and the height of the branches indicates the distance or dissimilarity between clusters.

### Example: Hirarchical clustering with different metrics using z-scores (for each metabolite)
![Hierarchical clustering](../image/hierarchical_clustering.png "Hierarchical clustering")

## Plotting {: #plotting }
With great power comes great responsibility. The plotting tool can generate impressive, and very complex plots, but it can be a bit overwhelming in the beginning. It uses the [Seaborn](http://seaborn.pydata.org/) library under the hood. Familiarity, with this library can help understanding what the different settings are doing. We recommend starting with a basic plot and then increase its complexity stepwisely.

- Bar plots
- Violin plots
- Boxen plot
- Scatter plots
- and more...

### Example: Compare log2 transformed and then z-scaled peak_area_top3 for between LC columns for all metabolites.

![](../image/plotting-example-1.png)
