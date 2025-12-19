
# Quickstart Guide for ms-mint-app
Welcome to the ms-mint-app quickstart guide! This guide will help you get up and running with the application, allowing you to start analyzing mass spectrometry data efficiently. Follow the steps below to install the app, create a workspace, and begin processing your data.

## 1. Open `ms-mint-app`

Download an executable compatible with your OS and open MINT (explore other options to install MINT [here](install.md)).

<!-- To install MINT, run:

```
pip install ms-mint-app
```

or follow the instructions [here](install.md).

Then start the application with

```
Mint
```

or, if you have a prefered directory for data you can specify it with `--data-dir` e.g.:

```
Mint --data-dir /data
```

The application will take a while until it starts up. In the mean time the browser window will show

> This site can’t be reached

Just wait a bit until the terminal shows `INFO:waitress:Serving on http://127.0.0.1:9999` and refresh the page.
The application is now served on port `9999` of your local machine. -->

## 2. Create a workspace
If you have never started the application before, you will not have any workspaces yet. A workspace is meant for easy access to all data files and results for a given project. 

![](quickstart/first-start.png)

In the `Workspaces` tab, click on  `+ Create Workspace` button. A dialogue will open asking for the name of the workspace and (optional) a brief description. Type `DEMO` into the text field, add a brief description `This is a DEMO` and click on `Create`.

![Create workspace](quickstart/create-workspace.png)

You can see which workspace is activated by looking at the blue toggle left side of the workspace name. By clicking in the `plus` sign, you can see in which folder is the workspace located as well as some stats pertaining how many samples were analyzed, how many compounds were included in the analysis, etc.

![Worspace active](quickstart/workspace-activated.png)

Now you have created your first workspace, but it is empty. We will need some input files to populate it.

## 3. Download the demo files

Some demo files are available for download on the `ms-mint` Google-Drive. Go on and download the files from [Google Drive](https://drive.google.com/drive/folders/1U4xMy5lfETk93sSVXPI79cCWyIMcAjeZ?usp=drive_link) and extract the archive.

You will find two `csv` files and 12 `mzXML` files.

```
.
├── README.md
├── metadata
│   └── metadata.csv
├── ms-files
│   ├── CA_B1.mzXML
│   ├── CA_B2.mzXML
│   ├── CA_B3.mzXML
│   ├── CA_B4.mzXML
│   ├── EC_B1.mzXML
│   ├── EC_B2.mzXML
│   ├── EC_B3.mzXML
│   ├── EC_B4.mzXML
│   ├── SA_B1.mzML
│   ├── SA_B2.mzML
│   ├── SA_B3.mzML
│   └── SA_B4.mzML
└── targets
    └── targets.csv

4 directories, 15 files
```

- A folder with 12 mass-spectrometry (MS) files from microbial samples. We have four files for each _Staphylococcus aureus_ (SA), _Escherichia coli_ (EC), and _Candida albicans_ (CA).
Each file belongs to one of four batches (B1-B4).
- `metadata.csv` contains this information in tabular format. Submit the metadata is optional, but highly recomended as will allow to make teh analysis more streamlined.
- `targets.csv` contains the extraction lists. The identification of the metabolites has been done before, so we know where the metabolites appear in the MS data.

## 4. Upload LCMS files 

Switch to `MS-Files` tab and upload the 12 MS files. Click `LOAD MS-FILES` on the top left, navigate to the folder where the files are located, select either the files individually or the folder, and click `Process Files`.

![](quickstart/ms-files-uploaded-1.png)

![](quickstart/ms-files-uploaded-2.png)

At this point you can proceed with the rest of steps without providing any metadata, however we strongly recomend using metadata to streamline downstream analyses.

## 4.1. Add metadata (Optional, but highly recommended)

In the same way as before, Click `LOAD METADATA` on the top left, navigate to the folder where the metadata file is located, select the file, and click `Process Files`. If colors are not provided, automatic ones will be assigned according to sample type.

![](quickstart/metadata-added.png)

This file contains important information about your samples.

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


## 5. Add targets (metabolites)
Switch to `Targets` and upload `MINT-targets.csv`.

![](quickstart/targets-table.png)

This is the data extraction protocol. This determines what data is extracted from the files. The same protocol is applied to all files. No fitting or peak optimization is done.

This file contains important information about the targets.

| Column Name             | Description                                                          |
|-------------------------|----------------------------------------------------------------------|
|`peak_label`             | Unique metabolite/feature name                                       |
|`peak_selection`         | True if selected for analysis                                        |
|`bookmark`               | True if bookmarked                                                   |
|`mz_mean`                | Mean m/z (centroid)                                                  |
|`mz_width`               | m/z window or tolerance                                              |
|`mz`                     | Precursor m/z (MS2)                                                  |
|`rt`                     | Retention time (default: in seconds)                                 |
|`rt_min`                 | Lower RT bound (default: in seconds)                                 |
|`rt_max`                 | Upper RT bound (default: in seconds)                                 |
|`rt_unit`                | RT unit (e.g. s or min; default: in seconds)                         |
|`intensity_threshold`    | Intensity cutoff (anything lower than this value is considered zero) |
|`polarity`               | Polarity (Positive or Negative)                                      |
|`filterLine`             | Filter ID for MS2 scans                                              |
|`ms_type`                | ms1 or ms2                                                           |
|`category`               | Category                                                             |
|`score`                  | Score                                                                |
|`notes`                  | Free-form notes                                                      |
|`source`                 | Data source or file                                                  |


## 6. Optimize retention times (Optional, but highly recommended)
Switch to the `Optimization` tab. Traditionally, and especially for large datasets, you select a representative set of samples including standards (with known concentrations of the target metabolites) to perform the optimization. However in MINT, you can perform the optimization with all the samples in most cases (see the files selected for optimization in the tree on the left side).

![](quickstart/peak-optimization-1.png)

The peak optimization takes longer the more files are used for it and the more targets are defined. Click on `COMPUTE CHROMATOGRAMS`. Here you can select how much resources you want to allocate to process the files, including CPU, RAM and batch size. In small datasets the defult values should suffice, as the number of files used for optimization grow, tweaking these parameters will guarantee better performance. Click `Generate` to compute the chromatograms.

![](quickstart/peak-optimization-2.png)
![](quickstart/peak-optimization-3.png)

This will show you the shapes of the data in the selected regions as an overview. This is a great way to validate that your target parameters are correct. 
However, you have to make sure that the metabolite you are looking for is present in the files. That is why you should always add some standard samples (samples with the metabolite of interest at different concentrations). The colors in the plots correspond to the sample type colors in the metadata table.

You can click on a card to use the interactive tool below and optimize the region of interest (ROI) or retention time span for each target manually. You can do that by moving the borders of the box towards the area that you want to select as peak and then click on `Save`. The green area is what is currently selected as retention time (RT) range. If the target is not present in any of the files, you can remove the target from the target list by clicking on `Delete target`.

![](quickstart/peak-optimization-4.png)

Once the optimization is done, you can proceed to `Processing`.

## 7. Process the data

Switch to `Processing` and start the data extraction with `RUN MINT`. In the same way that was done before for Optimization, here you can select how much resources you want to allocate to process the files, including CPU, RAM and batch size. In small datasets the defult values should suffice, as the number of files used for optimization grow, tweaking these parameters will guarantee better performance. Click `Run` to compute the results.

![](quickstart/run-mint-1.png)
![](quickstart/run-mint-2.png)


Now, you can download the results in long-format or the dense peak_max values by clicking on `DOWNLOAD RESULTS`. The tidy format contains all results, while the dense format only contains the a selected metric (`peak_max` as default) as a matrix values.

![](quickstart/run-mint-3.png)

## 8. Analyze the results.

Once the results are generated the 'Heatmap` tab will show an interactive heatmap.
You can change the size of the heatmap by changing your browser window and `UPDATE` the plot.
The heatmap shows the `peak_max` values. The dropdown menu provides some options.

## 9. Switch to `Analysis/Plotting`

The plotting tool is very powerful, but requires some practise. It is a wrapper of the powerful seaborn API. 
Let's create a few simple visualizations.

![](quickstart/01-demo-plot.png)


And click on `Update`. A very simple bar-graph is shown, and we will gradually make it more complex. 
This simple bar graph shows the average `peak_max` value across the whole dataset for all targets. 

### a) select `peak_label` for the `X` axis.
### b) set aspect-ratio to 5.
### c) select `Logarithmic y-scale` in the dropdown options.
### d) click on `UPDATE`.

![](quickstart/02-demo-plot.png)

### e) set figure height to `1.5` and aspect ratio to `2`.
### e) set `Column` to `Label`.
### f) set `Row` to `Batch`.

![](quickstart/03-demo-plot.png)

This way you can look at the whole dataset at once, sliced by `Batch` and `Label`

## Exercise: Try to create the following plot:

![](quickstart/05-demo-plot.png)

[Read more](gui.md)
