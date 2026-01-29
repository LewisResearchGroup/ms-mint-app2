<!-- [![Python package](https://github.com/LewisResearchGroup/ms-mint-app2/actions/workflows/pythonpackage.yml/badge.svg)](https://github.com/LewisResearchGroup/ms-mint-app2/actions/workflows/pythonpackage.yml)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/5178/badge)](https://bestpractices.coreinfrastructure.org/projects/5178)
![](images/coverage.svg)
[![Docker Image CI](https://github.com/LewisResearchGroup/ms-mint-app2/actions/workflows/docker-image.yml/badge.svg)](https://github.com/LewisResearchGroup/ms-mint-app2/actions/workflows/docker-image.yml)
![PyPI](https://img.shields.io/pypi/v/ms-mint-app2?label=pypi%20package)
![PyPI - Downloads](https://img.shields.io/pypi/dm/ms-mint-app2)
[![DOI](https://zenodo.org/badge/491654035.svg)](https://zenodo.org/doi/10.5281/zenodo.13121148) -->

<p align="center">
  <img src="docs/image/MINT-logo.jpg" alt="MINT Logo" width="400">
</p>

# MINT (Metabolomics Integrator)

A powerful post-processing tool for **LC-MS based metabolomics** that simplifies peak integration, quality control, and data analysis.

## Key Features

- **Targeted Peak Integration** - Extract chromatograms and quantify peaks from mzML/mzXML files
- **Interactive Visualization** - Explore chromatograms, heatmaps, and clustering results
- **RT Optimization** - Fine-tune retention time windows with visual feedback
- **Optional Quantification (SCALiR)** - Available in the Processing tab for absolute quantification when needed
- **DuckDB Backend** - Fast, efficient storage for large datasets
- **Desktop App** - Available as standalone Windows, Linux, and macOS executable

<p align="center">
  <img src="docs/quickstart/peak-preview.png" alt="Hierarchical Clustering" width="700">
</p>

## Quick Start

### Installation (pip)

```bash
pip install ms-mint-app2
```

Requires Python 3.12+.

### Run MINT

```bash
Mint
```

For detailed installation instructions, see the [Installation Guide](https://lewisresearchgroup.github.io/ms-mint-app2/install/).

## Documentation

- **[Full Documentation](https://LewisResearchGroup.github.io/ms-mint-app2/)** - Complete user guide
- **[Quick Start Tutorial](https://LewisResearchGroup.github.io/ms-mint-app2/quickstart/)** - Get up and running in 5 minutes

## Publications Using MINT

1. Brown K, et al. [Microbiota alters the metabolome in an age- and sex-dependent manner in mice.](https://pubmed.ncbi.nlm.nih.gov/36906623/) *Nat Commun.* 2023;14: 1348.

2. Ponce LF, et al. [SCALiR: A Web Application for Automating Absolute Quantification of Mass Spectrometry-Based Metabolomics Data.](https://pubs.acs.org/doi/10.1021/acs.analchem.3c04988) *Anal Chem.* 2024;96: 6566–6574.

## Contributing

All contributions are welcome! This includes:
- Bug reports and fixes
- Documentation improvements
- Feature requests and enhancements
- Code reviews

Please open a [GitHub issue](https://github.com/LewisResearchGroup/ms-mint-app2/issues) to get started.

## Acknowledgements

This project builds on the amazing open-source community:

Special thanks to [GitHub](https://github.com),[PyPI](https://pypi.org/), and the [Plotly Community](https://community.plotly.com/) for their invaluable resources.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---
