<!-- [![Python package](https://github.com/LewisResearchGroup/ms-mint-app/actions/workflows/pythonpackage.yml/badge.svg)](https://github.com/LewisResearchGroup/ms-mint-app/actions/workflows/pythonpackage.yml)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/5178/badge)](https://bestpractices.coreinfrastructure.org/projects/5178)
![](images/coverage.svg)
[![Docker Image CI](https://github.com/LewisResearchGroup/ms-mint-app/actions/workflows/docker-image.yml/badge.svg)](https://github.com/LewisResearchGroup/ms-mint-app/actions/workflows/docker-image.yml)
![PyPI](https://img.shields.io/pypi/v/ms-mint-app?label=pypi%20package)
![PyPI - Downloads](https://img.shields.io/pypi/dm/ms-mint-app)
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
- **Statistical Analysis** - Built-in tools including SCALiR for absolute quantification
- **DuckDB Backend** - Fast, efficient storage for large datasets
- **Desktop App** - Available as standalone Windows/Unix executable

<p align="center">
  <img src="docs/quickstart/peak-preview.png" alt="Hierarchical Clustering" width="700">
</p>

## Quick Start

### Installation (pip)

```bash
pip install ms-mint-app2
```

### Run MINT

```bash
Mint
```

For detailed installation instructions (conda, Docker, standalone builds), see the [Installation Guide](https://lewisresearchgroup.github.io/ms-mint-app/install/).

## Documentation

- **[Full Documentation](https://LewisResearchGroup.github.io/ms-mint-app/)** - Complete user guide
- **[Quick Start Tutorial](https://LewisResearchGroup.github.io/ms-mint-app/quickstart/)** - Get up and running in 5 minutes

## What's New (v1.x)

- **Modern Python packaging** - Now uses `pyproject.toml`
- **Simplified launch** - Run with `Mint` command (previously `Mint.py`)
- **DOI for citations** - Each release has a citable DOI
- **Smart RT derivation** - Automatically calculates missing RT parameters
- **Auto-save in optimization** - Changes saved automatically when navigating targets

## Publications Using MINT

1. Brown K, et al. [Microbiota alters the metabolome in an age- and sex-dependent manner in mice.](https://pubmed.ncbi.nlm.nih.gov/36906623/) *Nat Commun.* 2023;14: 1348.

2. Ponce LF, et al. [SCALiR: A Web Application for Automating Absolute Quantification of Mass Spectrometry-Based Metabolomics Data.](https://pubs.acs.org/doi/10.1021/acs.analchem.3c06988) *Anal Chem.* 2024;96: 6566–6574.

## Contributing

All contributions are welcome! This includes:
- Bug reports and fixes
- Documentation improvements
- Feature requests and enhancements
- Code reviews

Please open a [GitHub issue](https://github.com/LewisResearchGroup/ms-mint-app/issues) to get started.

### Code Standards

- Follows **PEP8** style guide
- Formatted with **Black**
- Linted with **Flake8**

## Acknowledgements

This project builds on the amazing open-source community:

- **@rokm** - Refactored PyInstaller specfile for Windows packaging
- **@bucknerns** - Helped configure versioneer

Special thanks to [GitHub](https://github.com), [Docker Hub](https://hub.docker.com/), [PyPI](https://pypi.org/), [Stack Overflow](https://stackoverflow.com), and the [Plotly Community](https://community.plotly.com/) for their invaluable resources.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---