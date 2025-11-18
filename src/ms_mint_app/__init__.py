from ._version import get_versions
__version__ = get_versions().get("version", "0+unknown")
del get_versions