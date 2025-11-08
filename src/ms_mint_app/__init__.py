from ._version import get_versions
__version__ = get_versions().get("version", "0+unknown")
del get_versions
from . import _version
__version__ = _version.get_versions()['version']
