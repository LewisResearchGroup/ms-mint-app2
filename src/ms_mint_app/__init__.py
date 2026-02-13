import os

from ._version import get_versions


def _resolve_version() -> str:
    version = get_versions().get("version", "0+unknown")
    if version != "0+unknown":
        return version

    # In frozen builds we may not have git metadata; use package metadata.
    try:
        from importlib.metadata import PackageNotFoundError, version as pkg_version
    except Exception:
        try:
            from importlib_metadata import PackageNotFoundError, version as pkg_version  # type: ignore
        except Exception:
            pkg_version = None
            PackageNotFoundError = Exception  # type: ignore

    if pkg_version is not None:
        for package_name in ("ms-mint-app2", "ms_mint_app2", "ms-mint-app"):
            try:
                resolved = pkg_version(package_name)
                if resolved:
                    return resolved
            except PackageNotFoundError:
                continue
            except Exception:
                continue

    # Optional final fallback for custom runtime branding.
    return os.environ.get("MINT_BUILD_ID", version)


__version__ = _resolve_version()

del get_versions
