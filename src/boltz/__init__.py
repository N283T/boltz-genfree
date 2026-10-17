from importlib.metadata import PackageNotFoundError, version

__version__ = "0+unknown"
for _distribution in ("boltz-genfree", "boltz-stfree", "boltz-community", "boltz"):
    try:
        __version__ = version(_distribution)
        break
    except PackageNotFoundError:
        continue
