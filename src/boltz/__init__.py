from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("boltz-stfree")
except PackageNotFoundError:
    try:
        __version__ = version("boltz-community")
    except PackageNotFoundError:
        try:
            __version__ = version("boltz")
        except PackageNotFoundError:
            __version__ = "0+unknown"
