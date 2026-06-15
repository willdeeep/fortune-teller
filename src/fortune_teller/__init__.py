"""Fortune Teller — local-first Tarot reading app."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fortune-teller")
except PackageNotFoundError:  # not installed (e.g. a raw source checkout)
    __version__ = "0.0.0+unknown"
