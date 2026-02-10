"""
sjanpy - A collection of Python utilities for single-cell analysis visualization
"""

__version__ = "0.1.0"

__all__ = [
    "deg",
    "dotplot",
    "embedding",
    "barplot",
    "nebulosa",
    "pres",
    "genecraft",
    "pynebulosa_2d",
    "pynebulosa_3d",
]


def __getattr__(name):
    """Lazy import modules to avoid loading all dependencies at once."""
    if name in __all__:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
