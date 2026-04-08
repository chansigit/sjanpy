"""
sjanpy - A collection of Python utilities for single-cell analysis visualization
"""

__version__ = "0.0.2a"

from . import pl
from . import tl
from . import pp
from . import ml


# Backward-compatible lazy imports for old flat API
# e.g. `from sjanpy import nebulosa` still works
def __getattr__(name):
    _compat = {
        "nebulosa": "pl.nebulosa",
        "pynebulosa_2d": "pl.nebulosa",
        "pynebulosa_3d": "pl.nebulosa",
        "embedding": "pl.embedding",
        "dotplot": "pl.dotplot",
        "barplot": "pl.barplot",
        "deg": "tl.deg",
        "pres": "tl.pres",
        "genecraft": "pp.genecraft",
    }
    if name in _compat:
        import importlib
        parts = _compat[name].split(".")
        return importlib.import_module(f".{parts[0]}.{parts[1]}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
