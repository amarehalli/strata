"""
STRATA — Spatial Transcriptomics Registration and Alignment Toolkit
"""

from .align   import auto_align, coarse_align
from .refine  import refine, RefineHistory
from .qc      import qc_alignment
from . import io

__version__ = "0.1.0"
__author__  = "Aditi Marehalli"


def _lazy(name):
    """Lazy-import a submodule only when first accessed."""
    import importlib
    return importlib.import_module(f"strata.{name}")


def launch_aligner(*args, **kwargs):
    return _lazy("launch").launch_aligner(*args, **kwargs)

def save_aligner(*args, **kwargs):
    return _lazy("launch").save_aligner(*args, **kwargs)

def launch_selector(*args, **kwargs):
    return _lazy("launch").launch_selector(*args, **kwargs)

def save_selector(*args, **kwargs):
    return _lazy("launch").save_selector(*args, **kwargs)


class Project:
    def __new__(cls, *args, **kwargs):
        return _lazy("project").Project(*args, **kwargs)


__all__ = [
    "auto_align", "coarse_align",
    "refine", "RefineHistory",
    "launch_aligner", "save_aligner",
    "launch_selector", "save_selector",
    "Project", "qc_alignment", "io",
]
