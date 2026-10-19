"""
STRATA — Spatial Transcriptomics Registration and Alignment Toolkit
====================================================================

A Python package for co-registering Xenium (single-cell resolution) and
Visium (transcriptome-depth) spatial transcriptomics datasets using
Fused Gromov-Wasserstein optimal transport.

Quickstart
----------
>>> import strata as st
>>> xen = st.io.load_xenium("path/to/spatial_annotated.h5ad")
>>> vis = st.io.load_visium("path/to/matrix.h5", "path/to/tissue_positions.csv")
>>> hist = st.auto_align(xen, vis)        # run FGW alignment
>>> srv  = st.save_aligner(xen, vis,      # bake animation into HTML
...           path="alignment.html", trajectory=hist)
>>> srv   # open in browser — watch animation, refine manually, Send
"""

from .align   import auto_align, coarse_align
from .refine  import refine, RefineHistory
from .launch  import (
    launch_aligner, save_aligner,
    launch_selector, save_selector,
)
from .project import Project
from .qc      import qc_alignment
from . import io

__version__ = "0.1.0"
__author__  = "Aditi Marehalli"

__all__ = [
    "auto_align",
    "coarse_align",
    "refine",
    "RefineHistory",
    "launch_aligner",
    "save_aligner",
    "launch_selector",
    "save_selector",
    "Project",
    "qc_alignment",
    "io",
]
