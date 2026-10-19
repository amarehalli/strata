"""
strata.io.loaders
-----------------
Load Xenium (h5ad) and Visium (h5 + tissue_positions.csv) into PointSet objects.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PointSet:
    """A 2-D point cloud with optional per-point features (gene expression).

    Attributes
    ----------
    coords : np.ndarray, shape (n, 2)
        Spatial coordinates (x, y) in microns or pixel space.
    ids : np.ndarray, shape (n,)
        Barcode or cell-id strings.
    feature_names : list[str]
        Gene names corresponding to the feature columns.
    features : np.ndarray, shape (n, g), optional
        Normalised expression matrix (log1p CPM recommended).
    role : str
        "moving" (Xenium, will be aligned) or "fixed" (Visium, reference).
    meta : dict
        Arbitrary metadata (sample name, platform, etc.).
    """
    coords       : np.ndarray
    ids          : np.ndarray
    feature_names: list
    features     : Optional[np.ndarray] = field(default=None, repr=False)
    role         : str = "moving"
    meta         : dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.coords)

    def feature(self, gene: str) -> np.ndarray:
        """Return expression vector for one gene."""
        idx = self.feature_names.index(gene)
        return self.features[:, idx]

    def shared_features(self, other: "PointSet") -> list:
        """Return genes present in both PointSets."""
        return sorted(set(self.feature_names) & set(other.feature_names))

    def subset_by_ids(self, ids) -> "PointSet":
        """Return a new PointSet containing only the supplied barcode IDs."""
        id_set = set(ids)
        mask   = np.array([i in id_set for i in self.ids])
        feat   = self.features[mask] if self.features is not None else None
        return PointSet(
            coords        = self.coords[mask],
            ids           = self.ids[mask],
            feature_names = self.feature_names,
            features      = feat,
            role          = self.role,
            meta          = self.meta.copy(),
        )

    def subsample(self, n: int, seed: int = 0) -> "PointSet":
        """Return a random subsample of n points."""
        rng = np.random.default_rng(seed)
        idx = rng.choice(self.n, min(n, self.n), replace=False)
        feat = self.features[idx] if self.features is not None else None
        return PointSet(
            coords        = self.coords[idx],
            ids           = self.ids[idx],
            feature_names = self.feature_names,
            features      = feat,
            role          = self.role,
            meta          = self.meta.copy(),
        )

    def __repr__(self):
        return (f"PointSet(n={self.n}, genes={len(self.feature_names)}, "
                f"role={self.role!r})")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_xenium(path: str, role: str = "moving",
                n_genes: int = 500, layer: str = "counts") -> PointSet:
    """Load a Xenium dataset from an AnnData (.h5ad) file.

    The AnnData object is expected to have:
    - ``obs['x_centroid']`` / ``obs['y_centroid']``  — cell coordinates
    - ``X`` or a layer named *layer*  — count matrix
    - ``var_names``  — gene names

    Parameters
    ----------
    path : str
        Path to the ``.h5ad`` file.
    role : str
        "moving" (default) — Xenium is the dataset that will be aligned.
    n_genes : int
        Keep only the top *n_genes* most-variable genes for alignment
        (speeds up OT; all genes are retained in ``feature_names``).
    layer : str
        AnnData layer to use for expression (default: raw ``X``).

    Returns
    -------
    PointSet
    """
    import scanpy as sc
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata = sc.read_h5ad(path)

    # Coordinates
    if "x_centroid" in adata.obs.columns:
        x = adata.obs["x_centroid"].values.astype(np.float32)
        y = adata.obs["y_centroid"].values.astype(np.float32)
    elif "spatial" in adata.obsm:
        x, y = adata.obsm["spatial"][:, 0], adata.obsm["spatial"][:, 1]
    else:
        raise ValueError("Cannot find spatial coordinates in AnnData. "
                         "Expected obs['x_centroid'] or obsm['spatial'].")

    coords = np.column_stack([x, y])

    # Expression
    mat = adata.layers[layer] if layer in adata.layers else adata.X
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    mat = mat.astype(np.float32)

    # Log-normalise
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    mat = np.log1p(mat / row_sums * 1e4)

    genes = list(adata.var_names)

    return PointSet(
        coords        = coords,
        ids           = np.array(adata.obs_names),
        feature_names = genes,
        features      = mat,
        role          = role,
        meta          = {"path": path, "platform": "xenium"},
    )


def load_visium(matrix_path: str, positions_path: str,
                role: str = "fixed") -> PointSet:
    """Load a Visium dataset from a filtered_feature_bc_matrix.h5 and
    tissue_positions.csv file.

    Parameters
    ----------
    matrix_path : str
        Path to ``filtered_feature_bc_matrix.h5``.
    positions_path : str
        Path to ``tissue_positions.csv`` (or ``tissue_positions_list.csv``).
    role : str
        "fixed" (default) — Visium is the reference dataset.

    Returns
    -------
    PointSet
    """
    import scanpy as sc
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata = sc.read_10x_h5(matrix_path)

    # Tissue positions
    pos = pd.read_csv(positions_path)
    # Normalise column names across Visium versions
    rename = {
        "barcode": "barcode",
        "in_tissue": "in_tissue",
        "array_row": "array_row",
        "array_col": "array_col",
        "pxl_col_in_fullres": "x",
        "pxl_row_in_fullres": "y",
    }
    if pos.columns[0] != "barcode":          # old format: no header
        pos.columns = list(rename.keys())
    pos = pos.rename(columns=rename)
    pos = pos[pos["in_tissue"] == 1].set_index("barcode")

    common = adata.obs_names.intersection(pos.index)
    adata  = adata[common]
    pos    = pos.loc[common]

    coords = pos[["x", "y"]].values.astype(np.float32)

    mat = adata.X
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    mat = mat.astype(np.float32)

    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    mat = np.log1p(mat / row_sums * 1e4)

    return PointSet(
        coords        = coords,
        ids           = np.array(common),
        feature_names = list(adata.var_names),
        features      = mat,
        role          = role,
        meta          = {"matrix_path": matrix_path,
                         "positions_path": positions_path,
                         "platform": "visium"},
    )
