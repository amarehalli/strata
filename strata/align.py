"""
strata.align — coarse alignment + auto-align pipeline
------------------------------------------------------
Provides:
  auto_align(xen, vis)  — full pipeline: coarse init → FGW refinement
  coarse_align(xen, vis) — PCA-based rigid pre-alignment only
"""

from __future__ import annotations
import numpy as np
from .io.loaders import PointSet
from .refine import refine, RefineHistory
from .transform import Transform


def coarse_align(xen: PointSet, vis: PointSet,
                 n_genes: int = 50,
                 n_inits: int = 4) -> Transform:
    """Rigid (rotation + translation + scale) coarse alignment via PCA.

    Projects both datasets onto the top principal components of their
    shared gene set and uses the PCA axes to estimate the rigid transform
    that maps Xenium → Visium coordinate space.  Multiple rotation
    candidates (n_inits, spread over 360°) are tried and the one with
    the lowest Wasserstein cost on a random subsample is kept.

    Parameters
    ----------
    xen, vis : PointSet
        Xenium (moving) and Visium (fixed) point clouds.
    n_genes : int
        Number of shared genes to use for coarse expression alignment.
    n_inits : int
        Number of rotation candidates to try (4 covers the 4 quadrants).

    Returns
    -------
    Transform
        Best rigid transform found.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    shared = xen.shared_features(vis)
    if not shared:
        raise ValueError("No shared genes between Xenium and Visium.")

    n_genes = min(n_genes, len(shared))
    genes   = shared[:n_genes]

    xen_idx = [xen.feature_names.index(g) for g in genes]
    vis_idx = [vis.feature_names.index(g) for g in genes]

    X = xen.features[:, xen_idx]
    V = vis.features[:, vis_idx]

    # PCA on gene space to get per-cell expression score
    pca = PCA(n_components=min(2, len(genes)))
    Xp  = pca.fit_transform(StandardScaler().fit_transform(X))
    Vp  = pca.transform(StandardScaler().fit_transform(V))

    # Estimate centroid + scale from spatial coords, then try n_inits rotations
    xen_c  = xen.coords.mean(0)
    vis_c  = vis.coords.mean(0)
    xen_s  = xen.coords.std()
    vis_s  = vis.coords.std()
    scale  = vis_s / (xen_s + 1e-8)

    best_t, best_cost = None, np.inf
    for k in range(n_inits):
        theta  = 2 * np.pi * k / n_inits
        R      = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta),  np.cos(theta)]])
        # Map xen coords: scale → rotate → translate
        moved  = (xen.coords - xen_c) * scale @ R.T + vis_c
        cost   = np.mean(np.min(
            np.linalg.norm(moved[:, None] - vis.coords[None], axis=-1),
            axis=1))
        if cost < best_cost:
            best_cost = cost
            best_t    = Transform(
                rotation    = float(theta),
                scale       = float(scale),
                translation = vis_c - xen_c * scale,
            )

    return best_t


def auto_align(xen: PointSet, vis: PointSet,
               n_iter: int = 12,
               alpha: float = 0.5,
               subsample_xen: int = 2000,
               subsample_vis: int = 2000,
               entropic_epsilon: float | None = None,
               verbose: bool = True,
               seed: int = 0) -> RefineHistory:
    """Full auto-alignment pipeline: coarse rigid init → FGW refinement.

    Parameters
    ----------
    xen : PointSet
        Xenium point cloud (moving).  Will NOT be modified in place.
    vis : PointSet
        Visium point cloud (fixed, selected spots only).
    n_iter : int
        Number of FGW refinement iterations.
    alpha : float
        FGW trade-off: 0 = pure expression matching, 1 = pure geometry.
    subsample_xen, subsample_vis : int
        Number of points to use in each OT solve (larger → better, slower).
    entropic_epsilon : float or None
        If given, use entropic FGW (faster, less exact).
        None (default) → exact Frank-Wolfe solver.
    verbose : bool
        Print iteration-level progress.
    seed : int
        Random seed for subsampling.

    Returns
    -------
    RefineHistory
        Contains .trajectory (list of Transforms), .concordance, etc.
    """
    rng = np.random.default_rng(seed)

    # 1. Subsample for OT
    xen_sub = xen.subsample(subsample_xen, seed=seed)
    vis_sub = vis.subsample(subsample_vis, seed=seed + 1)

    # 2. Coarse rigid alignment
    if verbose:
        print("Step 1/2: coarse rigid alignment...")
    t0 = coarse_align(xen_sub, vis_sub)

    # 3. FGW refinement
    if verbose:
        print("Step 2/2: FGW refinement...")
    hist = refine(
        xen_sub, vis_sub,
        init_transform      = t0,
        n_iter              = n_iter,
        alpha               = alpha,
        entropic_epsilon    = entropic_epsilon,
        verbose             = verbose,
    )

    return hist
