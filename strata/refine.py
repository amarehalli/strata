"""
strata.refine — Fused Gromov-Wasserstein refinement
----------------------------------------------------
Iteratively refines a coarse rigid transform using FGW optimal transport.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .io.loaders import PointSet
from .transform import Transform, compose


@dataclass
class RefineHistory:
    """Output of a refinement run.

    Attributes
    ----------
    trajectory : list[Transform]
        One Transform per iteration, from first to last.
    concordance : float
        Final per-gene Spearman concordance (mean across shared genes).
    mean_displacement : float
        Mean point displacement (pixels) on the last iteration.
    alpha_schedule : list[float]
        Alpha values used at each iteration.
    """
    trajectory        : list = field(default_factory=list)
    concordance       : float = 0.0
    mean_displacement : float = 0.0
    alpha_schedule    : list  = field(default_factory=list)


def refine(xen: PointSet,
           vis: PointSet,
           init_transform: Transform | None = None,
           n_iter: int = 12,
           alpha: float = 0.5,
           entropic_epsilon: float | None = None,
           verbose: bool = True) -> RefineHistory:
    """FGW-based iterative refinement.

    At each iteration:
    1. Compute pairwise cost matrices (spatial + expression).
    2. Solve FGW OT to get transport plan T.
    3. Derive the best rigid transform from T (Procrustes).
    4. Apply the transform to the Xenium point cloud.

    Parameters
    ----------
    xen, vis : PointSet
        Subsampled point clouds (moving and fixed).
    init_transform : Transform, optional
        Starting point. If None, uses identity.
    n_iter : int
        Number of iterations.
    alpha : float
        FGW trade-off: 0 = expression only, 1 = geometry only.
    entropic_epsilon : float or None
        Entropic regularisation strength. None → exact Frank-Wolfe solver.
    verbose : bool

    Returns
    -------
    RefineHistory
    """
    import ot
    from scipy.stats import spearmanr
    from sklearn.preprocessing import normalize

    hist = RefineHistory()

    # --- shared genes ---
    shared  = xen.shared_features(vis)
    if not shared:
        raise ValueError("No shared genes for FGW alignment.")
    xen_idx = [xen.feature_names.index(g) for g in shared]
    vis_idx = [vis.feature_names.index(g) for g in shared]

    xen_expr = normalize(xen.features[:, xen_idx], norm="l2")
    vis_expr = normalize(vis.features[:, vis_idx], norm="l2")

    # --- uniform marginals ---
    p = np.ones(xen.n) / xen.n
    q = np.ones(vis.n) / vis.n

    # --- apply init transform ---
    current_coords = (init_transform.apply(xen.coords)
                      if init_transform else xen.coords.copy())

    alpha_schedule = np.linspace(alpha, alpha, n_iter).tolist()
    hist.alpha_schedule = alpha_schedule

    for i in range(n_iter):
        a = alpha_schedule[i]

        # Spatial cost (Euclidean, row-normalised)
        C_xen = _pairwise_sq_dist_norm(current_coords)
        C_vis = _pairwise_sq_dist_norm(vis.coords)

        # Expression cost (cosine distance)
        C_expr = 1.0 - xen_expr @ vis_expr.T     # (n_xen, n_vis)
        C_expr = np.clip(C_expr, 0, 2)

        if entropic_epsilon is not None:
            T, _ = ot.gromov.entropic_fused_gromov_wasserstein(
                C_expr, C_xen, C_vis, p, q,
                epsilon=entropic_epsilon, alpha=a,
                log=True, verbose=False)
        else:
            T, _ = ot.gromov.fused_gromov_wasserstein(
                C_expr, C_xen, C_vis, p, q,
                alpha=a, log=True, verbose=False)

        # Derive new Xenium coords: barycentric projection
        new_coords = (T / T.sum(axis=1, keepdims=True)) @ vis.coords
        displacement = float(np.mean(np.linalg.norm(new_coords - current_coords, axis=1)))

        # Fit rigid transform (Procrustes)
        t = _fit_rigid(current_coords, new_coords, T)
        current_coords = t.apply(current_coords)

        hist.trajectory.append(t)
        hist.mean_displacement = displacement

        if verbose:
            print(f"  iter {i+1}/{n_iter}  displacement={displacement:.2f}")

    # Final concordance
    hist.concordance = _concordance(current_coords, vis.coords,
                                    xen_expr, vis_expr)
    return hist


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pairwise_sq_dist_norm(coords: np.ndarray) -> np.ndarray:
    """Normalised pairwise squared-distance matrix."""
    diff = coords[:, None] - coords[None]
    D    = (diff ** 2).sum(-1)
    dmax = D.max() + 1e-8
    return D / dmax


def _fit_rigid(src: np.ndarray,
               dst: np.ndarray,
               weights: np.ndarray) -> Transform:
    """Weighted Procrustes: find best rigid transform src → dst."""
    w = weights.sum(axis=1)
    w /= w.sum()
    src_c = (w[:, None] * src).sum(0)
    dst_c = (weights.sum(axis=0)[:, None] * dst).sum(0) / weights.sum()

    Hs  = src - src_c
    Hd  = (weights / weights.sum()) @ dst - dst_c
    # SVD
    U, _, Vt = np.linalg.svd(Hs.T @ Hd)
    R  = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    theta = float(np.arctan2(R[1, 0], R[0, 0]))
    scale = 1.0
    t     = dst_c - src_c @ R.T
    return Transform(rotation=theta, scale=scale, translation=t)


def _concordance(xen_c, vis_c, xen_expr, vis_expr,
                 k: int = 5) -> float:
    """Mean per-gene Spearman between aggregated Xenium and Visium expression."""
    from scipy.stats import spearmanr
    # kNN aggregate: for each Visium spot, find k nearest Xenium cells
    diff  = vis_c[:, None] - xen_c[None]            # (nv, nx, 2)
    dists = np.linalg.norm(diff, axis=-1)            # (nv, nx)
    idx   = np.argsort(dists, axis=1)[:, :k]        # (nv, k)
    xen_agg = xen_expr[idx].mean(axis=1)            # (nv, g)
    corrs = []
    for g in range(xen_agg.shape[1]):
        r, _ = spearmanr(xen_agg[:, g], vis_expr[:, g])
        if np.isfinite(r):
            corrs.append(r)
    return float(np.mean(corrs)) if corrs else 0.0
