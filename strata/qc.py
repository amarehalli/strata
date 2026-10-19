"""
strata.qc — alignment quality control
---------------------------------------
"""

from __future__ import annotations
import numpy as np
from .io.loaders import PointSet
from .transform import compose
from .refine import _concordance


def qc_alignment(xen: PointSet,
                 vis: PointSet,
                 transform_dicts: list[dict] | None = None,
                 k_neighbors: int = 5) -> dict:
    """Compute alignment quality metrics without ground truth.

    Parameters
    ----------
    xen, vis : PointSet
        Point clouds (xen should already be in Visium coordinate space if
        transform_dicts is None).
    transform_dicts : list[dict], optional
        Sequence of transform dicts (from Transform.to_dict()) to apply to
        xen.coords before scoring.  If None, coords are used as-is.
    k_neighbors : int
        kNN count for expression concordance aggregation.

    Returns
    -------
    dict with keys:
        concordance  float  mean per-gene Spearman (higher = better)
        coverage     float  fraction of Visium spots that have ≥1 Xenium
                            neighbour within 2× the median Visium spot spacing
        n_xen        int
        n_vis        int
    """
    from sklearn.preprocessing import normalize
    from .transform import Transform

    if transform_dicts:
        transforms = [Transform.from_dict(d) for d in transform_dicts]
        coords = compose(xen.coords, transforms)
    else:
        coords = xen.coords

    # Expression
    shared  = xen.shared_features(vis)
    xen_idx = [xen.feature_names.index(g) for g in shared]
    vis_idx = [vis.feature_names.index(g) for g in shared]
    xen_expr = normalize(xen.features[:, xen_idx], norm="l2")
    vis_expr = normalize(vis.features[:, vis_idx], norm="l2")

    # Concordance
    concordance = _concordance(coords, vis.coords, xen_expr, vis_expr, k=k_neighbors)

    # Coverage
    from sklearn.neighbors import NearestNeighbors
    spacing = np.median(np.diff(np.sort(vis.coords[:, 0])))
    nn = NearestNeighbors(n_neighbors=1).fit(coords)
    dists, _ = nn.kneighbors(vis.coords)
    coverage = float((dists[:, 0] < 2 * spacing).mean())

    return {
        "concordance": concordance,
        "coverage":    coverage,
        "n_xen":       xen.n,
        "n_vis":       vis.n,
    }
