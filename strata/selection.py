"""
strata.selection — Visium spot selection helper
-----------------------------------------------
Auto-proposes which Visium spots correspond to the Xenium tissue patch.
"""

from __future__ import annotations
import numpy as np
from .io.loaders import PointSet


def propose_selection(vis: PointSet,
                      xen: PointSet,
                      quantile: float = 0.5,
                      k_spatial: int = 6,
                      min_component_frac: float = 0.1) -> list[str]:
    """Score each Visium spot by expression similarity to Xenium and return
    the selected barcode IDs.

    Algorithm
    ---------
    1. Compute the mean Xenium expression over shared genes (a "signature").
    2. Score each Visium spot by cosine similarity to the signature.
    3. Smooth scores over a spatial k-NN graph.
    4. Threshold at *quantile* and keep the largest connected component.

    Parameters
    ----------
    vis : PointSet
        Full Visium slide (all in-tissue spots).
    xen : PointSet
        Xenium cells from the same sample.
    quantile : float
        Spots above this quantile of the smoothed score are candidates.
        Lower → select more spots. Default 0.5.
    k_spatial : int
        Number of spatial neighbours for score smoothing.
    min_component_frac : float
        Drop connected components smaller than this fraction of the largest.

    Returns
    -------
    list[str]
        Selected Visium barcode IDs.
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix

    shared = vis.shared_features(xen)
    if not shared:
        raise ValueError("No shared genes between Visium and Xenium.")

    vis_idx = [vis.feature_names.index(g) for g in shared]
    xen_idx = [xen.feature_names.index(g) for g in shared]

    vis_expr = vis.features[:, vis_idx]
    xen_expr = xen.features[:, xen_idx]

    # 1. Xenium mean expression signature
    sig = xen_expr.mean(axis=0)
    sig_norm = np.linalg.norm(sig) + 1e-8

    # 2. Cosine similarity
    norms = np.linalg.norm(vis_expr, axis=1) + 1e-8
    scores = (vis_expr @ sig) / (norms * sig_norm)

    # 3. Spatial smoothing
    nn = NearestNeighbors(n_neighbors=k_spatial + 1).fit(vis.coords)
    _, nbrs = nn.kneighbors(vis.coords)
    for i in range(len(scores)):
        scores[i] = scores[nbrs[i]].mean()

    # 4. Threshold + connected components
    thresh   = np.quantile(scores, quantile)
    selected = scores >= thresh
    sel_idx  = np.where(selected)[0]

    if len(sel_idx) == 0:
        return list(vis.ids)

    # Build adjacency for selected spots
    coords_sel = vis.coords[sel_idx]
    nn2 = NearestNeighbors(n_neighbors=min(k_spatial, len(sel_idx))).fit(coords_sel)
    _, nbrs2 = nn2.kneighbors(coords_sel)
    rows, cols = [], []
    for i, ns in enumerate(nbrs2):
        for j in ns:
            rows.append(i); cols.append(j)
    A = csr_matrix((np.ones(len(rows)), (rows, cols)),
                   shape=(len(sel_idx), len(sel_idx)))
    n_comp, labels = connected_components(A, directed=False)

    # Keep largest component(s)
    comp_sizes = np.bincount(labels)
    max_size   = comp_sizes.max()
    keep_comps = set(np.where(comp_sizes >= max_size * min_component_frac)[0])
    final_mask = np.array([labels[i] in keep_comps for i in range(len(sel_idx))])
    final_idx  = sel_idx[final_mask]

    return list(vis.ids[final_idx])
