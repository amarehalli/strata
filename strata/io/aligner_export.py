"""
strata.io.aligner_export
------------------------
Build CSV payloads for the browser-based aligner/selector tools.
"""
from __future__ import annotations
import io as _io
import numpy as np
import pandas as pd
from .loaders import PointSet


def build_aligner_csvs(xen: PointSet, vis: PointSet,
                       n_fiducials: int = 6,
                       xen_cap: int = 60000,
                       seed: int = 0):
    """Return (xen_csv, vis_csv, rank_table, fiducial_names).

    xen_csv / vis_csv : CSV strings with columns barcode,x,y,<genes>
    rank_table        : DataFrame of fiducial gene scores
    fiducial_names    : list of top n_fiducials gene names
    """
    shared = xen.shared_features(vis)
    if not shared:
        raise ValueError("No shared genes between Xenium and Visium.")

    # Pick fiducial genes: highest mean expression in Xenium, present in Visium
    xen_idx = [xen.feature_names.index(g) for g in shared]
    means   = xen.features[:, xen_idx].mean(axis=0)
    order   = np.argsort(means)[::-1]
    fids    = [shared[i] for i in order[:n_fiducials]]

    rank_df = pd.DataFrame({"gene": [shared[i] for i in order],
                             "mean_xen": means[order]})

    def _to_csv(ps: PointSet, genes: list, cap: int, s: int) -> str:
        sub = ps.subsample(cap, seed=s) if ps.n > cap else ps
        df  = pd.DataFrame({"barcode": sub.ids,
                             "x": sub.coords[:, 0],
                             "y": sub.coords[:, 1]})
        for g in genes:
            if g in ps.feature_names:
                df[g] = ps.feature(g)[
                    np.where(np.isin(ps.ids, sub.ids))[0]
                    if ps.n > cap else slice(None)]
        buf = _io.StringIO()
        df.to_csv(buf, index=False)
        return buf.getvalue()

    xen_csv = _to_csv(xen, fids, xen_cap, seed)
    vis_csv = _to_csv(vis, fids, vis.n,   seed)
    return xen_csv, vis_csv, rank_df, fids
