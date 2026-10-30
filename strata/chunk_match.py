"""
strata.chunk_match
------------------
Stage 1 of auto-alignment: automatically identify which Visium spots
correspond to the Xenium data, handling all sample types without
any hardcoding:

  Single biopsy  — Xenium = 1 chunk  → match 1 Visium chunk
  Multi-biopsy   — Xenium = N chunks → match N Visium chunks
  Full resection — Xenium covers most of the Visium extent → keep all

Sample type is detected automatically from the data.
"""

from __future__ import annotations
import numpy as np
from .io.loaders import PointSet


# ── Spatial clustering ─────────────────────────────────────────────────────

def _cluster_pointset(coords: np.ndarray, min_samples: int = 5,
                      eps_scale: float = 4.0) -> np.ndarray:
    """DBSCAN with adaptive eps based on nearest-neighbour distance distribution."""
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2).fit(coords)
    dists, _ = nn.kneighbors(coords)
    eps = float(np.quantile(dists[:, 1], 0.85)) * eps_scale
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)


def _chunks_from_labels(coords: np.ndarray, labels: np.ndarray,
                        min_size: int = 5) -> list[dict]:
    """Summarise each spatial cluster into shape descriptors."""
    chunks = []
    for lbl in np.unique(labels):
        if lbl < 0:
            continue
        mask = labels == lbl
        if mask.sum() < min_size:
            continue
        c = coords[mask]
        cov  = np.cov(c.T) if c.shape[0] > 2 else np.eye(2)
        eigs = np.maximum(np.linalg.eigvalsh(np.atleast_2d(cov)), 0)
        major = np.sqrt(eigs[-1]) * 2
        minor = np.sqrt(eigs[0])  * 2
        chunks.append({
            "label":    int(lbl),
            "n":        int(mask.sum()),
            "mask":     mask,
            "centroid": c.mean(axis=0),
            "major":    float(major),
            "minor":    float(minor),
            "aspect":   float(major / (minor + 1e-6)),
            "bbox_area": float((c[:, 0].ptp() + 1) * (c[:, 1].ptp() + 1)),
        })
    return sorted(chunks, key=lambda x: -x["n"])


# ── Sample-type detection ──────────────────────────────────────────────────

def _detect_sample_type(xen_chunks: list[dict],
                        vis_chunks: list[dict],
                        xen_coords: np.ndarray,
                        vis_coords: np.ndarray,
                        coverage_threshold: float = 0.4) -> str:
    """Return 'single', 'multi', or 'resection'.

    A sample is classified as a resection when the Xenium bounding-box
    area is large relative to the total Visium bounding-box area.
    This is data-driven: no sample names or counts are hardcoded.
    """
    # Bounding-box areas
    def _bbox_area(c):
        return (c[:, 0].ptp() + 1) * (c[:, 1].ptp() + 1)

    xen_area = _bbox_area(xen_coords)
    vis_area  = _bbox_area(vis_coords)
    coverage  = xen_area / (vis_area + 1e-6)

    n_xen = len(xen_chunks)

    if coverage >= coverage_threshold:
        return "resection"
    if n_xen > 1:
        return "multi"
    return "single"


# ── Shape similarity ───────────────────────────────────────────────────────

def _shape_score(a: dict, b: dict) -> float:
    """Lower = more similar shape."""
    asp = abs(a["aspect"] - b["aspect"]) / (max(a["aspect"], b["aspect"]) + 1e-6)
    return asp


# ── Main ───────────────────────────────────────────────────────────────────

def find_matching_chunks(
    xen: PointSet,
    vis: PointSet,
    coverage_threshold: float = 0.4,
    min_vis_spots: int = 20,
    merge_distance_factor: float = 1.5,
    verbose: bool = True,
) -> tuple[PointSet, list[int], str]:
    """Select the Visium spots corresponding to the Xenium data.

    Sample type (single biopsy / multi-biopsy / full resection) is
    detected automatically — nothing is hardcoded.

    Parameters
    ----------
    xen : PointSet
        Xenium cells.
    vis : PointSet
        Visium spots (may contain multiple tissue sections on one slide).
    coverage_threshold : float
        Fraction of Visium bounding-box area that Xenium must cover to
        be classified as a full-tissue resection (default 0.4).
    min_vis_spots : int
        Ignore Visium clusters smaller than this.
    merge_distance_factor : float
        After matching, also pull in Visium clusters whose centroid is
        within ``factor × major_axis`` of a matched cluster (catches
        split sub-chunks like G2's left piece).
    verbose : bool

    Returns
    -------
    (vis_subset, selected_labels, sample_type)
        vis_subset    — PointSet of selected Visium spots
        selected_labels — list of cluster labels that were selected
        sample_type   — "single" | "multi" | "resection"
    """
    vis_labels  = _cluster_pointset(vis.coords)
    xen_labels  = _cluster_pointset(xen.coords)

    vis_chunks = _chunks_from_labels(vis.coords, vis_labels, min_size=min_vis_spots)
    xen_chunks = _chunks_from_labels(xen.coords, xen_labels, min_size=5)

    sample_type = _detect_sample_type(
        xen_chunks, vis_chunks, xen.coords, vis.coords, coverage_threshold)

    if verbose:
        print(f"Sample type detected: {sample_type}")
        print(f"  Xenium: {len(xen_chunks)} spatial chunk(s), {xen.n} cells")
        print(f"  Visium: {len(vis_chunks)} spatial chunk(s), {vis.n} spots")

    # ── Full resection: keep all sizable Visium chunks ───────────
    if sample_type == "resection" or not vis_chunks:
        selected_lbls = [c["label"] for c in vis_chunks]
        mask = np.isin(vis_labels, selected_lbls)
        if verbose:
            print(f"  → Keeping all {mask.sum()} Visium spots")
        return _subset(vis, mask), selected_lbls, sample_type

    # ── Single / multi: match one Visium chunk per Xenium chunk ──
    matched_lbls: set[int] = set()
    for xc in xen_chunks:
        remaining = [vc for vc in vis_chunks if vc["label"] not in matched_lbls]
        if not remaining:
            break
        best = min(remaining, key=lambda vc: _shape_score(xc, vc))
        matched_lbls.add(best["label"])
        if verbose:
            score = _shape_score(xc, best)
            print(f"  Xenium chunk (n={xc['n']}) "
                  f"→ Visium chunk {best['label']} "
                  f"(n={best['n']}, score={score:.3f})")

    # ── Merge spatially adjacent orphan chunks ────────────────────
    # This catches cases where DBSCAN split one biological tissue section
    # into two clusters (e.g. a narrow gap between sections).
    selected_vis_chunks = [vc for vc in vis_chunks if vc["label"] in matched_lbls]
    sel_coords = vis.coords[np.isin(vis_labels, list(matched_lbls))]

    for vc in vis_chunks:
        if vc["label"] in matched_lbls:
            continue
        dists = np.linalg.norm(sel_coords - vc["centroid"], axis=1)
        # Reference distance: major axis of the nearest selected chunk
        near_chunk = min(selected_vis_chunks,
                         key=lambda c: np.linalg.norm(c["centroid"] - vc["centroid"]))
        threshold = near_chunk["major"] * merge_distance_factor
        if dists.min() < threshold:
            matched_lbls.add(vc["label"])
            if verbose:
                print(f"  Merging adjacent chunk {vc['label']} "
                      f"(n={vc['n']}, gap={dists.min():.0f})")

    selected_lbls = sorted(matched_lbls)
    mask = np.isin(vis_labels, selected_lbls)

    if verbose:
        print(f"  → Selected {mask.sum()} Visium spots from {len(selected_lbls)} cluster(s)")

    return _subset(vis, mask), selected_lbls, sample_type


def _subset(vis: PointSet, mask: np.ndarray) -> PointSet:
    return PointSet(
        coords        = vis.coords[mask],
        ids           = vis.ids[mask],
        feature_names = vis.feature_names,
        features      = vis.features[mask] if vis.features is not None else None,
        role          = "fixed",
    )


# ── Backwards-compatible alias ────────────────────────────────────
def find_matching_chunk(xen, vis, **kwargs):
    vis_sub, lbls, stype = find_matching_chunks(xen, vis, **kwargs)
    return vis_sub, lbls
