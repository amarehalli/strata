"""Basic smoke tests for STRATA core functionality."""
import numpy as np
import pytest
from strata.io.loaders import PointSet
from strata.transform import Transform, compose
from strata.selection import propose_selection
from strata.qc import qc_alignment


def _make_pointset(n=100, seed=0, role="moving"):
    rng = np.random.default_rng(seed)
    coords   = rng.random((n, 2)) * 1000
    features = rng.random((n, 20)).astype(np.float32)
    genes    = [f"GENE{i}" for i in range(20)]
    ids      = np.array([f"cell_{i}" for i in range(n)])
    return PointSet(coords=coords, ids=ids, feature_names=genes,
                    features=features, role=role)


def test_pointset_basics():
    ps = _make_pointset()
    assert ps.n == 100
    assert ps.feature("GENE0").shape == (100,)
    sub = ps.subsample(10)
    assert sub.n == 10


def test_shared_features():
    xen = _make_pointset()
    vis = _make_pointset(role="fixed")
    shared = xen.shared_features(vis)
    assert len(shared) == 20


def test_transform_apply():
    t = Transform(rotation=np.pi/4, scale=1.5,
                  translation=np.array([10.0, -5.0]))
    coords = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = t.apply(coords)
    assert out.shape == (2, 2)


def test_transform_roundtrip():
    t = Transform(rotation=0.3, scale=1.2,
                  translation=np.array([5.0, -3.0]))
    d = t.to_dict()
    t2 = Transform.from_dict(d)
    np.testing.assert_allclose(t.apply(np.eye(2)),
                               t2.apply(np.eye(2)), atol=1e-6)


def test_propose_selection_returns_subset():
    xen = _make_pointset(n=200)
    vis = _make_pointset(n=100, role="fixed")
    ids = propose_selection(vis, xen, quantile=0.5)
    assert len(ids) > 0
    assert len(ids) <= vis.n
    assert all(i in vis.ids for i in ids)


def test_qc_alignment_returns_dict():
    xen = _make_pointset(n=50)
    vis = _make_pointset(n=50, role="fixed")
    result = qc_alignment(xen, vis)
    assert "concordance" in result
    assert "coverage" in result
    assert -1.0 <= result["concordance"] <= 1.0
    assert  0.0 <= result["coverage"]    <= 1.0
