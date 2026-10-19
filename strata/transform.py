"""
strata.transform — affine transform representation
---------------------------------------------------
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Transform:
    """A 2-D similarity transform (rotation + isotropic scale + translation).

    Applies to a coordinate array as:
        coords_out = coords_in * scale @ R(rotation).T + translation
    """
    rotation   : float = 0.0          # radians
    scale      : float = 1.0          # isotropic scale factor
    translation: np.ndarray = field(
        default_factory=lambda: np.zeros(2))

    def __post_init__(self):
        self.translation = np.asarray(self.translation, dtype=float)

    def matrix(self) -> np.ndarray:
        """Return the 2×2 rotation-scale matrix."""
        c, s = np.cos(self.rotation), np.sin(self.rotation)
        return self.scale * np.array([[c, -s], [s, c]])

    def apply(self, coords: np.ndarray) -> np.ndarray:
        return coords @ self.matrix().T + self.translation

    def compose(self, other: "Transform") -> "Transform":
        """Return the transform equivalent to applying self then other."""
        M1 = self.matrix()
        M2 = other.matrix()
        M  = M1 @ M2
        # Decompose back to rotation + scale
        scale = float(np.linalg.norm(M[:, 0]))
        theta = float(np.arctan2(M[1, 0], M[0, 0]))
        t     = self.translation @ M2.T + other.translation
        return Transform(rotation=theta, scale=scale, translation=t)

    def to_dict(self) -> dict:
        return {
            "rotation":    float(self.rotation),
            "scale":       float(self.scale),
            "translation": self.translation.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Transform":
        return cls(
            rotation    = d["rotation"],
            scale       = d["scale"],
            translation = np.array(d["translation"]),
        )


def compose(coords: np.ndarray,
            transforms: list[Transform]) -> np.ndarray:
    """Apply a sequence of transforms to a coordinate array."""
    out = coords.copy()
    for t in transforms:
        out = t.apply(out)
    return out
