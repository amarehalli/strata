# STRATA

**S**patial **T**ranscriptomics **R**egistration **A**nd **T**ransfer **A**lignment

A Python toolkit for co-registering Xenium (single-cell resolution) and Visium
(transcriptome-depth) spatial transcriptomics datasets using Fused
Gromov-Wasserstein optimal transport.

---

## What it does

| Problem | STRATA's answer |
|---|---|
| Xenium and Visium cover the same tissue but live in different coordinate systems | FGW alignment maps Xenium → Visium physical space |
| Manually aligning 10+ samples is tedious | `strata align` runs batch alignment on SLURM |
| You can't see if the alignment is right | Browser tool plays back the FGW animation frame-by-frame |
| The browser tool doesn't work from an HPC Jupyter session | CLI + SSH tunnel — one command, no Jupyter proxy needed |
| You need both single-cell resolution AND whole-transcriptome depth | After alignment, reconvolution transfers Visium expression onto Xenium cells |

---

## Install

```bash
pip install strata-spatial        # PyPI (coming soon)
# or from source:
git clone https://github.com/amarehalli/strata
pip install -e strata/
```

**Requirements:** Python ≥ 3.10, numpy, pandas, scipy, scikit-learn, scanpy, POT (Python Optimal Transport), matplotlib.

---

## Quickstart

```python
import strata as st

# Load data
xen = st.io.load_xenium("path/to/spatial_annotated.h5ad")
vis = st.io.load_visium("path/to/filtered_feature_bc_matrix.h5",
                        "path/to/tissue_positions.csv")

# Auto-propose which Visium spots overlap the Xenium patch
from strata.selection import propose_selection
ids = propose_selection(vis, xen)
vis_sel = vis.subset_by_ids(ids)

# Run alignment
hist = st.auto_align(xen, vis_sel)

# Save self-contained HTML with baked animation
srv = st.save_aligner(xen, vis_sel,
                      path="alignment.html",
                      trajectory=hist)
```

Open `alignment.html` — the FGW animation auto-plays, then the canvas stays
live for manual refinement. Click **Send transform to notebook** to export.

---

## CLI (for HPC)

After `pip install`, a `strata` command is available:

```bash
# Check alignment status across all samples
strata status \
    --project /scratch/proj \
    --loader  loader.py \
    --samples B1-1 B2-1 B3-1

# Auto-propose spot selections
strata propose \
    --project /scratch/proj \
    --loader  loader.py \
    --samples B1-1 B2-1 \
    --output-dir /scratch/selections

# Align (sequential or SLURM array)
strata align  --project /scratch/proj --loader loader.py \
              --samples B1-1 B2-1 [--slurm]

# Open interactive aligner for one sample
strata view   --project /scratch/proj --loader loader.py --sample B1-1
# → prints port + ProxyJump tunnel command for your laptop
```

**HPC / Open OnDemand:** `strata view` prints a single `ssh -J` command to
run on your local machine. Open `http://localhost:<port>/` and the aligner
appears with no Jupyter proxy configuration needed.

---

## Cohort workflow (Project API)

```python
import strata as st

def load(sample):
    xen = st.io.load_xenium(f"/data/{sample}/spatial_annotated.h5ad")
    vis = st.io.load_visium(f"/data/{sample}/matrix.h5",
                            f"/data/{sample}/tissue_positions.csv")
    return xen, vis

proj = st.Project("/scratch/my_project", loader=load)
samples = ["B1-1", "B2-1", "G1", "G2"]

# 1. Full-tissue samples — select all spots (user-controlled, not hardcoded)
full_tissue = ["B1-2", "B2-2"]
for s in full_tissue:
    _, vis = load(s)
    proj.set_selection(s, vis.ids, source="manual")

# 2. Auto-propose remaining samples
proj.propose_all(samples, output_dir="/scratch/selections")

# 3. Preview — bright pink = selected, grey = rest, cyan = Xenium ref
fig = proj.preview_selections(samples)
fig.savefig("selections.png", dpi=150, bbox_inches="tight")

# 4. Override any bad samples manually (see docs/manual_override.md)

# 5. Align all
proj.align_all(samples)          # or align_all(samples, slurm=True)

# 6. Preview alignments (black bg, Xenium blue, Visium pink)
fig = proj.preview_alignments(samples)
fig.savefig("alignments.png", dpi=150, bbox_inches="tight")
```

---

## How it works

STRATA uses **Fused Gromov-Wasserstein (FGW) optimal transport** to jointly
minimise structural (geometric) and feature (expression) discrepancy between
the two point clouds.

1. **Coarse alignment** — PCA on shared genes → rigid Procrustes (handles
   different orientations and scales).
2. **FGW refinement** — iterative: solve FGW OT → derive rigid update →
   apply → repeat. Exact Frank-Wolfe solver by default; entropic solver
   available for large datasets.
3. **Concordance QC** — per-gene Spearman correlation between aggregated
   Xenium expression and Visium expression, used to flag poor alignments.

---

## Citation

If you use STRATA in your research, please cite:

> Marehalli A. *et al.* STRATA: spatial transcriptomics registration and
> alignment using Fused Gromov-Wasserstein optimal transport. *in prep* (2026).

---

## License

MIT © Aditi Marehalli
