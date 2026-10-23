"""
project.py
----------
The persistence + priority layer that makes the batch and interactive paths
coexist. One directory per project; per sample it stores:

  {project}/{sample}/selection.json   {"ids": [...], "source": "auto"|"manual"}
  {project}/{sample}/transform.json   {"transform": {...}, "source": "auto"|"manual"}

The rule — MANUAL OVERRIDES AUTO — is enforced in one place:
the batch runners (propose_all / align_all) skip any sample whose stored
source is "manual" unless force=True, and the interactive setters
(set_selection / set_transform) always write source="manual". So you can run
the auto pass over 100 samples, then hand-fix the few that came out wrong, and
re-running the auto pass won't clobber your fixes.

`loader` is a callable sample -> (moving_xenium, fixed_visium) PointSets, so
the batch runners can pull each sample's data on demand.

Batch / SLURM
-------------
align_all(..., slurm=True) writes a SLURM job-array script and submits it via
sbatch.  Each array task runs one sample, calling auto_align() and writing
transform.json.  The script is saved to {project_dir}/_slurm/align_array.sh
and logs land in {project_dir}/_slurm/logs/.

See Oscar docs: https://docs.ccv.brown.edu/oscar/submitting-jobs/array
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .io.loaders import PointSet
from .select import propose_selection
from .align import auto_align
from .transform import compose


class Project:
    def __init__(self, project_dir: str, loader: Callable[[str], tuple[PointSet, PointSet]]):
        self.dir = os.path.abspath(project_dir)
        os.makedirs(self.dir, exist_ok=True)
        self.loader = loader

    # ---- paths ----
    def _sdir(self, sample):
        d = os.path.join(self.dir, sample)
        os.makedirs(d, exist_ok=True)
        return d

    def _sel_path(self, sample):
        return os.path.join(self._sdir(sample), "selection.json")

    def _xf_path(self, sample):
        return os.path.join(self._sdir(sample), "transform.json")

    # ---- selection ----
    def get_selection(self, sample):
        p = self._sel_path(sample)
        return json.load(open(p)) if os.path.exists(p) else None

    def set_selection(self, sample, ids, source="manual"):
        ids = np.asarray(ids, dtype=str).tolist()
        json.dump({"ids": ids, "source": source}, open(self._sel_path(sample), "w"))
        return self

    def propose_all(self, samples, force=False, verbose=True,
                    output_dir=None, **kwargs):
        """Auto-propose spot selections for each sample.

        Skips samples whose stored selection is source='manual' unless force=True.

        output_dir : if provided, saves a barcode CSV per sample to
                     {output_dir}/{sample}_selection.csv in addition to the
                     project store.  Create the directory first if needed.
                     Example: output_dir="/oscar/scratch/amarehal/alignment_tool"
        """
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        for s in samples:
            cur = self.get_selection(s)
            if cur and cur.get("source") == "manual" and not force:
                if verbose: print(f"{s}: keeping manual selection")
                continue
            try:
                xen, vis = self.loader(s)
                ids = propose_selection(vis, xen, **kwargs)
                self.set_selection(s, ids, source="auto")
                if verbose: print(f"{s}: auto-selected {len(ids)}/{vis.n} spots")
                if output_dir is not None:
                    import pandas as pd
                    csv_path = os.path.join(output_dir, f"{s}_selection.csv")
                    pd.DataFrame({"barcode": ids}).to_csv(csv_path, index=False)
                    if verbose: print(f"  → saved {csv_path}")
            except Exception as e:
                if verbose: print(f"{s}: SKIP — {e}")
        return self

    def preview_selections(self, samples, ncols=4,
                           figsize_per_panel=(4, 3.5),
                           point_size_vis=8.0,
                           point_size_xen=2.0,
                           xen_cap=8000,
                           verbose=True):
        """Thumbnail grid showing auto-proposed Visium spot selections.

        Each row = one sample.  Left panel: all Visium spots (dim grey) with
        selected spots highlighted in bright pink.  Right panel: Xenium cells
        (cyan) for spatial reference.  Black background throughout.

        Title: "Visium spots selected: {sample}"

        Parameters
        ----------
        samples         : list of sample names
        ncols           : number of SAMPLES per row (each takes 2 subplot cols)
        figsize_per_panel : (w, h) per sample pair in inches
        point_size_vis  : scatter size for Visium spots
        point_size_xen  : scatter size for Xenium cells
        xen_cap         : max Xenium cells to plot (subsampled for speed)

        Returns
        -------
        matplotlib Figure — display inline or savefig().
        """
        import matplotlib.pyplot as plt
        import numpy as np

        n = len(samples)
        nrows = (n + ncols - 1) // ncols
        # Each sample = 2 subplot columns (Visium | Xenium)
        fig_w = figsize_per_panel[0] * 2 * ncols
        fig_h = figsize_per_panel[1] * nrows
        fig, axes = plt.subplots(nrows, ncols * 2,
                                  figsize=(fig_w, fig_h),
                                  facecolor="black")
        if nrows == 1 and ncols == 1:
            axes = [[axes[0], axes[1]]]
        elif nrows == 1:
            axes = [axes]
        # axes[row][col_pair*2], axes[row][col_pair*2+1]

        fig.patch.set_facecolor("black")

        def _ax(i):
            row = i // ncols
            col = i % ncols
            ax_vis = axes[row][col * 2]
            ax_xen = axes[row][col * 2 + 1]
            return ax_vis, ax_xen

        for i, sample in enumerate(samples):
            ax_vis, ax_xen = _ax(i)
            for ax in (ax_vis, ax_xen):
                ax.set_facecolor("black")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_aspect('equal', adjustable='datalim')
                for sp in ax.spines.values():
                    sp.set_visible(False)

            try:
                xen, vis = self.loader(sample)
                sel = self.get_selection(sample)
                sel_ids = set(sel["ids"]) if sel else set()
                src = sel.get("source", "?") if sel else "none"

                # Split Visium into selected vs unselected
                all_coords = vis.coords
                all_ids = vis.ids

                sel_mask = np.array([bid in sel_ids for bid in all_ids])
                unsel_mask = ~sel_mask

                # Left: Visium selection panel
                if unsel_mask.any():
                    ax_vis.scatter(
                        all_coords[unsel_mask, 0], all_coords[unsel_mask, 1],
                        c="#444444", s=point_size_vis, linewidths=0,
                        rasterized=True, alpha=0.5,
                    )
                if sel_mask.any():
                    ax_vis.scatter(
                        all_coords[sel_mask, 0], all_coords[sel_mask, 1],
                        c="#ff3aaa", s=point_size_vis * 1.5, linewidths=0,
                        rasterized=True, alpha=0.9,
                    )

                n_sel = int(sel_mask.sum())
                ax_vis.set_title(
                    f"Visium spots selected: {sample}\n"
                    f"{n_sel}/{vis.n} spots [{src}]",
                    color="white", fontsize=7, pad=3,
                )

                # Right: Xenium reference panel
                xen_coords = xen.coords
                if xen_coords.shape[0] > xen_cap:
                    idx = np.random.choice(xen_coords.shape[0], xen_cap, replace=False)
                    xen_coords = xen_coords[idx]
                ax_xen.scatter(
                    xen_coords[:, 0], xen_coords[:, 1],
                    c="#22d3d3", s=point_size_xen, linewidths=0,
                    rasterized=True, alpha=0.6,
                )
                ax_xen.set_title(
                    f"Xenium ref: {sample}",
                    color="#22d3d3", fontsize=7, pad=3,
                )

                if verbose:
                    print(f"{sample}: {n_sel}/{vis.n} selected [{src}]")

            except Exception as e:
                for ax in (ax_vis, ax_xen):
                    ax.text(0.5, 0.5, f"{sample}\n{e}",
                            color="red", fontsize=7,
                            ha="center", va="center",
                            transform=ax.transAxes)
                if verbose:
                    print(f"{sample}: ERROR — {e}")

        # Hide unused panels
        total_pairs = nrows * ncols
        for j in range(n, total_pairs):
            ax_v, ax_x = _ax(j)
            ax_v.set_visible(False)
            ax_x.set_visible(False)

        plt.tight_layout(pad=0.4)
        return fig

    def selected_visium(self, sample) -> PointSet:
        """The fixed (Visium) set cropped to this sample's stored selection.
        If no selection is stored, returns the full set."""
        xen, vis = self.loader(sample)
        sel = self.get_selection(sample)
        return vis.subset_by_ids(sel["ids"]) if sel else vis

    # ---- transform ----
    def get_transform(self, sample):
        p = self._xf_path(sample)
        return json.load(open(p)) if os.path.exists(p) else None

    def set_transform(self, sample, transform, source="manual"):
        json.dump({"transform": transform, "source": source}, open(self._xf_path(sample), "w"))
        return self

    # ---- align_all (inline or SLURM array) ----

    def align_all(self, samples, force=False, verbose=True,
                  slurm=False, slurm_partition="batch", slurm_time="60",
                  slurm_mem="16G", slurm_account=None, **kwargs):
        """Auto-align each sample (coarse init + FGW) on its SELECTED Visium.

        slurm=False (default): run each sample sequentially in this kernel.
        slurm=True: write + submit a SLURM job array (one task per sample).
          Each task imports this Project, calls auto_align, and writes
          transform.json.  Skips samples already marked source='manual' unless
          force=True.  Returns the sbatch job-id string (or raises on failure).

        SLURM array docs: https://docs.ccv.brown.edu/oscar/submitting-jobs/array
        """
        if slurm:
            return self._align_all_slurm(
                samples, force=force, verbose=verbose,
                partition=slurm_partition, time=slurm_time, mem=slurm_mem,
                account=slurm_account, **kwargs,
            )

        for s in samples:
            cur = self.get_transform(s)
            if cur and cur.get("source") == "manual" and not force:
                if verbose: print(f"{s}: keeping manual transform")
                continue
            try:
                xen, _ = self.loader(s)
                vis_sel = self.selected_visium(s)
                hist = auto_align(xen, vis_sel, verbose=False, **kwargs)
                traj = [t.to_dict() for t in hist.trajectory]
                self.set_transform(s, {"trajectory": traj}, source="auto")
                if verbose:
                    print(f"{s}: auto-aligned ({len(hist.trajectory)} steps)")
            except Exception as e:
                if verbose: print(f"{s}: SKIP — {e}")
        return self

    def _align_all_slurm(self, samples, force=False, verbose=True,
                          partition="batch", time="60", mem="16G",
                          account=None, **kwargs):
        """Write a SLURM array script and submit it via sbatch.

        The script writes transform.json for each sample into the project dir,
        which must be on a filesystem visible from compute nodes (e.g. /oscar/home
        or /oscar/scratch).  Each array task is identified by $SLURM_ARRAY_TASK_ID
        indexing into the samples list baked into the script.
        """
        # Filter out manually-set samples unless force=True
        to_run = []
        for s in samples:
            cur = self.get_transform(s)
            if cur and cur.get("source") == "manual" and not force:
                if verbose: print(f"{s}: keeping manual transform (skipped in array)")
                continue
            to_run.append(s)

        if not to_run:
            if verbose: print("Nothing to align (all samples have manual transforms).")
            return None

        slurm_dir = os.path.join(self.dir, "_slurm")
        log_dir = os.path.join(slurm_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        samples_json = json.dumps(to_run)
        project_dir = self.dir
        # We need the module path of the loader — user must pass loader_expr or
        # the script uses a sentinel that will raise a clear error.
        loader_expr = "None  # REPLACE: set loader_expr= in align_all() or edit this script"

        account_line = f"#SBATCH --account={account}" if account else ""
        kwarg_repr = repr(kwargs) if kwargs else "{}"

        script = textwrap.dedent(f"""\
            #!/bin/bash
            #SBATCH --job-name=strata_array
            #SBATCH --partition={partition}
            #SBATCH --time={time}
            #SBATCH --mem={mem}
            #SBATCH --array=0-{len(to_run)-1}
            #SBATCH --output={log_dir}/align_%a.out
            #SBATCH --error={log_dir}/align_%a.err
            {account_line}

            # ---- activate environment ----
            source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
            conda activate cellcharter

            # ---- run alignment for this array task ----
            python3 - <<'PYEOF'
            import sys, json, os
            sys.path.insert(0, os.path.dirname(os.path.dirname("{project_dir}")))

            SAMPLES = {samples_json}
            idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
            sample = SAMPLES[idx]

            project_dir = "{project_dir}"

            # Import the package
            import strata as sa
            from strata.project import Project

            # ---- User: replace this with your actual loader ----
            # loader = lambda s: (sa.io.load_xenium(...), sa.io.load_visium(...))
            loader = {loader_expr}
            if loader is None:
                raise RuntimeError(
                    "No loader configured in the SLURM script. "
                    "Edit {slurm_dir}/align_array.sh and set the loader."
                )

            proj = Project(project_dir, loader)

            # Skip if manual (double-check inside the task)
            cur = proj.get_transform(sample)
            force = {str(force).lower() == "true"}
            if cur and cur.get("source") == "manual" and not force:
                print(f"{{sample}}: keeping manual transform")
                sys.exit(0)

            xen, _ = proj.loader(sample)
            vis_sel = proj.selected_visium(sample)
            extra_kw = {kwarg_repr}
            hist = sa.auto_align(xen, vis_sel, verbose=True, **extra_kw)
            traj = [t.to_dict() for t in hist.trajectory]
            proj.set_transform(sample, {{"trajectory": traj}}, source="auto")
            print(f"{{sample}}: done ({{len(hist.trajectory)}} steps)")
            PYEOF
        """)

        script_path = os.path.join(slurm_dir, "align_array.sh")
        with open(script_path, "w") as f:
            f.write(script)
        os.chmod(script_path, 0o755)

        if verbose:
            print(f"SLURM array script written: {script_path}")
            print(f"Samples to align: {len(to_run)}")
            print(f"⚠  Edit {script_path} to set your loader before submitting.")
            print(f"Submit with: sbatch {script_path}")

        # Attempt auto-submit; warn if sbatch not found (e.g. login vs compute node)
        try:
            result = subprocess.run(
                ["sbatch", script_path],
                capture_output=True, text=True, check=True,
            )
            job_id = result.stdout.strip().split()[-1]
            if verbose:
                print(f"Submitted array job: {job_id}")
                print(f"Monitor: squeue -j {job_id}")
                print(f"Logs: {log_dir}/align_*.out")
            return job_id
        except FileNotFoundError:
            if verbose:
                print("sbatch not found on this node — script written but not submitted.")
                print(f"SSH to a login node and run: sbatch {script_path}")
            return None
        except subprocess.CalledProcessError as e:
            if verbose:
                print(f"sbatch failed: {e.stderr}")
            return None

    # ---- applying a stored transform ----

    def aligned_xenium_coords(self, sample) -> np.ndarray:
        """Full-resolution Xenium coordinates after this sample's stored
        AUTO transform (trajectory)."""
        from .transform import AffineTransform, LinearTransform, TPSTransform
        xen, _ = self.loader(sample)
        rec = self.get_transform(sample)
        if rec is None:
            raise RuntimeError(f"{sample}: no transform stored")
        traj = rec["transform"].get("trajectory")
        if traj is None:
            raise RuntimeError(
                f"{sample}: stored transform is a manual aligner transform "
                "(moves Visium). Use aligned_visium_coords(sample)."
            )
        objs = []
        for d in traj:
            if d["kind"] == "affine":
                objs.append(AffineTransform(
                    theta_deg=d["theta_deg"], scale=d["scale"],
                    flip=tuple(d["flip"]), pivot=tuple(d["pivot"]),
                    translate=tuple(d["translate"])))
            elif d["kind"] == "linear":
                objs.append(LinearTransform(A=np.array(d["A"]), t=np.array(d["t"])))
            else:
                raise RuntimeError(
                    f"{sample}: cannot replay a '{d['kind']}' step from disk "
                    "(TPS control points not serialized); re-run align in memory")
        return compose(xen.coords, objs)

    def aligned_visium_coords(self, sample) -> np.ndarray:
        """Full-resolution SELECTED-Visium coordinates after a MANUAL aligner
        transform (the browser tool's JSON, which moves Visium into Xenium's frame)."""
        rec = self.get_transform(sample)
        if rec is None or "trajectory" in rec.get("transform", {}):
            raise RuntimeError(f"{sample}: no manual aligner transform stored")
        vis_sel = self.selected_visium(sample)
        return apply_aligner_transform(vis_sel.coords, vis_sel.ids, rec["transform"])

    # ---- qc_all ----

    def qc_all(self, samples, min_concordance=0.3, min_coverage=0.5, verbose=True, **kwargs):
        """Score every sample's stored alignment and return a flagging DataFrame."""
        from .qc import concordance, QCResult
        rows = []
        for s in samples:
            rec = self.get_transform(s)
            if rec is None:
                rows.append((s, "-", np.nan, np.nan, np.nan, 0, True))
                if verbose: print(f"{s}: no transform")
                continue
            try:
                xen, _ = self.loader(s)
                vis_sel = self.selected_visium(s)
                src = rec.get("source", "?")
                if "trajectory" in rec.get("transform", {}):
                    moved = self.aligned_xenium_coords(s)
                    res = concordance(vis_sel, xen, agg_coords=moved, **kwargs)
                else:
                    moved_vis = self.aligned_visium_coords(s)
                    vis_moved = vis_sel.with_coords(moved_vis)
                    res = concordance(vis_moved, xen, **kwargs)
                flag = res.flagged(min_concordance, min_coverage)
                rows.append((s, src, res.concordance, res.coverage, res.residual, res.n_used, flag))
                if verbose:
                    print(f"{s}: concordance={res.concordance:.3f} coverage={res.coverage:.2f} "
                          f"{'<< FLAG' if flag else ''}")
            except Exception as e:
                rows.append((s, rec.get("source", "?"), np.nan, np.nan, np.nan, 0, True))
                if verbose: print(f"{s}: QC failed — {e}")
        df = pd.DataFrame(rows, columns=["sample", "source", "concordance",
                                          "coverage", "residual", "n_used", "flag"])
        if verbose:
            n_flag = int(df["flag"].sum())
            print(f"\n{n_flag}/{len(df)} sample(s) flagged for review")
        return df

    # ---- preview_alignments ----

    def preview_alignments(
        self,
        samples,
        ncols: int = 4,
        point_size: float = 3.0,
        xen_cap: int = 8000,
        vis_cap: int = 3000,
        figsize_per_panel: tuple = (3.5, 3.5),
        fiducial_gene: Optional[str] = None,
        title_prefix: str = "Overlay",
        show_flagged: bool = True,
        qc_df: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ):
        """Render a thumbnail grid of all stored alignments for visual QC.

        Each panel: black background, Xenium cells in blue, Visium spots in
        pink, both optionally colored by intensity of a shared fiducial gene.
        Title: "Overlay {sample} – Fiducial {gene}" (or "– no transform" if
        not yet aligned).

        Parameters
        ----------
        samples      : list of sample names
        ncols        : panels per row
        point_size   : matplotlib scatter size
        xen_cap      : max Xenium points to plot per panel (subsampled for speed)
        vis_cap      : max Visium points to plot per panel
        fiducial_gene: gene to color by; if None, uses rank_fiducials to pick
                       the top shared gene automatically
        show_flagged : add a red border around panels flagged by qc_df
        qc_df        : optional DataFrame from qc_all() for flag overlays
        verbose      : print per-sample status

        Returns
        -------
        matplotlib Figure — display inline with plt.show() or savefig().
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from .fiducial import rank_fiducials

        nrows = (len(samples) + ncols - 1) // ncols
        fig_w = figsize_per_panel[0] * ncols
        fig_h = figsize_per_panel[1] * nrows
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                                 facecolor="black")
        axes = np.array(axes).flatten()

        # Build a lookup for flagged samples
        flagged_set = set()
        if show_flagged and qc_df is not None:
            flagged_set = set(qc_df.loc[qc_df["flag"] == True, "sample"].tolist())

        for i, (ax, sample) in enumerate(zip(axes, samples)):
            ax.set_facecolor("black")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            try:
                xen, vis = self.loader(sample)
                vis_sel = self.selected_visium(sample)
                rec = self.get_transform(sample)

                # Resolve aligned coordinates
                if rec is None:
                    # No transform yet — plot raw (unaligned)
                    xen_coords = xen.coords
                    vis_coords = vis_sel.coords
                    suffix = "– no transform"
                elif "trajectory" in rec.get("transform", {}):
                    # Auto transform: Xenium moved into Visium frame
                    xen_coords = self.aligned_xenium_coords(sample)
                    vis_coords = vis_sel.coords
                    src = rec.get("source", "auto")
                    suffix = f"[{src}]"
                else:
                    # Manual transform: Visium moved into Xenium frame
                    xen_coords = xen.coords
                    vis_coords = self.aligned_visium_coords(sample)
                    suffix = "[manual]"

                # Pick fiducial gene
                gene = fiducial_gene
                if gene is None:
                    shared = xen.shared_features(vis_sel)
                    if shared:
                        try:
                            _, _, fids = rank_fiducials(xen, vis_sel, n=1)
                            gene = fids[0] if fids else shared[0]
                        except Exception:
                            gene = shared[0] if shared else None

                # Subsample for speed
                def _sub(coords, ps, cap):
                    if coords.shape[0] > cap:
                        idx = np.random.choice(coords.shape[0], cap, replace=False)
                        return coords[idx], ps.subset_by_idx(idx) if hasattr(ps, "subset_by_idx") else None
                    return coords, ps

                xen_c, xen_ps = _sub(xen_coords, xen, xen_cap)
                vis_c, vis_ps = _sub(vis_coords, vis_sel, vis_cap)

                # Color by gene expression if available
                def _get_expr(ps_orig, coords_sub, gene, cap, full_coords):
                    """Get expression values aligned to subsampled coords."""
                    if gene is None or ps_orig is None:
                        return None
                    try:
                        vals = ps_orig.feature(gene)
                        if full_coords.shape[0] > cap:
                            idx = np.random.choice(full_coords.shape[0], cap, replace=False)
                            return vals[idx]
                        return vals
                    except Exception:
                        return None

                xen_expr = None
                vis_expr = None
                if gene is not None:
                    try:
                        if xen_coords.shape[0] > xen_cap:
                            idx = np.random.choice(xen_coords.shape[0], xen_cap, replace=False)
                            xen_c = xen_coords[idx]
                            xen_expr = xen.feature(gene)[idx]
                        else:
                            xen_expr = xen.feature(gene)
                    except Exception:
                        pass
                    try:
                        if vis_coords.shape[0] > vis_cap:
                            idx = np.random.choice(vis_coords.shape[0], vis_cap, replace=False)
                            vis_c = vis_coords[idx]
                            vis_expr = vis_sel.feature(gene)[idx]
                        else:
                            vis_expr = vis_sel.feature(gene)
                    except Exception:
                        pass

                # Normalize expression to [0,1] for alpha modulation
                def _norm(v):
                    if v is None: return None
                    mn, mx = v.min(), v.max()
                    return (v - mn) / (mx - mn + 1e-9)

                xen_alpha = _norm(xen_expr)
                vis_alpha = _norm(vis_expr)

                # Plot Visium (pink) then Xenium (blue) on top
                if vis_alpha is not None:
                    # Color by expression: dim pink → bright pink
                    colors_vis = np.zeros((len(vis_c), 4))
                    colors_vis[:, 0] = 0.9          # R
                    colors_vis[:, 1] = 0.4 * vis_alpha  # G
                    colors_vis[:, 2] = 0.6          # B
                    colors_vis[:, 3] = 0.3 + 0.5 * vis_alpha
                    ax.scatter(vis_c[:, 0], vis_c[:, 1], c=colors_vis,
                               s=point_size * 2.5, linewidths=0, rasterized=True)
                else:
                    ax.scatter(vis_c[:, 0], vis_c[:, 1], c="#e066aa", alpha=0.35,
                               s=point_size * 2.5, linewidths=0, rasterized=True)

                if xen_alpha is not None:
                    colors_xen = np.zeros((len(xen_c), 4))
                    colors_xen[:, 0] = 0.1 * xen_alpha
                    colors_xen[:, 1] = 0.4 * xen_alpha
                    colors_xen[:, 2] = 1.0          # B
                    colors_xen[:, 3] = 0.25 + 0.65 * xen_alpha
                    ax.scatter(xen_c[:, 0], xen_c[:, 1], c=colors_xen,
                               s=point_size, linewidths=0, rasterized=True)
                else:
                    ax.scatter(xen_c[:, 0], xen_c[:, 1], c="#4488ff", alpha=0.5,
                               s=point_size, linewidths=0, rasterized=True)

                gene_label = gene if gene else "no gene"
                ax.set_title(
                    f"{title_prefix} {sample}\n– Fiducial {gene_label} {suffix}",
                    color="white", fontsize=7, pad=3,
                )

                # Red border for flagged samples
                if sample in flagged_set:
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_edgecolor("#ff4444")
                        spine.set_linewidth(2)

                if verbose:
                    print(f"{sample}: plotted (gene={gene_label}, "
                          f"xen={len(xen_c)}, vis={len(vis_c)})")

            except Exception as e:
                ax.text(0.5, 0.5, f"{sample}\n{e}", color="red", fontsize=7,
                        ha="center", va="center", transform=ax.transAxes)
                if verbose:
                    print(f"{sample}: ERROR — {e}")

        # Hide unused panels
        for ax in axes[len(samples):]:
            ax.set_visible(False)

        fig.patch.set_facecolor("black")
        plt.tight_layout(pad=0.5)
        return fig

    # ---- status ----

    def status(self, samples):
        """Quick table of what's stored and its source, per sample."""
        rows = []
        for s in samples:
            sel = self.get_selection(s)
            xf = self.get_transform(s)
            rows.append((s,
                         sel["source"] if sel else "-",
                         len(sel["ids"]) if sel else 0,
                         xf["source"] if xf else "-"))
        w = max(len(s) for s in samples) if samples else 6
        print(f"{'sample':<{w}}  {'selection':<10} {'n_spots':>7}  transform")
        for s, ssrc, n, xsrc in rows:
            print(f"{s:<{w}}  {ssrc:<10} {n:>7}  {xsrc}")
        return rows
