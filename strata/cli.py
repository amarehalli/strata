"""
cli.py
------
Command-line interface for strata.

Usage (after pip install -e .):

  # Show project status across all samples
  strata status --project /path/to/proj --samples B1-1 B2-1 B3-1

  # Auto-propose spot selections for all samples
  strata propose --project /path/to/proj --samples B1-1 B2-1 \
                     --loader my_loader.py --output-dir /scratch/selections

  # Auto-align all samples (sequential or SLURM array)
  strata align --project /path/to/proj --samples B1-1 B2-1 \
                   --loader my_loader.py [--slurm] [--partition batch] [--mem 16G]

  # Open the interactive aligner for one sample
  strata view --project /path/to/proj --sample B1-1 \
                  --loader my_loader.py [--port 0]

  # Run QC scoring across all samples
  strata qc --project /path/to/proj --samples B1-1 B2-1 \
                --loader my_loader.py

  # Preview selection grid (saves PNG)
  strata preview-selections --project /path/to/proj --samples B1-1 B2-1 \
                                --loader my_loader.py --out selections_grid.png

  # Preview alignment grid (saves PNG)
  strata preview-alignments --project /path/to/proj --samples B1-1 B2-1 \
                                --loader my_loader.py --out alignments_grid.png

HPC / Open OnDemand note
-------------------------
When running on an HPC login/compute node the browser cannot reach the tool
server directly.  Start the tool, note the port it prints, then on your
laptop run ONE ssh tunnel command:

  ssh -L <port>:<compute-node>:<port> <user>@ssh.ccv.brown.edu

Then open http://localhost:<port>/ in your browser.

Example for Brown Oscar:
  strata view --project /scratch/amarehal/proj --sample B1-1 --loader loader.py
  # prints: Aligner serving on port 43821 (node2310.oscar.ccv.brown.edu)
  # on laptop: ssh -L 43821:node2310.oscar.ccv.brown.edu:43821 amarehal@ssh.ccv.brown.edu
  # then open: http://localhost:43821/
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import socket
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Loader resolution
# ---------------------------------------------------------------------------

def _load_loader(loader_path: str):
    """Import a user-supplied loader.py and return its `load(sample)` function.

    The file must define a callable named `load` that accepts a sample name
    and returns (xenium_pointset, visium_pointset).
    """
    p = Path(loader_path).resolve()
    if not p.exists():
        sys.exit(f"Loader file not found: {p}")
    spec = importlib.util.spec_from_file_location("_strata_loader", p)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "load"):
        sys.exit(
            f"{p} must define a function named `load(sample)` "
            "that returns (xenium_pointset, visium_pointset)"
        )
    return mod.load


def _parse_samples(args) -> list[str]:
    if args.samples:
        return args.samples
    if args.samples_file:
        p = Path(args.samples_file)
        if not p.exists():
            sys.exit(f"Samples file not found: {p}")
        return [l.strip() for l in p.read_text().splitlines() if l.strip()]
    sys.exit("Provide --samples or --samples-file")


# ---------------------------------------------------------------------------
# Subcommand: status
# ---------------------------------------------------------------------------

def cmd_status(args):
    import strata as sa
    loader = _load_loader(args.loader)
    proj   = sa.Project(args.project, loader)
    proj.status(_parse_samples(args))


# ---------------------------------------------------------------------------
# Subcommand: propose
# ---------------------------------------------------------------------------

def cmd_propose(args):
    import strata as sa
    loader  = _load_loader(args.loader)
    proj    = sa.Project(args.project, loader)
    samples = _parse_samples(args)
    proj.propose_all(
        samples,
        force      = args.force,
        output_dir = args.output_dir,
    )
    proj.status(samples)


# ---------------------------------------------------------------------------
# Subcommand: align
# ---------------------------------------------------------------------------

def cmd_align(args):
    import strata as sa
    loader  = _load_loader(args.loader)
    proj    = sa.Project(args.project, loader)
    samples = _parse_samples(args)

    if args.slurm:
        job_id = proj.align_all(
            samples,
            force           = args.force,
            slurm           = True,
            slurm_partition = args.partition,
            slurm_time      = args.time,
            slurm_mem       = args.mem,
            slurm_account   = args.account,
        )
        if job_id:
            print(f"SLURM array submitted: job {job_id}")
            print(f"Monitor: squeue -j {job_id}")
        else:
            print("Script written but not submitted (sbatch not found on this node).")
    else:
        proj.align_all(samples, force=args.force)
        proj.status(samples)


# ---------------------------------------------------------------------------
# Subcommand: view (interactive aligner)
# ---------------------------------------------------------------------------

def cmd_view(args):
    import strata as sa

    loader = _load_loader(args.loader)
    proj   = sa.Project(args.project, loader)
    xen, vis = loader(args.sample)

    # If a selection is stored, use the selected Visium subset
    sel = proj.get_selection(args.sample)
    if sel:
        vis = vis.subset_by_ids(sel["ids"])
        print(f"Using stored selection: {vis.n} Visium spots")
    else:
        print("No selection stored — using full Visium slide")

    srv  = sa.launch_aligner(xen, vis, port=args.port)
    host = socket.gethostname()

    print(f"\nAligner serving on port {srv.port}  ({host})")
    print("\nTo open from a remote machine (e.g. your laptop), run:")
    print(f"  ssh -L {srv.port}:{host}:{srv.port} {os.environ.get('USER','USER')}@ssh.ccv.brown.edu")
    print(f"\nThen open:  http://localhost:{srv.port}/")
    print("\nPress Ctrl-C to stop.")

    try:
        while True:
            time.sleep(1)
            if srv._event.is_set():
                result = srv.result
                print("\nTransform received from browser.")
                proj.set_transform(args.sample, result, source="manual")
                print(f"Saved as manual transform for {args.sample!r}")
                print("Press Ctrl-C to exit or keep the server running for more edits.")
    except KeyboardInterrupt:
        print("\nStopping server.")
        srv.stop()


# ---------------------------------------------------------------------------
# Subcommand: qc
# ---------------------------------------------------------------------------

def cmd_qc(args):
    import strata as sa
    loader  = _load_loader(args.loader)
    proj    = sa.Project(args.project, loader)
    samples = _parse_samples(args)
    df      = proj.qc_all(samples)
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"QC results saved to {args.out}")
    else:
        print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# Subcommand: preview-selections
# ---------------------------------------------------------------------------

def cmd_preview_selections(args):
    import strata as sa
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loader  = _load_loader(args.loader)
    proj    = sa.Project(args.project, loader)
    samples = _parse_samples(args)
    fig     = proj.preview_selections(samples, ncols=args.ncols)
    out     = args.out or "selections_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Subcommand: preview-alignments
# ---------------------------------------------------------------------------

def cmd_preview_alignments(args):
    import strata as sa
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loader  = _load_loader(args.loader)
    proj    = sa.Project(args.project, loader)
    samples = _parse_samples(args)
    fig     = proj.preview_alignments(samples, ncols=args.ncols)
    out     = args.out or "alignments_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Shared sample args helper
# ---------------------------------------------------------------------------

def _add_sample_args(p):
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--samples", nargs="+", metavar="SAMPLE",
                     help="Sample names (space-separated)")
    grp.add_argument("--samples-file", metavar="FILE",
                     help="Text file with one sample name per line")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="strata",
        description="Xenium × Visium spatial alignment toolkit",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── status ───────────────────────────────────────────────────────────────
    p_status = sub.add_parser("status", help="Show project status per sample")
    p_status.add_argument("--project",  required=True, help="Project directory")
    p_status.add_argument("--loader",   required=True, help="Path to loader.py")
    _add_sample_args(p_status)
    p_status.set_defaults(func=cmd_status)

    # ── propose ───────────────────────────────────────────────────────────────
    p_propose = sub.add_parser("propose", help="Auto-propose Visium spot selections")
    p_propose.add_argument("--project",    required=True)
    p_propose.add_argument("--loader",     required=True)
    p_propose.add_argument("--output-dir", default=None,
                           help="Save per-sample CSVs here (optional)")
    p_propose.add_argument("--force", action="store_true",
                           help="Re-run even for manually set samples")
    _add_sample_args(p_propose)
    p_propose.set_defaults(func=cmd_propose)

    # ── align ─────────────────────────────────────────────────────────────────
    p_align = sub.add_parser("align", help="Auto-align samples (sequential or SLURM)")
    p_align.add_argument("--project",   required=True)
    p_align.add_argument("--loader",    required=True)
    p_align.add_argument("--force", action="store_true")
    p_align.add_argument("--slurm", action="store_true",
                         help="Submit a SLURM job array instead of running locally")
    p_align.add_argument("--partition", default="batch")
    p_align.add_argument("--time",      default="60", help="Wall time in minutes")
    p_align.add_argument("--mem",       default="16G")
    p_align.add_argument("--account",   default=None)
    _add_sample_args(p_align)
    p_align.set_defaults(func=cmd_align)

    # ── view ──────────────────────────────────────────────────────────────────
    p_view = sub.add_parser("view", help="Open interactive aligner for one sample")
    p_view.add_argument("--project", required=True)
    p_view.add_argument("--loader",  required=True)
    p_view.add_argument("--sample",  required=True, help="Single sample name")
    p_view.add_argument("--port",    type=int, default=0,
                        help="Port to serve on (0 = auto-assign)")
    p_view.set_defaults(func=cmd_view)

    # ── qc ────────────────────────────────────────────────────────────────────
    p_qc = sub.add_parser("qc", help="Score alignments without ground truth")
    p_qc.add_argument("--project", required=True)
    p_qc.add_argument("--loader",  required=True)
    p_qc.add_argument("--out",     default=None, help="Save QC CSV to this path")
    _add_sample_args(p_qc)
    p_qc.set_defaults(func=cmd_qc)

    # ── preview-selections ────────────────────────────────────────────────────
    p_ps = sub.add_parser("preview-selections",
                          help="Thumbnail grid of auto-proposed selections")
    p_ps.add_argument("--project", required=True)
    p_ps.add_argument("--loader",  required=True)
    p_ps.add_argument("--ncols",   type=int, default=3)
    p_ps.add_argument("--out",     default=None, help="Output PNG path")
    _add_sample_args(p_ps)
    p_ps.set_defaults(func=cmd_preview_selections)

    # ── preview-alignments ────────────────────────────────────────────────────
    p_pa = sub.add_parser("preview-alignments",
                          help="Thumbnail grid of stored alignments")
    p_pa.add_argument("--project", required=True)
    p_pa.add_argument("--loader",  required=True)
    p_pa.add_argument("--ncols",   type=int, default=3)
    p_pa.add_argument("--out",     default=None, help="Output PNG path")
    _add_sample_args(p_pa)
    p_pa.set_defaults(func=cmd_preview_alignments)

    # ── dispatch ──────────────────────────────────────────────────────────────
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
