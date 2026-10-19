"""
launch.py
---------
Serve the browser tools (aligner, selector) into a new tab and catch what they
send back into the kernel. Two mechanisms, because remote Jupyter setups vary:

  launch_*  -> kernel-side HTTP server; the tab POSTs the result straight back.
               Needs the tab to reach the kernel's port (local Jupyter, or
               remote with jupyter-server-proxy). Zero files.
  save_*    -> a self-contained HTML file (data baked in) opened through
               Jupyter's own file serving; the Send button downloads the result
               to re-upload. Works anywhere Jupyter can serve a file.

Both tools share one generic server/file core; aligner vs selector differ only
in which HTML they serve, what data they're handed, and how the returned JSON
is interpreted.

OOD / jupyter-server-proxy support
-----------------------------------
When running under Jupyter on an HPC Open OnDemand node the browser cannot
reach localhost:PORT directly.  jupyter-server-proxy exposes a kernel-side
port through the Jupyter server itself at:

    {base_url}proxy/{port}/

where {base_url} is the notebook's own base URL (e.g.
/node/compute001/8888/ on OOD, or / locally).

_jupyter_base_url() auto-detects this at runtime:
  1. JUPYTERHUB_SERVICE_PREFIX env var (set by JupyterHub/OOD)
  2. JPY_BASE_URL env var (older JupyterHub)
  3. Running notebook server's base_url via the jupyter_server API
  4. Falls back to / (works for localhost)

Install jupyter-server-proxy once in the env:
  pip install jupyter-server-proxy
Then restart the Jupyter server (not just the kernel).
"""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .io.loaders import PointSet
from .io.aligner_export import build_aligner_csvs
from .transform import apply_aligner_transform

_HERE = os.path.dirname(__file__)
_ALIGNER_HTML = os.path.join(_HERE, "aligner.html")
_SELECTOR_HTML = os.path.join(_HERE, "selector.html")


# ---------------------------------------------------------------------------
# OOD / jupyter-server-proxy URL detection
# ---------------------------------------------------------------------------

def _jupyter_base_url() -> str:
    """Return the Jupyter server's base URL (always ends with '/').

    Detection order (Brown OOD / classic notebook):
      1. JUPYTERHUB_SERVICE_PREFIX / JPY_BASE_URL / NB_PREFIX env vars
      2. JPY_PARENT_PID cmdline → --config= path → parse c.NotebookApp.base_url
         (most reliable for Brown OOD: base_url = /node/<host>/<port>/)
      3. jupyter_server list_running_servers() API
      4. '/' fallback (localhost)
    """
    import re as _re

    # 1. Env vars
    for var in ("JUPYTERHUB_SERVICE_PREFIX", "JPY_BASE_URL", "NB_PREFIX"):
        v = os.environ.get(var, "").strip()
        if v:
            return v.rstrip("/") + "/"

    # 2. JPY_PARENT_PID → cmdline → --config= → parse base_url
    #    Brown OOD sets JPY_PARENT_PID to the Jupyter server PID and passes
    #    --config=/oscar/home/.../config.py which contains c.NotebookApp.base_url
    pid = os.environ.get("JPY_PARENT_PID", "").strip()
    if pid:
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as _f:
                cmdline = _f.read().decode(errors="replace")
            args = cmdline.split("\x00") if "\x00" in cmdline else cmdline.split("\0")
            # also handle space-separated (some kernels)
            args = [a.strip() for a in _re.split(r"[\x00 ]", cmdline) if a.strip()]
            cfg_path = None
            for arg in args:
                m = _re.search(r"--config=(.+)", arg)
                if m:
                    cfg_path = m.group(1).strip()
                    break
            if cfg_path and os.path.exists(cfg_path):
                cfg_text = open(cfg_path, encoding="utf-8").read()
                # parse: c.NotebookApp.base_url = '/node/.../' or c.ServerApp.base_url
                m = _re.search(r'c\.(?:NotebookApp|ServerApp)\.base_url\s*=\s*[\'"]([^\'"]+)[\'"]', cfg_text)
                if m:
                    base = m.group(1)
                    return base if base.endswith("/") else base + "/"
        except Exception:
            pass

    # 3. jupyter_server API
    try:
        from jupyter_server.serverapp import list_running_servers as _ls
        for srv in _ls():
            base = srv.get("base_url", "/")
            if base and base != "/":
                return base if base.endswith("/") else base + "/"
    except Exception:
        pass

    return "/"


def _proxy_url(port: int) -> str | None:
    """Return the jupyter-server-proxy URL for this port, or None if
    jupyter-server-proxy is not installed."""
    try:
        import jupyter_server_proxy  # noqa: F401 — just checking it's installed
        base = _jupyter_base_url()
        return f"{base}proxy/{port}/"
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# generic kernel-side server
# ---------------------------------------------------------------------------

class _ToolServer:
    def __init__(self, html: str, data: dict, port: int = 0):
        self._html = html
        self._data = json.dumps(data)
        self.result = None
        self._event = threading.Event()

        # SSE alignment state (used by AlignerServer subclass)
        self._sse_clients: list = []
        self._sse_lock = threading.Lock()
        self._align_running = False

        server_self = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype, extra_headers=None):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                if extra_headers:
                    for k, v in extra_headers.items():
                        self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def do_OPTIONS(self):
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()

            def do_GET(self):
                path = self.path.split("?")[0].rstrip("/")
                if path in ("", "/index.html", "/aligner.html", "/selector.html"):
                    self._send(200, server_self._html.encode("utf-8"),
                               "text/html; charset=utf-8")
                elif path == "/data.json":
                    self._send(200, server_self._data.encode("utf-8"),
                               "application/json")
                elif path == "/align_events":
                    # SSE stream for alignment progress
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("X-Accel-Buffering", "no")
                    self.end_headers()
                    q = []
                    with server_self._sse_lock:
                        server_self._sse_clients.append(q)
                    try:
                        while True:
                            if q:
                                msg = q.pop(0)
                                data = json.dumps(msg).replace("\n", " ")
                                self.wfile.write(
                                    f"data: {data}\n\n".encode("utf-8"))
                                self.wfile.flush()
                                if msg.get("done"):
                                    break
                            else:
                                time.sleep(0.05)
                    except Exception:
                        pass
                    finally:
                        with server_self._sse_lock:
                            try:
                                server_self._sse_clients.remove(q)
                            except ValueError:
                                pass
                else:
                    self._send(404, b"not found", "text/plain")

            def do_POST(self):
                p = self.path.rstrip("/")
                if p.endswith("/transform") or p.endswith("/result"):
                    n = int(self.headers.get("Content-Length", 0))
                    try:
                        server_self.result = json.loads(self.rfile.read(n))
                        server_self._event.set()
                        self._send(200, b'{"ok":true}', "application/json")
                    except Exception as e:
                        self._send(400,
                                   json.dumps({"ok": False, "error": str(e)}).encode(),
                                   "application/json")
                elif p.endswith("/run_align"):
                    # Trigger auto-alignment in a background thread
                    n = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(n)) if n else {}
                    if server_self._align_running:
                        self._send(409, b'{"error":"alignment already running"}',
                                   "application/json")
                        return
                    threading.Thread(
                        target=server_self._run_alignment,
                        kwargs={"params": body},
                        daemon=True,
                    ).start()
                    self._send(202, b'{"ok":true,"message":"alignment started"}',
                               "application/json")
                else:
                    self._send(404, b"not found", "text/plain")

        self._httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def _broadcast_sse(self, msg: dict):
        """Push a message to all connected SSE clients."""
        with self._sse_lock:
            for q in self._sse_clients:
                q.append(msg)

    def _run_alignment(self, params: dict):
        """Override in subclasses (AlignerServer) to run auto_align and stream
        progress via SSE.  Base class does nothing."""
        self._broadcast_sse({"done": True, "error": "no alignment configured"})

    def _candidate_urls(self):
        urls = []
        proxy = _proxy_url(self.port)
        if proxy:
            urls.append(proxy)
        # Always also offer the plain prefix-based path and localhost as fallbacks
        prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "").strip()
        if prefix and not proxy:
            urls.append(f"{prefix.rstrip('/')}/proxy/{self.port}/")
        if not proxy:
            urls.append(f"/proxy/{self.port}/")
        urls.append(f"http://localhost:{self.port}/")
        return urls

    def _links_html(self, title, note, extra=""):
        proxy = _proxy_url(self.port)
        if proxy:
            proxy_note = (
                f'<div style="color:#5cb85c;font-size:12px;margin-top:4px">'
                f'✓ jupyter-server-proxy detected — use the first link on OOD</div>'
            )
        else:
            proxy_note = (
                '<div style="color:#e8a838;font-size:12px;margin-top:4px">'
                '⚠ jupyter-server-proxy not found — install it for OOD support, '
                'or use save_aligner() instead</div>'
            )
        links = "".join(
            f'<div style="margin:3px 0"><a href="{u}" target="_blank" '
            f'style="font-family:monospace;color:#66c2a5">{u}</a></div>'
            for u in self._candidate_urls()
        )
        return (
            '<div style="padding:10px 14px;border:1px solid #24292c;border-radius:10px;'
            'background:#15181a;color:#e9ebea;font-family:sans-serif;max-width:660px">'
            f'<b>{title}</b> Open it in a new tab (click whichever link connects):{links}'
            f'{proxy_note}'
            f'<div style="color:#7f868b;font-size:12px;margin-top:6px">port {self.port} · {note}</div>'
            f'{extra}</div>'
        )

    def wait(self, timeout=None) -> bool:
        return self._event.wait(timeout)

    def stop(self):
        self._httpd.shutdown()


# ---------------------------------------------------------------------------
# generic self-contained file
# ---------------------------------------------------------------------------

class _ToolFile:
    def __init__(self, html_path: str, data: dict, out_path: str):
        html = open(html_path, encoding="utf-8").read()
        payload = json.dumps(data).replace("</", "<\\/")
        embed = ("var _d=" + payload + ";"
                 "if(_d.xenium_csv){loadXen(_d.xenium_csv);}"
                 "if(_d.visium_csv){loadVis(_d.visium_csv);}"
                 "if(_d.csv){parseCSV(_d.csv);if(_d.preselect)applyPreselect(_d.preselect);}"
                 "if(_d.auto_align_trajectory){setTimeout(function(){playBakedTrajectory(_d.auto_align_trajectory);},400);}")
        html = html.replace("/*__SPATIALIGN_EMBED__*/", embed)
        self.path = os.path.abspath(out_path)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(html)
        self.result = None

    def _link_html(self, title, note):
        rel = os.path.relpath(self.path)
        try:
            from IPython.display import FileLink
            link = FileLink(rel)._repr_html_()
        except Exception:
            link = f"<code>{self.path}</code>"
        return (
            '<div style="padding:10px 14px;border:1px solid #24292c;border-radius:10px;'
            'background:#15181a;color:#e9ebea;font-family:sans-serif;max-width:680px">'
            f'<b>{title}</b> Open it in a new tab:<div style="margin:6px 0">{link}</div>'
            f'<div style="color:#7f868b;font-size:12.5px;line-height:1.5">{note}</div></div>'
        )


# ---------------------------------------------------------------------------
# ALIGNER
# ---------------------------------------------------------------------------

def _apply_aligner_result(result, vis: PointSet) -> PointSet:
    if result is None:
        raise RuntimeError("No transform received yet. Align and click Send, then retry.")
    new_coords = apply_aligner_transform(vis.coords, vis.ids, result)
    return vis.with_coords(new_coords)


class AlignerServer(_ToolServer):
    def __init__(self, xen, vis, n_fiducials=6, xen_cap=60000, port=0, seed=0):
        xen_csv, vis_csv, rank, fids = build_aligner_csvs(xen, vis, n_fiducials, xen_cap, seed)
        super().__init__(open(_ALIGNER_HTML, encoding="utf-8").read(),
                         {"xenium_csv": xen_csv, "visium_csv": vis_csv}, port=port)
        self.fiducials, self.rank_table = fids, rank
        self._xen = xen
        self._vis = vis

    def _run_alignment(self, params: dict):
        """Run auto_align() in a background thread, streaming each FGW
        iteration as an SSE frame to the browser."""
        from .align import auto_align
        from .refine import RefineHistory
        import numpy as np

        self._align_running = True
        try:
            n_iter = params.get("n_iter", 12)
            xen_sub = self._xen.subsample(min(2000, self._xen.n))
            vis_sub = self._vis.subsample(min(1000, self._vis.n))

            # Patch RefineHistory to broadcast each iteration via SSE
            original_refine = None
            import strata.refine as _refine_mod
            original_refine_fn = _refine_mod.refine

            captured = {"history": None}

            def _streaming_refine(moving, fixed, n_iter=n_iter, verbose=False, **kw):
                from .refine import _alpha_at, RefineHistory, AffineTransform, TPSTransform
                from scipy.spatial.distance import cdist
                import ot.gromov as gromov

                shared = moving.shared_features(fixed)
                mov_feat = moving.subset_features(shared).features
                fix_feat = fixed.subset_features(shared).features
                fixed_coords = fixed.coords

                current = moving.coords.copy()
                trajectory, entropies, displacements, alphas = [], [], [], []
                p = np.full(current.shape[0], 1.0 / current.shape[0])
                q_w = np.full(fixed_coords.shape[0], 1.0 / fixed_coords.shape[0])
                rigid_iters = kw.get("rigid_iters", 6)
                tps_smoothing = kw.get("tps_smoothing", 10.0)
                allow_scale = kw.get("allow_scale", False)
                tol = kw.get("tol", 1e-3)
                alpha_sched = kw.get("alpha", (0.85, 0.3))

                for i in range(n_iter):
                    a = _alpha_at(alpha_sched, i, n_iter)
                    alphas.append(a)
                    C1 = cdist(current, current)
                    C2 = cdist(fixed_coords, fixed_coords)
                    M = cdist(mov_feat, fix_feat)
                    C1 /= (C1.max() or 1.0)
                    C2 /= (C2.max() or 1.0)
                    M /= (M.max() or 1.0)
                    T = gromov.fused_gromov_wasserstein(M, C1, C2, p, q_w, alpha=a)
                    row_sums = T.sum(1, keepdims=True)
                    row_sums[row_sums == 0] = 1e-12
                    target = (T @ fixed_coords) / row_sums
                    row_p = T / row_sums
                    ent = -(row_p * np.log(row_p + 1e-12)).sum(1).mean()
                    entropies.append(float(ent))
                    if i < rigid_iters:
                        tf = AffineTransform.fit(current, target, allow_flip=False,
                                                  allow_scale=allow_scale)
                    else:
                        tf = TPSTransform(control_source=current.copy(),
                                          control_target=target.copy(),
                                          smoothing=tps_smoothing)
                    new_coords = tf.apply(current)
                    disp = float(np.mean(np.linalg.norm(new_coords - current, axis=1)))
                    displacements.append(disp)
                    trajectory.append(tf)
                    current = new_coords

                    # Broadcast this frame to all SSE clients
                    self._broadcast_sse({
                        "iter": i + 1,
                        "total": n_iter,
                        "mean_disp": round(disp, 4),
                        "alpha": round(a, 3),
                        "coords_moving": current.tolist(),
                        "done": False,
                    })

                    if disp < tol:
                        break

                return RefineHistory(
                    trajectory=trajectory, coupling_entropy=entropies,
                    mean_displacement=displacements, alpha_schedule=alphas,
                )

            hist = _streaming_refine(xen_sub, vis_sub)
            # Apply the full trajectory to the original subsampled coords to get
            # the true final aligned positions (compose is the correct path per
            # refine.py docs — _streaming_refine never mutates xen_sub.coords)
            from .transform import compose as _compose
            final_coords = _compose(xen_sub.coords, hist.trajectory)
            # Broadcast completion with final coords and trajectory summary
            traj_dicts = []
            for t in hist.trajectory:
                try:
                    traj_dicts.append(t.to_dict())
                except Exception:
                    pass
            self._broadcast_sse({
                "done": True,
                "iter": len(hist.trajectory),
                "total": len(hist.trajectory),
                "coords_moving": final_coords.tolist(),
                "n_iters": len(hist.trajectory),
                "final_mean_disp": round(hist.mean_displacement[-1], 4) if hist.mean_displacement else 0,
            })
            self._history = hist

        except Exception as e:
            self._broadcast_sse({"done": True, "error": str(e)})
        finally:
            self._align_running = False

    @property
    def transform(self):
        return self.result

    def _repr_html_(self):
        return self._links_html(
            "Aligner is serving.",
            f"fiducials: {', '.join(self.fiducials)}",
            extra=(
                '<div style="color:#aaa;font-size:12px;margin-top:6px">'
                'Click <b>▶ Run Auto-Align</b> in the tool to run FGW alignment '
                'and watch the animation, then drag manually to refine.</div>'
            )
        )

    def aligned_visium(self, vis):
        return _apply_aligner_result(self.result, vis)


class AlignerFile(_ToolFile):
    def __init__(self, xen, vis, path, n_fiducials=6, xen_cap=60000, seed=0,
                 trajectory=None):
        xen_csv, vis_csv, rank, fids = build_aligner_csvs(xen, vis, n_fiducials, xen_cap, seed)
        data = {"xenium_csv": xen_csv, "visium_csv": vis_csv}
        if trajectory is not None:
            # Serialize trajectory frames for baked-in animation playback.
            # Each frame: the Xenium coords after applying transforms[:i+1].
            from .transform import compose as _compose
            frames = []
            for i in range(len(trajectory.trajectory)):
                coords = _compose(xen.coords, trajectory.trajectory[:i+1])
                frames.append(coords.tolist())
            data["auto_align_trajectory"] = {
                "frames": frames,
                "n_iters": len(frames),
                "mean_displacement": trajectory.mean_displacement,
                "alpha_schedule": trajectory.alpha_schedule,
            }
        super().__init__(_ALIGNER_HTML, data, path)
        self.fiducials, self.rank_table, self.transform = fids, rank, None

    def _repr_html_(self):
        traj_note = (
            " Auto-align trajectory baked in — animation plays on open. "
            "Refine manually, then Send."
            if hasattr(self, '_has_trajectory') and self._has_trajectory
            else ""
        )
        return self._link_html(
            "Aligner written.",
            "Align, then click <b>Send transform to notebook</b> — this downloads "
            "<code>visium_transform.json</code>. Upload it back and run "
            "<code>srv.load_transform('visium_transform.json')</code>.<br>"
            f"fiducials: {', '.join(self.fiducials)}{traj_note}")

    def load_transform(self, path):
        for c in [path, os.path.join(os.getcwd(), os.path.basename(path)),
                  os.path.join(os.path.expanduser("~"), os.path.basename(path))]:
            if os.path.exists(c):
                self.transform = json.load(open(c)); return self
        raise FileNotFoundError(
            f"{path!r} not found; upload it to your Jupyter dir, "
            "or use set_transform(json_string).")

    def set_transform(self, obj):
        self.transform = json.loads(obj) if isinstance(obj, str) else obj
        return self

    def aligned_visium(self, vis):
        return _apply_aligner_result(self.transform, vis)


def launch_aligner(xen, vis, n_fiducials=6, xen_cap=60000, port=0, seed=0):
    """Serve the aligner (kernel-side server).  On OOD/HPC, requires
    jupyter-server-proxy to be installed in the env and the Jupyter server
    restarted.  Falls back gracefully to localhost URLs for local use."""
    return AlignerServer(xen, vis, n_fiducials, xen_cap, port, seed)


def save_aligner(xen, vis, path=None, n_fiducials=6, xen_cap=60000, seed=0,
                 trajectory=None):
    """Write a self-contained aligner HTML; return via download + re-upload.

    trajectory : optional RefineHistory from auto_align() or refine().
                 If provided, the animation frames are baked into the HTML
                 and auto-play when the tool opens — no server proxy needed.
                 After playback the canvas stays live for manual refinement.
    """
    if path is None:
        path = f"aligner_{np.random.randint(1e6):06d}.html"
    return AlignerFile(xen, vis, path, n_fiducials, xen_cap, seed, trajectory=trajectory)


# ---------------------------------------------------------------------------
# SELECTOR
# ---------------------------------------------------------------------------

def _selector_csv(ps: PointSet, n_channels=6, cap=200000):
    """Build a barcode,x,y,<genes> CSV for the selector from a PointSet."""
    import io as _io, pandas as pd
    p = ps.subsample(cap) if ps.n > cap else ps
    df = pd.DataFrame({"barcode": p.ids, "x": p.coords[:, 0], "y": p.coords[:, 1]})
    for g in p.feature_names[:n_channels]:
        df[g] = p.feature(g)
    buf = _io.StringIO(); df.to_csv(buf, index=False); return buf.getvalue()


def _result_barcodes(result):
    if result is None:
        raise RuntimeError("No selection received yet. Select and click Send, then retry.")
    return np.asarray(result["barcodes"], dtype=str)


class SelectorServer(_ToolServer):
    def __init__(self, ps, preselect=None, n_channels=6, port=0):
        data = {"csv": _selector_csv(ps, n_channels)}
        if preselect is not None:
            data["preselect"] = np.asarray(preselect, dtype=str).tolist()
        super().__init__(open(_SELECTOR_HTML, encoding="utf-8").read(), data, port=port)
        self.platform = ps.platform

    def _repr_html_(self):
        return self._links_html(f"Selector is serving ({self.platform}).",
                                "lasso the region; Send returns the barcodes")

    def selected_ids(self):
        return _result_barcodes(self.result)

    def selected(self, ps):
        return ps.subset_by_ids(self.selected_ids())


class SelectorFile(_ToolFile):
    def __init__(self, ps, path, preselect=None, n_channels=6):
        data = {"csv": _selector_csv(ps, n_channels)}
        if preselect is not None:
            data["preselect"] = np.asarray(preselect, dtype=str).tolist()
        super().__init__(_SELECTOR_HTML, data, path)
        self.platform = ps.platform
        self.result = None

    def _repr_html_(self):
        return self._link_html(
            f"Selector written ({self.platform}).",
            "Lasso the region, then click <b>Send selection to notebook</b> — this downloads "
            "<code>selected_spots.csv</code>. Upload it back and run "
            "<code>sel.load_selection('selected_spots.csv')</code>.")

    def load_selection(self, path):
        import pandas as pd
        for c in [path, os.path.join(os.getcwd(), os.path.basename(path)),
                  os.path.join(os.path.expanduser("~"), os.path.basename(path))]:
            if os.path.exists(c):
                self.result = {"barcodes": pd.read_csv(c)["barcode"].astype(str).tolist()}
                return self
        raise FileNotFoundError(f"{path!r} not found; upload it to your Jupyter dir.")

    def selected_ids(self):
        return _result_barcodes(self.result)

    def selected(self, ps):
        return ps.subset_by_ids(self.selected_ids())


def launch_selector(ps, preselect=None, n_channels=6, port=0):
    """Serve the spot selector (kernel-side server) for a PointSet (Visium OR
    Xenium).  On OOD/HPC requires jupyter-server-proxy."""
    return SelectorServer(ps, preselect, n_channels, port)


def save_selector(ps, path=None, preselect=None, n_channels=6):
    """Write a self-contained spot selector for a PointSet (Visium OR Xenium)."""
    if path is None:
        path = f"selector_{np.random.randint(1e6):06d}.html"
    return SelectorFile(ps, path, preselect, n_channels)
