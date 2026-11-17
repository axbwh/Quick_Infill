"""
Worker processes for Quick Infill tools that work on many objects one by one (Trim Edges, Trim Thin).

Why: meshlib holds Python's lock for the whole of a call, so threads can't run two objects at
once (measured: 4 threads = 1 thread). Separate processes can: on 140 stones Trim Edges ran
2.4 times faster with 8 workers, with results identical bit for bit (2026-11-18).

The workers are Blender's own python.exe, started once and kept for the session (start-up is
about a second). Anything that goes wrong with them (no python.exe, a worker dies) and the
caller falls back to working in Blender as before.
"""

import glob
import importlib.util
import os
import sys
import threading

ADDON_DIR = os.path.dirname(os.path.abspath(__file__))
# Below this many objects the start-up and the copying are not worth it.
MIN_JOBS = int(os.environ.get("QI_WORKER_MIN_JOBS", "12"))   # env: for tests
WORKERS = max(2, min(8, (os.cpu_count() or 4) // 4))

_pool = None
_worker_mod = None
_lock = threading.Lock()
_disabled = False


def _python_exe():
    names = ("python.exe",) if os.name == "nt" else ("python3", "python")
    for n in names:
        p = os.path.join(sys.prefix, "bin", n)
        if os.path.isfile(p):
            return p
    found = sorted(glob.glob(os.path.join(sys.prefix, "bin", "python3*")))
    return found[0] if found else None


def worker_module():
    """qi_worker, imported under that plain name, so its functions pickle by a name the workers know."""
    global _worker_mod
    if _worker_mod is None:
        mod = sys.modules.get("qi_worker")
        if mod is None or os.path.dirname(os.path.abspath(getattr(mod, "__file__", ""))) != ADDON_DIR:
            spec = importlib.util.spec_from_file_location("qi_worker", os.path.join(ADDON_DIR, "qi_worker.py"))
            mod = importlib.util.module_from_spec(spec)
            sys.modules["qi_worker"] = mod
            spec.loader.exec_module(mod)
        _worker_mod = mod
    return _worker_mod


def get_pool():
    """The running pool, started on first use; None if workers can't be used here."""
    global _pool, _disabled
    if _disabled:
        return None
    with _lock:
        if _pool is not None:
            return _pool
        exe = _python_exe()
        if exe is None:
            _disabled = True
            return None
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        wm = worker_module()
        ctx = multiprocessing.get_context("spawn")
        ctx.set_executable(exe)
        # A spawned worker re-runs the parent's __main__ script if it has a file (Blender's
        # --python script): hide it while the workers start. The add-on folder goes on the path
        # only while they start, so "import qi_worker" works in them.
        main = sys.modules.get("__main__")
        saved = {k: getattr(main, k) for k in ("__file__", "__spec__") if main is not None and hasattr(main, k)}
        sys.path.insert(0, ADDON_DIR)
        try:
            for k in saved:
                setattr(main, k, None) if k == "__spec__" else delattr(main, k)
            pool = ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx, initializer=wm.init,
                                       initargs=(ADDON_DIR,))
            # start every worker now, while the path and __main__ are set up for it
            list(pool.map(wm.ping, range(WORKERS * 2)))
        except Exception as e:
            print(f"[Quick Infill] worker processes unavailable, working in Blender: {e}")
            _disabled = True
            return None
        finally:
            for k, v in saved.items():
                setattr(main, k, v)
            try:
                sys.path.remove(ADDON_DIR)
            except ValueError:
                pass
        _pool = pool
        return _pool


def reset():
    """Drop a broken pool; the next get_pool() starts a new one."""
    global _pool
    with _lock:
        pool, _pool = _pool, None
    if pool is not None:
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


def shutdown():
    reset()
