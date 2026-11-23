"""
Entry module of Quick Infill's worker processes (see process_pool.py). No bpy here.

A worker is Blender's own python.exe. It loads the add-on folder as the package "qi_core" without
running its __init__ (which needs bpy), so the pure-meshlib modules (offset_utils, trim_round, ...)
import as they do in Blender.
"""

import importlib
import os
import sys
import types

_trim = None


def _exit_with_parent(parent):
    """End this worker when the Blender that started it is gone. A Blender that crashes or is killed never shuts its
    pool down, and its workers stayed behind for good, each holding a CUDA context: 32 of them from crashed test runs
    took 12 GB of GPU memory on 2026-11-23. parent: Blender's process id (from process_pool; the worker's own idea of
    its parent is not reliable on Windows)."""
    import threading
    if not parent:
        return

    def watch():
        try:
            if os.name == "nt":
                import ctypes
                from ctypes import wintypes
                k32 = ctypes.WinDLL("kernel32", use_last_error=True)
                k32.OpenProcess.restype = wintypes.HANDLE
                k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
                k32.WaitForSingleObject.restype = wintypes.DWORD
                k32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
                h = k32.OpenProcess(0x00100000, False, int(parent))     # SYNCHRONIZE
                if not h:
                    return                                              # (cannot watch it: carry on as before)
                if k32.WaitForSingleObject(h, 0xFFFFFFFF) != 0:         # (0: the parent ended)
                    return
            else:
                import time
                while os.getppid() == int(parent):
                    time.sleep(2.0)
        except Exception:
            return
        os._exit(0)
    threading.Thread(target=watch, name="qi-parent-watch", daemon=True).start()


def init(addon_dir, parent_pid=None):
    global _trim
    _exit_with_parent(parent_pid)
    if "qi_core" not in sys.modules:
        pkg = types.ModuleType("qi_core")
        pkg.__path__ = [addon_dir]
        sys.modules["qi_core"] = pkg
    _trim = importlib.import_module("qi_core.trim_round")
    mm, _ = importlib.import_module("qi_core.meshlib_utils").get_meshlib()   # load meshlib + CUDA now


def ping(_=None):
    return os.getpid()


def trim_edges_job(args):
    return _trim.job(*args)   # kind: "voxel" (Round (old)) or "lean" (Round)


def trim_thin_job(args):
    return _trim.job(*args)   # kind "thin"
