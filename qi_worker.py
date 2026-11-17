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


def init(addon_dir):
    global _trim
    if "qi_core" not in sys.modules:
        pkg = types.ModuleType("qi_core")
        pkg.__path__ = [addon_dir]
        sys.modules["qi_core"] = pkg
    _trim = importlib.import_module("qi_core.trim_round")
    mm, _ = importlib.import_module("qi_core.meshlib_utils").get_meshlib()   # load meshlib + CUDA now


def ping(_=None):
    return os.getpid()


def trim_edges_job(args):
    return _trim.job(*args)
