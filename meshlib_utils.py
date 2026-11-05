"""
Meshlib import utilities for Quick Infill.

meshlib's own package __init__ registers its bundled DLL folder
(meshlib_core.libs) with Windows, so nothing extra is needed here. The add-on
used to add DLL folders to PATH by hand; that found the wrong (older, other
Python version) meshlib build on Blender 5.2 and has been removed.
"""

import threading

_modules = None

# CUDA work (FastWindingNumber offsets) must not run concurrently: several
# simultaneous CUDA jobs on the GPU that also drives the display can trip a
# driver timeout (TDR) and freeze the whole machine. CPU-side meshlib work
# (decimation, voxel booleans, undercut fixing) may still run in parallel.
CUDA_LOCK = threading.Lock()


def get_meshlib():
    """Return (mrmeshpy, mrcudapy), importing them once (on first use, not at startup)."""
    global _modules
    if _modules is None:
        try:
            import meshlib.mrmeshpy as mm
            import meshlib.mrcudapy as mc
        except ImportError as e:
            raise ImportError(
                f"Quick Infill: failed to load meshlib - {e}\n"
                "Reinstall the add-on via Edit > Preferences > Get Extensions > Install from Disk"
            )
        _prevent_quit_hang(mm)
        _modules = (mm, mc)
    return _modules


def _prevent_quit_hang(mm):
    """Stop Blender from hanging when it quits after meshlib was loaded.

    meshlib installs an exception handler that prints a stack trace using the
    Windows debug engine (dbgeng). When Blender quits, a harmless exception
    inside a thread that is shutting down reaches that handler while Windows
    holds its DLL loader lock; dbgeng then initialises for the first time,
    needs that same lock, and the process deadlocks (Blender never closes).
    Initialising dbgeng once now, outside the lock, avoids it: in testing,
    quitting hung in 2 of 4 normal and 4 of 5 background runs without this,
    and in 0 of 14 runs with it. Costs about 0.6 s, once per session.
    """
    try:
        mm.getCurrentStacktrace()
    except Exception:
        pass


def get_mrmeshnumpy():
    get_meshlib()
    import meshlib.mrmeshnumpy as mn
    return mn


def get_mrmeshpy():
    return get_meshlib()[0]


def get_mrcudapy():
    return get_meshlib()[1]
