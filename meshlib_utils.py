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
    """Return (mrmeshpy, mrcudapy), importing them once."""
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
        _modules = (mm, mc)
    return _modules


def get_mrmeshnumpy():
    get_meshlib()
    import meshlib.mrmeshnumpy as mn
    return mn


def get_mrmeshpy():
    return get_meshlib()[0]


def get_mrcudapy():
    return get_meshlib()[1]
