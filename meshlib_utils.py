"""
Meshlib import utilities for Quick Infill addon.
Handles wheel loading and provides safe imports.
"""


def get_meshlib():
    """
    Import and return meshlib modules with proper error handling.
    Returns (mrmeshpy, mrcudapy) tuple.
    """
    # Use the main module's ensure_wheels_loaded which handles DLL directories
    from . import ensure_wheels_loaded
    try:
        ensure_wheels_loaded()
    except ImportError:
        pass  # Will fail below with better error
    
    try:
        import meshlib.mrmeshpy as mm
        import meshlib.mrcudapy as mc
        return mm, mc
    except ImportError as e:
        raise ImportError(f"Quick Infill: Failed to load meshlib - {str(e)}")


def get_mrmeshpy():
    """Get just the mrmeshpy module."""
    mm, _ = get_meshlib()
    return mm

def get_mrcudapy():
    """Get just the mrcudapy module."""
    _, mc = get_meshlib()
    return mc