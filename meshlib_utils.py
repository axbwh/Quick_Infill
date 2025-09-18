"""
Meshlib import utilities for Quick Infill addon.
Handles wheel loading and provides safe imports.
"""

def get_meshlib():
    """
    Import and return meshlib modules with proper error handling.
    Returns (mrmeshpy, mrcudapy) tuple.
    """
    try:
        import meshlib.mrmeshpy as mm
        import meshlib.mrcudapy as mc
        return mm, mc
    except ImportError as e:
        # Try to load from the main init module's wheel loader
        try:
            from . import ensure_wheels_loaded
            ensure_wheels_loaded()
            import meshlib.mrmeshpy as mm
            import meshlib.mrcudapy as mc
            return mm, mc
        except:
            raise ImportError(f"Quick Infill: Failed to load meshlib - {str(e)}")

def get_mrmeshpy():
    """Get just the mrmeshpy module."""
    mm, _ = get_meshlib()
    return mm

def get_mrcudapy():
    """Get just the mrcudapy module."""
    _, mc = get_meshlib()
    return mc