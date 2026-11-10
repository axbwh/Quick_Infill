# Quick Infill — Dev Setup

This repo is a Blender add-on that uses MeshLib (mrmeshpy, mrviewerpy, mrcudapy) via Python.

## What you get
- A local Python virtual environment workflow for linting and quick scripts
- Scripts to install meshlib into Blender's bundled Python on Windows
- A solid .gitignore and VS Code settings to use .venv

## Prereqs
- Windows with PowerShell
- Python 3.10+ installed and on PATH (for local venv)
- Blender 4.0+ installed (manifest targets 4.2)

## 1) Local venv (for dev tooling)
This lets you run tools like ruff/mypy and experiment with code outside Blender.

```powershell
# From repo root
./scripts/setup-venv.ps1
# Activate for this session
. ./.venv/Scripts/Activate.ps1
```

This installs:
- meshlib (PyPI) — modules used by heal_cavity.py
- ruff, mypy (optional)

## 2) Bundle meshlib wheel for Blender 4.5 (recommended)
Following Blender's official guidance, we ship a wheel in `wheels/` and reference it in `blender_manifest.toml`.

```powershell
# Auto-detect Blender 5.2 and download a compatible meshlib wheel into wheels/
./scripts/fetch-meshlib-wheel.ps1

# Or specify a custom Blender path or meshlib version
./scripts/fetch-meshlib-wheel.ps1 -BlenderPath "C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe" -MeshlibVersion 2.2.2

# Package the extension for distribution
./scripts/package-extension.ps1
```

This updates `blender_manifest.toml` with a `wheels = ["./wheels/<file>.whl"]` entry.

## 3) Using in Blender
- Copy/symlink this folder into a path Blender recognizes for add-ons or use the Extensions manager
- Ensure CUDA is available if you plan to use mrcudapy paths in heal_cavity.py

## Tips
- Some NVIDIA/CUDA features require NVIDIA drivers; check output of `mc.isCudaAvailable()` in the console
- If you still prefer installing into Blender's Python for local dev, the helper remains available:
	```powershell
	./scripts/install-meshlib-into-blender.ps1
	```

## Linting
```powershell
. ./.venv/Scripts/Activate.ps1
ruff check .
```

## License
GPL-2.0-or-later (see blender_manifest.toml).

## Print Check, Seal Cavities, Make Manifold (1.2.0)

Sidebar > Quick Infill > Print Check. Select everything that prints as one piece (stones, blank, infill) first.

- **Print Check** reads the selection the way a resin slicer does, layer by layer from the build plate, and reports
  sealed cavities (trapped resin), suction cups, islands, ledges and plate contact. Problem areas are added as coloured
  objects in the `Print Check` collection (`PC_Sealed` red, `PC_Cups` orange, `PC_Islands` yellow, `PC_Ledges` purple,
  `PC_NoContact` blue).
- **Seal Cavities** adds a `<active>_Plugs` object that fills every sealed cavity. Sealed cavities cannot be seen from
  outside, so nothing visible changes.
- **Make Manifold** unites the selection into one watertight `<active>_Solid` object (Blender's Manifold solver on
  clean copies), repairs self-intersections and zero-size faces with meshlib, and makes the flat cuts LAST
  (*Intersect With* a blank, *Cut at Plate*), so a repair can never pull the bottom out of flat. It reports how much
  of the bottom lies on the plate.
- *Min Drain Gap*: openings narrower than this count as closed. 0 is what a slicer reports.

`fill_support.fill_and_support(mesh, wall_dir, plate_z)` runs the whole wall-face recipe (meshes A to H of the stone
builder notes) in memory; it has no button here, it is meant to be called by the stone packer.
