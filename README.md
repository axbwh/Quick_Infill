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
- **Make Manifold** unites the selection into one watertight `<active>_Solid` object in the `STLs` collection
  (Blender's Manifold solver) and intersects it with the *Intersect With* object, if one is set. Unconnected islands
  are deleted. *Regrow* grows the pieces that reach the blank's surface a hair first, so they are cut once, fresh.
  Self-intersections and zero-size faces are reported, not repaired. Ashlarize shows the same button under
  Make Print Ready.
- *Min Drain Gap*: openings narrower than this count as closed. 0 is what a slicer reports.
- Sealed space that reaches the first layer is listed apart as *plate pockets* (`PC_PlatePockets`, pink): it is closed
  only by the build plate and opens when the print comes off. The method follows UVtools' resin trap / suction cup
  detection and the 3D Systems trapped-volume patent (US 5,945,058); see the notes at the top of `print_check.py`.

`fill_support.fill_and_support(mesh, wall_dir, plate_z)` runs the whole wall-face recipe (meshes A to H of the stone
builder notes) in memory; it has no button here, it is meant to be called by the stone packer.

## Smart scoop (Fix Undercuts)

Fix Undercuts builds its scoop as slopes tilted by the Angle. On a slope every printed layer steps
out past the one below by `layer * tan(angle)`, so the angle decides how the scoop prints. **Smart**
(Support Tools, on by default) asks for that step directly:

- **Stringing** is the step in mm per layer, counted for 0.2 mm layers: 0.2 (a 45 degree scoop, all
  green in Ashlarize's Undercuts overlay) to 0.6 (72 degrees, the outer line droops). The default,
  0.35, is the 60 degree scoop. Four buttons under the slider set 0.2, 0.35, 0.45 and 0.6, and a
  line under them says what the value prints like. Lower is a deeper scoop that hides more of the
  stone; higher is shallower and rougher. (`settings.smart_lip`, `smart_angle`, `stringing_status`.)
- **Several objects selected**: an object's scoop is the wedge under it, whatever is there, so in a
  wall it used to run into the stone below. With Smart each scoop stops at the other selected
  objects and goes half a voxel into what it meets, so it rests on the object below and joins it. A
  piece that another object cuts off from the rest is dropped. Each object is then worked whole,
  not island by island. (`undercut_volume.fix_undercuts_resting`, `rest_on_others`.)

Smart off gives the Angle and its presets back, and each object is worked on its own as before.
From View takes the angle from Stringing too, but leaves the other objects alone.
