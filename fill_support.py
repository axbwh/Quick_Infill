"""
Fill and Support: Alex's wall-face recipe as one in-memory meshlib run.

It is the manual Quick Infill sequence from the stone builder notes (meshes A
to H), with no Blender round trips between the steps:

    A  stones (+ support wall) grown and shrunk 0.2 at voxel 0.2
    B  Fix Undercuts "from view" 90 degrees, +Y and -Y only, voxel 0.3,
       then grown and shrunk 0.7
    C  Fix Undercuts "from view" 60 degrees, every direction but +X / -X,
       voxel 0.3, Auto Shrink 0.2
    D  B intersect C
    E  Cavity Infill (Accurate) on D + A
    F  E + D grown and shrunk 0.2
    G  bottom 4 mm strip of F, Fix Undercuts 45 degrees all directions, voxel 0.2
    F' F shrunk 0.2
    H  F' + G grown and shrunk 0.2

"From view" means looking straight down at the wall with the view's X axis
running along the wall, so +Y / -Y point through the wall. `wall_dir` is that
X axis in world space.

No bpy in here: pure meshlib, usable from a worker thread or from another
add-on (the stone packer's Fill and Support button).

Sizes are in meshlib space (mm for Alex's files: Blender units x 10).
"""

import math
import time

import numpy as np

from .meshlib_utils import get_meshlib, check_cancelled
from .offset_utils import cuda_offset, decimate_mesh, face_count, INFILL_PRESETS
from .support_tools import fix_undercuts_multi, intersect_meshes, DIAG, Z_UP
from .heal_cavity import build_infill

X_POS, X_NEG, Y_POS, Y_NEG = (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)
DIAGONALS = [(DIAG, DIAG, 0), (DIAG, -DIAG, 0), (-DIAG, DIAG, 0), (-DIAG, -DIAG, 0)]


class Recipe:
    """The numbers of the recipe. Defaults are Alex's manual settings."""

    def __init__(self):
        self.a_voxel, self.a_dist = 0.2, 0.2
        self.b_voxel, self.b_angle, self.b_dist = 0.3, 90.0, 0.7
        self.c_voxel, self.c_angle, self.c_auto_shrink, self.c_shrink_angle = 0.3, 60.0, 0.2, 70.0
        self.infill_preset = 'BUILDING_ACCURATE'
        self.f_voxel, self.f_dist = 0.2, 0.2
        self.strip_height = 4.0
        self.g_voxel, self.g_angle = 0.2, 45.0
        self.h_voxel, self.h_dist = 0.2, 0.2
        # Faces kept by the decimation after each step (as the manual tools do with
        # Auto Decimation on Ratio); None = keep full resolution until the end.
        self.step_ratio = None
        # Faces kept in the final H.
        self.final_ratio = 0.03


class _InfillSettings:
    def __init__(self, preset):
        for k, v in INFILL_PRESETS[preset].items():
            setattr(self, k, v)


def wall_view_up_fn(wall_dir):
    """Up-vector function for 'from view' looking down at a wall running along wall_dir."""
    right = np.array([wall_dir[0], wall_dir[1], 0.0], dtype=np.float64)
    right /= np.linalg.norm(right)
    forward = np.array([0.0, 0.0, 1.0])
    up = np.cross(forward, right)

    def fn(direction, angle_degrees):
        if direction == Z_UP:
            return tuple(forward)
        tilt = right * direction[0] + up * direction[1]
        tilt /= np.linalg.norm(tilt)
        a = math.radians(angle_degrees)
        v = forward * math.cos(a) + tilt * math.sin(a)
        return tuple(v / np.linalg.norm(v))
    return fn


def merged(*meshes):
    mm, _ = get_meshlib()
    out = mm.Mesh()
    for m in meshes:
        out.addMesh(m)
    return out


def bottom_strip(mesh, plate_z, height):
    """The part of mesh between the plate and plate + height (exact cut)."""
    mm, _ = get_meshlib()
    box = mesh.computeBoundingBox()
    pad = 10.0
    size = mm.Vector3f(box.max.x - box.min.x + 2 * pad, box.max.y - box.min.y + 2 * pad, float(height) + pad)
    base = mm.Vector3f(box.min.x - pad, box.min.y - pad, float(plate_z) - pad)
    cutter = mm.makeCube(size, base)
    out = mm.boolean(mesh, cutter, mm.BooleanOperation.Intersection).mesh
    out.pack()   # drop the points of `mesh` the strip no longer uses
    return out


def fill_and_support(mesh, wall_dir=(1.0, 0.0, 0.0), plate_z=0.0, recipe=None, timings=None, keep=None):
    """Run the recipe on one wall face. `mesh` = that face's stones and its
    support wall as ONE meshlib mesh (loose shells are fine). Returns H.

    timings: optional list, gets (step, seconds) appended.
    keep: optional dict, gets the intermediate meshes by letter.
    """
    mm, _ = get_meshlib()
    r = recipe or Recipe()
    up_fn = wall_view_up_fn(wall_dir)

    def step(label, fn, voxel=None, decimate=True):
        check_cancelled()
        t0 = time.perf_counter()
        out = fn()
        if decimate and r.step_ratio and voxel:
            target = max(1, int(face_count(out) * r.step_ratio))
            out = decimate_mesh(out, target_face_count=target, resolution=voxel)
        if timings is not None:
            timings.append((label, time.perf_counter() - t0))
        if keep is not None:
            keep[label.split(" ")[0]] = out
        return out

    def close(m, voxel, dist, label):
        g = step(label + " grow", lambda: cuda_offset(m, voxel, dist), voxel)
        return step(label + " shrink", lambda: cuda_offset(g, voxel, -dist), voxel)

    A = close(mesh, r.a_voxel, r.a_dist, "A")

    b = step("B undercuts", lambda: fix_undercuts_multi(mm.copyMesh(mesh), [Y_POS, Y_NEG], r.b_angle, r.b_voxel,
                                                         0.0, 70.0, up_fn)[0], r.b_voxel)
    B = close(b, r.b_voxel, r.b_dist, "B")

    C = step("C undercuts", lambda: fix_undercuts_multi(mm.copyMesh(mesh), [Y_POS, Y_NEG] + DIAGONALS, r.c_angle,
                                                         r.c_voxel, r.c_auto_shrink, r.c_shrink_angle, up_fn)[0],
             r.c_voxel)

    D = step("D intersect", lambda: intersect_meshes(B, C, r.b_voxel), r.b_voxel)

    E = step("E cavity infill", lambda: build_infill(merged(D, A), _InfillSettings(r.infill_preset)), decimate=False)

    F = close(merged(E, D), r.f_voxel, r.f_dist, "F")

    strip = step("G0 strip", lambda: bottom_strip(F, plate_z, r.strip_height), decimate=False)
    if face_count(strip) > 0:
        G = step("G undercuts", lambda: fix_undercuts_multi(strip, [X_POS, X_NEG, Y_POS, Y_NEG] + DIAGONALS, r.g_angle,
                                                             r.g_voxel, 0.0, 70.0, up_fn)[0], r.g_voxel)
    else:
        G = None

    Fs = step("F' shrink", lambda: cuda_offset(F, r.f_voxel, -r.f_dist), r.f_voxel)
    H = close(merged(Fs, G) if G is not None else Fs, r.h_voxel, r.h_dist, "H")

    if r.final_ratio and r.final_ratio < 1.0:
        H = step("H decimate", lambda: decimate_mesh(H, target_face_count=max(1, int(face_count(H) * r.final_ratio)),
                                                      resolution=r.h_voxel), decimate=False)
    return H


def join_faces(face_meshes, preset='BUILDING_ACCURATE', final_ratio=0.03, timings=None):
    """I: all the H meshes merged and run through Cavity Infill once more, which joins the corners."""
    t0 = time.perf_counter()
    out = build_infill(merged(*face_meshes), _InfillSettings(preset))
    if final_ratio and final_ratio < 1.0:
        out = decimate_mesh(out, target_face_count=max(1, int(face_count(out) * final_ratio)))
    if timings is not None:
        timings.append(("I join faces", time.perf_counter() - t0))
    return out
