"""
Openings through a wall (windows, doors, arches, arrow slits), found from the wall itself, and a
solid that fills them, to cut out of the support (Alex 2026-11-17: "make print ready needs to keep
like arches and windows and arrow slits clear of any infill").

The wall is looked at straight through (meshlib computeDistanceMap along the wall's normal: one ray
per pixel, the wall's silhouette). A pixel no ray hits is clear through the wall. Clear pixels that
reach the left, right or top edge of the picture are outside the wall; the rest are openings (the
build plate closes the bottom, so a door that stands on it counts). Gaps narrower than `min_width`
(the joints between stones when no support wall is selected) are not openings. The openings'
outlines (distanceMapTo2DIsoPolyline) are filled flat (PlanarTriangulation) and pushed through the
wall into a solid.

The solid holds no part of the wall (every ray through it is clear), so cutting it out of the support
cannot touch a stone.

Pure meshlib + numpy, no bpy. Sizes in mm.
"""
import os
import tempfile

import numpy as np

from .meshlib_utils import get_meshlib, get_mrmeshnumpy

PIXEL = 0.1          # mm: the picture of the wall the openings are found on
MIN_WIDTH = 1.2      # mm: narrower gaps are joints between stones, not openings (an arrow slit is wider)
PAD = 1.0            # mm the solid reaches past the wall's front and back


def _dm_array(dm):
    """A meshlib DistanceMap as an array [y, x] (meshlib's raw file; its Python side gives the values
    one by one)."""
    mm, _ = get_meshlib()
    path = os.path.join(tempfile.gettempdir(), "qi_through_%d_%d.raw" % (os.getpid(), id(dm)))
    mm.DistanceMapSave.toRAW(dm, path)
    out = np.fromfile(path, np.float32, offset=16).reshape(dm.resY(), dm.resX())
    os.remove(path)
    return out


def _array_dm(field):
    """An array [y, x] as a meshlib DistanceMap (in memory where meshlib allows it)."""
    import ctypes
    mm, _ = get_meshlib()
    ny, nx = field.shape
    flat = np.ascontiguousarray(field, np.float32)
    vec = mm.std_vector_float()
    if hasattr(vec, "data_pointer"):
        vec.resize(nx * ny)
        ctypes.memmove(int(vec.data_pointer()), flat.ctypes.data, nx * ny * 4)
        dm = mm.DistanceMap(nx, ny)
        dm.set(vec)
        return dm
    path = os.path.join(tempfile.gettempdir(), "qi_through_%d.raw" % os.getpid())
    with open(path, "wb") as f:
        np.array([nx, ny], np.uint64).tofile(f)
        flat.tofile(f)
    dm = mm.DistanceMapLoad.fromRaw(path)
    os.remove(path)
    return dm


def _erode(a, n):
    out = a
    for _ in range(int(n)):
        b = out.copy()
        b[1:, :] &= out[:-1, :]
        b[:-1, :] &= out[1:, :]
        b[:, 1:] &= out[:, :-1]
        b[:, :-1] &= out[:, 1:]
        out = b
    return out


def _dilate(a, n):
    out = a
    for _ in range(int(n)):
        b = out.copy()
        b[1:, :] |= out[:-1, :]
        b[:-1, :] |= out[1:, :]
        b[:, 1:] |= out[:, :-1]
        b[:, :-1] |= out[:, 1:]
        out = b
    return out


def wall_frame(wall_dir):
    """(right, normal, up) unit vectors of a wall running along wall_dir (horizontal)."""
    right = np.array([wall_dir[0], wall_dir[1], 0.0], np.float64)
    right /= np.linalg.norm(right)
    up = np.array([0.0, 0.0, 1.0])
    return right, np.cross(up, right), up


def opening_mask(mesh, wall_dir, plate_z, pixel=PIXEL, min_width=MIN_WIDTH):
    """(mask [z, u] of the openings, frame) or (None, None). frame = (right, normal, up, u0, z0,
    n0, n1, pixel): pixel (i, j) is centred on u0 + (j + 0.5) * pixel along right, z0 + (i + 0.5) *
    pixel up; the wall lies between n0 and n1 along normal."""
    from .print_check import flood2d
    mm, _ = get_meshlib()
    mn = get_mrmeshnumpy()
    right, normal, up = wall_frame(wall_dir)
    pts = np.asarray(mn.getNumpyVerts(mesh), np.float64)
    box = mesh.computeBoundingBox()
    blo = np.array([box.min.x, box.min.y, box.min.z]); bhi = np.array([box.max.x, box.max.y, box.max.z])
    pts = pts[((pts >= blo - 1e-6) & (pts <= bhi + 1e-6)).all(1)]
    if len(pts) == 0:
        return None, None
    u, n = pts @ right, pts @ normal
    u0, u1 = float(u.min()) - 2 * pixel, float(u.max()) + 2 * pixel
    n0, n1 = float(n.min()), float(n.max())
    z0, z1 = float(plate_z), float(pts[:, 2].max()) + 2 * pixel
    nx, ny = int(np.ceil((u1 - u0) / pixel)), int(np.ceil((z1 - z0) / pixel))
    if nx < 3 or ny < 3:
        return None, None
    p = mm.MeshToDistanceMapParams()
    org = right * u0 + normal * (n0 - 1.0) + up * z0
    p.orgPoint = mm.Vector3f(*(float(v) for v in org))
    p.direction = mm.Vector3f(*(float(v) for v in normal))
    p.xRange = mm.Vector3f(*(float(v) for v in right * (nx * pixel)))
    p.yRange = mm.Vector3f(*(float(v) for v in up * (ny * pixel)))
    p.resolution = mm.Vector2i(nx, ny)
    p.allowNegativeValues = True
    dm = mm.computeDistanceMap(mm.MeshPart(mesh), p)
    a = _dm_array(dm)
    clear = ~np.isfinite(a) | (np.abs(a) >= 1e30)          # meshlib marks a pixel no ray hit with the largest float
    seed = np.zeros_like(clear)
    seed[:, 0] = seed[:, -1] = True                    # the wall's ends
    seed[-1, :] = True                                 # above it (the plate closes the bottom)
    outside = flood2d(seed, clear)
    holes = clear & ~outside
    if not holes.any():
        return None, None
    r = int(np.ceil(0.5 * float(min_width) / pixel))
    core = _erode(holes, r)
    if not core.any():
        return None, None
    # what a disc min_width across can reach inside them: the opening without the joints that
    # run into it from the side (the infill in a joint holds the stones together)
    holes = _dilate(core, r) & holes
    return holes, (right, normal, up, u0, z0, n0, n1, pixel)


def opening_solid(mesh, wall_dir, plate_z, pixel=PIXEL, min_width=MIN_WIDTH, pad=PAD):
    """A closed meshlib mesh filling the wall's openings (front to back, `pad` past the wall), or
    None when it has none."""
    mm, _ = get_meshlib()
    mn = get_mrmeshnumpy()
    holes, frame = opening_mask(mesh, wall_dir, plate_z, pixel, min_width)
    if holes is None:
        return None
    right, normal, up, u0, z0, n0, n1, pixel = frame
    field = np.zeros((holes.shape[0] + 2, holes.shape[1] + 2), np.float32)
    field[1:-1, 1:-1] = holes
    lines = mm.distanceMapTo2DIsoPolyline(_array_dm(field), 0.5)
    cap = mm.PlanarTriangulation.triangulateContours(lines.contours())
    v2 = np.asarray(mn.getNumpyVerts(cap), np.float64)[:, :2] - 1.0      # back from the padded picture
    f = np.asarray(mn.getNumpyFaces(cap.topology), np.int64)
    if len(f) == 0:
        return None
    k = len(v2)
    # (meshlib counts from a pixel's corner, the picture's pixel j spans u0 + [j, j + 1] * pixel)
    base = right[None, :] * (u0 + v2[:, :1] * pixel) + up[None, :] * (z0 + v2[:, 1:2] * pixel)
    front = base + normal[None, :] * (n0 - pad)
    back = base + normal[None, :] * (n1 + pad)
    verts = np.vstack([front, back])
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    key = edges[:, 0] * k + edges[:, 1]
    rev = edges[:, 1] * k + edges[:, 0]
    border = edges[~np.isin(key, rev)]                 # the outline, in the caps' own order
    a, b = border[:, 0], border[:, 1]
    sides = np.concatenate([np.stack([a, b, b + k], 1), np.stack([a, b + k, a + k], 1)])
    faces = np.vstack([f[:, ::-1], f + k, sides])
    solid = mn.meshFromFacesVerts(faces.astype(np.int32), verts.astype(np.float32))
    if solid.volume() < 0:
        solid = mn.meshFromFacesVerts(faces[:, ::-1].astype(np.int32), verts.astype(np.float32))
    return solid


def cut_out(mesh, solid):
    """mesh less the openings' solid (exact boolean); mesh itself when there is nothing to cut or
    the cut fails."""
    if solid is None or mesh is None:
        return mesh
    mm, _ = get_meshlib()
    try:
        res = mm.boolean(mesh, solid, mm.BooleanOperation.DifferenceAB)
        if not res.valid():
            raise RuntimeError(res.errorString)
        out = res.mesh
        out.pack()
        return out if out.topology.numValidFaces() else mesh
    except Exception as ex:
        print("Quick Infill: openings not cut out of the support:", ex)
        return mesh
