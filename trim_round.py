"""
Trim Edges, Round method (the voxel way), and Trim Thin, in pure meshlib: no bpy, so the same code
runs in Blender and in Quick Infill's worker processes (process_pool.py, qi_worker.py).

Grow 2d, shrink 3d (rounds off thin edges), regrow (1+x)d, decimate to the density target,
then intersect with the original to clip back to its silhouette. A stone worked at Stone that
is too small for it is trimmed with the Detail preset instead and comes out tagged Detail.
"""

import numpy as np

from .meshlib_utils import get_meshlib, get_mrmeshnumpy
from .offset_utils import (
    cuda_offset, decimate_mesh, target_faces_for_density, ensure_not_collapsed, bbox_of, face_count,
    auto_decimate as shared_auto_decimate, PARALLEL_DECIMATE_FACES, OFFSET_PRESETS, RESOLUTION_PRESETS,
    nearest_resolution_tag, MeshCollapsedError,
)

# At the Stone tag a mesh keeps at least this share of its volume, or it is trimmed at Detail
# instead (Alex 2026-11-17).
STONE_KEEPS = 2.0 / 3.0

# Every decimation decimates in parallel parts above this many faces: the one shared setting
# (offset_utils.PARALLEL_DECIMATE_FACES; it was Trim Edges' own, Ashlarize-library/trim-edges-speed).
PARALLEL_FACES = PARALLEL_DECIMATE_FACES


def trim(mesh, res, distance, x, density):
    """The round trim at one voxel size. The input mesh is not modified."""
    mm, _ = get_meshlib()
    w = cuda_offset(mesh, res, 2.0 * distance)
    w = ensure_not_collapsed(cuda_offset(w, res, -3.0 * distance))
    w = cuda_offset(w, res, (1.0 + x) * distance)
    current = face_count(w)
    target = target_faces_for_density(w, faces_per_sq_unit=density, min_faces=20, max_faces=current)
    if target < current:
        lo, hi = bbox_of(w)
        diag = float(np.linalg.norm(hi - lo))
        reduction = max(0.0, 1.0 - float(target) / float(max(1, current)))
        max_err = max(float(res) * 10.0, diag * (0.01 + 0.49 * reduction))
        w = decimate_mesh(w, target_face_count=target, max_error=max_err)
    return mm.voxelBooleanIntersect(w, mesh, res)


def kept(out, mesh):
    """Share of the mesh's volume the trimmed result still has."""
    v0 = abs(float(mesh.volume()))
    return abs(float(out.volume())) / v0 if v0 > 1e-12 else 1.0


def trim_edges(mesh, res, distance, x, density):
    """(trimmed mesh, tag it was worked at or None). Raises MeshCollapsedError."""
    try:
        out = trim(mesh, res, distance, x, density)
        if face_count(out) == 0:
            out = None
    except MeshCollapsedError:
        out = None
    if nearest_resolution_tag(res) != 'STONE':
        if out is None:
            raise MeshCollapsedError()
        return out, None
    share = kept(out, mesh) if out is not None else 0.0
    if share >= STONE_KEEPS:
        return out, None
    d = OFFSET_PRESETS['DETAIL']
    try:
        fine = trim(mesh, d["voxel"], min(distance, d["distance"]), d["x"], density)
    except MeshCollapsedError:
        fine = None
    if fine is None or face_count(fine) == 0 or kept(fine, mesh) <= share:
        if out is None:
            raise MeshCollapsedError()
        return out, None
    return fine, 'DETAIL'


def lean_trim(mesh, res, distance, x, density, unit):
    """Round (lean, trim_chip.lean_round), the way the operator calls it: (mesh, tag or None).
    It decimates to today's face count itself (no pipeline decimation afterwards)."""
    from .trim_chip import lean_round
    tag = nearest_resolution_tag(res)
    ratio = RESOLUTION_PRESETS.get(tag, RESOLUTION_PRESETS['STONE'])[1]
    info = {}
    out = lean_round(mesh, unit=unit, density=density, info=info, ratio=ratio,
                     stone=dict(voxel=float(res) / unit, distance=distance / unit, x=x),
                     fallback=(tag == 'STONE'))
    if face_count(out) == 0:
        raise MeshCollapsedError()
    return out, ('DETAIL' if info.get("tag") == "DETAIL" else None)


def trim_thin(mesh, res):
    """Trim Thin: shrink by one voxel, grow back by one voxel; anything thinner than two voxels goes.
    (mesh, None). Raises MeshCollapsedError when nothing is left."""
    shrunk = ensure_not_collapsed(cuda_offset(mesh, res, -res))
    return cuda_offset(shrunk, res, res), None


def chip_trim(mesh, res, dents, size, amount, unit):
    """Trim Edges, Chip: the stone damage chips (trim_dent), then Trim Thin. (mesh, None)."""
    from .trim_dent import dent_trim
    return dent_trim(mesh, res, dents, size, amount, unit)


KINDS = {"voxel": trim_edges, "lean": lean_trim, "thin": trim_thin, "chip": chip_trim}


def job(verts, faces, res, ratio, mode, auto_decimate, kind, *params):
    """One object, the way the pipeline does it (trim, then its auto-decimate), on numpy arrays.
    For worker processes: returns ("ok", verts, faces, voxel size used) | ("collapsed",) | ("error", text)."""
    try:
        mn = get_mrmeshnumpy()
        src = mn.meshFromFacesVerts(faces, verts)
        out, tag = KINDS[kind](src, res, *params)
        if tag is not None:
            res, ratio = RESOLUTION_PRESETS[tag]
        if face_count(out) == 0:
            return ("collapsed",)
        if auto_decimate:   # the same as blender_meshlib_utils._decimate
            out = shared_auto_decimate(out, face_count(src), res, mode, ratio)
        if out.topology.numValidFaces() != out.topology.faceSize():
            out.pack()
        return ("ok", np.asarray(mn.getNumpyVerts(out), np.float32),
                np.asarray(mn.getNumpyFaces(out.topology), np.int32), res)
    except MeshCollapsedError:
        return ("collapsed",)
    except Exception as e:
        return ("error", f"{type(e).__name__}: {e}")
