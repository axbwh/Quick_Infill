"""
Presets and reusable meshlib operations for Quick Infill.

All sizes here are in meshlib space: Blender geometry is scaled by
STL_EXPORT_SCALE (10x) on the way into meshlib (see pipeline.py), so a voxel
size of 0.075 here is 0.0075 Blender units.
"""

from typing import Optional
import numpy as np

from .meshlib_utils import (
    get_meshlib, get_mrmeshnumpy, CUDA_LOCK, progress_callback, check_cancelled, reraise_if_cancelled,
)

# ---------------------------------------------------------------------------
# Presets (single source of truth; the UI reads these)
# ---------------------------------------------------------------------------

# Voxel tool presets: voxel size, offset distance, Trim Edges X, decimation ratio.
OFFSET_PRESETS = {
    'STONE': {"voxel": 0.075, "distance": 0.7, "x": 0.25, "ratio": 0.0175},
    'DETAIL': {"voxel": 0.05, "distance": 0.3, "x": 0.4, "ratio": 0.03},
    'FINE': {"voxel": 0.03, "distance": 0.2, "x": 0.25, "ratio": 0.02},
    'MINI': {"voxel": 0.025, "distance": 0.2, "x": 0.25, "ratio": 0.04},
}

# Resolution tags: (voxel size, decimation ratio). Order defines the tag index
# stored in the per-face "qi_res_tag" attribute; COARSE is last so existing
# stored indices for Stone/Detail/Fine/Mini stay valid.
RESOLUTION_PRESETS = {k: (v["voxel"], v["ratio"]) for k, v in OFFSET_PRESETS.items()}
RESOLUTION_PRESETS['COARSE'] = (0.2, 0.01)
RESOLUTION_TAG_ORDER = list(RESOLUTION_PRESETS.keys())

# Voxel sizes above this are always bucketed as Coarse.
COARSE_RESOLUTION_THRESHOLD = 0.1

# Cavity Infill presets (its own resolution / grow / shrink settings).
INFILL_PRESETS = {
    'BUILDING_FAST': {"voxel_mode": 'TARGET_VOXELS', "target_res": 0.5, "resolution": 0.1, "grow": 0.75,
                      "shrink_mult": 2.0, "method": 'NAIVE', "trim_thin": True},
    'BUILDING_ACCURATE': {"voxel_mode": 'TARGET_VOXELS', "target_res": 3.0, "resolution": 0.1, "grow": 1.0,
                          "shrink_mult": 1.2, "method": 'ACCURATE', "trim_thin": True},
    'MINI_LARGE_HOLES': {"voxel_mode": 'RESOLUTION', "target_res": 1.0, "resolution": 0.1, "grow": 1.0,
                         "shrink_mult": 1.5, "method": 'ACCURATE', "trim_thin": True},
    'MINI_ACCURATE': {"voxel_mode": 'RESOLUTION', "target_res": 2.0, "resolution": 0.075, "grow": 0.35,
                      "shrink_mult": 1.2, "method": 'ACCURATE', "trim_thin": True},
}

ANGLE_PRESETS = (10.0, 15.0, 30.0, 45.0, 60.0, 90.0)


class MeshCollapsedError(Exception):
    """An operation left no geometry (e.g. Trim Thin on a part thinner than a voxel).

    Only this exception means "collapsed"; any other exception is a real error
    and must never cause the source object to be deleted.
    """


def nearest_resolution_tag(voxel_size: float) -> str:
    """Bucket a voxel size into the closest known resolution tag.

    Anything above COARSE_RESOLUTION_THRESHOLD is Coarse. A small epsilon
    keeps float32 round-off (0.1 read back as 0.100000001) out of Coarse.
    """
    voxel_size = float(voxel_size)
    if voxel_size > COARSE_RESOLUTION_THRESHOLD + 1e-6:
        return 'COARSE'
    fine_tags = [name for name in RESOLUTION_TAG_ORDER if name != 'COARSE']
    return min(fine_tags, key=lambda name: abs(RESOLUTION_PRESETS[name][0] - voxel_size))


def voxel_size_to_decimation_ratio(voxel_size: float) -> float:
    """Fallback ratio from voxel size (0.025 -> 0.01 ... clamped 0.01-0.2)."""
    return max(0.01, min(0.2, float(voxel_size) / 5.0))


def should_auto_decimate_faces(initial_faces: int, final_faces: int, voxel_size: Optional[float] = None,
                               mode: str = "VOXEL_RATIO", ratio: Optional[float] = None) -> tuple:
    """Return (decimate?, target_faces) for the shared decimate modes.

    OFF: never. ORIGINAL: back to the original face count if it grew.
    VOXEL_RATIO: keep `ratio` of the final faces (falls back to a voxel-size mapping).
    """
    if mode == "OFF":
        return False, final_faces
    if mode == "ORIGINAL":
        if final_faces > initial_faces:
            return True, initial_faces
        return False, final_faces
    target_ratio = ratio if ratio is not None else (voxel_size_to_decimation_ratio(voxel_size) if voxel_size is not None else 0.015)
    target_faces = max(1, int(final_faces * float(target_ratio)))
    if target_faces < final_faces:
        return True, target_faces
    return False, final_faces


def should_auto_decimate_cavity_faces(initial_faces: int, final_faces: int) -> tuple:
    """Cavity Infill's own rule: decimate back to the source face count if it grew."""
    if final_faces > initial_faces:
        return True, initial_faces
    return False, final_faces


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

def face_count(mesh) -> int:
    return mesh.topology.numValidFaces()


def bbox_of(mesh):
    """(min xyz, max xyz) numpy arrays. Much faster than computeBoundingBox() on big meshes."""
    mn = get_mrmeshnumpy()
    v = mn.getNumpyVerts(mesh)
    if len(v) == 0:
        z = np.zeros(3)
        return z, z
    return v.min(axis=0), v.max(axis=0)


def ensure_not_collapsed(mesh):
    if face_count(mesh) == 0:
        raise MeshCollapsedError()
    return mesh


def cuda_offset(mesh, resolution: float, distance: float):
    """CUDA winding-number offset. Positive distance grows, negative shrinks, 0 re-voxelises.

    Serialised with CUDA_LOCK so only one CUDA job touches the GPU at a time.
    """
    mm, mc = get_meshlib()
    check_cancelled()
    p = mm.GeneralOffsetParameters()
    p.voxelSize = float(resolution)
    p.signDetectionMode = mm.SignDetectionMode.HoleWindingRule
    cb = progress_callback()
    if cb is not None:
        p.callBack = cb
    with CUDA_LOCK:
        try:
            p.fwn = mc.FastWindingNumber(mesh)
            return mm.generalOffsetMesh(mp=mesh, offset=float(distance), params=p)
        except Exception as e:
            reraise_if_cancelled(e)
            raise


# Cavity Infill (Accurate) carves the closed-up mesh back with a skin that is
# one voxel thick where it lies on the model and thicker where it bridges a gap.
# True = build that skin from an inward offset (the even part) plus meshlib's
# point-based weighted shell (the thick part): about 8 times faster than the
# triangle-based weighted shell, and on a wall of 150 stones the carved result
# differs by 0.013 mm on average (0.06 mm at the 99th percentile) at a 0.34 mm voxel.
FAST_SHELL = True


def _vert_scalars(values):
    mm, _ = get_meshlib()
    sc = mm.VertScalars()
    sc.vec = mm.std_vector_float(np.asarray(values, dtype=np.float32).tolist())
    return sc


def _shell_inputs(mesh_to_offset, reference_mesh, voxel_size, shrink_mult, max_vertices, target_resolution):
    """(working mesh, per-vertex weights, voxel size) shared by both shell routes."""
    mm, _ = get_meshlib()
    MAX_VERTICES = max_vertices if max_vertices is not None else 1_000_000

    working_mesh = mesh_to_offset
    working_ref = reference_mesh
    if mesh_to_offset.points.size() > MAX_VERTICES:
        ratio = (MAX_VERTICES // 2) / mesh_to_offset.points.size()
        working_mesh = decimate_mesh(mesh_to_offset, reduction_ratio=ratio)
    if reference_mesh.points.size() > MAX_VERTICES:
        ratio = (MAX_VERTICES // 2) / reference_mesh.points.size()
        working_ref = decimate_mesh(reference_mesh, reduction_ratio=ratio)

    sd = mm.findSignedDistances(working_ref, working_mesh)
    n = working_mesh.points.size()
    weights = np.abs(np.fromiter((float(d) for d in sd), dtype=np.float64, count=n)) * float(shrink_mult)

    if target_resolution is not None and max_vertices is not None:
        target_voxel_count = min(int(max_vertices), int(target_resolution))
        suggested = mm.suggestVoxelSize(mm.MeshPart(working_mesh), target_voxel_count)
        adaptive_voxel_size = max(float(voxel_size), suggested)
    else:
        adaptive_voxel_size = float(voxel_size)
    return working_mesh, weights, adaptive_voxel_size


def weighted_dist_shell(mesh_to_offset, reference_mesh, voxel_size: float, shrink_mult: float,
                        max_vertices: Optional[int] = None, target_resolution: Optional[float] = None):
    """Variable-width shell, thicker where farther from reference_mesh (Cavity Infill, Accurate)."""
    mm, _ = get_meshlib()
    working_mesh, weights, adaptive_voxel_size = _shell_inputs(mesh_to_offset, reference_mesh, voxel_size, shrink_mult,
                                                               max_vertices, target_resolution)
    params = mm.WeightedShell.ParametersMetric()
    params.voxelSize = float(adaptive_voxel_size)
    params.offset = float(adaptive_voxel_size)
    params.dist.maxWeight = float(weights.max()) if len(weights) else 1.0
    return mm.WeightedShell.meshShell(working_mesh, _vert_scalars(weights), params)


def carve_weighted(mesh_to_offset, reference_mesh, voxel_size: float, shrink_mult: float,
                   max_vertices: Optional[int] = None, target_resolution: Optional[float] = None):
    """mesh_to_offset minus the weighted shell (see FAST_SHELL)."""
    mm, _ = get_meshlib()
    if not FAST_SHELL:
        shell = weighted_dist_shell(mesh_to_offset, reference_mesh, voxel_size, shrink_mult, max_vertices, target_resolution)
        return mm.boolean(mesh_to_offset, shell, mm.BooleanOperation.DifferenceAB).mesh

    working_mesh, weights, av = _shell_inputs(mesh_to_offset, reference_mesh, voxel_size, shrink_mult,
                                              max_vertices, target_resolution)
    # Even part of the skin: everything within one (shell) voxel of the surface.
    thin = cuda_offset(mesh_to_offset, av, -av)
    if face_count(thin) == 0 or len(weights) == 0 or float(weights.max()) <= 1e-6:
        return thin
    # Thick part: balls of radius av + weight round the surface points.
    mn = get_mrmeshnumpy()
    check_cancelled()
    cloud = mn.pointCloudFromPoints(mn.getNumpyVerts(working_mesh).astype(np.float32))
    params = mm.WeightedShell.ParametersMetric()
    params.voxelSize = float(av)
    params.offset = float(av)
    params.dist.maxWeight = float(weights.max())
    shell = mm.WeightedShell.pointsShell(cloud, _vert_scalars(weights), params)
    check_cancelled()
    if face_count(shell) == 0:
        return thin
    return mm.boolean(thin, shell, mm.BooleanOperation.DifferenceAB).mesh


def compute_voxel_size(mesh, target_voxels: int, min_resolution: float) -> float:
    """Suggested voxel size for a target voxel count, never finer than min_resolution."""
    mm, _ = get_meshlib()
    suggested = mm.suggestVoxelSize(mm.MeshPart(mesh), float(target_voxels))
    return max(float(suggested), float(min_resolution))


# Above this many faces, decimate in parallel parts (about 4x faster on a 16-core CPU).
PARALLEL_DECIMATE_FACES = 200_000


def decimate_mesh(mesh, target_face_count: Optional[int] = None, reduction_ratio: Optional[float] = None,
                  max_error: Optional[float] = None, resolution: Optional[float] = None):
    """Decimate in place (and return) with meshlib.

    resolution: voxel size in meshlib units. maxError defaults to 10x the
    voxel size - the value the tools have always used, kept so results don't
    change. (An old comment claimed resolution was in Blender units; it is not.)
    """
    mm, _ = get_meshlib()
    current_faces = face_count(mesh)
    if target_face_count is None and reduction_ratio is None:
        reduction_ratio = 0.5

    if target_face_count is not None:
        target = int(target_face_count)
    else:
        target = max(1, int(current_faces * float(reduction_ratio)))

    settings = mm.DecimateSettings()
    settings.maxDeletedFaces = max(0, current_faces - target)
    if max_error is not None:
        settings.maxError = float(max_error)
    elif resolution is not None:
        settings.maxError = float(resolution) * 10.0
    else:
        lo, hi = bbox_of(mesh)
        settings.maxError = float(np.linalg.norm(hi - lo)) * 0.005
    settings.packMesh = True
    if current_faces > PARALLEL_DECIMATE_FACES:
        settings.subdivideParts = 16

    check_cancelled()
    mm.decimateMesh(mesh, settings)
    check_cancelled()  # a cancelled decimation stops early without raising
    return mesh


def target_faces_for_density(mesh, faces_per_sq_unit: float, min_faces: int = 200, max_faces: Optional[int] = None) -> int:
    """Target face count from bounding-box surface area, for roughly consistent density."""
    lo, hi = bbox_of(mesh)
    dx, dy, dz = (hi - lo).tolist()
    surface_area = max(1e-6, 2.0 * (dx * dy + dy * dz + dz * dx))
    target_faces = int(max(float(min_faces), surface_area * float(faces_per_sq_unit)))
    if max_faces is not None:
        target_faces = min(target_faces, int(max_faces))
    return target_faces
