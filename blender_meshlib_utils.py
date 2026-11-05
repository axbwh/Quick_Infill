"""
Blender <-> meshlib pipeline for Quick Infill.

Geometry moves between Blender and meshlib IN MEMORY (numpy arrays), not
through temporary STL files: no disk I/O, no import/export operators, no
selection juggling, and no temporary objects linked into the scene (which used
to dirty Geometry Nodes setups that scan the user's collections).

Coordinates are scaled by STL_EXPORT_SCALE on the way in and back out, exactly
as the old STL round trip did, so every voxel size and distance in the add-on
keeps its meaning.

One pipeline (process_objects) serves single objects, multi-object batches and
per-island processing.
"""

from concurrent.futures import ThreadPoolExecutor

import bpy
import numpy as np

from .meshlib_utils import get_meshlib, get_mrmeshnumpy, JobCancelled, check_cancelled
from .offset_utils import (
    RESOLUTION_PRESETS, RESOLUTION_TAG_ORDER, nearest_resolution_tag,
    decimate_mesh, should_auto_decimate_faces, face_count, MeshCollapsedError, PARALLEL_DECIMATE_FACES,
)

# Scale applied to Blender coordinates when entering meshlib (and undone on the way back).
STL_EXPORT_SCALE = 10.0

# Per-face integer attribute + object custom property tagging the resolution
# (Stone/Detail/Fine/Mini/Coarse) a mesh was processed at.
RES_TAG_ATTR = "qi_res_tag"
# Colour attribute mirroring the tag, for viewport shading.
RES_COLOR_ATTR = "qi_res_color"
RES_TAG_COLORS = {
    'STONE': (0.55, 0.35, 0.2, 1.0),
    'DETAIL': (0.2, 0.55, 0.9, 1.0),
    'FINE': (0.25, 0.8, 0.3, 1.0),
    'MINI': (0.95, 0.75, 0.1, 1.0),
    'COARSE': (0.8, 0.15, 0.15, 1.0),
}
RES_UNTAGGED_COLOR = (0.5, 0.5, 0.5, 1.0)

MAX_WORKERS = 4


# ---------------------------------------------------------------------------
# Resolution tags
# ---------------------------------------------------------------------------

def _tag_to_index(tag_name):
    try:
        return RESOLUTION_TAG_ORDER.index(tag_name)
    except (ValueError, TypeError):
        return -1


def _index_to_tag(index):
    if index is None or index < 0 or index >= len(RESOLUTION_TAG_ORDER):
        return None
    return RESOLUTION_TAG_ORDER[int(index)]


def _majority(values):
    """Most common value in an int array (ties -> smallest), or None if empty."""
    values = np.asarray(values)
    if values.size == 0:
        return None
    shifted = values.astype(np.int64) + 1  # -1 (untagged) -> 0
    shifted = np.clip(shifted, 0, None)
    return int(np.bincount(shifted).argmax()) - 1


def _dominant_tag_label(face_tags):
    """One tag name if all faces share it, "MIXED" if they differ, None if untagged."""
    face_tags = np.asarray(face_tags)
    valid = face_tags[face_tags >= 0]
    if valid.size == 0:
        return None
    unique = np.unique(valid)
    if unique.size == 1:
        return _index_to_tag(int(unique[0]))
    return "MIXED"


def read_face_tags(mesh):
    """Per-face qi_res_tag values as an int array, or None if the mesh has no tags."""
    if mesh is None:
        return None
    attr = mesh.attributes.get(RES_TAG_ATTR)
    if attr is None or attr.domain != 'FACE' or len(attr.data) == 0:
        return None
    out = np.empty(len(attr.data), dtype=np.int32)
    attr.data.foreach_get("value", out)
    return out


def _dominant_face_tag(mesh):
    tags = read_face_tags(mesh)
    if tags is None:
        return None
    return _index_to_tag(_majority(tags))


def _approximate_tag_from_remesh_voxel_size(mesh):
    """Guess a tag from mesh.remesh_voxel_size (Blender units), trusted within 0.1 of a preset."""
    if mesh is None:
        return None
    blender_voxel_size = getattr(mesh, "remesh_voxel_size", 0.0)
    if not blender_voxel_size:
        return None
    voxel_size = blender_voxel_size * STL_EXPORT_SCALE
    tag = nearest_resolution_tag(voxel_size)
    if abs(voxel_size - RESOLUTION_PRESETS[tag][0]) <= 0.1:
        return tag
    return None


def read_object_resolution_tag(obj):
    """Whole-object tag for Smart resolution: per-face attribute, then the object
    property, then a guess from remesh_voxel_size."""
    if obj is None:
        return None
    if obj.type == 'MESH':
        face_tag = _dominant_face_tag(obj.data)
        if face_tag:
            return face_tag
    tag = obj.get(RES_TAG_ATTR)
    if tag in RESOLUTION_TAG_ORDER:
        return tag
    if obj.type == 'MESH':
        return _approximate_tag_from_remesh_voxel_size(obj.data)
    return None


def write_mesh_resolution_tags(mesh, face_tag_indices, remesh_voxel_size=None):
    """Write qi_res_tag (per face), the matching colour attribute (per corner) and
    remesh_voxel_size (converted to Blender units) onto a mesh. Vectorised."""
    if mesh is None or len(mesh.polygons) == 0:
        return
    tags = np.asarray(face_tag_indices, dtype=np.int32)
    if tags.size == len(mesh.polygons):
        attr = mesh.attributes.get(RES_TAG_ATTR)
        if attr is None:
            attr = mesh.attributes.new(name=RES_TAG_ATTR, type='INT', domain='FACE')
        attr.data.foreach_set("value", tags)

        palette = np.array([RES_TAG_COLORS.get(t, RES_UNTAGGED_COLOR) for t in RESOLUTION_TAG_ORDER] + [RES_UNTAGGED_COLOR],
                           dtype=np.float32)
        idx = np.where((tags >= 0) & (tags < len(RESOLUTION_TAG_ORDER)), tags, len(RESOLUTION_TAG_ORDER))
        loop_totals = np.empty(len(mesh.polygons), dtype=np.int32)
        mesh.polygons.foreach_get("loop_total", loop_totals)
        corner_colors = palette[np.repeat(idx, loop_totals)]
        color_attr = mesh.color_attributes.get(RES_COLOR_ATTR)
        if color_attr is None:
            color_attr = mesh.color_attributes.new(name=RES_COLOR_ATTR, type='BYTE_COLOR', domain='CORNER')
        color_attr.data.foreach_set("color", corner_colors.ravel())
        try:
            mesh.color_attributes.active_color = color_attr
        except (AttributeError, TypeError):
            pass
    if remesh_voxel_size is not None:
        mesh.remesh_voxel_size = float(remesh_voxel_size) / STL_EXPORT_SCALE


def set_object_resolution_label(obj, label):
    if obj is None:
        return
    if label:
        obj[RES_TAG_ATTR] = label
    elif RES_TAG_ATTR in obj:
        del obj[RES_TAG_ATTR]


# ---------------------------------------------------------------------------
# Blender <-> meshlib (in memory)
# ---------------------------------------------------------------------------

def _triangles(me):
    """(tri vertex indices (n,3) int32, tri -> polygon index (n,) int32)."""
    if hasattr(me, "calc_loop_triangles"):
        me.calc_loop_triangles()
    n = len(me.loop_triangles)
    tris = np.empty(n * 3, dtype=np.int32)
    me.loop_triangles.foreach_get("vertices", tris)
    polys = np.empty(n, dtype=np.int32)
    me.loop_triangles.foreach_get("polygon_index", polys)
    return tris.reshape(-1, 3), polys


def blender_to_meshlib(obj, apply_modifiers=False, depsgraph=None, with_face_map=False):
    """World-space (x STL_EXPORT_SCALE) meshlib mesh of a Blender mesh object.

    with_face_map=True also returns, per meshlib face, the Blender polygon it
    came from (or None if meshlib reordered faces).
    """
    mn = get_mrmeshnumpy()
    if obj.type != 'MESH':
        raise ValueError(f"'{obj.name}' is not a mesh.")

    eval_obj = None
    if apply_modifiers:
        depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(depsgraph)
        me = eval_obj.to_mesh()
    else:
        me = obj.data
    try:
        verts = np.empty(len(me.vertices) * 3, dtype=np.float32)
        me.vertices.foreach_get("co", verts)
        verts = verts.reshape(-1, 3).astype(np.float64)
        mw = np.array(obj.matrix_world, dtype=np.float64)
        verts = (verts @ mw[:3, :3].T + mw[:3, 3]) * STL_EXPORT_SCALE
        tris, tri_polys = _triangles(me)
    finally:
        if eval_obj is not None:
            eval_obj.to_mesh_clear()

    if len(tris) == 0:
        raise MeshCollapsedError()
    mesh = mn.meshFromFacesVerts(tris, verts.astype(np.float32))
    if not with_face_map:
        return mesh
    face_map = tri_polys if face_count(mesh) == len(tri_polys) else None
    return mesh, face_map


# Kept for callers of the old name
def blender_to_meshlib_via_stl(blender_obj, tmp_dir=None, apply_modifiers=True):
    return blender_to_meshlib(blender_obj, apply_modifiers=apply_modifiers)


def meshlib_to_blender_mesh(ml_mesh, name):
    """New bpy Mesh (world space, Blender units) from a meshlib mesh."""
    mn = get_mrmeshnumpy()
    topo = ml_mesh.topology
    if hasattr(ml_mesh, "pack") and topo.numValidFaces() != topo.faceSize():
        ml_mesh.pack()
    verts = (mn.getNumpyVerts(ml_mesh) / STL_EXPORT_SCALE).astype(np.float32)
    faces = mn.getNumpyFaces(ml_mesh.topology).astype(np.int32)
    if faces.size == 0:
        raise MeshCollapsedError()

    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.ravel())
    me.loops.add(faces.size)
    me.loops.foreach_set("vertex_index", faces.ravel())
    me.polygons.add(len(faces))
    me.polygons.foreach_set("loop_start", np.arange(0, faces.size, 3, dtype=np.int32))
    me.update(calc_edges=True)
    return me


def link_new_object(name, mesh, collections):
    obj = bpy.data.objects.new(name, mesh)
    for coll in collections:
        coll.objects.link(obj)
    return obj


def move_object_to_collections(obj, target_collections):
    for coll in list(obj.users_collection):
        coll.objects.unlink(obj)
    for coll in target_collections:
        coll.objects.link(obj)


def swap_mesh_keep_transforms(obj, world_mesh):
    """Give obj the geometry of world_mesh (world-space coordinates) without
    moving the object. Materials of the old mesh are kept. Returns the old mesh,
    which the caller frees after the depsgraph has been updated."""
    world_mesh.transform(obj.matrix_world.inverted())
    old = obj.data
    for mat in old.materials:
        world_mesh.materials.append(mat)
    obj.data = world_mesh
    world_mesh.name = obj.name
    return old


def free_unused_meshes(meshes):
    """Remove meshes left without users. The view layer is updated first so the
    depsgraph no longer references them (removing a mesh it still holds made the
    next file save freeze)."""
    meshes = [m for m in meshes if m is not None]
    if not meshes:
        return
    bpy.context.view_layer.update()
    for m in meshes:
        try:
            if m.users == 0 and not m.use_fake_user:
                bpy.data.meshes.remove(m)
        except ReferenceError:
            pass


def remove_objects(objs):
    """Delete objects and any meshes they leave unused."""
    meshes = []
    for obj in objs:
        try:
            if obj.type == 'MESH':
                meshes.append(obj.data)
            bpy.data.objects.remove(obj, do_unlink=True)
        except ReferenceError:
            pass
    free_unused_meshes(meshes)


def select_results(result_objs):
    """Select the given objects and make the first one active."""
    # Walk the view layer rather than context.selected_objects: that list can
    # still hold objects this operator just deleted (e.g. voxel boolean Replace).
    for obj in bpy.context.view_layer.objects:
        if obj.select_get():
            obj.select_set(False)
    alive = [o for o in result_objs if o is not None and o.name in bpy.data.objects]
    for obj in alive:
        try:
            obj.select_set(True)
        except RuntimeError:
            pass  # not in the view layer
    if alive:
        bpy.context.view_layer.objects.active = alive[0]


# ---------------------------------------------------------------------------
# The one processing pipeline
# ---------------------------------------------------------------------------

class ProcessResult:
    def __init__(self):
        self.results = []      # (obj, initial_verts, final_verts)
        self.collapsed = []    # (obj, MeshCollapsedError) - geometry vanished
        self.errors = []       # (obj, exception) - real failures; object left untouched
        self.dropped_islands = 0

    def error_summary(self):
        return "; ".join(f"{o.name}: {e}" for o, e in self.errors[:3]) + (" ..." if len(self.errors) > 3 else "")


class _Job:
    __slots__ = ("obj", "name", "src", "face_map", "face_tags", "res", "ratio", "mode",
                 "out", "out_tags", "voxels_used", "initial_verts", "dropped", "error")


def _effective_settings(obj, settings, resolution, smart):
    ratio = settings.decimation_ratio
    mode = settings.decimate_mode
    if smart:
        tag = read_object_resolution_tag(obj)
        if tag:
            res, ratio = RESOLUTION_PRESETS[tag]
            return res, ratio, "VOXEL_RATIO"
    return resolution, ratio, mode


def _decimate(out, initial_faces, res, mode, ratio):
    do, target = should_auto_decimate_faces(initial_faces, face_count(out), voxel_size=res, mode=mode, ratio=ratio)
    if do:
        out = decimate_mesh(out, target_face_count=target, resolution=res)
    return out


def _tag_index(res):
    return _tag_to_index(nearest_resolution_tag(res)) if res is not None else -1


def _compute(job, operation_fn, auto_decimate, islands, smart):
    """Worker: pure meshlib (no bpy). Fills job.out / job.out_tags or raises."""
    mm, _ = get_meshlib()
    mn = get_mrmeshnumpy()
    src = job.src

    components = mm.MeshComponents.getAllComponents(mm.MeshPart(src)) if islands else []
    if len(components) <= 1:
        out = operation_fn(src, job.res)
        if face_count(out) == 0:
            raise MeshCollapsedError()
        if auto_decimate:
            out = _decimate(out, face_count(src), job.res, job.mode, job.ratio)
        job.out = out
        job.out_tags = np.full(face_count(out), _tag_index(job.res), dtype=np.int32)
        job.voxels_used = [job.res] if job.res is not None else []
        return

    n_faces = src.topology.faceSize()
    merged = mm.Mesh()
    tags, voxels = [], []
    for bitset in components:
        island = mm.Mesh()
        island.addMeshPart(mm.MeshPart(src, bitset))
        res, ratio, mode = job.res, job.ratio, job.mode
        if smart and job.face_tags is not None:
            mask = mn.getNumpyBitSet(bitset)
            mask = np.pad(mask, (0, max(0, n_faces - mask.size)))[:n_faces]
            tag = _index_to_tag(_majority(job.face_tags[mask[:job.face_tags.size]]))
            if tag:
                res, ratio = RESOLUTION_PRESETS[tag]
                mode = "VOXEL_RATIO"
        check_cancelled()
        try:
            out = operation_fn(island, res)
            if face_count(out) == 0:
                raise MeshCollapsedError()
        except JobCancelled:
            raise
        except Exception:
            # An island that collapses (or that meshlib can't process) is dropped
            # from the merged result; the rest of the object is still processed.
            job.dropped += 1
            continue
        if auto_decimate:
            out = _decimate(out, face_count(island), res, mode, ratio)
        tags.append(np.full(face_count(out), _tag_index(res), dtype=np.int32))
        if res is not None:
            voxels.append(res)
        merged.addMesh(out)
    if face_count(merged) == 0:
        raise MeshCollapsedError()
    job.out = merged
    job.out_tags = np.concatenate(tags) if tags else np.zeros(0, dtype=np.int32)
    job.voxels_used = voxels


class PipelineRun:
    """One run of the pipeline, split into three steps so the meshlib part can
    run on a background thread while Blender stays responsive:

        prepare()  main thread  - read the Blender meshes
        compute()  any thread   - meshlib only, cancellable
        finish()   main thread  - write the results back into Blender
    """

    def __init__(self, objects, operation_fn, output_suffix, *, resolution, auto_decimate=True,
                 replace_original=True, settings=None, islands=None, smart=None, apply_modifiers=None,
                 parallel=True):
        s = settings or bpy.context.scene.quick_infill_settings
        self.objects = list(objects)
        self.operation_fn = operation_fn
        self.suffix = output_suffix
        self.resolution = resolution
        self.auto_decimate = auto_decimate
        self.replace = replace_original
        self.settings = s
        self.islands = s.process_islands if islands is None else islands
        self.smart = s.smart_resolution if smart is None else smart
        self.apply_modifiers = s.apply_modifiers_on_export if apply_modifiers is None else apply_modifiers
        self.parallel = parallel
        self.jobs = []
        self.pr = ProcessResult()
        self.cancelled = False

    # 1. main thread
    def prepare(self):
        depsgraph = bpy.context.evaluated_depsgraph_get() if self.apply_modifiers else None
        for obj in self.objects:
            job = _Job()
            job.obj, job.name, job.dropped, job.error, job.out = obj, obj.name, 0, None, None
            try:
                job.src, job.face_map = blender_to_meshlib(obj, self.apply_modifiers, depsgraph, with_face_map=True)
            except MeshCollapsedError as e:
                self.pr.collapsed.append((obj, e))
                continue
            except Exception as e:
                self.pr.errors.append((obj, e))
                continue
            job.res, job.ratio, job.mode = _effective_settings(obj, self.settings, self.resolution, self.smart)
            job.face_tags = None
            if self.islands and self.smart and job.face_map is not None:
                poly_tags = read_face_tags(obj.data) if not self.apply_modifiers else None
                if poly_tags is not None and poly_tags.size and job.face_map.max(initial=-1) < poly_tags.size:
                    job.face_tags = poly_tags[job.face_map]
            job.initial_verts = job.src.topology.numValidVerts()
            self.jobs.append(job)
        return self

    # 2. any thread (no bpy)
    def compute(self, control=None):
        def work(job):
            try:
                if control is not None:
                    control.check()
                _compute(job, self.operation_fn, self.auto_decimate, self.islands, self.smart)
            except Exception as e:  # recorded per job, handled on the main thread
                job.error = e
            finally:
                if control is not None:
                    control.item_done()

        if self.parallel and len(self.jobs) > 1:
            with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(self.jobs))) as ex:
                list(ex.map(work, self.jobs))
        else:
            for job in self.jobs:
                work(job)
        self.cancelled = any(isinstance(j.error, JobCancelled) for j in self.jobs) or \
            (control is not None and control.cancelled)
        return self

    # 3. main thread
    def finish(self):
        """Write results back. After a cancel nothing is changed in the scene."""
        pr = self.pr
        if self.cancelled:
            return pr
        old_meshes = []
        for job in self.jobs:
            pr.dropped_islands += job.dropped
            try:
                job.obj.name  # the object may have been deleted while the job ran
            except ReferenceError:
                continue
            if isinstance(job.error, MeshCollapsedError):
                pr.collapsed.append((job.obj, job.error))
                continue
            if job.error is not None:
                pr.errors.append((job.obj, job.error))
                continue
            try:
                new_mesh = meshlib_to_blender_mesh(job.out, job.name + self.suffix)
            except MeshCollapsedError as e:
                pr.collapsed.append((job.obj, e))
                continue
            voxel = min(job.voxels_used) if job.voxels_used else self.resolution
            write_mesh_resolution_tags(new_mesh, job.out_tags, remesh_voxel_size=voxel)
            if self.replace:
                old_meshes.append(swap_mesh_keep_transforms(job.obj, new_mesh))
                result = job.obj
            else:
                result = link_new_object(job.name + self.suffix, new_mesh,
                                         list(job.obj.users_collection) or [bpy.context.scene.collection])
            set_object_resolution_label(result, _dominant_tag_label(job.out_tags))
            pr.results.append((result, job.initial_verts, job.out.topology.numValidVerts()))

        # One depsgraph update for the whole run, then free the replaced meshes.
        if old_meshes:
            free_unused_meshes(old_meshes)
        else:
            bpy.context.view_layer.update()
        return pr


def process_objects(objects, operation_fn, output_suffix, **kwargs):
    """Run operation_fn(meshlib_mesh, voxel_size) -> meshlib_mesh on each object, synchronously.

    - Islands: each loose part processed separately, then merged into one object.
    - Smart: per object (or per island) voxel size + ratio from the resolution tag.
    - Replace: the object keeps its transform and materials and gets the new mesh;
      otherwise a new object named <name><suffix> is added to the same collections.
    Objects are processed in parallel (meshlib releases the GIL); CUDA steps are
    serialised inside cuda_offset. Returns a ProcessResult.
    """
    return PipelineRun(objects, operation_fn, output_suffix, **kwargs).prepare().compute().finish()


# ---------------------------------------------------------------------------
# Run-time estimate (asked before starting very long jobs)
# ---------------------------------------------------------------------------
#
# How it works: a trial offset at a 4x coarser voxel (a few ms) measures how big
# the real result will be (result faces = 16 x trial faces, accurate to ~1% on
# 6,240 timed runs). The work is then priced with coefficients fitted on 168
# real operator runs on Infill_Test.blend (Grow/Remesh/Trim Thin/Trim Edges on
# the four test objects at all presets with Islands on and off, plus batches of
# 12 and 48 wall pieces), RTX 5090 / Ryzen 9950X3D, Nov 2026. On those runs 99%
# of predictions were within 2x of the real time. Held-out check on 53 different runs
# (other wall pieces, Smart Res, Fix Undercuts): 79% within 1.25x, 91% within
# 1.5x, all within 2x (worst 0.51x: Smart + Islands on a sculpt with mixed tags).
# Fix Undercuts terms come from 1,088 per-direction timings (median 1.03x).
# The estimate then keeps learning: see EstimateLearner below.

COEF = {
    "mvox": 0.0096,        # s per million voxels, per offset
    "in_faces": 0.412,     # s per million input faces, per offset
    "dec_single": 0.384,   # s per million result faces decimated on one thread (< 200k faces)
    "bool_faces": 0.2036,  # s per million result faces, per voxel boolean
    "bool_call": 0.0025,   # s per voxel boolean
    "result": 0.1529,      # s per million result faces (decimation, write-back, tags)
    "job": 0.12,           # s per job (operator overhead, scene update)
    "object": 0.03,        # s per object
    "island": 0.0158,      # s per island when Islands is on
    "uc_mvox": 0.0124,     # Fix Undercuts: s per million voxels, per direction
    "uc_in_faces": 0.41,   # Fix Undercuts: s per million input faces, per direction
    "uc_call": 0.024,      # Fix Undercuts: s per direction
}
PROBE_FACTOR = 4.0          # trial voxel = 4x coarser -> 64x fewer voxels
PROBE_MAX_MVOX = 20.0       # keep the trial itself fast on huge grids


def _grid_mvox(lo, hi, voxel, grow):
    ext = (np.asarray(hi) - np.asarray(lo)) + 2.0 * abs(grow) + 4.0 * voxel
    return float(np.prod(np.ceil(np.maximum(ext, voxel) / voxel))) / 1e6


class ToolCost:
    """How much work a tool does per processed part (object or island)."""

    def __init__(self, offsets=1, decimates=1, booleans=0, undercut_dirs=0, scale=1.0):
        self.offsets = offsets
        self.decimates = decimates
        self.booleans = booleans
        self.undercut_dirs = undercut_dirs
        self.scale = scale  # fitted correction for work the terms don't capture


GROW_LIKE = ToolCost()


def object_parts(obj, voxel, grow=0.0, *, islands=False, apply_modifiers=False):
    """[(million voxels, million input faces, million result faces)] per processed part."""
    from .offset_utils import cuda_offset
    mm, _ = get_meshlib()
    src = blender_to_meshlib(obj, apply_modifiers)
    box = src.computeBoundingBox()
    mvox = _grid_mvox([box.min.x, box.min.y, box.min.z], [box.max.x, box.max.y, box.max.z], voxel, grow)
    factor = PROBE_FACTOR * max(1.0, (mvox / PROBE_FACTOR ** 3 / PROBE_MAX_MVOX) ** (1.0 / 3.0))
    probe = cuda_offset(src, voxel * factor, grow)
    mfaces = face_count(probe) * factor * factor / 1e6
    in_faces = face_count(src) / 1e6

    comps = mm.MeshComponents.getAllComponents(mm.MeshPart(src)) if islands else []
    if len(comps) <= 1:
        return [(mvox, in_faces, mfaces)]
    total_in = max(1, face_count(src))
    parts = []
    for bs in comps:
        b = src.computeBoundingBox(bs)
        share = bs.count() / total_in
        parts.append((_grid_mvox([b.min.x, b.min.y, b.min.z], [b.max.x, b.max.y, b.max.z], voxel, grow),
                      in_faces * share, mfaces * share))
    return parts


def parts_seconds(parts, cost, decimate=True):
    k = COEF
    t = k["object"]
    if len(parts) > 1:
        t += k["island"] * len(parts)
    for pv, pin, pf in parts:
        t += cost.offsets * (k["mvox"] * pv + k["in_faces"] * pin)
        t += cost.undercut_dirs * (k["uc_mvox"] * pv + k["uc_in_faces"] * pin + k["uc_call"])
        t += cost.booleans * (k["bool_faces"] * pf + k["bool_call"])
        if decimate and pf * 1e6 <= PARALLEL_DECIMATE_FACES:
            t += cost.decimates * k["dec_single"] * pf
        t += k["result"] * pf
    return t * cost.scale


def predict_object(obj, voxel, grow=0.0, *, islands=False, decimate=True, apply_modifiers=False, cost=GROW_LIKE):
    return parts_seconds(object_parts(obj, voxel, grow, islands=islands, apply_modifiers=apply_modifiers), cost, decimate)


def estimate_seconds(objects, voxel, *, grow=0.0, decimate=True, islands=False, cost=GROW_LIKE, apply_modifiers=False,
                     smart=False):
    """Predicted seconds for a job on `objects` (before learning; see EstimateLearner).

    With Smart on, each object is priced at its own tagged voxel size, as the job will run it.
    """
    total = COEF["job"] * cost.scale
    for obj in objects:
        if obj.type != 'MESH' or len(obj.data.polygons) == 0:
            continue
        v = voxel
        if smart:
            tag = read_object_resolution_tag(obj)
            if tag:
                v = RESOLUTION_PRESETS[tag][0]
        total += predict_object(obj, v, grow, islands=islands, decimate=decimate, cost=cost,
                                apply_modifiers=apply_modifiers)
    return total


class EstimateLearner:
    """Corrects estimates with this machine's real timings, per tool.

    After each job that was estimated, actual/predicted is remembered (last 12
    per tool, this Blender session); later estimates are multiplied by the
    median of those ratios, clamped to 1/3x..3x.
    """
    _ratios = {}

    @classmethod
    def correct(cls, tool, seconds):
        r = cls._ratios.get(tool)
        if not r:
            return seconds
        return seconds * float(np.clip(np.median(r), 1 / 3, 3.0))

    @classmethod
    def record(cls, tool, predicted, actual):
        if predicted > 0.05 and actual > 0.05:
            r = cls._ratios.setdefault(tool, [])
            r.append(actual / predicted)
            del r[:-12]