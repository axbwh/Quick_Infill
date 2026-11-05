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

from .meshlib_utils import get_meshlib, get_mrmeshnumpy
from .offset_utils import (
    RESOLUTION_PRESETS, RESOLUTION_TAG_ORDER, nearest_resolution_tag,
    decimate_mesh, should_auto_decimate_faces, face_count, MeshCollapsedError,
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
    for obj in bpy.context.selected_objects:
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
        try:
            out = operation_fn(island, res)
            if face_count(out) == 0:
                raise MeshCollapsedError()
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


def process_objects(objects, operation_fn, output_suffix, *, resolution, auto_decimate=True,
                    replace_original=True, settings=None, islands=None, smart=None, apply_modifiers=None,
                    parallel=True):
    """Run operation_fn(meshlib_mesh, voxel_size) -> meshlib_mesh on each object.

    - Islands: each loose part processed separately, then merged into one object.
    - Smart: per object (or per island) voxel size + ratio from the resolution tag.
    - Replace: the object keeps its transform and materials and gets the new mesh;
      otherwise a new object named <name><suffix> is added to the same collections.
    Objects are processed in parallel (meshlib releases the GIL); CUDA steps are
    serialised inside cuda_offset. Returns a ProcessResult.
    """
    s = settings or bpy.context.scene.quick_infill_settings
    islands = s.process_islands if islands is None else islands
    smart = s.smart_resolution if smart is None else smart
    apply_modifiers = s.apply_modifiers_on_export if apply_modifiers is None else apply_modifiers
    pr = ProcessResult()

    # 1. Gather inputs (main thread: bpy access)
    jobs = []
    depsgraph = bpy.context.evaluated_depsgraph_get() if apply_modifiers else None
    for obj in objects:
        job = _Job()
        job.obj, job.name, job.dropped, job.error, job.out = obj, obj.name, 0, None, None
        try:
            job.src, job.face_map = blender_to_meshlib(obj, apply_modifiers, depsgraph, with_face_map=True)
        except MeshCollapsedError as e:
            pr.collapsed.append((obj, e))
            continue
        except Exception as e:
            pr.errors.append((obj, e))
            continue
        job.res, job.ratio, job.mode = _effective_settings(obj, s, resolution, smart)
        job.face_tags = None
        if islands and smart and job.face_map is not None:
            poly_tags = read_face_tags(obj.data) if not apply_modifiers else None
            if poly_tags is not None and poly_tags.size and job.face_map.max(initial=-1) < poly_tags.size:
                job.face_tags = poly_tags[job.face_map]
        job.initial_verts = job.src.topology.numValidVerts()
        jobs.append(job)

    # 2. Compute (worker threads: meshlib only)
    def work(job):
        try:
            _compute(job, operation_fn, auto_decimate, islands, smart)
        except Exception as e:  # recorded per job, handled on the main thread
            job.error = e

    if parallel and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(jobs))) as ex:
            list(ex.map(work, jobs))
    else:
        for job in jobs:
            work(job)

    # 3. Write results back (main thread, original order)
    old_meshes = []
    for job in jobs:
        pr.dropped_islands += job.dropped
        if isinstance(job.error, MeshCollapsedError):
            pr.collapsed.append((job.obj, job.error))
            continue
        if job.error is not None:
            pr.errors.append((job.obj, job.error))
            continue
        try:
            new_mesh = meshlib_to_blender_mesh(job.out, job.name + output_suffix)
        except MeshCollapsedError as e:
            pr.collapsed.append((job.obj, e))
            continue
        voxel = min(job.voxels_used) if job.voxels_used else resolution
        write_mesh_resolution_tags(new_mesh, job.out_tags, remesh_voxel_size=voxel)
        if replace_original:
            old_meshes.append(swap_mesh_keep_transforms(job.obj, new_mesh))
            result = job.obj
        else:
            result = link_new_object(job.name + output_suffix, new_mesh,
                                     list(job.obj.users_collection) or [bpy.context.scene.collection])
        set_object_resolution_label(result, _dominant_tag_label(job.out_tags))
        pr.results.append((result, job.initial_verts, job.out.topology.numValidVerts()))

    # One depsgraph update for the whole run, then free the replaced meshes.
    if old_meshes:
        free_unused_meshes(old_meshes)
    else:
        bpy.context.view_layer.update()
    return pr
