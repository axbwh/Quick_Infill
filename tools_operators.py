"""
Offset tool operators: Grow, Shrink, Remesh, Trim Thin, Trim Edges.

They share one base class (_OffsetToolBase). Each tool only defines how to build
its meshlib operation; selection, settings, batching, islands, decimation,
reporting and collapse handling live in the base class and the pipeline.
"""

import bpy
from bpy.types import Operator

from .meshlib_utils import get_meshlib
from .offset_utils import (
    cuda_offset, decimate_mesh, target_faces_for_density, bbox_of, face_count,
    OFFSET_PRESETS, nearest_resolution_tag, MeshCollapsedError,
)
from .blender_meshlib_utils import PipelineRun, select_results, remove_objects, estimate_seconds, ToolCost, Retagged
from .job_runner import ModalJobMixin, JobSpec
import numpy as np


def RESOLUTION_PRESETS_RATIO(tag):
    from .offset_utils import RESOLUTION_PRESETS
    return RESOLUTION_PRESETS.get(tag, RESOLUTION_PRESETS['STONE'])[1]


def selected_meshes(context):
    return [o for o in context.selected_objects if o.type == 'MESH']


def report_process(op, pr, verb, label):
    """Shared INFO / WARNING / ERROR reporting for a ProcessResult."""
    n = len(pr.results)
    if n == 1:
        r = pr.results[0][0]
        op.report({'INFO'}, f"{label} completed. {verb} '{r.name}'")
    elif n > 1:
        op.report({'INFO'}, f"{label} completed. {verb} {n} objects")
    if pr.dropped_islands:
        op.report({'WARNING'}, f"{label}: {pr.dropped_islands} island(s) collapsed or failed and were left out")
    if pr.errors:
        op.report({'ERROR'}, f"{label} failed on {len(pr.errors)} object(s), left unchanged: {pr.error_summary()}")


def pipeline_spec(objs, operation, suffix, s, finish, on_prepared=None, **kwargs):
    """JobSpec running `operation` on `objs` through the shared pipeline;
    finish(ProcessResult) -> operator result set.
    on_prepared(run): called on the main thread once the meshes are read (run.jobs), before the
    work starts: for a tool whose work on one object looks at the others."""
    run = PipelineRun(objs, operation, suffix,
                      resolution=kwargs.pop("resolution", float(s.shared_voxel_size)),
                      auto_decimate=kwargs.pop("auto_decimate", s.decimate_mode != "OFF"),
                      replace_original=s.replace_original, settings=s, **kwargs)

    def prepare():
        out = run.prepare()
        if on_prepared is not None:
            on_prepared(run)
        return out

    return JobSpec(prepare=prepare, compute=run.compute, finish=lambda: finish(run.finish()), items=len(objs))


class _OffsetToolBase(ModalJobMixin):
    bl_options = {'REGISTER', 'UNDO'}
    suffix = ""
    label = ""
    # Trim tools delete objects whose geometry fully collapses; other tools leave them alone.
    delete_collapsed = False
    collapse_hint = "geometry collapsed"
    # For the run-time estimate: work per object and the grow distance used (x Distance).
    est_cost = ToolCost(offsets=1, decimates=1)
    est_grow = 1.0

    def process_job(self, s):
        """(qi_worker function name, extra args) to run the objects in worker processes, or None."""
        return None

    @classmethod
    def poll(cls, context):
        return any(o.type == 'MESH' for o in context.selected_objects)

    def make_operation(self, s):
        """Return fn(meshlib_mesh, voxel_size) -> meshlib_mesh."""
        raise NotImplementedError

    def job_estimate(self, context):
        s = context.scene.quick_infill_settings
        return estimate_seconds(selected_meshes(context), float(s.shared_voxel_size), cost=self.est_cost,
                                grow=self.est_grow * float(s.distance), decimate=s.decimate_mode != "OFF",
                                islands=s.process_islands, apply_modifiers=s.apply_modifiers_on_export,
                                smart=s.smart_resolution)

    def job_setup(self, context):
        s = context.scene.quick_infill_settings
        objs = selected_meshes(context)
        if not objs:
            self.report({'ERROR'}, "No mesh selected.")
            return None
        # bpy.context (not the captured one): finish may run later, from the modal handler.
        return pipeline_spec(objs, self.make_operation(s), self.suffix, s, lambda pr: self.job_finish(bpy.context, pr),
                             process_job=self.process_job(s))

    def job_finish(self, context, pr):
        s = context.scene.quick_infill_settings
        if pr.collapsed:
            names = ", ".join(f"'{o.name}'" for o, _ in pr.collapsed)
            if self.delete_collapsed:
                remove_objects([o for o, _ in pr.collapsed])
                self.report({'WARNING'}, f"{self.label}: {len(pr.collapsed)} object(s) fully removed ({self.collapse_hint}): {names}")
            else:
                self.report({'WARNING'}, f"{self.label}: {len(pr.collapsed)} object(s) collapsed and were left unchanged: {names}")

        select_results([r[0] for r in pr.results])
        report_process(self, pr, "Updated" if s.replace_original else "Created", self.label)
        return {'FINISHED'} if pr.results or pr.collapsed else {'CANCELLED'}


class _FlatAwareOffset(_OffsetToolBase):
    """Grow / Shrink. An object made only of flat polygons is offset in its plane with the
    outline method (flat_outline). An object with real volume AND flat polygons loses the
    flat polygons and the rest goes through the voxel offset as usual (Alex 2026-11-16)."""
    sign = 1.0

    def make_operation(self, s):
        d = self.sign * float(s.distance)
        return lambda mesh, res: cuda_offset(mesh, res, d)

    def job_setup(self, context):
        from . import flat_outline
        s = context.scene.quick_infill_settings
        objs = selected_meshes(context)
        if not objs:
            self.report({'ERROR'}, "No mesh selected.")
            return None
        apply_mods = s.apply_modifiers_on_export
        kinds = {o: flat_outline.classify(o, apply_mods) for o in objs}
        flat = [o for o in objs if kinds[o] == 'flat']
        volume = [o for o in objs if kinds[o] != 'flat']
        mixed = [o for o in volume if kinds[o] == 'mixed']
        d = self.sign * float(s.distance)

        def finish(pr):
            for o in flat:
                try:
                    r = flat_outline.offset_object(o, d, replace=s.replace_original, suffix=self.suffix,
                                                   apply_modifiers=apply_mods,
                                                   clean=flat_outline.clean_from_settings(s))
                except Exception as e:
                    pr.errors.append((o, e))
                    continue
                if r is None:
                    pr.collapsed.append((o, "polygon vanished"))
                else:
                    pr.results.append((r, 0, len(r.data.vertices)))
            if flat:
                bpy.context.view_layer.update()
            return self.job_finish(bpy.context, pr)

        run = PipelineRun(volume, self.make_operation(s), self.suffix,
                          resolution=float(s.shared_voxel_size), auto_decimate=s.decimate_mode != "OFF",
                          replace_original=s.replace_original, settings=s, strip_flat=mixed)
        return JobSpec(prepare=run.prepare, compute=run.compute, finish=lambda: finish(run.finish()),
                       items=max(1, len(volume)))


class QUICKINFILL_OT_grow(_FlatAwareOffset, Operator):
    bl_idname = "quick_infill.grow"
    bl_label = "Grow"
    bl_description = ("Push the surface of the selected meshes outward by Distance.\n"
                      "Each object is worked on its own. Flat polygons grow in their own plane")
    suffix, label = "_Grown", "Grow"
    sign = 1.0


class QUICKINFILL_OT_shrink(_FlatAwareOffset, Operator):
    bl_idname = "quick_infill.shrink"
    bl_label = "Shrink"
    bl_description = ("Pull the surface of the selected meshes inward by Distance.\n"
                      "Each object is worked on its own. Flat polygons shrink in their own plane")
    suffix, label = "_Shrunk", "Shrink"
    est_grow = 0.0
    sign = -1.0


class QUICKINFILL_OT_remesh(_OffsetToolBase, Operator):
    bl_idname = "quick_infill.remesh"
    bl_label = "Remesh"
    bl_description = ("Rebuild the selected meshes at the voxel size.\n"
                      "Each object is worked on its own. The shape stays, the faces are made new")
    suffix, label = "_Remeshed", "Remesh"
    est_grow = 0.0

    def make_operation(self, s):
        return lambda mesh, res: cuda_offset(mesh, res, 0.0)


# The Round method of Trim Edges and Trim Thin live in trim_round.py (pure meshlib, also run by worker
# processes).
from .trim_round import STONE_KEEPS  # noqa: F401, E402


class QUICKINFILL_OT_trim_thin(_OffsetToolBase, Operator):
    bl_idname = "quick_infill.trim_thin"
    bl_label = "Trim Thin"
    bl_description = ("Remove the parts of the selected meshes that are too thin to print.\n"
                      "Anything thinner than two voxels goes. A mesh that is thin all over is deleted")
    suffix, label = "_TrimThin", "Trim Thin"
    est_cost, est_grow = ToolCost(offsets=2, decimates=1), 0.0
    delete_collapsed = True
    collapse_hint = "too thin for current resolution"

    def make_operation(self, s):
        from .trim_round import trim_thin
        return lambda mesh, res: trim_thin(mesh, res)[0]

    def process_job(self, s):
        """Many objects run in worker processes (the same trim_round.trim_thin and decimation there)."""
        return ("trim_thin_job", ("thin",))


DENT_COLL = "Dent_S"     # Ashlarize's small stone damage meshes: the default Chip Meshes


def _dent_collection(context):
    """The Chip Meshes collection: the setting, else Ashlarize's Dent_S (its own pointer, this file, or
    linked from Ashlarize's stone library file, the way Ashlarize links it). None when there is none."""
    s = context.scene.quick_infill_settings
    if s.trim_edges_chip_meshes is not None:
        return s.trim_edges_chip_meshes
    sp = getattr(context.scene, "stone_pack", None)
    c = getattr(sp, "dent_lib_s", None) if sp is not None else None
    if c is not None:
        return c
    c = bpy.data.collections.get(DENT_COLL)
    if c is not None:
        return c
    path = getattr(sp, "library_path", "") if sp is not None else ""
    if path:
        try:
            with bpy.data.libraries.load(bpy.path.abspath(path), link=True) as (src, dst):
                if DENT_COLL not in src.collections:
                    return None
                dst.collections = [DENT_COLL]
            return dst.collections[0]
        except Exception as e:
            print("Quick Infill chip meshes:", e)
    return None


def chip_meshes(context):
    """The Chip Meshes as [(V (n,3), F (k,3))] in Blender units (each object's scale applied, not its
    place or turn: Z is out of the stone)."""
    c = _dent_collection(context)
    if c is None:
        return []
    out = []
    for o in c.all_objects:
        if o.type != 'MESH' or not len(o.data.polygons):
            continue
        me = o.data
        V = np.empty(len(me.vertices) * 3, np.float64)
        me.vertices.foreach_get("co", V)
        me.calc_loop_triangles()
        T = np.empty(len(me.loop_triangles) * 3, np.int32)
        me.loop_triangles.foreach_get("vertices", T)
        out.append((V.reshape(-1, 3) * np.array(o.scale[:], np.float64), T.reshape(-1, 3)))
    return out


class QUICKINFILL_OT_trim_edges(_OffsetToolBase, Operator):
    bl_idname = "quick_infill.trim_edges"
    bl_label = "Trim Edges"
    bl_description = ("Take the thin, sharp edges off the selected meshes.\n"
                      "Round (default): rounds them off as the old Trim Edges did, about 5 times faster; a stone "
                      "too small for the Stone resolution is trimmed at Detail instead and tagged Detail. "
                      "Chip: stone damage meshes knock chips off the sharp edges and corners, then Trim Thin")
    suffix, label = "_TrimEdges", "Trim Edges"
    est_cost, est_grow = ToolCost(offsets=3, decimates=2, booleans=1), 2.0
    delete_collapsed = True
    collapse_hint = "mesh collapsed during trim"

    def job_setup(self, context):
        s = context.scene.quick_infill_settings
        if getattr(s, "trim_edges_method", "ROUND") != "CHIP":
            return self._lean_round_setup(context, s)
        return self._chip_setup(context, s)

    def _chip_setup(self, context, s):
        """Chip (Alex 2026-11-18, in place of the old Flat Chip): stone damage meshes knocked off the
        sharp edges and corners (trim_dent), then Trim Thin. Trim Thin works through voxels, so the
        result is decimated the usual way afterwards."""
        objs = selected_meshes(context)
        if not objs:
            self.report({'ERROR'}, "No mesh selected.")
            return None
        dents = chip_meshes(context)
        if not dents:
            self.report({'ERROR'}, "Trim Edges (Chip): no Chip Meshes. Set Chip Meshes to a collection of "
                                   "damage meshes (Ashlarize's Dent_S), or set Ashlarize's stone library")
            return None
        from .trim_dent import dent_trim
        from .trim_chip import detect_cut_faces
        from .trim_round import trim_thin
        from .blender_meshlib_utils import STL_EXPORT_SCALE
        size, amount = float(s.trim_edges_chip_size), float(s.trim_edges_chip_amount)
        only_cut = bool(s.trim_edges_only_cut)
        marks = {}   # id(meshlib mesh) -> cut flag per meshlib face, from Ashlarize's sp_cut face marks

        def on_prepared(run):
            if not only_cut:
                return
            for job in run.jobs:
                att = job.obj.data.attributes.get("sp_cut")
                if att is None or att.domain != 'FACE' or job.face_map is None:
                    continue           # no marks: the cut faces are found by shape
                vals = [False] * len(job.obj.data.polygons)
                att.data.foreach_get("value", vals)
                marks[id(job.src)] = np.asarray(vals, bool)[job.face_map]

        def chip(mesh, res):
            cut = None
            if only_cut:
                cut = marks.get(id(mesh))
                if cut is None:
                    cut = detect_cut_faces(mesh, STL_EXPORT_SCALE)
                if cut is None or not np.any(cut):
                    return trim_thin(mesh, res)[0]          # no cut faces: no chips, Trim Thin only
            return dent_trim(mesh, res, dents, size, amount, STL_EXPORT_SCALE, cut_faces=cut)[0]

        # Many stones run in worker processes (the same trim_dent there); not with Only Cut Edges,
        # whose marks are per object
        return pipeline_spec(objs, chip, self.suffix, s, lambda pr: self.job_finish(bpy.context, pr),
                             on_prepared=on_prepared,
                             process_job=None if only_cut else
                             ("trim_edges_job", ("chip", dents, size, amount, STL_EXPORT_SCALE)))

    def _lean_round_setup(self, context, s):
        """Round: today's Round with fewer steps (trim_chip.lean_round): offsets, one decimate, an
        exact intersect with the stone, one decimate to the same face count. Stone -> Detail kept."""
        objs = selected_meshes(context)
        if not objs:
            self.report({'ERROR'}, "No mesh selected.")
            return None
        from .trim_round import lean_trim
        from .blender_meshlib_utils import STL_EXPORT_SCALE
        distance, x, density = float(s.distance), float(s.trim_edges_x), float(s.trim_edges_density)

        def rnd(mesh, res):
            out, tag = lean_trim(mesh, res, distance, x, density, STL_EXPORT_SCALE)
            return out if tag is None else Retagged(out, tag)

        # lean_round decimates to today's face count itself: no pipeline decimation afterwards.
        # Many stones run in worker processes (the same lean_trim there).
        return pipeline_spec(objs, rnd, self.suffix, s, lambda pr: self.job_finish(bpy.context, pr),
                             auto_decimate=False,
                             process_job=("trim_edges_job", ("lean", distance, x, density, STL_EXPORT_SCALE)))

    def job_estimate(self, context):
        s = context.scene.quick_infill_settings
        if getattr(s, "trim_edges_method", "ROUND") == "CHIP":
            return 0.15 * len(selected_meshes(context))
        return 0.12 * len(selected_meshes(context))


class QUICKINFILL_OT_create_outline(Operator):
    bl_idname = "quick_infill.create_outline"
    bl_label = "Create Outline"
    bl_description = ("Make a flat outline of each selected mesh, as seen from the current view.\n"
                      "A new flat polygon (the same outline method as Ashlarize). Slits are closed by Distance, "
                      "bumps trimmed by half of it, and the number of points follows Decimate")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return any(o.type == 'MESH' for o in context.selected_objects)

    def execute(self, context):
        from . import flat_outline
        from mathutils import Vector
        # From the sidebar button context.region_data is not the viewport: use the 3D view's own.
        sp = context.space_data
        rv3d = sp.region_3d if sp is not None and sp.type == 'VIEW_3D' else None
        if rv3d is None:   # no 3D view (scripts): look straight down
            view_dir = Vector((0.0, 0.0, -1.0))
        else:
            view_dir = rv3d.view_rotation @ Vector((0.0, 0.0, -1.0))
        s = context.scene.quick_infill_settings
        made = []
        for o in selected_meshes(context):
            try:
                r = flat_outline.create_outline(o, tuple(view_dir),
                                                clean=flat_outline.clean_from_settings(s))
            except Exception as e:
                self.report({'ERROR'}, f"Create Outline failed on '{o.name}': {e}")
                continue
            if r is None:
                self.report({'WARNING'}, f"Create Outline: '{o.name}' has no outline from this view")
            else:
                made.append(r)
        if not made:
            return {'CANCELLED'}
        select_results(made)
        self.report({'INFO'}, f"Create Outline: made {len(made)} outline(s)")
        return {'FINISHED'}


classes = (
    QUICKINFILL_OT_grow,
    QUICKINFILL_OT_shrink,
    QUICKINFILL_OT_remesh,
    QUICKINFILL_OT_trim_thin,
    QUICKINFILL_OT_trim_edges,
    QUICKINFILL_OT_create_outline,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
