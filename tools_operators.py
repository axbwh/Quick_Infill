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
    cuda_offset, decimate_mesh, target_faces_for_density, ensure_not_collapsed, bbox_of, face_count,
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
    # The pipeline's decimation runs in parallel parts above this many faces (None: its default).
    decimate_parallel_above = None

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
                             decimate_parallel_above=self.decimate_parallel_above,
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
        def trim_thin(mesh, res):
            shrunk = ensure_not_collapsed(cuda_offset(mesh, res, -res))
            return cuda_offset(shrunk, res, res)
        return trim_thin


# The Round method of Trim Edges lives in trim_round.py (pure meshlib, also run by worker processes).
from .trim_round import STONE_KEEPS, PARALLEL_FACES as TRIM_EDGES_PARALLEL_FACES  # noqa: F401


class QUICKINFILL_OT_trim_edges(_OffsetToolBase, Operator):
    bl_idname = "quick_infill.trim_edges"
    bl_label = "Trim Edges"
    bl_description = ("Take the thin, sharp edges off the selected meshes.\n"
                      "Flat Chip (default): a flat chip off each sharp edge, sized by Thin; Only Cut Edges "
                      "keeps it to where a cut face meets the stone. Round (old): the voxel way; a stone too "
                      "small for the Stone resolution is trimmed at Detail instead and tagged Detail")
    suffix, label = "_TrimEdges", "Trim Edges"
    est_cost, est_grow = ToolCost(offsets=3, decimates=2, booleans=1), 2.0
    delete_collapsed = True
    collapse_hint = "mesh collapsed during trim"
    decimate_parallel_above = TRIM_EDGES_PARALLEL_FACES

    def job_setup(self, context):
        s = context.scene.quick_infill_settings
        method = getattr(s, "trim_edges_method", "VOXEL")
        if method == "ROUND":
            return self._lean_round_setup(context, s)
        if method != "CHIP":
            return super().job_setup(context)
        objs = selected_meshes(context)
        if not objs:
            self.report({'ERROR'}, "No mesh selected.")
            return None
        from .trim_chip import fast_chip
        from .blender_meshlib_utils import STL_EXPORT_SCALE
        thin = float(s.trim_edges_thin)
        only_cut = bool(s.trim_edges_only_cut)
        marks = {}   # id(meshlib mesh) -> cut flag per meshlib face, from Ashlarize's sp_cut face marks

        def on_prepared(run):
            if not only_cut:
                return
            for job in run.jobs:
                att = job.obj.data.attributes.get("sp_cut")
                if att is None or att.domain != 'FACE' or job.face_map is None:
                    continue           # no marks: fast_chip finds the cut faces by shape
                vals = [False] * len(job.obj.data.polygons)
                att.data.foreach_get("value", vals)
                marks[id(job.src)] = np.asarray(vals, bool)[job.face_map]

        def chip(mesh, res):
            return fast_chip(mesh, thin, unit=STL_EXPORT_SCALE, only_cut=only_cut,
                             cut_faces=marks.get(id(mesh)), detect=True)

        # the chipped stone keeps its own mesh: no voxel-style decimation afterwards
        return pipeline_spec(objs, chip, self.suffix, s, lambda pr: self.job_finish(bpy.context, pr),
                             on_prepared=on_prepared, auto_decimate=False)

    def _lean_round_setup(self, context, s):
        """Round: today's Round with fewer steps (trim_chip.lean_round): offsets, one decimate, an
        exact intersect with the stone, one decimate to the same face count. Stone -> Detail kept."""
        objs = selected_meshes(context)
        if not objs:
            self.report({'ERROR'}, "No mesh selected.")
            return None
        from .trim_chip import lean_round
        from .blender_meshlib_utils import STL_EXPORT_SCALE
        distance, x, density = float(s.distance), float(s.trim_edges_x), float(s.trim_edges_density)

        def rnd(mesh, res):
            tag = nearest_resolution_tag(res)
            ratio = RESOLUTION_PRESETS_RATIO(tag)
            info = {}
            out = lean_round(mesh, unit=STL_EXPORT_SCALE, density=density, info=info, ratio=ratio,
                             stone=dict(voxel=float(res) / STL_EXPORT_SCALE, distance=distance / STL_EXPORT_SCALE, x=x),
                             fallback=(tag == 'STONE'))
            if face_count(out) == 0:
                raise MeshCollapsedError()
            return Retagged(out, 'DETAIL') if info.get("tag") == "DETAIL" else out

        # lean_round decimates to today's face count itself: no pipeline decimation afterwards
        return pipeline_spec(objs, rnd, self.suffix, s, lambda pr: self.job_finish(bpy.context, pr),
                             auto_decimate=False)

    def job_estimate(self, context):
        s = context.scene.quick_infill_settings
        method = getattr(s, "trim_edges_method", "VOXEL")
        if method == "CHIP":
            return 0.05 * len(selected_meshes(context))
        if method == "ROUND":
            return 0.12 * len(selected_meshes(context))
        return super().job_estimate(context)

    def make_operation(self, s):
        from .trim_round import trim_edges as round_trim
        distance, x, density = float(s.distance), float(s.trim_edges_x), float(s.trim_edges_density)

        # A stone too small for Stone comes back worked at Detail: tagged Detail, so every later tool
        # with Smart Res on treats it as Detail too (Alex 2026-11-17).
        def trim_edges(mesh, res):
            out, tag = round_trim(mesh, res, distance, x, density)
            return out if tag is None else Retagged(out, tag)
        return trim_edges

    def process_job(self, s):
        return ("trim_edges_job", (float(s.distance), float(s.trim_edges_x), float(s.trim_edges_density)))


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
