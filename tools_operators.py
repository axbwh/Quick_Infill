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
)
from .blender_meshlib_utils import PipelineRun, select_results, remove_objects, estimate_seconds, ToolCost
from .job_runner import ModalJobMixin, JobSpec
import numpy as np


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


def pipeline_spec(objs, operation, suffix, s, finish, **kwargs):
    """JobSpec running `operation` on `objs` through the shared pipeline;
    finish(ProcessResult) -> operator result set."""
    run = PipelineRun(objs, operation, suffix,
                      resolution=kwargs.pop("resolution", float(s.shared_voxel_size)),
                      auto_decimate=kwargs.pop("auto_decimate", s.decimate_mode != "OFF"),
                      replace_original=s.replace_original, settings=s, **kwargs)
    return JobSpec(prepare=run.prepare, compute=run.compute, finish=lambda: finish(run.finish()), items=len(objs))


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
        return pipeline_spec(objs, self.make_operation(s), self.suffix, s, lambda pr: self.job_finish(bpy.context, pr))

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


class QUICKINFILL_OT_grow(_OffsetToolBase, Operator):
    bl_idname = "quick_infill.grow"
    bl_label = "Grow"
    bl_description = "Grow selected mesh(es) by the distance value. With multiple selections, processes each object individually"
    suffix, label = "_Grown", "Grow"

    def make_operation(self, s):
        d = float(s.distance)
        return lambda mesh, res: cuda_offset(mesh, res, d)


class QUICKINFILL_OT_shrink(_OffsetToolBase, Operator):
    bl_idname = "quick_infill.shrink"
    bl_label = "Shrink"
    bl_description = "Shrink selected mesh(es) by the distance value. With multiple selections, processes each object individually"
    suffix, label = "_Shrunk", "Shrink"
    est_grow = 0.0

    def make_operation(self, s):
        d = float(s.distance)
        return lambda mesh, res: cuda_offset(mesh, res, -d)


class QUICKINFILL_OT_remesh(_OffsetToolBase, Operator):
    bl_idname = "quick_infill.remesh"
    bl_label = "Remesh"
    bl_description = "Remesh selected mesh(es) at the resolution value. With multiple selections, processes each object individually"
    suffix, label = "_Remeshed", "Remesh"
    est_grow = 0.0

    def make_operation(self, s):
        return lambda mesh, res: cuda_offset(mesh, res, 0.0)


class QUICKINFILL_OT_trim_thin(_OffsetToolBase, Operator):
    bl_idname = "quick_infill.trim_thin"
    bl_label = "Trim Thin"
    bl_description = "Remove thin sections from selected mesh(es). With multiple selections, processes each object individually"
    suffix, label = "_TrimThin", "Trim Thin"
    est_cost, est_grow = ToolCost(offsets=2, decimates=1), 0.0
    delete_collapsed = True
    collapse_hint = "too thin for current resolution"

    def make_operation(self, s):
        def trim_thin(mesh, res):
            shrunk = ensure_not_collapsed(cuda_offset(mesh, res, -res))
            return cuda_offset(shrunk, res, res)
        return trim_thin


class QUICKINFILL_OT_trim_edges(_OffsetToolBase, Operator):
    bl_idname = "quick_infill.trim_edges"
    bl_label = "Trim Edges"
    bl_description = "Trim edges by offset sequence and intersect with original mesh. With multiple selections, processes each object individually"
    suffix, label = "_TrimEdges", "Trim Edges"
    est_cost, est_grow = ToolCost(offsets=3, decimates=2, booleans=1), 2.0
    delete_collapsed = True
    collapse_hint = "mesh collapsed during trim"

    def make_operation(self, s):
        from .support_tools import intersect_meshes
        distance = float(s.distance)
        x = float(s.trim_edges_x)
        density = float(s.trim_edges_density)

        # Grow 2d, shrink 3d (rounds off thin edges), regrow (1+x)d, decimate to the
        # density target, then intersect with the original to clip back to its silhouette.
        # The input mesh is never modified by these steps, so no copies are needed.
        def trim_edges(mesh, res):
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
            return intersect_meshes(w, mesh, res)
        return trim_edges


classes = (
    QUICKINFILL_OT_grow,
    QUICKINFILL_OT_shrink,
    QUICKINFILL_OT_remesh,
    QUICKINFILL_OT_trim_thin,
    QUICKINFILL_OT_trim_edges,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
