"""
Create Cavity Infill: builds an infill object (<name>Infill) that fills
resin-trapping cavities. Uses the Infill settings (resolution, grow, shrink
multiplier, working resolution), which are separate from the voxel tools.
"""

import bpy
from bpy.types import Operator

from .meshlib_utils import get_meshlib
from .offset_utils import (
    cuda_offset, weighted_dist_shell, compute_voxel_size, decimate_mesh,
    should_auto_decimate_cavity_faces, face_count,
)
from .blender_meshlib_utils import (
    blender_to_meshlib, meshlib_to_blender_mesh, link_new_object, select_results, estimate_seconds, object_voxel_stats,
)
from .meshlib_utils import JobCancelled
from .job_runner import ModalJobMixin, JobSpec


def build_infill(src_mesh, s, report=None):
    """meshlib mesh in -> infill meshlib mesh out, using the Infill settings `s`."""
    mm, _ = get_meshlib()
    working_res = int(float(s.target_res) * 1_000_000)
    resolution = float(s.resolution)
    grow = float(s.grow)
    shrink_mult = float(s.shrink_mult)

    initial_faces = face_count(src_mesh)
    initial_verts = src_mesh.topology.numValidVerts()
    if initial_verts > working_res:
        src_mesh = decimate_mesh(src_mesh, reduction_ratio=working_res / initial_verts)
        if report:
            report({'INFO'}, f"Decimated mesh: {initial_verts} → {src_mesh.topology.numValidVerts()} vertices")

    vox = resolution if s.voxel_mode == 'RESOLUTION' else compute_voxel_size(src_mesh, working_res, resolution)

    if s.method == "NAIVE":
        grown = cuda_offset(src_mesh, vox, grow)
        out = cuda_offset(grown, vox, -grow * shrink_mult)
    else:
        grown = cuda_offset(src_mesh, vox, grow)
        shrunk = cuda_offset(grown, vox, -grow)
        shell = weighted_dist_shell(shrunk, src_mesh, vox, shrink_mult, max_vertices=working_res, target_resolution=working_res)
        out = mm.boolean(shrunk, shell, mm.BooleanOperation.DifferenceAB).mesh

    if s.trim_thin:
        out = cuda_offset(cuda_offset(out, vox, -vox), vox, vox)

    do, target = should_auto_decimate_cavity_faces(initial_faces, face_count(out))
    if do:
        out = decimate_mesh(out, target_face_count=target, resolution=vox)
    return out


class _InfillSettingsSnapshot:
    """Plain copy of the Infill settings, safe to read from the worker thread."""

    def __init__(self, s):
        for k in ("target_res", "resolution", "grow", "shrink_mult", "voxel_mode", "method", "trim_thin"):
            setattr(self, k, getattr(s, k))


class QUICKINFILL_OT_heal_cavity(ModalJobMixin, Operator):
    bl_idname = "quick_infill.heal_cavity"
    bl_label = "Heal Cavity"
    bl_description = "Generate an infill object for each selected mesh using CUDA offset operations"
    bl_options = {'REGISTER', 'UNDO'}

    def _objects(self, context):
        objs = [o for o in context.selected_objects if o.type == 'MESH']
        # Active object first, so a single click behaves as before.
        if context.active_object in objs:
            objs.remove(context.active_object)
            objs.insert(0, context.active_object)
        return objs

    def job_estimate(self, context):
        s = context.scene.quick_infill_settings
        total = 0.0
        for obj in self._objects(context):
            vox = float(s.resolution)
            if s.voxel_mode == 'TARGET_VOXELS':
                mvox, _ = object_voxel_stats(obj, vox, s.grow)
                # suggestVoxelSize targets ~target_res voxels, never finer than Resolution
                target = max(1e-6, float(s.target_res))
                if mvox > target:
                    vox *= (mvox / target) ** (1.0 / 3.0)
            total += estimate_seconds([obj], vox, offsets=2 if s.method == 'NAIVE' else 4,
                                      grow=float(s.grow), decimate=True)
        return total

    def job_setup(self, context):
        s = context.scene.quick_infill_settings
        objs = self._objects(context)
        if not objs:
            self.report({'ERROR'}, "No mesh selected.")
            return None
        snap = _InfillSettingsSnapshot(s)
        apply_mods = s.apply_modifiers_on_export
        state = {"inputs": [], "outputs": [], "failed": [], "notes": []}

        def prepare():
            for obj in objs:
                try:
                    state["inputs"].append((obj, obj.name, blender_to_meshlib(obj, apply_mods)))
                except Exception as e:
                    state["failed"].append((obj.name, e))

        def compute(control):
            for obj, name, src in state["inputs"]:
                try:
                    if control is not None:
                        control.check()
                    out = build_infill(src, snap, lambda lvl, msg: state["notes"].append(msg))
                    if face_count(out) == 0:
                        raise RuntimeError("the infill came out empty")
                    state["outputs"].append((obj, name, out))
                except JobCancelled:
                    raise
                except Exception as e:
                    state["failed"].append((name, e))
                    print(f"[Quick Infill] Heal Cavity error on '{name}': {e}")
                finally:
                    if control is not None:
                        control.item_done()

        def finish():
            created = []
            for obj, name, out in state["outputs"]:
                try:
                    colls = list(obj.users_collection) or [bpy.context.scene.collection]
                except ReferenceError:
                    colls = [bpy.context.scene.collection]
                created.append(link_new_object(name + "Infill", meshlib_to_blender_mesh(out, name + "Infill"), colls))
            bpy.context.view_layer.update()
            for msg in state["notes"]:
                self.report({'INFO'}, msg)
            if created:
                select_results(created)
                self.report({'INFO'}, "Heal Cavity completed. Created " + ", ".join(f"'{o.name}'" for o in created))
            if state["failed"]:
                self.report({'ERROR'}, "Heal Cavity failed: " + "; ".join(f"{n}: {e}" for n, e in state["failed"][:3]))
            return {'FINISHED'} if created else {'CANCELLED'}

        return JobSpec(prepare, compute, finish, items=len(objs))


def register():
    bpy.utils.register_class(QUICKINFILL_OT_heal_cavity)


def unregister():
    bpy.utils.unregister_class(QUICKINFILL_OT_heal_cavity)
