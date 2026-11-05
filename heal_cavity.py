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
from .blender_meshlib_utils import blender_to_meshlib, meshlib_to_blender_mesh, link_new_object, select_results


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


class QUICKINFILL_OT_heal_cavity(Operator):
    bl_idname = "quick_infill.heal_cavity"
    bl_label = "Heal Cavity"
    bl_description = "Generate an infill object for each selected mesh using CUDA offset operations"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.quick_infill_settings
        objs = [o for o in context.selected_objects if o.type == 'MESH']
        if not objs:
            self.report({'ERROR'}, "No mesh selected.")
            return {'CANCELLED'}
        # Active object first, so a single click behaves as before.
        if context.active_object in objs:
            objs.remove(context.active_object)
            objs.insert(0, context.active_object)

        created, failed = [], []
        for obj in objs:
            try:
                src = blender_to_meshlib(obj, s.apply_modifiers_on_export)
                out = build_infill(src, s, self.report)
                if face_count(out) == 0:
                    raise RuntimeError("the infill came out empty")
                name = obj.name + "Infill"
                created.append(link_new_object(name, meshlib_to_blender_mesh(out, name),
                                               list(obj.users_collection) or [context.scene.collection]))
            except Exception as e:
                failed.append((obj.name, e))
                print(f"[Quick Infill] Heal Cavity error on '{obj.name}': {e}")

        bpy.context.view_layer.update()
        if created:
            select_results(created)
            names = ", ".join(f"'{o.name}'" for o in created)
            self.report({'INFO'}, f"Heal Cavity completed. Created {names}")
        if failed:
            self.report({'ERROR'}, "Heal Cavity failed: " + "; ".join(f"{n}: {e}" for n, e in failed[:3]))
        return {'FINISHED'} if created else {'CANCELLED'}


def register():
    bpy.utils.register_class(QUICKINFILL_OT_heal_cavity)


def unregister():
    bpy.utils.unregister_class(QUICKINFILL_OT_heal_cavity)
