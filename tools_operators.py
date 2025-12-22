"""
Tool operators for Quick Infill addon.
Provides Grow, Shrink, Remesh, and Trim Thin operations.
"""

import bpy
from bpy.types import Operator
from .offset_utils import cuda_offset, weighted_dist_shell
from .blender_meshlib_utils import process_mesh_operation, blender_to_meshlib_via_stl, meshlib_to_blender_via_stl


class QUICKINFILL_OT_grow(Operator):
    bl_idname = "quick_infill.grow"
    bl_label = "Grow"
    bl_description = "Grow the selected mesh by the distance value"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            settings = context.scene.quick_infill_tools_settings
            distance = float(settings.distance)
            resolution = float(settings.resolution)
            auto_decimate = settings.auto_decimate
            replace_original = settings.replace_original

            selected_objs = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
            if not selected_objs:
                self.report({'ERROR'}, "No mesh selected.")
                return {'CANCELLED'}

            src_obj = selected_objs[0]
            
            # Grow = positive offset
            def grow_op(mesh):
                return cuda_offset(mesh, resolution, distance)
            
            result_obj, initial_verts, final_verts = process_mesh_operation(
                src_obj, grow_op, "_Grown", auto_decimate=auto_decimate, replace_original=replace_original
            )
            
            print(f"[Quick Infill] Grow: {initial_verts} → {final_verts} vertices, offset +{distance}mm")
            if replace_original:
                self.report({'INFO'}, f"Grow completed. Updated '{result_obj.name}'")
            else:
                self.report({'INFO'}, f"Grow completed. Created '{result_obj.name}'")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Grow failed: {e}")
            print(f"[Quick Infill] Grow error: {e}")
            return {'CANCELLED'}


class QUICKINFILL_OT_shrink(Operator):
    bl_idname = "quick_infill.shrink"
    bl_label = "Shrink"
    bl_description = "Shrink the selected mesh by the distance value"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            settings = context.scene.quick_infill_tools_settings
            distance = float(settings.distance)
            resolution = float(settings.resolution)
            auto_decimate = settings.auto_decimate
            replace_original = settings.replace_original

            selected_objs = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
            if not selected_objs:
                self.report({'ERROR'}, "No mesh selected.")
                return {'CANCELLED'}

            src_obj = selected_objs[0]
            
            # Shrink = negative offset
            def shrink_op(mesh):
                return cuda_offset(mesh, resolution, -distance)
            
            result_obj, initial_verts, final_verts = process_mesh_operation(
                src_obj, shrink_op, "_Shrunk", auto_decimate=auto_decimate, replace_original=replace_original
            )
            
            print(f"[Quick Infill] Shrink: {initial_verts} → {final_verts} vertices, offset -{distance}mm")
            if replace_original:
                self.report({'INFO'}, f"Shrink completed. Updated '{result_obj.name}'")
            else:
                self.report({'INFO'}, f"Shrink completed. Created '{result_obj.name}'")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Shrink failed: {e}")
            print(f"[Quick Infill] Shrink error: {e}")
            return {'CANCELLED'}


class QUICKINFILL_OT_remesh(Operator):
    bl_idname = "quick_infill.remesh"
    bl_label = "Remesh"
    bl_description = "Remesh the selected mesh at the resolution value"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            settings = context.scene.quick_infill_tools_settings
            resolution = float(settings.resolution)
            auto_decimate = settings.auto_decimate
            replace_original = settings.replace_original

            selected_objs = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
            if not selected_objs:
                self.report({'ERROR'}, "No mesh selected.")
                return {'CANCELLED'}

            src_obj = selected_objs[0]
            
            # Remesh = offset with 0 distance (re-voxelizes)
            def remesh_op(mesh):
                return cuda_offset(mesh, resolution, 0.0)
            
            result_obj, initial_verts, final_verts = process_mesh_operation(
                src_obj, remesh_op, "_Remeshed", auto_decimate=auto_decimate, replace_original=replace_original
            )
            
            print(f"[Quick Infill] Remesh: {initial_verts} → {final_verts} vertices at {resolution}mm")
            if replace_original:
                self.report({'INFO'}, f"Remesh completed. Updated '{result_obj.name}'")
            else:
                self.report({'INFO'}, f"Remesh completed. Created '{result_obj.name}'")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Remesh failed: {e}")
            print(f"[Quick Infill] Remesh error: {e}")
            return {'CANCELLED'}


class QUICKINFILL_OT_trim_thin(Operator):
    bl_idname = "quick_infill.trim_thin"
    bl_label = "Trim Thin"
    bl_description = "Remove thin sections from the selected mesh (shrink then grow by resolution)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            settings = context.scene.quick_infill_tools_settings
            resolution = float(settings.resolution)
            auto_decimate = settings.auto_decimate
            replace_original = settings.replace_original

            selected_objs = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
            if not selected_objs:
                self.report({'ERROR'}, "No mesh selected.")
                return {'CANCELLED'}

            src_obj = selected_objs[0]
            
            # Trim thin = shrink then grow by resolution (removes thin features)
            def trim_thin_op(mesh):
                shrunk = cuda_offset(mesh, resolution, -resolution)
                return cuda_offset(shrunk, resolution, resolution)
            
            result_obj, initial_verts, final_verts = process_mesh_operation(
                src_obj, trim_thin_op, "_TrimThin", auto_decimate=auto_decimate, replace_original=replace_original
            )
            
            print(f"[Quick Infill] Trim Thin: {initial_verts} → {final_verts} vertices at {resolution}mm")
            if replace_original:
                self.report({'INFO'}, f"Trim Thin completed. Updated '{result_obj.name}'")
            else:
                self.report({'INFO'}, f"Trim Thin completed. Created '{result_obj.name}'")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Trim Thin failed: {e}")
            print(f"[Quick Infill] Trim Thin error: {e}")
            return {'CANCELLED'}


class QUICKINFILL_OT_shrinkwrap(Operator):
    bl_idname = "quick_infill.shrinkwrap"
    bl_label = "Shrinkwrap"
    bl_description = "Create a variable-width shell on selected mesh based on distance to active mesh. Select the mesh to offset, then Ctrl+select the reference mesh (active)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            settings = context.scene.quick_infill_tools_settings
            resolution = float(settings.resolution)
            shrink_mult = float(settings.shrink_mult)
            auto_decimate = settings.auto_decimate
            replace_original = settings.replace_original

            # Get active object (reference mesh)
            active_obj = context.active_object
            if not active_obj or active_obj.type != 'MESH':
                self.report({'ERROR'}, "Active object must be a mesh (reference mesh).")
                return {'CANCELLED'}

            # Get selected mesh that is NOT active (mesh to offset)
            selected_objs = [obj for obj in context.selected_objects if obj.type == 'MESH' and obj != active_obj]
            if not selected_objs:
                self.report({'ERROR'}, "Select a mesh to offset, then Ctrl+click to set a reference mesh as active.")
                return {'CANCELLED'}

            mesh_to_offset_obj = selected_objs[0]
            reference_obj = active_obj

            # Convert both meshes to meshlib
            mesh_to_offset = blender_to_meshlib_via_stl(mesh_to_offset_obj)
            reference_mesh = blender_to_meshlib_via_stl(reference_obj)
            
            initial_vertex_count = mesh_to_offset.topology.numValidVerts()

            # Perform weighted distance shell operation
            # shrink_mult controls shell thickness
            shell_mesh = weighted_dist_shell(
                mesh_to_offset=mesh_to_offset,
                reference_mesh=reference_mesh,
                voxel_size=resolution,
                shrink_mult=shrink_mult,
                max_vertices=3_000_000,
            )
            
            # Boolean subtract the shell from the original mesh
            from meshlib import mrmeshpy as mm
            out_mesh = mm.boolean(mesh_to_offset, shell_mesh, mm.BooleanOperation.DifferenceAB).mesh
            
            final_vertex_count = out_mesh.topology.numValidVerts()
            
            # Auto decimate if enabled
            if auto_decimate and final_vertex_count > initial_vertex_count:
                from .offset_utils import decimate_mesh
                out_mesh = decimate_mesh(out_mesh, target_vertex_count=initial_vertex_count)
                final_vertex_count = out_mesh.topology.numValidVerts()

            # Convert back to Blender
            result_obj = meshlib_to_blender_via_stl(
                out_mesh, 
                mesh_to_offset_obj.name + "_Shrinkwrap", 
                import_scale=0.1
            )
            
            # If replace_original is enabled, swap mesh data
            if replace_original:
                from .blender_meshlib_utils import replace_mesh_keep_transforms
                result_obj = replace_mesh_keep_transforms(mesh_to_offset_obj, result_obj)

            print(f"[Quick Infill] Shrinkwrap: {initial_vertex_count} → {final_vertex_count} vertices")
            if replace_original:
                self.report({'INFO'}, f"Shrinkwrap completed. Updated '{result_obj.name}'")
            else:
                self.report({'INFO'}, f"Shrinkwrap completed. Created '{result_obj.name}'")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Shrinkwrap failed: {e}")
            print(f"[Quick Infill] Shrinkwrap error: {e}")
            return {'CANCELLED'}


classes = (
    QUICKINFILL_OT_grow,
    QUICKINFILL_OT_shrink,
    QUICKINFILL_OT_remesh,
    QUICKINFILL_OT_trim_thin,
    QUICKINFILL_OT_shrinkwrap,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
