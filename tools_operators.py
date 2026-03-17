"""
Tool operators for Quick Infill addon.
Provides Grow, Shrink, Remesh, and Trim Thin operations.
"""

import bpy
from bpy.types import Operator
from .offset_utils import cuda_offset
from .blender_meshlib_utils import process_mesh_operation, batch_process_mesh_operation, blender_to_meshlib_via_stl, meshlib_to_blender_via_stl


class QUICKINFILL_OT_grow(Operator):
    bl_idname = "quick_infill.grow"
    bl_label = "Grow"
    bl_description = "Grow selected mesh(es) by the distance value. With multiple selections, processes each object individually"
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

            # Grow = positive offset
            def grow_op(mesh):
                return cuda_offset(mesh, resolution, distance)
            
            # Use batch processing for multiple objects, single processing for one
            if len(selected_objs) == 1:
                result_obj, initial_verts, final_verts = process_mesh_operation(
                    selected_objs[0], grow_op, "_Grown", auto_decimate=auto_decimate, replace_original=replace_original, resolution=resolution
                )
                print(f"[Quick Infill] Grow: {initial_verts} → {final_verts} vertices, offset +{distance}mm")
                if replace_original:
                    self.report({'INFO'}, f"Grow completed. Updated '{result_obj.name}'")
                else:
                    self.report({'INFO'}, f"Grow completed. Created '{result_obj.name}'")
            else:
                # Batch process all selected objects
                results = batch_process_mesh_operation(
                    selected_objs, grow_op, "_Grown", auto_decimate=auto_decimate, replace_original=replace_original, resolution=resolution
                )
                
                total_initial = sum(r[1] for r in results)
                total_final = sum(r[2] for r in results)
                obj_count = len(results)
                
                print(f"[Quick Infill] Grow (batch): {obj_count} objects, {total_initial} → {total_final} total vertices, offset +{distance}mm")
                if replace_original:
                    self.report({'INFO'}, f"Grow completed. Updated {obj_count} objects")
                else:
                    self.report({'INFO'}, f"Grow completed. Created {obj_count} new objects")
            
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Grow failed: {e}")
            print(f"[Quick Infill] Grow error: {e}")
            return {'CANCELLED'}


class QUICKINFILL_OT_shrink(Operator):
    bl_idname = "quick_infill.shrink"
    bl_label = "Shrink"
    bl_description = "Shrink selected mesh(es) by the distance value. With multiple selections, processes each object individually"
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

            # Shrink = negative offset
            def shrink_op(mesh):
                return cuda_offset(mesh, resolution, -distance)
            
            # Use batch processing for multiple objects, single processing for one
            if len(selected_objs) == 1:
                result_obj, initial_verts, final_verts = process_mesh_operation(
                    selected_objs[0], shrink_op, "_Shrunk", auto_decimate=auto_decimate, replace_original=replace_original, resolution=resolution
                )
                print(f"[Quick Infill] Shrink: {initial_verts} → {final_verts} vertices, offset -{distance}mm")
                if replace_original:
                    self.report({'INFO'}, f"Shrink completed. Updated '{result_obj.name}'")
                else:
                    self.report({'INFO'}, f"Shrink completed. Created '{result_obj.name}'")
            else:
                # Batch process all selected objects
                results = batch_process_mesh_operation(
                    selected_objs, shrink_op, "_Shrunk", auto_decimate=auto_decimate, replace_original=replace_original, resolution=resolution
                )
                
                total_initial = sum(r[1] for r in results)
                total_final = sum(r[2] for r in results)
                obj_count = len(results)
                
                print(f"[Quick Infill] Shrink (batch): {obj_count} objects, {total_initial} → {total_final} total vertices, offset -{distance}mm")
                if replace_original:
                    self.report({'INFO'}, f"Shrink completed. Updated {obj_count} objects")
                else:
                    self.report({'INFO'}, f"Shrink completed. Created {obj_count} new objects")
            
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Shrink failed: {e}")
            print(f"[Quick Infill] Shrink error: {e}")
            return {'CANCELLED'}


class QUICKINFILL_OT_remesh(Operator):
    bl_idname = "quick_infill.remesh"
    bl_label = "Remesh"
    bl_description = "Remesh selected mesh(es) at the resolution value. With multiple selections, processes each object individually"
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

            # Remesh = offset with 0 distance (re-voxelizes)
            def remesh_op(mesh):
                return cuda_offset(mesh, resolution, 0.0)
            
            # Use batch processing for multiple objects, single processing for one
            if len(selected_objs) == 1:
                result_obj, initial_verts, final_verts = process_mesh_operation(
                    selected_objs[0], remesh_op, "_Remeshed", auto_decimate=auto_decimate, replace_original=replace_original, resolution=resolution
                )
                print(f"[Quick Infill] Remesh: {initial_verts} → {final_verts} vertices at {resolution}mm")
                if replace_original:
                    self.report({'INFO'}, f"Remesh completed. Updated '{result_obj.name}'")
                else:
                    self.report({'INFO'}, f"Remesh completed. Created '{result_obj.name}'")
            else:
                # Batch process all selected objects
                results = batch_process_mesh_operation(
                    selected_objs, remesh_op, "_Remeshed", auto_decimate=auto_decimate, replace_original=replace_original, resolution=resolution
                )
                
                total_initial = sum(r[1] for r in results)
                total_final = sum(r[2] for r in results)
                obj_count = len(results)
                
                print(f"[Quick Infill] Remesh (batch): {obj_count} objects, {total_initial} → {total_final} total vertices at {resolution}mm")
                if replace_original:
                    self.report({'INFO'}, f"Remesh completed. Updated {obj_count} objects")
                else:
                    self.report({'INFO'}, f"Remesh completed. Created {obj_count} new objects")
            
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Remesh failed: {e}")
            print(f"[Quick Infill] Remesh error: {e}")
            return {'CANCELLED'}


class QUICKINFILL_OT_trim_thin(Operator):
    bl_idname = "quick_infill.trim_thin"
    bl_label = "Trim Thin"
    bl_description = "Remove thin sections from selected mesh(es). With multiple selections, processes each object individually"
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

            # Trim thin = shrink then grow by resolution (removes thin features)
            def trim_thin_op(mesh):
                shrunk = cuda_offset(mesh, resolution, -resolution)
                return cuda_offset(shrunk, resolution, resolution)
            
            # Use batch processing for multiple objects, single processing for one
            if len(selected_objs) == 1:
                result_obj, initial_verts, final_verts = process_mesh_operation(
                    selected_objs[0], trim_thin_op, "_TrimThin", auto_decimate=auto_decimate, replace_original=replace_original, resolution=resolution
                )
                print(f"[Quick Infill] Trim Thin: {initial_verts} → {final_verts} vertices at {resolution}mm")
                if replace_original:
                    self.report({'INFO'}, f"Trim Thin completed. Updated '{result_obj.name}'")
                else:
                    self.report({'INFO'}, f"Trim Thin completed. Created '{result_obj.name}'")
            else:
                # Batch process all selected objects
                results = batch_process_mesh_operation(
                    selected_objs, trim_thin_op, "_TrimThin", auto_decimate=auto_decimate, replace_original=replace_original, resolution=resolution
                )
                
                total_initial = sum(r[1] for r in results)
                total_final = sum(r[2] for r in results)
                obj_count = len(results)
                
                print(f"[Quick Infill] Trim Thin (batch): {obj_count} objects, {total_initial} → {total_final} total vertices at {resolution}mm")
                if replace_original:
                    self.report({'INFO'}, f"Trim Thin completed. Updated {obj_count} objects")
                else:
                    self.report({'INFO'}, f"Trim Thin completed. Created {obj_count} new objects")
            
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Trim Thin failed: {e}")
            print(f"[Quick Infill] Trim Thin error: {e}")
            return {'CANCELLED'}
            return {'CANCELLED'}


classes = (
    QUICKINFILL_OT_grow,
    QUICKINFILL_OT_shrink,
    QUICKINFILL_OT_remesh,
    QUICKINFILL_OT_trim_thin,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
