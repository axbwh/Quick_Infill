"""
Tool operators for Quick Infill addon.
Provides Grow, Shrink, Remesh, Trim Thin, and Trim Edges operations.
"""

import bpy
from bpy.types import Operator
from .meshlib_utils import get_meshlib
from .offset_utils import cuda_offset, decimate_mesh, target_faces_for_density, should_auto_decimate_faces
from .blender_meshlib_utils import process_mesh_operation, batch_process_mesh_operation, blender_to_meshlib_via_stl, meshlib_to_blender_via_stl, select_results


class _MeshCollapsedError(Exception):
    """Raised when a mesh offset collapses the mesh to no remaining geometry."""
    pass


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
                # print(f"[Quick Infill] Grow: {initial_verts} → {final_verts} vertices, offset +{distance}mm")
                if replace_original:
                    self.report({'INFO'}, f"Grow completed. Updated '{result_obj.name}'")
                else:
                    self.report({'INFO'}, f"Grow completed. Created '{result_obj.name}'")
            else:
                # Batch process all selected objects
                results, _ = batch_process_mesh_operation(
                    selected_objs, grow_op, "_Grown", auto_decimate=auto_decimate, replace_original=replace_original, resolution=resolution
                )
                
                total_initial = sum(r[1] for r in results)
                total_final = sum(r[2] for r in results)
                obj_count = len(results)
                
                # print(f"[Quick Infill] Grow (batch): {obj_count} objects, {total_initial} → {total_final} total vertices, offset +{distance}mm")
                if replace_original:
                    self.report({'INFO'}, f"Grow completed. Updated {obj_count} objects")
                else:
                    self.report({'INFO'}, f"Grow completed. Created {obj_count} new objects")
            
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Grow failed: {e}")
            # print(f"[Quick Infill] Grow error: {e}")
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
                # print(f"[Quick Infill] Shrink: {initial_verts} → {final_verts} vertices, offset -{distance}mm")
                if replace_original:
                    self.report({'INFO'}, f"Shrink completed. Updated '{result_obj.name}'")
                else:
                    self.report({'INFO'}, f"Shrink completed. Created '{result_obj.name}'")
            else:
                # Batch process all selected objects
                results, _ = batch_process_mesh_operation(
                    selected_objs, shrink_op, "_Shrunk", auto_decimate=auto_decimate, replace_original=replace_original, resolution=resolution
                )
                
                total_initial = sum(r[1] for r in results)
                total_final = sum(r[2] for r in results)
                obj_count = len(results)
                
                # print(f"[Quick Infill] Shrink (batch): {obj_count} objects, {total_initial} → {total_final} total vertices, offset -{distance}mm")
                if replace_original:
                    self.report({'INFO'}, f"Shrink completed. Updated {obj_count} objects")
                else:
                    self.report({'INFO'}, f"Shrink completed. Created {obj_count} new objects")
            
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Shrink failed: {e}")
            # print(f"[Quick Infill] Shrink error: {e}")
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
                # print(f"[Quick Infill] Remesh: {initial_verts} → {final_verts} vertices at {resolution}mm")
                if replace_original:
                    self.report({'INFO'}, f"Remesh completed. Updated '{result_obj.name}'")
                else:
                    self.report({'INFO'}, f"Remesh completed. Created '{result_obj.name}'")
            else:
                # Batch process all selected objects
                results, _ = batch_process_mesh_operation(
                    selected_objs, remesh_op, "_Remeshed", auto_decimate=auto_decimate, replace_original=replace_original, resolution=resolution
                )
                
                total_initial = sum(r[1] for r in results)
                total_final = sum(r[2] for r in results)
                obj_count = len(results)
                
                # print(f"[Quick Infill] Remesh (batch): {obj_count} objects, {total_initial} → {total_final} total vertices at {resolution}mm")
                if replace_original:
                    self.report({'INFO'}, f"Remesh completed. Updated {obj_count} objects")
                else:
                    self.report({'INFO'}, f"Remesh completed. Created {obj_count} new objects")
            
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Remesh failed: {e}")
            # print(f"[Quick Infill] Remesh error: {e}")
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

            # Trim thin = shrink then grow by resolution (removes thin features).
            # If the shrink collapses the mesh to nothing, raise _MeshCollapsedError
            # so batch_process_mesh_operation can skip it and return it as collapsed.
            def trim_thin_op(mesh):
                shrunk = cuda_offset(mesh, resolution, -resolution)
                if shrunk.topology.numValidFaces() == 0:
                    raise _MeshCollapsedError()
                return cuda_offset(shrunk, resolution, resolution)

            # Use batch processing for all objects. Collapsed meshes are returned
            # separately in the second element without aborting the batch.
            if len(selected_objs) == 1:
                try:
                    result_obj, initial_verts, final_verts = process_mesh_operation(
                        selected_objs[0], trim_thin_op, "_TrimThin",
                        auto_decimate=auto_decimate, replace_original=replace_original, resolution=resolution
                    )
                    results = [(result_obj, initial_verts, final_verts)]
                    collapsed = []
                except _MeshCollapsedError:
                    results = []
                    collapsed = [(selected_objs[0], _MeshCollapsedError())]
            else:
                results, collapsed = batch_process_mesh_operation(
                    selected_objs, trim_thin_op, "_TrimThin",
                    auto_decimate=auto_decimate, replace_original=replace_original, resolution=resolution
                )

            # Delete any objects whose mesh fully collapsed
            removed_names = []
            for obj, exc in collapsed:
                obj_name = obj.name
                removed_names.append(obj_name)
                bpy.data.objects.remove(obj, do_unlink=True)
                # print(f"[Quick Infill] Trim Thin ({obj_name}): mesh fully collapsed, object deleted")

            if removed_names:
                names_str = ", ".join(f"'{n}'" for n in removed_names)
                self.report({'WARNING'}, f"Trim Thin: {len(removed_names)} object(s) fully removed (too thin for current resolution): {names_str}")

            if results:
                total_initial = sum(r[1] for r in results)
                total_final = sum(r[2] for r in results)
                obj_count = len(results)
                # print(f"[Quick Infill] Trim Thin: {obj_count} objects, {total_initial} → {total_final} vertices at {resolution}mm")
                select_results([r[0] for r in results])
                if obj_count == 1:
                    result_obj, _, _ = results[0]
                    if replace_original:
                        self.report({'INFO'}, f"Trim Thin completed. Updated '{result_obj.name}'")
                    else:
                        self.report({'INFO'}, f"Trim Thin completed. Created '{result_obj.name}'")
                else:
                    if replace_original:
                        self.report({'INFO'}, f"Trim Thin completed. Updated {obj_count} objects")
                    else:
                        self.report({'INFO'}, f"Trim Thin completed. Created {obj_count} new objects")

            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Trim Thin failed: {e}")
            # print(f"[Quick Infill] Trim Thin error: {e}")
            return {'CANCELLED'}


class QUICKINFILL_OT_trim_edges(Operator):
    bl_idname = "quick_infill.trim_edges"
    bl_label = "Trim Edges"
    bl_description = "Trim edges by offset sequence and intersect with original mesh. With multiple selections, processes each object individually"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            mm, _ = get_meshlib()
            settings = context.scene.quick_infill_tools_settings
            distance = float(settings.distance)
            resolution = float(settings.resolution)
            trim_edges_x = float(settings.trim_edges_x)
            trim_edges_density = float(settings.trim_edges_density)
            auto_decimate = settings.auto_decimate
            replace_original = settings.replace_original

            selected_objs = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
            if not selected_objs:
                self.report({'ERROR'}, "No mesh selected.")
                return {'CANCELLED'}

            # Trim Edges sequence per object:
            # 1) Keep original mesh copy
            # 2) Grow by +2*distance
            # 3) Shrink by -3*distance
            # 4) Grow by +(1+x)*distance
            # 5) Intersect with original mesh using voxel boolean (same style as support tools)
            results = []
            removed_names = []
            for src_obj in selected_objs:
                original_mesh = blender_to_meshlib_via_stl(src_obj)
                working_mesh = mm.copyMesh(original_mesh)
                initial_faces = original_mesh.topology.numValidFaces()
                initial_verts = original_mesh.topology.numValidVerts()

                working_mesh = cuda_offset(working_mesh, resolution, 2.0 * distance)
                working_mesh = cuda_offset(working_mesh, resolution, -3.0 * distance)

                # If the shrink collapsed the mesh entirely, delete the object and skip.
                if working_mesh.topology.numValidFaces() == 0:
                    # print(f"[Quick Infill] Trim Edges ({src_obj.name}): mesh fully collapsed, object deleted")
                    removed_names.append(src_obj.name)
                    bpy.data.objects.remove(src_obj, do_unlink=True)
                    continue

                working_mesh = cuda_offset(working_mesh, resolution, (1.0 + trim_edges_x) * distance)

                # Normalize triangle density before boolean so output quality is more
                # consistent across meshes of different physical size.
                target_faces = target_faces_for_density(
                    working_mesh,
                    faces_per_sq_unit=trim_edges_density,
                    min_faces=20,
                    max_faces=working_mesh.topology.numValidFaces(),
                )
                current_faces = working_mesh.topology.numValidFaces()
                if target_faces < current_faces:
                    bbox = working_mesh.computeBoundingBox()
                    diag = (bbox.max - bbox.min).length()
                    target_ratio = float(target_faces) / float(max(1, current_faces))
                    reduction_strength = max(0.0, 1.0 - target_ratio)
                    adaptive_max_error = max(
                        float(resolution) * 10.0,
                        diag * (0.01 + 0.49 * reduction_strength),
                    )
                    working_mesh = decimate_mesh(
                        working_mesh,
                        target_face_count=target_faces,
                        max_error=adaptive_max_error,
                    )

                from .support_tools import intersect_meshes
                result_mesh = intersect_meshes(working_mesh, original_mesh, resolution)

                # Optionally auto-decimate final output similar to other offset operators.
                if auto_decimate:
                    final_faces_before = result_mesh.topology.numValidFaces()
                    do_decimate, target_faces = should_auto_decimate_faces(initial_faces, final_faces_before)
                    if do_decimate:
                        result_mesh = decimate_mesh(
                            result_mesh,
                            target_face_count=target_faces,
                            resolution=resolution,
                        )

                final_verts = result_mesh.topology.numValidVerts()

                result_name = src_obj.name + "_TrimEdges"
                result_obj = meshlib_to_blender_via_stl(result_mesh, name=result_name)

                if replace_original:
                    from .blender_meshlib_utils import replace_mesh_keep_transforms
                    result_obj = replace_mesh_keep_transforms(src_obj, result_obj)

                results.append((result_obj, initial_verts, final_verts))
                # print(
                #     f"[Quick Infill] Trim Edges ({src_obj.name}): {initial_verts} → {final_verts} vertices "
                #     f"(distance={distance}mm, x={trim_edges_x:.3f}, density={trim_edges_density:.2f})"
                # )

            if removed_names:
                names_str = ", ".join(f"'{n}'" for n in removed_names)
                self.report({'WARNING'}, f"Trim Edges: {len(removed_names)} object(s) fully removed (mesh collapsed during trim): {names_str}")

            obj_count = len(results)
            if obj_count == 1:
                result_obj, _, _ = results[0]
                if replace_original:
                    self.report({'INFO'}, f"Trim Edges completed. Updated '{result_obj.name}'")
                else:
                    self.report({'INFO'}, f"Trim Edges completed. Created '{result_obj.name}'")
            elif obj_count > 1:
                if replace_original:
                    self.report({'INFO'}, f"Trim Edges completed. Updated {obj_count} objects")
                else:
                    self.report({'INFO'}, f"Trim Edges completed. Created {obj_count} new objects")

            # Restore selection to result objects
            if results:
                select_results([r[0] for r in results])

            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Trim Edges failed: {e}")
            # print(f"[Quick Infill] Trim Edges error: {e}")
            return {'CANCELLED'}


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
