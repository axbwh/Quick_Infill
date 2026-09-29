"""
Tool operators for Quick Infill addon.
Provides Grow, Shrink, Remesh, Trim Thin, and Trim Edges operations.
"""

import bpy
from bpy.types import Operator
from .meshlib_utils import get_meshlib
from .offset_utils import cuda_offset, decimate_mesh, target_faces_for_density, should_auto_decimate_faces
from .blender_meshlib_utils import process_mesh_operation, batch_process_mesh_operation, select_results


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
            resolution = float(settings.voxel_size)
            auto_decimate = settings.decimate_mode != "OFF"
            replace_original = settings.replace_original

            selected_objs = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
            if not selected_objs:
                self.report({'ERROR'}, "No mesh selected.")
                return {'CANCELLED'}

            # Grow = positive offset
            def grow_op(mesh, res):
                return cuda_offset(mesh, res, distance)
            
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

            # Force the depsgraph to fully evaluate the result mesh now, inside
            # the operator where the user expects a wait.  Without this, Blender
            # defers the evaluation (split-normal computation, BVH construction,
            # etc.) until the next time it is needed – which is the file save –
            # causing the save to appear frozen.
            bpy.context.view_layer.update()

            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Grow failed: {e}")
            import traceback
            traceback.print_exc()
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
            resolution = float(settings.voxel_size)
            auto_decimate = settings.decimate_mode != "OFF"
            replace_original = settings.replace_original

            selected_objs = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
            if not selected_objs:
                self.report({'ERROR'}, "No mesh selected.")
                return {'CANCELLED'}

            # Shrink = negative offset
            def shrink_op(mesh, res):
                return cuda_offset(mesh, res, -distance)
            
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

            # Force the depsgraph to fully evaluate the result mesh now, inside
            # the operator where the user expects a wait.  Without this, Blender
            # defers the evaluation (split-normal computation, BVH construction,
            # etc.) until the next time it is needed – which is the file save –
            # causing the save to appear frozen.
            bpy.context.view_layer.update()

            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Shrink failed: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


class QUICKINFILL_OT_remesh(Operator):
    bl_idname = "quick_infill.remesh"
    bl_label = "Remesh"
    bl_description = "Remesh selected mesh(es) at the resolution value. With multiple selections, processes each object individually"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            settings = context.scene.quick_infill_tools_settings
            resolution = float(settings.voxel_size)
            auto_decimate = settings.decimate_mode != "OFF"
            replace_original = settings.replace_original

            selected_objs = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
            if not selected_objs:
                self.report({'ERROR'}, "No mesh selected.")
                return {'CANCELLED'}

            # Remesh = offset with 0 distance (re-voxelizes)
            def remesh_op(mesh, res):
                return cuda_offset(mesh, res, 0.0)
            
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

            # Force the depsgraph to fully evaluate the result mesh now, inside
            # the operator where the user expects a wait.  Without this, Blender
            # defers the evaluation (split-normal computation, BVH construction,
            # etc.) until the next time it is needed – which is the file save –
            # causing the save to appear frozen.
            bpy.context.view_layer.update()

            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Remesh failed: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


class QUICKINFILL_OT_trim_thin(Operator):
    bl_idname = "quick_infill.trim_thin"
    bl_label = "Trim Thin"
    bl_description = "Remove thin sections from selected mesh(es). With multiple selections, processes each object individually"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            settings = context.scene.quick_infill_tools_settings
            resolution = float(settings.voxel_size)
            auto_decimate = settings.decimate_mode != "OFF"
            replace_original = settings.replace_original

            selected_objs = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
            if not selected_objs:
                self.report({'ERROR'}, "No mesh selected.")
                return {'CANCELLED'}

            # Trim thin = shrink then grow by resolution (removes thin features).
            # If the shrink collapses the mesh to nothing, raise _MeshCollapsedError
            # so batch_process_mesh_operation can skip it and return it as collapsed.
            def trim_thin_op(mesh, res):
                shrunk = cuda_offset(mesh, res, -res)
                if shrunk.topology.numValidFaces() == 0:
                    raise _MeshCollapsedError()
                return cuda_offset(shrunk, res, res)

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
                # Save mesh reference before removal: bpy.data.objects.remove()
                # decrements the mesh's user count but never removes the mesh
                # itself, leaving it as an orphan that gets serialised on save.
                mesh_to_del = obj.data if obj.type == 'MESH' else None
                bpy.data.objects.remove(obj, do_unlink=True)
                if mesh_to_del and mesh_to_del.users == 0 and not mesh_to_del.use_fake_user:
                    try:
                        bpy.data.meshes.remove(mesh_to_del)
                    except Exception:
                        pass

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

            # Force the depsgraph to fully evaluate the result mesh now, inside
            # the operator where the user expects a wait.  Without this, Blender
            # defers the evaluation (split-normal computation, BVH construction,
            # etc.) until the next time it is needed – which is the file save –
            # causing the save to appear frozen.
            bpy.context.view_layer.update()

            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Trim Thin failed: {e}")
            import traceback
            traceback.print_exc()
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
            resolution = float(settings.voxel_size)
            trim_edges_x = float(settings.trim_edges_x)
            trim_edges_density = float(settings.trim_edges_density)
            auto_decimate = settings.decimate_mode != "OFF"
            replace_original = settings.replace_original

            selected_objs = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
            if not selected_objs:
                self.report({'ERROR'}, "No mesh selected.")
                return {'CANCELLED'}

            from .support_tools import intersect_meshes

            # Trim edges = grow then shrink back further (rounds off thin edges),
            # regrow slightly, decimate for density, then intersect with the
            # original mesh to clip the result back to the original silhouette.
            def trim_edges_op(mesh, res):
                original_mesh = mm.copyMesh(mesh)
                working_mesh = mm.copyMesh(mesh)
                working_mesh = cuda_offset(working_mesh, res, 2.0 * distance)
                working_mesh = cuda_offset(working_mesh, res, -3.0 * distance)

                if working_mesh.topology.numValidFaces() == 0:
                    raise _MeshCollapsedError()

                working_mesh = cuda_offset(working_mesh, res, (1.0 + trim_edges_x) * distance)

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
                        float(res) * 10.0,
                        diag * (0.01 + 0.49 * reduction_strength),
                    )
                    working_mesh = decimate_mesh(working_mesh, target_face_count=target_faces, max_error=adaptive_max_error)

                return intersect_meshes(working_mesh, original_mesh, res)

            # Use batch processing for all objects. Collapsed meshes are returned
            # separately in the second element without aborting the batch.
            if len(selected_objs) == 1:
                try:
                    result_obj, initial_verts, final_verts = process_mesh_operation(
                        selected_objs[0], trim_edges_op, "_TrimEdges",
                        auto_decimate=auto_decimate, replace_original=replace_original, resolution=resolution
                    )
                    results = [(result_obj, initial_verts, final_verts)]
                    collapsed = []
                except _MeshCollapsedError:
                    results = []
                    collapsed = [(selected_objs[0], _MeshCollapsedError())]
            else:
                results, collapsed = batch_process_mesh_operation(
                    selected_objs, trim_edges_op, "_TrimEdges",
                    auto_decimate=auto_decimate, replace_original=replace_original, resolution=resolution
                )

            # Delete any objects whose mesh fully collapsed
            removed_names = []
            for obj, exc in collapsed:
                obj_name = obj.name
                removed_names.append(obj_name)
                # Save mesh reference before removal: bpy.data.objects.remove()
                # decrements the mesh's user count but never removes the mesh
                # itself, leaving it as an orphan that gets serialised on save.
                mesh_to_del = obj.data if obj.type == 'MESH' else None
                bpy.data.objects.remove(obj, do_unlink=True)
                if mesh_to_del and mesh_to_del.users == 0 and not mesh_to_del.use_fake_user:
                    try:
                        bpy.data.meshes.remove(mesh_to_del)
                    except Exception:
                        pass

            if removed_names:
                names_str = ", ".join(f"'{n}'" for n in removed_names)
                self.report({'WARNING'}, f"Trim Edges: {len(removed_names)} object(s) fully removed (mesh collapsed during trim): {names_str}")

            if results:
                obj_count = len(results)
                select_results([r[0] for r in results])
                if obj_count == 1:
                    result_obj, _, _ = results[0]
                    if replace_original:
                        self.report({'INFO'}, f"Trim Edges completed. Updated '{result_obj.name}'")
                    else:
                        self.report({'INFO'}, f"Trim Edges completed. Created '{result_obj.name}'")
                else:
                    if replace_original:
                        self.report({'INFO'}, f"Trim Edges completed. Updated {obj_count} objects")
                    else:
                        self.report({'INFO'}, f"Trim Edges completed. Created {obj_count} new objects")

            # Force the depsgraph to fully evaluate the result mesh now, inside
            # the operator where the user expects a wait.  Without this, Blender
            # defers the evaluation (split-normal computation, BVH construction,
            # etc.) until the next time it is needed – which is the file save –
            # causing the save to appear frozen.
            bpy.context.view_layer.update()

            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Trim Edges failed: {e}")
            import traceback
            traceback.print_exc()
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
