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

            from .support_tools import intersect_meshes
            from .blender_meshlib_utils import replace_mesh_keep_transforms
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import os, tempfile

            tmp_dir = tempfile.gettempdir()
            import_scale = 0.1

            # ── Phase 1: Export each Blender object to a meshlib mesh (Blender API, sequential) ──
            meshlib_meshes = []      # (original_mesh, initial_faces, initial_verts) per obj
            collapsed_on_load = []   # indices of objects that failed to load

            view_layer = bpy.context.view_layer
            prev_active_name = view_layer.objects.active.name if view_layer.objects.active else None
            prev_selection_names = [obj.name for obj in bpy.context.selected_objects]

            for obj in selected_objs:
                try:
                    original_mesh = blender_to_meshlib_via_stl(obj)
                    meshlib_meshes.append((
                        original_mesh,
                        original_mesh.topology.numValidFaces(),
                        original_mesh.topology.numValidVerts(),
                    ))
                except Exception:
                    meshlib_meshes.append(None)

            # ── Phase 2: Process meshes in parallel (pure meshlib, no Blender API) ──
            def _process_one(args):
                i, entry = args
                if entry is None:
                    raise RuntimeError("failed to load mesh")
                original_mesh, initial_faces, initial_verts = entry

                working_mesh = mm.copyMesh(original_mesh)
                working_mesh = cuda_offset(working_mesh, resolution, 2.0 * distance)
                working_mesh = cuda_offset(working_mesh, resolution, -3.0 * distance)

                if working_mesh.topology.numValidFaces() == 0:
                    raise _MeshCollapsedError()

                working_mesh = cuda_offset(working_mesh, resolution, (1.0 + trim_edges_x) * distance)

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
                    working_mesh = decimate_mesh(working_mesh, target_face_count=target_faces, max_error=adaptive_max_error)

                result_mesh = intersect_meshes(working_mesh, original_mesh, resolution)

                if auto_decimate:
                    final_faces_before = result_mesh.topology.numValidFaces()
                    do_decimate, tgt = should_auto_decimate_faces(initial_faces, final_faces_before)
                    if do_decimate:
                        result_mesh = decimate_mesh(result_mesh, target_face_count=tgt, resolution=resolution)

                return i, result_mesh, initial_verts, result_mesh.topology.numValidVerts()

            n_workers = min(len(selected_objs), 4)
            success_map = {}   # i → (result_mesh, initial_verts, final_verts)
            collapsed_map = {} # i → exception

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_process_one, (i, meshlib_meshes[i])): i
                           for i in range(len(selected_objs))}
                for future in as_completed(futures):
                    i = futures[future]
                    try:
                        idx, result_mesh, initial_verts, final_verts = future.result()
                        success_map[idx] = (result_mesh, initial_verts, final_verts)
                    except Exception as exc:
                        collapsed_map[i] = exc

            # ── Phase 3: Save processed meshes to STL in parallel (file I/O) ──
            surviving_indices = sorted(success_map.keys())
            output_stl_paths = {}

            for i in surviving_indices:
                fd, stl_path = tempfile.mkstemp(prefix=f"qi_te_{selected_objs[i].name}_", suffix=".stl", dir=tmp_dir)
                os.close(fd)
                output_stl_paths[i] = stl_path

            def _save_one(args):
                i, stl_path = args
                mesh = success_map[i][0]
                try:
                    mm.saveMesh(mesh, stl_path)
                except Exception:
                    mm.saveMeshAs(mesh, stl_path)

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                list(executor.map(_save_one, output_stl_paths.items()))

            # ── Phase 4: Import results back to Blender (Blender API, sequential) ──
            result_objs = {}
            for i in surviving_indices:
                stl_path = output_stl_paths[i]
                prev_objs = set(bpy.data.objects)
                if hasattr(bpy.ops.wm, "stl_import"):
                    bpy.ops.wm.stl_import('EXEC_DEFAULT', filepath=stl_path, global_scale=float(import_scale))
                else:
                    bpy.ops.import_mesh.stl('EXEC_DEFAULT', filepath=stl_path, global_scale=float(import_scale))
                new_objs = [obj for obj in bpy.data.objects if obj not in prev_objs] or list(bpy.context.selected_objects)
                if new_objs:
                    result_obj = new_objs[0]
                    result_obj.name = selected_objs[i].name + "_TrimEdges"
                    result_objs[i] = result_obj
                try:
                    os.remove(stl_path)
                except Exception:
                    pass

            # ── Phase 5: Handle replace_original, deletions, and build final results ──
            # Pre-capture names before any removal so stale StructRNA is never accessed.
            collapsed_names = {i: selected_objs[i].name for i in collapsed_map}
            removed_names = []
            for i, obj_name in collapsed_names.items():
                removed_names.append(obj_name)
                if obj_name in bpy.data.objects:
                    bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)

            results = []
            for i in surviving_indices:
                if i not in result_objs:
                    continue
                result_obj = result_objs[i]
                src_obj = selected_objs[i]
                _, initial_verts, final_verts = success_map[i]
                if replace_original:
                    result_obj = replace_mesh_keep_transforms(src_obj, result_obj)
                results.append((result_obj, initial_verts, final_verts))

            # Restore selection state (use names to avoid stale StructRNA references)
            for obj in bpy.context.selected_objects:
                obj.select_set(False)
            for name in prev_selection_names:
                if name in bpy.data.objects:
                    bpy.data.objects[name].select_set(True)
            if prev_active_name and prev_active_name in bpy.data.objects:
                view_layer.objects.active = bpy.data.objects[prev_active_name]

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
