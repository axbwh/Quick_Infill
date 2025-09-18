import bpy
from bpy.types import Operator
from .offset_utils import cuda_offset, weighted_dist_shell, compute_voxel_size
from .blender_meshlib_utils import (
    blender_to_meshlib,
    blender_to_meshlib_via_stl,
    meshlib_to_blender,
    meshlib_to_blender_via_stl,
)

class QUICKINFILL_OT_heal_cavity(Operator):
    bl_idname = "quick_infill.heal_cavity"
    bl_label = "Heal Cavity"
    bl_description = "Generate infill for selected mesh using CUDA offset operations"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            # Safe meshlib import
            from .meshlib_utils import get_meshlib
            mm, mc = get_meshlib()
            
            # Read settings from Scene to avoid _PropertyDeferred
            s = getattr(context.scene, 'quick_infill_settings', None)
            # Coerce robustly with fallbacks
            def _cf(v, d):
                try:
                    return float(v)
                except Exception:
                    return float(d)

            # New UI uses millions input (0.5 - 2.0). Convert to absolute count.
            # target_res drives both voxel count and decimation limit (~1M vertices)
            target_res_millions = getattr(s, 'target_res', 2.0)
            target_voxels_val = int(float(target_res_millions) * 1_000_000)
            max_vertices_limit = int(float(target_res_millions) * 1_000_000)  # Use same limit for decimation

            resolution_val = _cf(getattr(s, 'resolution', 0.1), 0.1)
            voxel_mode = getattr(s, 'voxel_mode', 'TARGET_VOXELS')
            grow_val = _cf(getattr(s, 'grow', 2.0), 2.0)
            shrink_mult_val = _cf(getattr(s, 'shrink_mult', 1.5), 1.5)
            method = getattr(s, 'method', 'NAIVE')
            trim_thin_val = getattr(s, 'trim_thin', False)


            # Get selected mesh
            selected_objs = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
            if not selected_objs:
                self.report({'ERROR'}, "No mesh selected.")
                return {'CANCELLED'}

            src_mesh_blender = selected_objs[0]
            # Convert to meshlib; fallback to STL route for very dense meshes
            src_mesh = blender_to_meshlib_via_stl(src_mesh_blender)
            INITIAL_VERTEX_COUNT = src_mesh.topology.numValidVerts()
            print(f"Initial vertex count: {INITIAL_VERTEX_COUNT}")
            
            # Decimate if mesh exceeds target resolution limit
            if INITIAL_VERTEX_COUNT > max_vertices_limit:
                from .offset_utils import decimate_mesh
                reduction_ratio = max_vertices_limit / INITIAL_VERTEX_COUNT
                src_mesh = decimate_mesh(src_mesh, reduction_ratio=reduction_ratio)
                new_vertex_count = src_mesh.topology.numValidVerts()
                print(f"Decimated mesh from {INITIAL_VERTEX_COUNT} to {new_vertex_count} vertices (target: {max_vertices_limit})")
                self.report({'INFO'}, f"Decimated mesh: {INITIAL_VERTEX_COUNT} → {new_vertex_count} vertices")

            obj_name = src_mesh_blender.name

            # Calculate voxel size via helper or direct, based on mode
            if voxel_mode == 'RESOLUTION':
                vox = float(resolution_val)
            else:
                vox = compute_voxel_size(src_mesh, int(target_voxels_val), float(resolution_val))
            print(f"Voxel Size: {vox}")

            if method == "NAIVE":
                grow_mesh = cuda_offset(src_mesh, vox, grow_val)
                shrink_mesh = cuda_offset(grow_mesh, vox, -grow_val*shrink_mult_val)
                # Smooth step: shrink then grow by one voxel to clean artifacts
                out_mesh = shrink_mesh
            else:
                # Use helpers from offset_utils
                # Process mesh
                grow_mesh = cuda_offset(src_mesh, vox, grow_val)
                shrink_mesh = cuda_offset(grow_mesh, vox, -grow_val)
                shell_mesh = weighted_dist_shell(shrink_mesh, src_mesh, vox, shrink_mult_val, max_vertices=max_vertices_limit, target_resolution=int(target_res_millions * 1_000_000))
                trim_mesh = mm.boolean(shrink_mesh, shell_mesh, mm.BooleanOperation.DifferenceAB).mesh
                out_mesh = trim_mesh

            # Apply trimThin if enabled
            if trim_thin_val:
                ss_mesh_shrink = cuda_offset(out_mesh, vox, -vox)
                out_mesh = cuda_offset(ss_mesh_shrink, vox, vox)
            
            # Decimate output mesh if it's higher resolution than initial mesh
            final_vertex_count = out_mesh.topology.numValidVerts()
            if final_vertex_count > INITIAL_VERTEX_COUNT:
                from .offset_utils import decimate_mesh
                target_vertices = INITIAL_VERTEX_COUNT
                out_mesh = decimate_mesh(out_mesh, target_face_count=None, reduction_ratio=None, target_vertex_count=target_vertices)
                new_final_count = out_mesh.topology.numValidVerts()
                print(f"Decimated output mesh from {final_vertex_count} to {new_final_count} vertices (target: {target_vertices})")
                self.report({'INFO'}, f"Decimated result: {final_vertex_count} → {new_final_count} vertices")
        
            infill_obj = meshlib_to_blender_via_stl(out_mesh, obj_name + "Infill", import_scale=0.1)
            
            self.report({'INFO'}, f"Heal Cavity completed. Created '{obj_name}Infill'")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Heal Cavity failed: {e}")
            print(f"[Quick Infill] Heal Cavity error: {e}")
            return {'CANCELLED'}


def register():
    import bpy
    bpy.utils.register_class(QUICKINFILL_OT_heal_cavity)


def unregister():
    import bpy
    bpy.utils.unregister_class(QUICKINFILL_OT_heal_cavity)