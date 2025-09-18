import bpy
from bpy.types import Operator
from .offset_utils import cuda_offset, weighted_dist_shell, compute_voxel_size
from meshlib import mrmeshpy as mm
from meshlib import mrcudapy as mc
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
            # Read settings from Scene to avoid _PropertyDeferred
            s = getattr(context.scene, 'quick_infill_settings', None)
            # Coerce robustly with fallbacks
            def _cf(v, d):
                try:
                    return float(v)
                except Exception:
                    return float(d)
            def _ci(v, d):
                try:
                    return int(v)
                except Exception:
                    return int(d)

            # New UI uses millions input (0.5 - 2.0). Convert to absolute count.
            tvm = getattr(s, 'target_voxels', 2.0)
            target_voxels_val = int(float(tvm) * 1_000_000)

            resolution_val = _cf(getattr(s, 'resolution', 0.1), 0.1)
            voxel_mode = getattr(s, 'voxel_mode', 'TARGET_VOXELS')
            grow_val = _cf(getattr(s, 'grow', 2.0), 2.0)
            shrink_mult_val = _cf(getattr(s, 'shrink_mult', 1.5), 1.5)


            # Get selected mesh
            selected_objs = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
            if not selected_objs:
                self.report({'ERROR'}, "No mesh selected.")
                return {'CANCELLED'}

            src_mesh_blender = selected_objs[0]
            # Convert to meshlib; fallback to STL route for very dense meshes
            try:
                src_mesh = blender_to_meshlib(src_mesh_blender)
            except Exception as conv_err:
                print(f"[Quick Infill] Direct conversion failed ({conv_err}); falling back to STL export/import.")
                src_mesh = blender_to_meshlib_via_stl(src_mesh_blender)
            INITIAL_VERTEX_COUNT = src_mesh.topology.numValidVerts()
            print(f"Initial vertex count: {INITIAL_VERTEX_COUNT}")

            obj_name = src_mesh_blender.name

            # Calculate voxel size via helper or direct, based on mode
            if voxel_mode == 'RESOLUTION':
                vox = float(resolution_val)
            else:
                vox = compute_voxel_size(src_mesh, int(target_voxels_val), float(resolution_val))
            print(f"Voxel Size: {vox}")

            # Use helpers from offset_utils
            # Process mesh
            grow_mesh = cuda_offset(src_mesh, vox, grow_val)
            shrink_mesh = cuda_offset(grow_mesh, vox, -grow_val)
            shell_mesh = weighted_dist_shell(shrink_mesh, src_mesh, vox, shrink_mult_val)
            trim_mesh = mm.boolean(shrink_mesh, shell_mesh, mm.BooleanOperation.DifferenceAB).mesh

            # Smooth step: shrink then grow by one voxel to clean artifacts
            ss_mesh_shrink = cuda_offset(trim_mesh, vox, -vox)
            ss_mesh_grow = cuda_offset(ss_mesh_shrink, vox, vox)

            out_mesh = ss_mesh_grow
            # Prefer direct creation; if needed, switch to STL-based import at 0.1 scale
            try:
                infill_obj = meshlib_to_blender(out_mesh, obj_name + "Infill")
            except Exception as conv_back_err:
                print(f"[Quick Infill] Direct back-conversion failed ({conv_back_err}); importing via STL.")
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