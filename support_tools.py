"""
Support tools for Quick Infill addon.
Provides Fix Undercuts operation using MeshLib.
"""

import bpy
import math
import mathutils
from bpy.types import Operator, PropertyGroup
from bpy.props import FloatProperty, BoolProperty, EnumProperty
from .meshlib_utils import get_meshlib
from .blender_meshlib_utils import process_mesh_operation, blender_to_meshlib_via_stl, meshlib_to_blender_via_stl


class QuickInfillSupportSettings(PropertyGroup):
    voxel_size: FloatProperty(
        name="Voxel Size",
        description="Voxel size for undercut fixing (smaller = more precise but slower)",
        default=0.2,
        min=0.05,
        max=1.0,
        precision=3,
    )
    
    undercut_angle: FloatProperty(
        name="Angle",
        description="Undercut angle in degrees. 0° = vertical (Z up), 90° = horizontal (selected axis)",
        default=60.0,
        min=0.0,
        max=90.0,
        precision=1,
    )
    
    # Multi-select direction booleans
    dir_x_pos: BoolProperty(
        name="+X",
        description="Include +X direction",
        default=False,
    )
    dir_x_neg: BoolProperty(
        name="-X",
        description="Include -X direction",
        default=False,
    )
    dir_y_pos: BoolProperty(
        name="+Y",
        description="Include +Y direction",
        default=False,
    )
    dir_y_neg: BoolProperty(
        name="-Y",
        description="Include -Y direction",
        default=False,
    )
    
    # Diagonal directions
    dir_xpos_ypos: BoolProperty(
        name="+X+Y",
        description="Include +X+Y diagonal direction",
        default=False,
    )
    dir_xpos_yneg: BoolProperty(
        name="+X-Y",
        description="Include +X-Y diagonal direction",
        default=False,
    )
    dir_xneg_ypos: BoolProperty(
        name="-X+Y",
        description="Include -X+Y diagonal direction",
        default=False,
    )
    dir_xneg_yneg: BoolProperty(
        name="-X-Y",
        description="Include -X-Y diagonal direction",
        default=False,
    )
    
    replace_original: BoolProperty(
        name="Replace Original",
        description="Replace the original object with the result instead of creating a new object",
        default=False,
    )
    
    auto_decimate: BoolProperty(
        name="Auto Decimate",
        description="Automatically decimate after operations to match original vertex count",
        default=True,
    )
    
    auto_shrink: BoolProperty(
        name="Auto Shrink",
        description="Shrink the result to compensate for material added by undercut fixing",
        default=False,
    )
    
    shrink_amount: FloatProperty(
        name="Shrink Amount",
        description="Amount to shrink the mesh after undercut fixing",
        default=0.1,
        min=0.01,
        max=1.0,
        precision=2,
    )
    
    voxel_intersect_keep_original: BoolProperty(
        name="Keep Original",
        description="Keep original objects after voxel intersect (off = delete originals)",
        default=False,
    )
    
    show_support_tools: BoolProperty(
        name="Show Support Tools",
        description="Expand/collapse support tools panel",
        default=False,
    )


def compute_up_vector(horizontal_axis: tuple, angle_degrees: float):
    """
    Compute the up vector by tilting from Z-up toward a horizontal axis.
    
    Args:
        horizontal_axis: (x, y, z) tuple for the horizontal direction
        angle_degrees: Tilt angle. 0° = Z up, 90° = horizontal_axis
    
    Returns:
        MeshLib Vector3f
    """
    mm, _ = get_meshlib()
    angle_rad = math.radians(angle_degrees)
    
    # Blend between Z-up and horizontal axis
    # At 0°: pure Z (0, 0, 1)
    # At 90°: pure horizontal axis
    z_component = math.cos(angle_rad)
    horiz_component = math.sin(angle_rad)
    
    x = horizontal_axis[0] * horiz_component
    y = horizontal_axis[1] * horiz_component
    z = z_component
    
    return mm.Vector3f(x, y, z)


def get_selected_directions(settings) -> list:
    """Get list of selected horizontal direction tuples (normalized)."""
    directions = []
    DIAG = 0.7071067811865476  # sqrt(2)/2 for normalized diagonals
    
    # Cardinal directions
    if settings.dir_x_pos:
        directions.append((1, 0, 0))
    if settings.dir_x_neg:
        directions.append((-1, 0, 0))
    if settings.dir_y_pos:
        directions.append((0, 1, 0))
    if settings.dir_y_neg:
        directions.append((0, -1, 0))
    
    # Diagonal directions (normalized)
    if settings.dir_xpos_ypos:
        directions.append((DIAG, DIAG, 0))
    if settings.dir_xpos_yneg:
        directions.append((DIAG, -DIAG, 0))
    if settings.dir_xneg_ypos:
        directions.append((-DIAG, DIAG, 0))
    if settings.dir_xneg_yneg:
        directions.append((-DIAG, -DIAG, 0))
    
    return directions


def fix_undercuts_for_direction(mesh, up_vector, voxel_size):
    """Run undercut fix for a single direction, returns (result_mesh, undercut_count)."""
    mm, _ = get_meshlib()
    
    params = mm.FixUndercuts.FixParams()
    params.findParameters.upDirection = up_vector
    params.voxelSize = voxel_size
    
    # Find undercuts
    undercuts = mm.FaceBitSet()
    mm.FixUndercuts.find(mesh, params.findParameters, undercuts)
    undercut_count = undercuts.count()
    
    if undercut_count > 0:
        mm.FixUndercuts.fix(mesh, params)
    
    return mesh, undercut_count


def intersect_meshes(mesh_a, mesh_b, voxel_size):
    """Perform voxel-based boolean intersection of two meshlib meshes."""
    mm, _ = get_meshlib()
    
    # Use voxel boolean intersection - more robust than mesh boolean
    return mm.voxelBooleanIntersect(mesh_a, mesh_b, voxel_size)


class QUICKINFILL_OT_fix_undercuts(Operator):
    bl_idname = "quick_infill.fix_undercuts"
    bl_label = "Fix Undercuts"
    bl_description = "Fix undercuts on the selected mesh for better printability"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            mm, _ = get_meshlib()
            settings = context.scene.quick_infill_support_settings
            voxel_size = float(settings.voxel_size)
            angle = float(settings.undercut_angle)
            replace_original = settings.replace_original
            auto_decimate = settings.auto_decimate
            auto_shrink = settings.auto_shrink

            selected_objs = [obj for obj in context.selected_objects if obj.type == 'MESH']
            if not selected_objs:
                self.report({'ERROR'}, "No mesh selected.")
                return {'CANCELLED'}

            src_obj = selected_objs[0]
            
            # Get selected directions
            directions = get_selected_directions(settings)
            
            if not directions:
                # Default to Z-up if no directions selected
                directions = [(0, 0, 0)]  # Special case: pure Z-up (angle ignored)
            
            # Convert to meshlib
            mesh = blender_to_meshlib_via_stl(src_obj)
            initial_verts = mesh.topology.numValidVerts()
            total_undercuts = 0
            
            # If angle is 0, all directions produce the same Z-up vector, so just do one pass
            if angle == 0 or (len(directions) == 1 and directions[0] == (0, 0, 0)):
                # Pure Z-up mode (no horizontal tilt)
                up_vector = mm.Vector3f(0, 0, 1)
                mesh, undercut_count = fix_undercuts_for_direction(mesh, up_vector, voxel_size)
                total_undercuts = undercut_count
            else:
                # Process each direction and voxel-intersect results
                results = []
                for horiz_dir in directions:
                    # Make a copy of the mesh for each direction
                    mesh_copy = mm.copyMesh(mesh)
                    up_vector = compute_up_vector(horiz_dir, angle)
                    print(f"[Quick Infill] Fixing undercuts for direction {horiz_dir} at {angle}° with up=({up_vector.x:.3f}, {up_vector.y:.3f}, {up_vector.z:.3f})")
                    result_mesh, undercut_count = fix_undercuts_for_direction(mesh_copy, up_vector, voxel_size)
                    results.append(result_mesh)
                    total_undercuts += undercut_count
                
                # Voxel intersect all results together
                if len(results) == 1:
                    mesh = results[0]
                else:
                    mesh = results[0]
                    for i in range(1, len(results)):
                        mesh = intersect_meshes(mesh, results[i], voxel_size)
            
            # Auto shrink if enabled
            if auto_shrink:
                from .offset_utils import cuda_offset
                shrink_amount = float(settings.shrink_amount)
                mesh = cuda_offset(mesh, voxel_size, -shrink_amount)
            
            # Auto decimate if enabled
            if auto_decimate:
                from .offset_utils import decimate_mesh
                current_verts = mesh.topology.numValidVerts()
                if current_verts > initial_verts:
                    mesh = decimate_mesh(mesh, target_vertex_count=initial_verts)
            
            final_verts = mesh.topology.numValidVerts()
            
            # Convert back to Blender
            new_name = src_obj.name + "_NoUndercuts"
            result_obj = meshlib_to_blender_via_stl(mesh, name=new_name)
            
            # Handle transforms
            if replace_original:
                from .blender_meshlib_utils import replace_mesh_keep_transforms
                result_obj = replace_mesh_keep_transforms(src_obj, result_obj)
            # Note: non-replace leaves object at origin with correct scale
            
            if total_undercuts == 0:
                self.report({'INFO'}, "No undercuts found on mesh.")
            else:
                dir_count = len(directions) if directions[0] != (0, 0, 0) else 1
                print(f"[Quick Infill] Fix Undercuts: {initial_verts} → {final_verts} vertices, {dir_count} direction(s), {total_undercuts} total undercut faces")
                self.report({'INFO'}, f"Fixed undercuts from {dir_count} direction(s). Result: '{result_obj.name}'")
            
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Fix Undercuts failed: {e}")
            print(f"[Quick Infill] Fix Undercuts error: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


class QUICKINFILL_OT_fix_undercuts_from_view(Operator):
    bl_idname = "quick_infill.fix_undercuts_from_view"
    bl_label = "From View"
    bl_description = "Fix undercuts using the current viewport direction as the up axis"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            mm, _ = get_meshlib()
            settings = context.scene.quick_infill_support_settings
            voxel_size = float(settings.voxel_size)
            replace_original = settings.replace_original
            auto_decimate = settings.auto_decimate

            selected_objs = [obj for obj in context.selected_objects if obj.type == 'MESH']
            if not selected_objs:
                self.report({'ERROR'}, "No mesh selected.")
                return {'CANCELLED'}

            # Get viewport view direction
            region_3d = None
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    region_3d = area.spaces.active.region_3d
                    break
            
            if region_3d is None:
                self.report({'ERROR'}, "No 3D viewport found.")
                return {'CANCELLED'}
            
            # Get view direction - the direction pointing toward the viewer
            # view_rotation is a quaternion; local -Z is where the camera looks
            # So local +Z (toward viewer) in world space is the "up" for undercuts
            view_dir = region_3d.view_rotation @ mathutils.Vector((0, 0, 1))
            view_dir.normalize()
            
            up_vector = mm.Vector3f(view_dir.x, view_dir.y, view_dir.z)
            
            print(f"[Quick Infill] From View: up direction = ({view_dir.x:.3f}, {view_dir.y:.3f}, {view_dir.z:.3f})")

            src_obj = selected_objs[0]
            mesh = blender_to_meshlib_via_stl(src_obj)
            initial_verts = mesh.topology.numValidVerts()
            
            mesh, undercut_count = fix_undercuts_for_direction(mesh, up_vector, voxel_size)
            
            # Auto decimate if enabled
            if auto_decimate:
                from .offset_utils import decimate_mesh
                current_verts = mesh.topology.numValidVerts()
                if current_verts > initial_verts:
                    mesh = decimate_mesh(mesh, target_vertex_count=initial_verts)
            
            final_verts = mesh.topology.numValidVerts()
            
            # Convert back to Blender
            new_name = src_obj.name + "_NoUndercuts"
            result_obj = meshlib_to_blender_via_stl(mesh, name=new_name)
            
            # Handle transforms
            if replace_original:
                from .blender_meshlib_utils import replace_mesh_keep_transforms
                result_obj = replace_mesh_keep_transforms(src_obj, result_obj)
            # Note: non-replace leaves object at origin with correct scale
            
            if undercut_count == 0:
                self.report({'INFO'}, "No undercuts found from view direction.")
            else:
                print(f"[Quick Infill] Fix Undercuts (View): {initial_verts} → {final_verts} vertices, {undercut_count} undercut faces")
                self.report({'INFO'}, f"Fixed {undercut_count} undercut faces from view. Result: '{result_obj.name}'")
            
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Fix Undercuts (View) failed: {e}")
            print(f"[Quick Infill] Fix Undercuts (View) error: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


class QUICKINFILL_OT_voxel_intersect(Operator):
    """Intersect all selected mesh objects using voxel boolean"""
    bl_idname = "quick_infill.voxel_intersect"
    bl_label = "Voxel Intersect"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        # Need at least 2 selected mesh objects
        selected_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']
        return len(selected_meshes) >= 2
    
    def execute(self, context):
        try:
            mm, _ = get_meshlib()
            settings = context.scene.quick_infill_support_settings
            voxel_size = settings.voxel_size
            keep_original = settings.voxel_intersect_keep_original
            
            # Get all selected mesh objects
            selected_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']
            if len(selected_meshes) < 2:
                self.report({'ERROR'}, "Select at least 2 mesh objects to intersect.")
                return {'CANCELLED'}
            
            # Use active object or first selected as starting point
            active_obj = context.active_object if context.active_object in selected_meshes else selected_meshes[0]
            
            # Convert first mesh to meshlib
            result_mesh = blender_to_meshlib_via_stl(active_obj)
            mesh_names = [active_obj.name]
            
            # Iteratively intersect with remaining meshes
            for obj in selected_meshes:
                if obj == active_obj:
                    continue
                
                other_mesh = blender_to_meshlib_via_stl(obj)
                result_mesh = intersect_meshes(result_mesh, other_mesh, voxel_size)
                mesh_names.append(obj.name)
            
            # Check if result is valid
            if result_mesh.topology.numValidVerts() == 0:
                self.report({'ERROR'}, "Intersection resulted in empty mesh. Objects may not overlap.")
                return {'CANCELLED'}
            
            final_verts = result_mesh.topology.numValidVerts()
            
            # Convert back to Blender
            new_name = active_obj.name + "_Intersect"
            result_obj = meshlib_to_blender_via_stl(result_mesh, name=new_name)
            
            # Handle transforms
            if not keep_original:
                from .blender_meshlib_utils import replace_mesh_keep_transforms
                result_obj = replace_mesh_keep_transforms(active_obj, result_obj)
                
                # Delete other selected meshes
                for obj in selected_meshes:
                    if obj != active_obj:
                        bpy.data.objects.remove(obj, do_unlink=True)
            
            print(f"[Quick Infill] Voxel Intersect: {len(mesh_names)} objects → {final_verts} vertices")
            self.report({'INFO'}, f"Intersected {len(mesh_names)} objects. Result: '{result_obj.name}'")
            
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Voxel Intersect failed: {e}")
            print(f"[Quick Infill] Voxel Intersect error: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


def draw_support_tools(layout, context):
    """Draw the Support Tools UI into the given layout.
    
    Args:
        layout: The parent Blender UI layout to draw into
        context: The Blender context
    """
    settings = context.scene.quick_infill_support_settings
    col = layout.column(align=True)
    
    # Collapsible box similar to Settings panel
    box = col.box()
    header = box.row()
    
    # Collapsible header with arrow icon
    show_tools = getattr(settings, 'show_support_tools', False)
    icon = 'DOWNARROW_HLT' if show_tools else 'RIGHTARROW'
    header.prop(settings, "show_support_tools", text="Support Tools", icon=icon, emboss=False)
    
    # Show tools when expanded
    if show_tools:
        tools_col = box.column(align=True)
        
        def prop_with_suffix(parent_layout, data, attr, label="", suffix="mm"):
            split = parent_layout.split(factor=0.9, align=True)
            c = split.column(align=True)
            c.use_property_split = False
            c.use_property_decorate = False
            c.prop(data, attr, text=label)
            split.label(text=suffix)
        
        # Auto Decimate, Replace Original checkboxes
        row = tools_col.row(align=True)
        row.prop(settings, "auto_decimate", text="Auto Decimate")
        row.prop(settings, "replace_original", text="Replace Original")
        
        # Auto Shrink checkbox
        tools_col.prop(settings, "auto_shrink", text="Auto Shrink")
        
        # Shrink amount slider (only shown when auto_shrink is enabled)
        if settings.auto_shrink:
            prop_with_suffix(tools_col, settings, "shrink_amount", "Shrink Amount", "mm")
        
        tools_col.separator()
        
        # Directions label and multi-select toggles
        tools_col.label(text="Directions:")
        dir_row = tools_col.row(align=True)
        dir_row.prop(settings, "dir_x_pos", text="+X", toggle=True)
        dir_row.prop(settings, "dir_x_neg", text="-X", toggle=True)
        dir_row.prop(settings, "dir_y_pos", text="+Y", toggle=True)
        dir_row.prop(settings, "dir_y_neg", text="-Y", toggle=True)
        
        # Diagonal directions row
        diag_row = tools_col.row(align=True)
        diag_row.prop(settings, "dir_xpos_ypos", text="+X+Y", toggle=True)
        diag_row.prop(settings, "dir_xpos_yneg", text="+X-Y", toggle=True)
        diag_row.prop(settings, "dir_xneg_ypos", text="-X+Y", toggle=True)
        diag_row.prop(settings, "dir_xneg_yneg", text="-X-Y", toggle=True)
        
        # Angle slider (degrees, shown without subtype conversion)
        angle_row = tools_col.row(align=True)
        angle_row.prop(settings, "undercut_angle", text="Angle")
        
        # Voxel size slider
        prop_with_suffix(tools_col, settings, "voxel_size", "Voxel Size", "mm")
        
        tools_col.separator()
        
        # Fix Undercuts and From View buttons on same line
        row = tools_col.row(align=True)
        row.operator("quick_infill.fix_undercuts", text="Fix Undercuts", icon='MOD_SMOOTH')
        row.operator("quick_infill.fix_undercuts_from_view", text="From View", icon='HIDE_OFF')
        
        tools_col.separator()
        
        # Voxel Intersect with Keep Original on same line
        row = tools_col.row(align=True)
        row.operator("quick_infill.voxel_intersect", text="Voxel Intersect", icon='MOD_BOOLEAN')
        row.prop(settings, "voxel_intersect_keep_original", text="Keep Original")


classes = (
    QuickInfillSupportSettings,
    QUICKINFILL_OT_fix_undercuts,
    QUICKINFILL_OT_fix_undercuts_from_view,
    QUICKINFILL_OT_voxel_intersect,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.quick_infill_support_settings = bpy.props.PointerProperty(
        type=QuickInfillSupportSettings
    )


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    del bpy.types.Scene.quick_infill_support_settings
