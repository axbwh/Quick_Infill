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
from .blender_meshlib_utils import process_mesh_operation, batch_process_mesh_operation, blender_to_meshlib_via_stl, meshlib_to_blender_via_stl, select_results


def reset_angle_preset(self, context):
    """Clear the active angle preset only when the value is manually edited."""
    active = getattr(self, 'active_angle_preset', 'NONE')
    if active == 'NONE':
        return

    target = float(self.undercut_angle)
    if active == '10' and abs(target - 10.0) > 1e-6:
        self.active_angle_preset = 'NONE'
    elif active == '15' and abs(target - 15.0) > 1e-6:
        self.active_angle_preset = 'NONE'
    elif active == '30' and abs(target - 30.0) > 1e-6:
        self.active_angle_preset = 'NONE'
    elif active == '45' and abs(target - 45.0) > 1e-6:
        self.active_angle_preset = 'NONE'
    elif active == '60' and abs(target - 60.0) > 1e-6:
        self.active_angle_preset = 'NONE'
    elif active == '90' and abs(target - 90.0) > 1e-6:
        self.active_angle_preset = 'NONE'


class QuickInfillSupportSettings(PropertyGroup):
    active_angle_preset: EnumProperty(
        name="Active Angle Preset",
        description="Current angle preset selection",
        items=[
            ("NONE", "None", "No preset selected"),
            ("10", "10°", "10 degree preset"),
            ("15", "15°", "15 degree preset"),
            ("30", "30°", "30 degree preset"),
            ("45", "45°", "45 degree preset"),
            ("60", "60°", "60 degree preset"),
            ("90", "90°", "90 degree preset"),
        ],
        default="60",
    )

    undercut_angle: FloatProperty(
        name="Angle",
        description="Undercut angle in degrees. 0° = vertical (Z up), +/-90° = horizontal (selected axis). Negative angles use voxel union when combining directions",
        default=60.0,
        min=-90.0,
        max=90.0,
        soft_min=-90.0,
        soft_max=90.0,
        precision=1,
        update=reset_angle_preset,
    )
    
    # Multi-select direction booleans
    dir_x_pos: BoolProperty(
        name="+X",
        description="Include +X direction",
        default=True,
    )
    dir_x_neg: BoolProperty(
        name="-X",
        description="Include -X direction",
        default=True,
    )
    dir_y_pos: BoolProperty(
        name="+Y",
        description="Include +Y direction",
        default=True,
    )
    dir_y_neg: BoolProperty(
        name="-Y",
        description="Include -Y direction",
        default=True,
    )
    
    # Diagonal directions
    dir_xpos_ypos: BoolProperty(
        name="+X+Y",
        description="Include +X+Y diagonal direction",
        default=True,
    )
    dir_xpos_yneg: BoolProperty(
        name="+X-Y",
        description="Include +X-Y diagonal direction",
        default=True,
    )
    dir_xneg_ypos: BoolProperty(
        name="-X+Y",
        description="Include -X+Y diagonal direction",
        default=True,
    )
    dir_xneg_yneg: BoolProperty(
        name="-X-Y",
        description="Include -X-Y diagonal direction",
        default=True,
    )
    
    # Center Z direction (straight on, no tilt)
    dir_z: BoolProperty(
        name="Z",
        description="Include straight-on direction (no tilt, like 0° angle)",
        default=False,
    )
    
    replace_original: BoolProperty(
        name="Replace Original",
        description="Replace the original object with the result instead of creating a new object",
        default=True,
    )

    process_islands: BoolProperty(
        name="Islands",
        description="Process each mesh island individually, then join the result back into one object",
        default=False,
    )

    smart_resolution: BoolProperty(
        name="Smart",
        description="Use each island/object's existing resolution tag (if any) for its voxel size and decimation ratio",
        default=False,
    )
    
    decimate_mode: EnumProperty(
        name="Decimate Mode",
        description="Choose when auto-decimation is active",
        items=[
            ("OFF", "Off", "Disable auto-decimation"),
            ("ORIGINAL", "Original", "Decimate back to the original polycount"),
            ("VOXEL_RATIO", "Ratio", "Decimate based on the current ratio value"),
        ],
        default="VOXEL_RATIO",
        options={'HIDDEN'},
    )

    decimation_ratio: FloatProperty(
        name="Decimation Ratio",
        description="Fraction of final faces to keep during auto-decimation",
        default=0.015,
        min=0.0,
        max=1.0,
        precision=4,
        options={'HIDDEN'},
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
    
    shrink_angle_threshold: FloatProperty(
        name="Shrink Angle",
        description="Only shrink faces within this angle (degrees) of pointing straight up",
        default=70.0,
        min=5.0,
        max=90.0,
        precision=1,
    )
    
    voxel_intersect_keep_original: BoolProperty(
        name="Keep Original",
        description="Keep original objects after voxel intersect (off = delete originals)",
        default=False,
    )

    apply_modifiers_on_export: BoolProperty(
        name="Apply Modifiers",
        description="Apply each object's modifiers (viewport result) when exporting for Voxel Intersect. Off = ignore modifiers",
        default=False,
    )
    
    show_support_tools: BoolProperty(
        name="Show Support Tools",
        description="Expand/collapse support tools panel",
        default=False,
    )

    voxel_size: FloatProperty(
        name="Voxel Size",
        description="Internal compatibility value for the shared voxel size",
        default=0.05,
        min=0.025,
        max=0.4,
        precision=3,
        options={'HIDDEN'},
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
    """Get list of selected horizontal direction tuples (normalized).
    
    Returns tuples of (x, y, z) where z=0 for horizontal directions,
    or (0, 0, 1) for the special 'Z' center direction.
    """
    directions = []
    DIAG = 0.7071067811865476  # sqrt(2)/2 for normalized diagonals
    
    # Center Z direction (straight on, no tilt)
    if settings.dir_z:
        directions.append((0, 0, 1))  # Special marker for pure up
    
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


def compute_view_space_up_vector(screen_direction, view_rotation, angle_degrees):
    """
    Compute an up vector in world space from a screen-space direction.
    
    The direction grid is interpreted as screen space:
    - +X = right side of screen
    - -X = left side of screen
    - +Y = top of screen
    - -Y = bottom of screen
    - Z (0,0,1) = straight at camera (no tilt)
    
    Args:
        screen_direction: (x, y, z) tuple in screen space. z=1 means pure camera direction.
        view_rotation: Blender quaternion from camera space to world space
        angle_degrees: Tilt angle from camera direction toward screen direction
    
    Returns:
        MeshLib Vector3f for the up direction in world space
    """
    mm, _ = get_meshlib()
    
    # Camera forward direction (toward viewer) = camera +Z in world space
    camera_forward = view_rotation @ mathutils.Vector((0, 0, 1))
    camera_forward.normalize()
    
    # If direction is Z (0,0,1), return pure camera forward (no tilt)
    if screen_direction[2] == 1:
        return mm.Vector3f(camera_forward.x, camera_forward.y, camera_forward.z)
    
    # Transform screen direction to world space
    # screen_direction.x corresponds to camera right (+X in camera space)
    # screen_direction.y corresponds to camera up (+Y in camera space)
    camera_right = view_rotation @ mathutils.Vector((1, 0, 0))
    camera_up = view_rotation @ mathutils.Vector((0, 1, 0))
    
    # World-space tilt direction (combination of camera_right and camera_up)
    tilt_dir = mathutils.Vector((
        screen_direction[0] * camera_right.x + screen_direction[1] * camera_up.x,
        screen_direction[0] * camera_right.y + screen_direction[1] * camera_up.y,
        screen_direction[0] * camera_right.z + screen_direction[1] * camera_up.z
    ))
    tilt_dir.normalize()
    
    # Blend between camera_forward and tilt_dir based on angle
    angle_rad = math.radians(angle_degrees)
    forward_component = math.cos(angle_rad)
    tilt_component = math.sin(angle_rad)
    
    result = mathutils.Vector((
        camera_forward.x * forward_component + tilt_dir.x * tilt_component,
        camera_forward.y * forward_component + tilt_dir.y * tilt_component,
        camera_forward.z * forward_component + tilt_dir.z * tilt_component
    ))
    result.normalize()
    
    return mm.Vector3f(result.x, result.y, result.z)


def shrink_top_faces_along_normals(mesh, up_vector, shrink_amount, angle_threshold=30.0):
    """
    Shrink vertices of top-facing faces (opposite to undercuts) along their normals.
    Only affects faces nearly perpendicular to the up direction.
    
    Args:
        mesh: MeshLib mesh
        up_vector: The up direction used for undercut fixing
        shrink_amount: Distance to move vertices inward (positive = shrink)
        angle_threshold: Only shrink faces within this angle (degrees) of pointing straight up
    
    Returns:
        Modified mesh
    """
    mm, _ = get_meshlib()
    
    # Find top-facing faces by looking for "undercuts" from the opposite direction
    # These are faces that would be undercuts if printed upside-down
    opposite_up = mm.Vector3f(-up_vector.x, -up_vector.y, -up_vector.z)
    
    # Use FixParams to get findParameters (same pattern as fix_undercuts_for_direction)
    params = mm.FixUndercuts.FixParams()
    params.findParameters.upDirection = opposite_up
    
    top_faces = mm.FaceBitSet()
    mm.FixUndercuts.find(mesh, params.findParameters, top_faces)
    
    if top_faces.count() == 0:
        return mesh
    
    # Get vertex normals
    vert_normals = mm.computePerVertNormals(mesh)
    
    # Compute threshold: cos(angle_threshold) - normals must have dot product >= this with up_vector
    cos_threshold = math.cos(math.radians(angle_threshold))
    
    # Get the points buffer for modification
    points = mesh.points
    
    # Get vertices belonging to top faces
    top_verts = mm.getIncidentVerts(mesh.topology, top_faces)
    
    # Move each top vertex inward along its normal, but only if it's nearly horizontal
    shrunk_count = 0
    for vid in range(mesh.topology.numValidVerts()):
        vert_id = mm.VertId(vid)
        if top_verts.test(vert_id):
            normal = vert_normals[vert_id]
            
            # Check if this vertex's normal is within the angle threshold of pointing "up"
            # Dot product with up_vector: 1.0 = pointing straight up, 0.0 = horizontal
            dot = normal.x * up_vector.x + normal.y * up_vector.y + normal.z * up_vector.z
            
            if dot >= cos_threshold:
                # Move inward (opposite to normal direction)
                old_pos = points[vert_id]
                new_pos = mm.Vector3f(
                    old_pos.x - normal.x * shrink_amount,
                    old_pos.y - normal.y * shrink_amount,
                    old_pos.z - normal.z * shrink_amount
                )
                points[vert_id] = new_pos
                shrunk_count += 1
    
    print(f"[Quick Infill] Shrunk {shrunk_count} vertices (of {top_verts.count()} candidates) within {angle_threshold}° of horizontal by {shrink_amount}mm")
    return mesh


def fix_undercuts_for_direction(mesh, up_vector, voxel_size, shrink_amount=0.0, shrink_angle=30.0):
    """Run undercut fix for a single direction, returns (result_mesh, undercut_count).
    
    Args:
        mesh: MeshLib mesh to fix
        up_vector: The up direction for undercut detection
        voxel_size: Voxel size for the fix operation
        shrink_amount: If > 0, shrink top-facing vertices by this amount after fixing
        shrink_angle: Angle threshold for shrinking (degrees from straight up)
    """
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
    
    # Shrink top faces if requested
    if shrink_amount > 0:
        mesh = shrink_top_faces_along_normals(mesh, up_vector, shrink_amount, shrink_angle)
    
    return mesh, undercut_count


def intersect_meshes(mesh_a, mesh_b, voxel_size):
    """Perform voxel-based boolean intersection of two meshlib meshes."""
    mm, _ = get_meshlib()
    
    # Use voxel boolean intersection - more robust than mesh boolean
    return mm.voxelBooleanIntersect(mesh_a, mesh_b, voxel_size)


def union_meshes(mesh_a, mesh_b, voxel_size):
    """Perform voxel-based boolean union of two meshlib meshes."""
    mm, _ = get_meshlib()

    return mm.voxelBooleanUnite(mesh_a, mesh_b, voxel_size)


def diff_meshes(mesh_a, mesh_b, voxel_size):
    """Perform voxel-based boolean subtraction (mesh_a - mesh_b)."""
    mm, _ = get_meshlib()

    return mm.voxelBooleanSubtract(mesh_a, mesh_b, voxel_size)


def fix_undercuts_single_mesh(mesh, directions, angle, voxel_size, shrink_amount, shrink_angle):
    """
    Process undercut fixing for a single meshlib mesh.
    
    Args:
        mesh: MeshLib mesh to process
        directions: List of direction tuples for undercut fixing
        angle: Undercut angle in degrees
        voxel_size: Voxel size for fixing
        shrink_amount: Amount to shrink top faces (0 = disabled)
        shrink_angle: Angle threshold for shrinking
    
    Returns:
        tuple: (processed_mesh, total_undercuts)
    """
    mm, _ = get_meshlib()
    
    total_undercuts = 0
    abs_angle = abs(float(angle))
    use_union_combine = float(angle) < 0.0
    only_z_direction = len(directions) == 1 and directions[0] == (0, 0, 1)
    
    if abs_angle == 0 or only_z_direction:
        # Pure Z-up mode (no horizontal tilt)
        up_vector = mm.Vector3f(0, 0, 1)
        mesh, undercut_count = fix_undercuts_for_direction(mesh, up_vector, voxel_size, shrink_amount, shrink_angle)
        total_undercuts = undercut_count
    else:
        # In union mode (negative angle), Z is applied as a post-process on the
        # merged result rather than being unioned as a separate direction.
        has_z = (0, 0, 1) in directions
        apply_z_after = use_union_combine and has_z
        process_dirs = [d for d in directions if d != (0, 0, 1)] if apply_z_after else directions

        # Process each direction, then combine with voxel intersect (positive angle)
        # or voxel union (negative angle).
        results = []
        for horiz_dir in process_dirs:
            if horiz_dir == (0, 0, 1):
                # Z direction = pure Z-up
                mesh_copy = mm.copyMesh(mesh)
                up_vector = mm.Vector3f(0, 0, 1)
                result_mesh, undercut_count = fix_undercuts_for_direction(mesh_copy, up_vector, voxel_size, shrink_amount, shrink_angle)
                results.append(result_mesh)
                total_undercuts += undercut_count
                continue
            
            # Make a copy of the mesh for each direction
            mesh_copy = mm.copyMesh(mesh)
            up_vector = compute_up_vector(horiz_dir, abs_angle)
            result_mesh, undercut_count = fix_undercuts_for_direction(mesh_copy, up_vector, voxel_size, shrink_amount, shrink_angle)
            results.append(result_mesh)
            total_undercuts += undercut_count
        
        # Combine all results together.
        if len(results) == 1:
            mesh = results[0]
        else:
            mesh = results[0]
            for i in range(1, len(results)):
                if use_union_combine:
                    mesh = union_meshes(mesh, results[i], voxel_size)
                else:
                    mesh = intersect_meshes(mesh, results[i], voxel_size)

        # In union mode, apply Z fix on the merged result instead of the original.
        if apply_z_after:
            up_vector = mm.Vector3f(0, 0, 1)
            mesh, undercut_count = fix_undercuts_for_direction(mesh, up_vector, voxel_size, shrink_amount, shrink_angle)
            total_undercuts += undercut_count
    
    return mesh, total_undercuts


def fix_undercuts_from_view_single_mesh(mesh, directions, angle, voxel_size, shrink_amount, shrink_angle, view_rotation):
    """
    Process undercut fixing from view for a single meshlib mesh.
    
    Args:
        mesh: MeshLib mesh to process
        directions: List of screen-space direction tuples
        angle: Undercut angle in degrees
        voxel_size: Voxel size for fixing
        shrink_amount: Amount to shrink top faces (0 = disabled)
        shrink_angle: Angle threshold for shrinking
        view_rotation: Blender quaternion for camera rotation
    
    Returns:
        tuple: (processed_mesh, total_undercuts)
    """
    mm, _ = get_meshlib()
    
    total_undercuts = 0
    abs_angle = abs(float(angle))
    use_union_combine = float(angle) < 0.0
    only_z_direction = len(directions) == 1 and directions[0] == (0, 0, 1)
    
    if only_z_direction or abs_angle == 0:
        # Pure camera direction (no tilt)
        camera_forward = view_rotation @ mathutils.Vector((0, 0, 1))
        camera_forward.normalize()
        up_vector = mm.Vector3f(camera_forward.x, camera_forward.y, camera_forward.z)
        mesh, undercut_count = fix_undercuts_for_direction(mesh, up_vector, voxel_size, shrink_amount, shrink_angle)
        total_undercuts = undercut_count
    else:
        # In union mode (negative angle), Z is applied as a post-process on the
        # merged result rather than being unioned as a separate direction.
        has_z = (0, 0, 1) in directions
        apply_z_after = use_union_combine and has_z
        process_dirs = [d for d in directions if d != (0, 0, 1)] if apply_z_after else directions

        # Process each direction in view space, then combine with voxel intersect
        # (positive angle) or voxel union (negative angle).
        results = []
        for screen_dir in process_dirs:
            mesh_copy = mm.copyMesh(mesh)
            up_vector = compute_view_space_up_vector(screen_dir, view_rotation, abs_angle)
            result_mesh, undercut_count = fix_undercuts_for_direction(mesh_copy, up_vector, voxel_size, shrink_amount, shrink_angle)
            results.append(result_mesh)
            total_undercuts += undercut_count
        
        # Combine all results together.
        if len(results) == 1:
            mesh = results[0]
        else:
            mesh = results[0]
            for i in range(1, len(results)):
                if use_union_combine:
                    mesh = union_meshes(mesh, results[i], voxel_size)
                else:
                    mesh = intersect_meshes(mesh, results[i], voxel_size)

        # In union mode, apply Z fix on the merged result instead of the original.
        if apply_z_after:
            camera_forward = view_rotation @ mathutils.Vector((0, 0, 1))
            camera_forward.normalize()
            up_vector = mm.Vector3f(camera_forward.x, camera_forward.y, camera_forward.z)
            mesh, undercut_count = fix_undercuts_for_direction(mesh, up_vector, voxel_size, shrink_amount, shrink_angle)
            total_undercuts += undercut_count
    
    return mesh, total_undercuts


class QUICKINFILL_OT_fix_undercuts(Operator):
    bl_idname = "quick_infill.fix_undercuts"
    bl_label = "Fix Undercuts"
    bl_description = "Fix undercuts on selected mesh(es) for better printability. With multiple selections, processes each object individually"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            settings = context.scene.quick_infill_support_settings
            voxel_size = float(settings.voxel_size)
            angle = float(settings.undercut_angle)
            replace_original = settings.replace_original
            auto_decimate = settings.decimate_mode != "OFF"
            auto_shrink = settings.auto_shrink

            selected_objs = [obj for obj in context.selected_objects if obj.type == 'MESH']
            if not selected_objs:
                self.report({'ERROR'}, "No mesh selected.")
                return {'CANCELLED'}

            directions = get_selected_directions(settings)
            if not directions:
                directions = [(0, 0, 1)]

            shrink_amount = float(settings.shrink_amount) if auto_shrink else 0.0
            shrink_angle = float(settings.shrink_angle_threshold) if auto_shrink else 30.0

            # List append is safe here even when batch_process_mesh_operation
            # runs this concurrently across objects (CPython list.append is
            # atomic under the GIL); only the aggregate total is reported.
            undercut_counts = []

            def fix_undercuts_op(mesh, res):
                result_mesh, undercut_count = fix_undercuts_single_mesh(
                    mesh, directions, angle, res, shrink_amount, shrink_angle
                )
                undercut_counts.append(undercut_count)
                return result_mesh

            if len(selected_objs) == 1:
                result_obj, initial_verts, final_verts = process_mesh_operation(
                    selected_objs[0], fix_undercuts_op, "_NoUndercuts",
                    auto_decimate=auto_decimate, replace_original=replace_original, resolution=voxel_size
                )
                results = [(result_obj, initial_verts, final_verts)]
            else:
                results, _ = batch_process_mesh_operation(
                    selected_objs, fix_undercuts_op, "_NoUndercuts",
                    auto_decimate=auto_decimate, replace_original=replace_original, resolution=voxel_size
                )

            total_obj_undercuts = sum(undercut_counts)
            obj_count = len(results)
            dir_count = len(directions)
            if obj_count == 1:
                result_obj, _, _ = results[0]
                if total_obj_undercuts == 0:
                    self.report({'INFO'}, "No undercuts found on mesh.")
                else:
                    self.report({'INFO'}, f"Fixed undercuts from {dir_count} direction(s). Result: '{result_obj.name}'")
            else:
                if total_obj_undercuts == 0:
                    self.report({'INFO'}, f"No undercuts found on {obj_count} meshes.")
                else:
                    if replace_original:
                        self.report({'INFO'}, f"Fixed undercuts on {obj_count} objects from {dir_count} direction(s)")
                    else:
                        self.report({'INFO'}, f"Fixed undercuts, created {obj_count} new objects from {dir_count} direction(s)")

            select_results([r[0] for r in results])
            bpy.context.view_layer.update()
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Fix Undercuts failed: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


class QUICKINFILL_OT_fix_undercuts_from_view(Operator):
    bl_idname = "quick_infill.fix_undercuts_from_view"
    bl_label = "From View"
    bl_description = "Fix undercuts using the viewport direction. With multiple selections, processes each object individually"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            settings = context.scene.quick_infill_support_settings
            voxel_size = float(settings.voxel_size)
            angle = float(settings.undercut_angle)
            replace_original = settings.replace_original
            auto_decimate = settings.decimate_mode != "OFF"
            auto_shrink = settings.auto_shrink

            selected_objs = [obj for obj in context.selected_objects if obj.type == 'MESH']
            if not selected_objs:
                self.report({'ERROR'}, "No mesh selected.")
                return {'CANCELLED'}

            region_3d = None
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    region_3d = area.spaces.active.region_3d
                    break
            if region_3d is None:
                self.report({'ERROR'}, "No 3D viewport found.")
                return {'CANCELLED'}

            # Capture view rotation before entering threads
            view_rotation = region_3d.view_rotation.copy()

            directions = get_selected_directions(settings)
            if not directions:
                directions = [(0, 0, 1)]

            shrink_amount = float(settings.shrink_amount) if auto_shrink else 0.0
            shrink_angle = float(settings.shrink_angle_threshold) if auto_shrink else 70.0

            undercut_counts = []

            def fix_undercuts_view_op(mesh, res):
                result_mesh, undercut_count = fix_undercuts_from_view_single_mesh(
                    mesh, directions, angle, res, shrink_amount, shrink_angle, view_rotation
                )
                undercut_counts.append(undercut_count)
                return result_mesh

            if len(selected_objs) == 1:
                result_obj, initial_verts, final_verts = process_mesh_operation(
                    selected_objs[0], fix_undercuts_view_op, "_NoUndercuts",
                    auto_decimate=auto_decimate, replace_original=replace_original, resolution=voxel_size
                )
                results = [(result_obj, initial_verts, final_verts)]
            else:
                results, _ = batch_process_mesh_operation(
                    selected_objs, fix_undercuts_view_op, "_NoUndercuts",
                    auto_decimate=auto_decimate, replace_original=replace_original, resolution=voxel_size
                )

            total_obj_undercuts = sum(undercut_counts)
            obj_count = len(results)
            dir_count = len(directions)
            if obj_count == 1:
                result_obj, _, _ = results[0]
                if total_obj_undercuts == 0:
                    self.report({'INFO'}, "No undercuts found from view direction.")
                else:
                    self.report({'INFO'}, f"Fixed undercuts from {dir_count} view direction(s). Result: '{result_obj.name}'")
            else:
                if total_obj_undercuts == 0:
                    self.report({'INFO'}, f"No undercuts found on {obj_count} meshes.")
                else:
                    if replace_original:
                        self.report({'INFO'}, f"Fixed undercuts on {obj_count} objects from {dir_count} view direction(s)")
                    else:
                        self.report({'INFO'}, f"Fixed undercuts, created {obj_count} new objects from {dir_count} view direction(s)")

            select_results([r[0] for r in results])
            bpy.context.view_layer.update()
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Fix Undercuts (View) failed: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


def _voxel_boolean_combine(context, combine_fn, op_label, suffix):
    """Shared pipeline for Voxel Intersect/Union/Diff: starting from the
    active object, combine each other selected mesh into it individually
    using combine_fn(result_mesh, other_mesh, voxel_size).

    Returns (result_obj, message) on success, or (None, error_message) on failure.
    """
    settings = context.scene.quick_infill_support_settings
    voxel_size = settings.voxel_size
    replace_original = settings.replace_original
    apply_modifiers = settings.apply_modifiers_on_export

    selected_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']
    if len(selected_meshes) < 2:
        return None, f"Select at least 2 mesh objects to {op_label.lower()}."

    # Use active object or first selected as starting point
    active_obj = context.active_object if context.active_object in selected_meshes else selected_meshes[0]

    result_mesh = blender_to_meshlib_via_stl(active_obj, apply_modifiers=apply_modifiers)
    mesh_names = [active_obj.name]

    # Iteratively combine with remaining meshes
    for obj in selected_meshes:
        if obj == active_obj:
            continue

        other_mesh = blender_to_meshlib_via_stl(obj, apply_modifiers=apply_modifiers)
        result_mesh = combine_fn(result_mesh, other_mesh, voxel_size)
        mesh_names.append(obj.name)

    # Respect the shared auto-decimation option before exporting back to Blender.
    if settings.decimate_mode != "OFF":
        from .offset_utils import should_auto_decimate_faces, decimate_mesh
        final_faces_before = result_mesh.topology.numValidFaces()
        do_decimate, target_faces = should_auto_decimate_faces(
            sum(obj.data.vertices.__len__() for obj in selected_meshes if obj.type == 'MESH' and obj.data is not None),
            final_faces_before,
            voxel_size=float(voxel_size),
            mode=settings.decimate_mode,
            ratio=settings.decimation_ratio,
        )
        if do_decimate:
            result_mesh = decimate_mesh(result_mesh, target_face_count=target_faces, resolution=float(voxel_size))

    if result_mesh.topology.numValidVerts() == 0:
        return None, f"{op_label} resulted in empty mesh. Objects may not overlap."

    final_verts = result_mesh.topology.numValidVerts()

    new_name = active_obj.name + suffix
    result_obj = meshlib_to_blender_via_stl(result_mesh, name=new_name)

    # Handle transforms using the shared Replace toggle.
    if replace_original:
        from .blender_meshlib_utils import replace_mesh_keep_transforms
        result_obj = replace_mesh_keep_transforms(active_obj, result_obj)

        # Delete other selected meshes
        for obj in selected_meshes:
            if obj != active_obj:
                bpy.data.objects.remove(obj, do_unlink=True)

    print(f"[Quick Infill] {op_label}: {len(mesh_names)} objects → {final_verts} vertices")
    return result_obj, f"{op_label} completed on {len(mesh_names)} objects. Result: '{result_obj.name}'"


class QUICKINFILL_OT_voxel_intersect(Operator):
    """Intersect all selected mesh objects using voxel boolean"""
    bl_idname = "quick_infill.voxel_intersect"
    bl_label = "Voxel Intersect"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        selected_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']
        return len(selected_meshes) >= 2

    def execute(self, context):
        try:
            result_obj, message = _voxel_boolean_combine(context, intersect_meshes, "Voxel Intersect", "_Intersect")
            if result_obj is None:
                self.report({'ERROR'}, message)
                return {'CANCELLED'}
            self.report({'INFO'}, message)
            bpy.context.view_layer.update()
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Voxel Intersect failed: {e}")
            print(f"[Quick Infill] Voxel Intersect error: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


class QUICKINFILL_OT_voxel_union(Operator):
    """Union all selected mesh objects using voxel boolean"""
    bl_idname = "quick_infill.voxel_union"
    bl_label = "Voxel Union"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        selected_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']
        return len(selected_meshes) >= 2

    def execute(self, context):
        try:
            result_obj, message = _voxel_boolean_combine(context, union_meshes, "Voxel Union", "_Union")
            if result_obj is None:
                self.report({'ERROR'}, message)
                return {'CANCELLED'}
            self.report({'INFO'}, message)
            bpy.context.view_layer.update()
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Voxel Union failed: {e}")
            print(f"[Quick Infill] Voxel Union error: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


class QUICKINFILL_OT_voxel_diff(Operator):
    """Subtract every other selected mesh object from the active object, one at a time, using voxel boolean"""
    bl_idname = "quick_infill.voxel_diff"
    bl_label = "Voxel Diff"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        selected_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']
        return len(selected_meshes) >= 2

    def execute(self, context):
        try:
            result_obj, message = _voxel_boolean_combine(context, diff_meshes, "Voxel Diff", "_Diff")
            if result_obj is None:
                self.report({'ERROR'}, message)
                return {'CANCELLED'}
            self.report({'INFO'}, message)
            bpy.context.view_layer.update()
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Voxel Diff failed: {e}")
            print(f"[Quick Infill] Voxel Diff error: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


class QUICKINFILL_OT_shrink_from_view(Operator):
    """Shrink top-facing faces based on the current viewport direction"""
    bl_idname = "quick_infill.shrink_from_view"
    bl_label = "Shrink from View"
    bl_description = "Shrink top-facing faces on selected mesh(es) based on viewport direction. With multiple selections, processes each object individually"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            mm, _ = get_meshlib()
            settings = context.scene.quick_infill_support_settings
            shrink_amount = float(settings.shrink_amount)
            shrink_angle = float(settings.shrink_angle_threshold)
            replace_original = settings.replace_original

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
            view_dir = region_3d.view_rotation @ mathutils.Vector((0, 0, 1))
            view_dir.normalize()
            up_vector = mm.Vector3f(view_dir.x, view_dir.y, view_dir.z)

            def shrink_from_view_op(mesh, res):
                return shrink_top_faces_along_normals(mesh, up_vector, shrink_amount, shrink_angle)

            if len(selected_objs) == 1:
                result_obj, initial_verts, final_verts = process_mesh_operation(
                    selected_objs[0], shrink_from_view_op, "_Shrunk",
                    auto_decimate=False, replace_original=replace_original, resolution=None
                )
                results = [(result_obj, initial_verts, final_verts)]
            else:
                results, _ = batch_process_mesh_operation(
                    selected_objs, shrink_from_view_op, "_Shrunk",
                    auto_decimate=False, replace_original=replace_original, resolution=None
                )

            # Report results
            obj_count = len(results)
            
            if obj_count == 1:
                result_obj, _, _ = results[0]
                self.report({'INFO'}, f"Shrunk top faces from view. Result: '{result_obj.name}'")
            else:
                if replace_original:
                    self.report({'INFO'}, f"Shrunk top faces on {obj_count} objects from view")
                else:
                    self.report({'INFO'}, f"Shrunk top faces, created {obj_count} new objects from view")
            
            # Restore selection to result objects
            select_results([r[0] for r in results])
            bpy.context.view_layer.update()
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Shrink from View failed: {e}")
            print(f"[Quick Infill] Shrink from View error: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


class QUICKINFILL_OT_support_angle_preset(Operator):
    bl_idname = "quick_infill.support_angle_preset"
    bl_label = "Angle Preset"
    bl_description = "Set the support undercut angle"

    angle: FloatProperty(name="Angle", default=60.0)

    def execute(self, context):
        settings = context.scene.quick_infill_support_settings
        settings.active_angle_preset = str(int(self.angle))
        settings.undercut_angle = self.angle
        return {'FINISHED'}


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
        
        tools_col.separator()
        
        # Directions 3x3 grid layout (top-down view)
        tools_col.label(text="Directions:")
        
        # Row 1: -X+Y, +Y, +X+Y
        row1 = tools_col.row(align=True)
        row1.prop(settings, "dir_xneg_ypos", text="-X+Y", toggle=True)
        row1.prop(settings, "dir_y_pos", text="+Y", toggle=True)
        row1.prop(settings, "dir_xpos_ypos", text="+X+Y", toggle=True)
        
        # Row 2: -X, Z (center), +X
        row2 = tools_col.row(align=True)
        row2.prop(settings, "dir_x_neg", text="-X", toggle=True)
        row2.prop(settings, "dir_z", text="Z", toggle=True)
        row2.prop(settings, "dir_x_pos", text="+X", toggle=True)
        
        # Row 3: -X-Y, -Y, +X-Y
        row3 = tools_col.row(align=True)
        row3.prop(settings, "dir_xneg_yneg", text="-X-Y", toggle=True)
        row3.prop(settings, "dir_y_neg", text="-Y", toggle=True)
        row3.prop(settings, "dir_xpos_yneg", text="+X-Y", toggle=True)
        
        tools_col.separator()
        
        # Angle field and preset row
        angle_row = tools_col.row(align=True)
        angle_row.prop(settings, "undercut_angle", text="Angle")
        tools_col.separator(factor=0.2)

        preset_row = tools_col.row(align=True)
        for angle_value in (10.0, 15.0, 30.0, 45.0, 60.0, 90.0):
            active = settings.active_angle_preset == str(int(angle_value))
            op = preset_row.operator("quick_infill.support_angle_preset", text=str(int(angle_value)), icon='NONE', depress=active)
            op.angle = angle_value

        tools_col.separator()
        
        # Fix Undercuts and From View buttons on same line
        row = tools_col.row(align=True)
        row.operator("quick_infill.fix_undercuts", text="Fix Undercuts", icon='MOD_SMOOTH')
        row.operator("quick_infill.fix_undercuts_from_view", text="From View", icon='HIDE_OFF')
        
        tools_col.separator()
        
        # Auto Shrink checkbox
        tools_col.prop(settings, "auto_shrink", text="Auto Shrink")
        
        # Shrink sliders (always visible)
        prop_with_suffix(tools_col, settings, "shrink_amount", "Shrink Amount", "mm")
        prop_with_suffix(tools_col, settings, "shrink_angle_threshold", "Shrink Angle", "°")
        
        tools_col.separator()
        
        # Shrink from View button
        tools_col.operator("quick_infill.shrink_from_view", text="Shrink from View", icon='FULLSCREEN_EXIT')
        


classes = (
    QuickInfillSupportSettings,
    QUICKINFILL_OT_support_angle_preset,
    QUICKINFILL_OT_fix_undercuts,
    QUICKINFILL_OT_fix_undercuts_from_view,
    QUICKINFILL_OT_voxel_intersect,
    QUICKINFILL_OT_voxel_union,
    QUICKINFILL_OT_voxel_diff,
    QUICKINFILL_OT_shrink_from_view,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.quick_infill_support_settings = bpy.props.PointerProperty(
        type=QuickInfillSupportSettings
    )


def unregister():
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
    
    if hasattr(bpy.types.Scene, 'quick_infill_support_settings'):
        del bpy.types.Scene.quick_infill_support_settings
