"""
Support tools for Quick Infill: Fix Undercuts (world or view directions),
Shrink from View, and the voxel booleans (Intersect / Union / Diff).
"""

import math

import bpy
import mathutils
import numpy as np
from bpy.types import Operator
from bpy.props import FloatProperty

from .meshlib_utils import get_meshlib, get_mrmeshnumpy
from .offset_utils import ANGLE_PRESETS, should_auto_decimate_faces, decimate_mesh, face_count
from .blender_meshlib_utils import (
    process_objects, blender_to_meshlib, meshlib_to_blender_mesh, select_results, swap_mesh_keep_transforms,
    link_new_object, free_unused_meshes, remove_objects,
)
from .tools_operators import selected_meshes, report_process
from .settings import get_active_angle_preset

DIAG = 0.7071067811865476  # sqrt(2)/2
Z_UP = (0, 0, 1)


# ---------------------------------------------------------------------------
# Directions
# ---------------------------------------------------------------------------

def get_selected_directions(s) -> list:
    """Enabled direction tuples. (0, 0, 1) is the special straight-on 'Z' entry."""
    table = (
        (s.dir_z, Z_UP),
        (s.dir_x_pos, (1, 0, 0)), (s.dir_x_neg, (-1, 0, 0)),
        (s.dir_y_pos, (0, 1, 0)), (s.dir_y_neg, (0, -1, 0)),
        (s.dir_xpos_ypos, (DIAG, DIAG, 0)), (s.dir_xpos_yneg, (DIAG, -DIAG, 0)),
        (s.dir_xneg_ypos, (-DIAG, DIAG, 0)), (s.dir_xneg_yneg, (-DIAG, -DIAG, 0)),
    )
    return [d for on, d in table if on]


def world_up_vector(direction, angle_degrees):
    """Tilt from world Z toward a horizontal direction. Z_UP = pure Z."""
    if direction == Z_UP:
        return (0.0, 0.0, 1.0)
    a = math.radians(angle_degrees)
    return (direction[0] * math.sin(a), direction[1] * math.sin(a), math.cos(a))


def view_up_vector_fn(view_rotation):
    """Up-vector function for screen-space directions (x = right, y = up, Z_UP = toward the viewer)."""
    forward = (view_rotation @ mathutils.Vector((0, 0, 1))).normalized()
    right = view_rotation @ mathutils.Vector((1, 0, 0))
    up = view_rotation @ mathutils.Vector((0, 1, 0))

    def fn(direction, angle_degrees):
        if direction == Z_UP:
            return tuple(forward)
        tilt = (right * direction[0] + up * direction[1]).normalized()
        a = math.radians(angle_degrees)
        return tuple((forward * math.cos(a) + tilt * math.sin(a)).normalized())
    return fn


def _vec3(t):
    mm, _ = get_meshlib()
    return mm.Vector3f(float(t[0]), float(t[1]), float(t[2]))


# Kept for callers of the old helpers
def compute_up_vector(horizontal_axis, angle_degrees):
    return _vec3(world_up_vector(tuple(horizontal_axis), angle_degrees))


def compute_view_space_up_vector(screen_direction, view_rotation, angle_degrees):
    return _vec3(view_up_vector_fn(view_rotation)(tuple(screen_direction), angle_degrees))


# ---------------------------------------------------------------------------
# meshlib operations
# ---------------------------------------------------------------------------

def shrink_top_faces_along_normals(mesh, up_vector, shrink_amount, angle_threshold=30.0):
    """Move vertices of up-facing faces inward along their normals, but only where
    the vertex normal is within angle_threshold of up_vector. Vectorised."""
    mm, _ = get_meshlib()
    mn = get_mrmeshnumpy()
    up = np.array([up_vector.x, up_vector.y, up_vector.z], dtype=np.float64)

    params = mm.FixUndercuts.FixParams()
    params.findParameters.upDirection = mm.Vector3f(-up[0], -up[1], -up[2])
    top_faces = mm.FaceBitSet()
    mm.FixUndercuts.find(mesh, params.findParameters, top_faces)
    if top_faces.count() == 0:
        return mesh

    verts = mn.getNumpyVerts(mesh)
    normals = mn.toNumpyArray(mm.computePerVertNormals(mesh))
    top = mn.getNumpyBitSet(mm.getIncidentVerts(mesh.topology, top_faces))
    top = np.pad(top, (0, max(0, len(verts) - top.size)))[:len(verts)]
    move = top & (normals @ up >= math.cos(math.radians(angle_threshold)))
    if not move.any():
        return mesh
    verts = verts.copy()
    verts[move] -= normals[move] * float(shrink_amount)
    faces = mn.getNumpyFaces(mesh.topology)
    return mn.meshFromFacesVerts(faces.astype(np.int32), verts.astype(np.float32), duplicateNonManifoldVertices=False)


def fix_undercuts_for_direction(mesh, up_vector, voxel_size, shrink_amount=0.0, shrink_angle=30.0):
    """Fix undercuts for one direction (in place). Returns (mesh, undercut_count)."""
    mm, _ = get_meshlib()
    params = mm.FixUndercuts.FixParams()
    params.findParameters.upDirection = up_vector
    params.voxelSize = voxel_size
    undercuts = mm.FaceBitSet()
    mm.FixUndercuts.find(mesh, params.findParameters, undercuts)
    count = undercuts.count()
    if count > 0:
        mm.FixUndercuts.fix(mesh, params)
    if shrink_amount > 0:
        mesh = shrink_top_faces_along_normals(mesh, up_vector, shrink_amount, shrink_angle)
    return mesh, count


def intersect_meshes(mesh_a, mesh_b, voxel_size):
    mm, _ = get_meshlib()
    return mm.voxelBooleanIntersect(mesh_a, mesh_b, voxel_size)


def union_meshes(mesh_a, mesh_b, voxel_size):
    mm, _ = get_meshlib()
    return mm.voxelBooleanUnite(mesh_a, mesh_b, voxel_size)


def diff_meshes(mesh_a, mesh_b, voxel_size):
    mm, _ = get_meshlib()
    return mm.voxelBooleanSubtract(mesh_a, mesh_b, voxel_size)


def fix_undercuts_multi(mesh, directions, angle, voxel_size, shrink_amount, shrink_angle, up_fn=world_up_vector):
    """Fix undercuts from every direction and combine the results.

    Positive angle: per-direction results are voxel-intersected. Negative angle:
    they are voxel-unioned and the straight-on Z direction is applied to the
    merged result afterwards. Returns (mesh, total_undercuts).
    """
    mm, _ = get_meshlib()
    abs_angle = abs(float(angle))
    union_mode = float(angle) < 0.0

    if abs_angle == 0 or directions == [Z_UP]:
        return fix_undercuts_for_direction(mesh, _vec3(up_fn(Z_UP, 0.0)), voxel_size, shrink_amount, shrink_angle)

    apply_z_after = union_mode and Z_UP in directions
    process_dirs = [d for d in directions if d != Z_UP] if apply_z_after else list(directions)

    # Directions run one after another and are combined in order. Running them in
    # parallel was measured to give no speed-up (meshlib already uses every core
    # inside each fix), and a tree-shaped combine changed results by up to 0.5%.
    total = 0
    combine = union_meshes if union_mode else intersect_meshes
    result = None
    for direction in process_dirs:
        m, c = fix_undercuts_for_direction(mm.copyMesh(mesh), _vec3(up_fn(direction, abs_angle)), voxel_size,
                                           shrink_amount, shrink_angle)
        total += c
        result = m if result is None else combine(result, m, voxel_size)

    if apply_z_after:
        result, c = fix_undercuts_for_direction(result, _vec3(up_fn(Z_UP, 0.0)), voxel_size, shrink_amount, shrink_angle)
        total += c
    return result, total


# Kept for callers of the old names
def fix_undercuts_single_mesh(mesh, directions, angle, voxel_size, shrink_amount, shrink_angle):
    return fix_undercuts_multi(mesh, directions, angle, voxel_size, shrink_amount, shrink_angle)


def fix_undercuts_from_view_single_mesh(mesh, directions, angle, voxel_size, shrink_amount, shrink_angle, view_rotation):
    return fix_undercuts_multi(mesh, directions, angle, voxel_size, shrink_amount, shrink_angle, view_up_vector_fn(view_rotation))


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

def _view_rotation(context):
    rv3d = getattr(context, "region_data", None)
    if rv3d is not None and hasattr(rv3d, "view_rotation"):
        return rv3d.view_rotation.copy()
    for area in (context.screen.areas if context.screen else []):
        if area.type == 'VIEW_3D':
            return area.spaces.active.region_3d.view_rotation.copy()
    return None


def _run_fix_undercuts(op, context, view_rotation=None):
    s = context.scene.quick_infill_settings
    objs = selected_meshes(context)
    if not objs:
        op.report({'ERROR'}, "No mesh selected.")
        return {'CANCELLED'}
    directions = get_selected_directions(s) or [Z_UP]
    shrink_amount = float(s.shrink_amount) if s.auto_shrink else 0.0
    shrink_angle = float(s.shrink_angle_threshold)
    angle = float(s.undercut_angle)
    up_fn = view_up_vector_fn(view_rotation) if view_rotation is not None else world_up_vector
    counts = []

    def operation(mesh, res):
        m, c = fix_undercuts_multi(mesh, directions, angle, res, shrink_amount, shrink_angle, up_fn)
        counts.append(c)
        return m

    pr = process_objects(objs, operation, "_NoUndercuts", resolution=float(s.shared_voxel_size),
                         auto_decimate=s.decimate_mode != "OFF", replace_original=s.replace_original, settings=s)
    where = "view direction(s)" if view_rotation is not None else "direction(s)"
    if pr.results:
        if sum(counts) == 0:
            op.report({'INFO'}, f"No undercuts found on {len(pr.results)} mesh(es).")
        else:
            op.report({'INFO'}, f"Fixed undercuts on {len(pr.results)} object(s) from {len(directions)} {where}")
    if pr.collapsed:
        op.report({'WARNING'}, f"Fix Undercuts: {len(pr.collapsed)} object(s) produced no geometry and were left unchanged")
    if pr.errors:
        op.report({'ERROR'}, f"Fix Undercuts failed on {len(pr.errors)} object(s): {pr.error_summary()}")
    select_results([r[0] for r in pr.results])
    return {'FINISHED'} if pr.results else {'CANCELLED'}


class QUICKINFILL_OT_fix_undercuts(Operator):
    bl_idname = "quick_infill.fix_undercuts"
    bl_label = "Fix Undercuts"
    bl_description = "Fix undercuts on selected mesh(es) for better printability. With multiple selections, processes each object individually"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            return _run_fix_undercuts(self, context)
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
        rot = _view_rotation(context)
        if rot is None:
            self.report({'ERROR'}, "No 3D viewport found.")
            return {'CANCELLED'}
        try:
            return _run_fix_undercuts(self, context, rot)
        except Exception as e:
            self.report({'ERROR'}, f"Fix Undercuts (View) failed: {e}")
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
        s = context.scene.quick_infill_settings
        objs = selected_meshes(context)
        if not objs:
            self.report({'ERROR'}, "No mesh selected.")
            return {'CANCELLED'}
        rot = _view_rotation(context)
        if rot is None:
            self.report({'ERROR'}, "No 3D viewport found.")
            return {'CANCELLED'}
        try:
            up = _vec3((rot @ mathutils.Vector((0, 0, 1))).normalized())
            amount, angle = float(s.shrink_amount), float(s.shrink_angle_threshold)
            pr = process_objects(objs, lambda mesh, res: shrink_top_faces_along_normals(mesh, up, amount, angle),
                                 "_Shrunk", resolution=None, auto_decimate=False,
                                 replace_original=s.replace_original, settings=s, smart=False)
        except Exception as e:
            self.report({'ERROR'}, f"Shrink from View failed: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
        select_results([r[0] for r in pr.results])
        report_process(self, pr, "Updated" if s.replace_original else "Created", "Shrink from View")
        return {'FINISHED'} if pr.results else {'CANCELLED'}


def _voxel_boolean_combine(op, context, combine_fn, op_label, suffix):
    """Combine every other selected mesh into the active one, one at a time."""
    s = context.scene.quick_infill_settings
    voxel_size = float(s.shared_voxel_size)
    meshes = selected_meshes(context)
    if len(meshes) < 2:
        op.report({'ERROR'}, f"Select at least 2 mesh objects to {op_label.lower()}.")
        return {'CANCELLED'}
    active = context.active_object if context.active_object in meshes else meshes[0]
    others = [o for o in meshes if o != active]

    result = blender_to_meshlib(active, s.apply_modifiers_on_export)
    initial_faces = face_count(result)
    for obj in others:
        other = blender_to_meshlib(obj, s.apply_modifiers_on_export)
        initial_faces += face_count(other)
        result = combine_fn(result, other, voxel_size)

    if face_count(result) == 0:
        op.report({'ERROR'}, f"{op_label} resulted in an empty mesh. Objects may not overlap.")
        return {'CANCELLED'}
    if s.decimate_mode != "OFF":
        do, target = should_auto_decimate_faces(initial_faces, face_count(result), voxel_size=voxel_size,
                                                mode=s.decimate_mode, ratio=s.decimation_ratio)
        if do:
            result = decimate_mesh(result, target_face_count=target, resolution=voxel_size)

    new_mesh = meshlib_to_blender_mesh(result, active.name + suffix)
    if s.replace_original:
        old = swap_mesh_keep_transforms(active, new_mesh)
        remove_objects(others)
        free_unused_meshes([old])
        result_obj = active
    else:
        result_obj = link_new_object(active.name + suffix, new_mesh, list(active.users_collection) or [context.scene.collection])
    select_results([result_obj])
    op.report({'INFO'}, f"{op_label} completed on {len(meshes)} objects. Result: '{result_obj.name}'")
    return {'FINISHED'}


class _VoxelBoolBase:
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return sum(1 for o in context.selected_objects if o.type == 'MESH') >= 2

    def execute(self, context):
        try:
            return _voxel_boolean_combine(self, context, self.combine, self.bl_label, self.suffix)
        except Exception as e:
            self.report({'ERROR'}, f"{self.bl_label} failed: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


class QUICKINFILL_OT_voxel_intersect(_VoxelBoolBase, Operator):
    """Intersect all selected mesh objects using voxel boolean"""
    bl_idname = "quick_infill.voxel_intersect"
    bl_label = "Voxel Intersect"
    suffix = "_Intersect"
    combine = staticmethod(intersect_meshes)


class QUICKINFILL_OT_voxel_union(_VoxelBoolBase, Operator):
    """Union all selected mesh objects using voxel boolean"""
    bl_idname = "quick_infill.voxel_union"
    bl_label = "Voxel Union"
    suffix = "_Union"
    combine = staticmethod(union_meshes)


class QUICKINFILL_OT_voxel_diff(_VoxelBoolBase, Operator):
    """Subtract every other selected mesh object from the active object, one at a time, using voxel boolean"""
    bl_idname = "quick_infill.voxel_diff"
    bl_label = "Voxel Diff"
    suffix = "_Diff"
    combine = staticmethod(diff_meshes)


class QUICKINFILL_OT_support_angle_preset(Operator):
    bl_idname = "quick_infill.support_angle_preset"
    bl_label = "Angle Preset"
    bl_description = "Set the support undercut angle"

    angle: FloatProperty(name="Angle", default=60.0)

    def execute(self, context):
        context.scene.quick_infill_settings.undercut_angle = self.angle
        return {'FINISHED'}


def draw_support_tools(layout, context):
    s = context.scene.quick_infill_settings
    box = layout.column(align=True).box()
    icon = 'DOWNARROW_HLT' if s.show_support_tools else 'RIGHTARROW'
    box.row().prop(s, "show_support_tools", text="Support Tools", icon=icon, emboss=False)
    if not s.show_support_tools:
        return

    col = box.column(align=True)

    def prop_with_suffix(attr, label, suffix):
        split = col.split(factor=0.9, align=True)
        c = split.column(align=True)
        c.use_property_split = False
        c.use_property_decorate = False
        c.prop(s, attr, text=label)
        split.label(text=suffix)

    col.separator()
    col.label(text="Directions:")
    for row_props in (("dir_xneg_ypos", "dir_y_pos", "dir_xpos_ypos"),
                      ("dir_x_neg", "dir_z", "dir_x_pos"),
                      ("dir_xneg_yneg", "dir_y_neg", "dir_xpos_yneg")):
        row = col.row(align=True)
        for p in row_props:
            row.prop(s, p, text=s.bl_rna.properties[p].name, toggle=True)
    col.separator()
    col.row(align=True).prop(s, "undercut_angle", text="Angle")
    col.separator(factor=0.2)
    active = get_active_angle_preset(s)
    row = col.row(align=True)
    for a in ANGLE_PRESETS:
        op = row.operator("quick_infill.support_angle_preset", text=str(int(a)), depress=(active == str(int(a))))
        op.angle = a
    col.separator()
    row = col.row(align=True)
    row.operator("quick_infill.fix_undercuts", text="Fix Undercuts", icon='MOD_SMOOTH')
    row.operator("quick_infill.fix_undercuts_from_view", text="From View", icon='HIDE_OFF')
    col.separator()
    col.prop(s, "auto_shrink", text="Auto Shrink")
    prop_with_suffix("shrink_amount", "Shrink Amount", "mm")
    prop_with_suffix("shrink_angle_threshold", "Shrink Angle", "°")
    col.separator()
    col.operator("quick_infill.shrink_from_view", text="Shrink from View", icon='FULLSCREEN_EXIT')


classes = (
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


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
