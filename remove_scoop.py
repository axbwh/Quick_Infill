"""
Remove Scoop: cut the printing support scoop (the 45-60 degree wedge that Fix
Undercuts builds under a stone) off the bottom of each island, flat, so the
stone gets a flat contact face and can be used either way up.

The cut height uses the same contact-area rule as the SystemHEX stone packer's
flat cuts (stone_pack/core.py, _end_profiles / cut_depth): slice the mesh with
a flat plane at depth d up from the bottom, measure the outline area of the
section and compare it with the stone's full footprint (length x depth of its
smallest bounding rectangle, so it works however the stone is turned). The cut
goes at the lowest d where the section reaches the Contact share (default 90%).

Only faces crossing the cut plane change: every face above it keeps its exact
vertices (no remesh), so the stone's front, sides and damage stay intact. The
new cap faces get the face attribute "sp_cut" (the same name the stone packer
writes on its cut faces), which drives Stone Damage on the cut face only.

Optional follow-ups, in Alex's order: Stone Damage on the cut face (needs the
Stone_Damage node group in the file), then Quick Infill's Trim Edges.
"""

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from bpy.types import Operator

from .tools_operators import selected_meshes
from .blender_meshlib_utils import select_results, link_new_object, STL_EXPORT_SCALE

CUT_ATTR = "sp_cut"
DAMAGE_GROUP = "Stone_Damage"
DAMAGE_CUT_GROUP = "Stone_Damage_Cut"
STEP = 0.01            # BU between trial cut heights
MAX_CUT = 0.5          # BU: never cut more than this off the bottom (stone_pack caps flat cuts at 0.35-0.5)
MAX_CUT_SHARE = 0.45   # ...nor more than this share of the stone's height


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _hull2d(p):
    """Convex hull (monotone chain) of (n, 2) points."""
    p = np.unique(np.round(p, 6), axis=0)
    if len(p) < 3:
        return p

    def half(pts):
        h = []
        for q in pts:
            while len(h) >= 2 and _cross(h[-1] - h[-2], q - h[-2]) <= 0:
                h.pop()
            h.append(q)
        return h[:-1]
    return np.array(half(p) + half(p[::-1]))


def _hull_area(p):
    h = _hull2d(p)
    if len(h) < 3:
        return 0.0
    x, y = h[:, 0], h[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _footprint_area(xy):
    """Area of the smallest bounding rectangle of the points (= length x depth of
    the stone in its own frame, like the packer's L * D)."""
    h = _hull2d(xy)
    best = None
    for i in range(len(h)):
        e = h[(i + 1) % len(h)] - h[i]
        n = float(np.linalg.norm(e))
        if n < 1e-9:
            continue
        u = e / n
        w = np.array([-u[1], u[0]])
        a = float(np.ptp(h @ u) * np.ptp(h @ w))
        if best is None or a < best:
            best = a
    return best or 0.0


def contact_profile(v, edges):
    """Contact share of a flat cut at each trial height up from the bottom.
    v: (n, 3) world vertices of one island; edges: (m, 2) vertex indices.
    Returns (heights, shares)."""
    z = v[:, 2]
    zmin, H = float(z.min()), float(np.ptp(z))
    full = _footprint_area(v[:, :2])
    if H <= 1e-6 or full <= 1e-9:
        return np.zeros(0), np.zeros(0)
    top = min(MAX_CUT, MAX_CUT_SHARE * H)
    heights = np.arange(STEP * 0.5, top + 1e-9, STEP)
    a, b = v[edges[:, 0]], v[edges[:, 1]]
    shares = []
    for d in heights:
        za, zb = a[:, 2] - (zmin + d), b[:, 2] - (zmin + d)
        m = (za * zb) < 0
        if m.sum() < 3:
            shares.append(0.0)
            continue
        t = (za[m] / (za[m] - zb[m]))[:, None]
        pts = a[m] + t * (b[m] - a[m])
        shares.append(_hull_area(pts[:, :2]) / full)
    return heights, np.array(shares)


def contact_cut_height(v, edges, target):
    """Height above the island's lowest point where a flat cut reaches `target`
    contact share; if it never does (within the cap), where it comes closest."""
    heights, shares = contact_profile(v, edges)
    if not len(heights):
        return 0.0
    hit = np.nonzero(shares >= target)[0]
    i = int(hit[0]) if len(hit) else int(np.argmax(shares))
    return float(heights[i])


def _islands(bm):
    """Vertex lists of each connected part."""
    bm.verts.index_update()
    seen = np.zeros(len(bm.verts), bool)
    out = []
    for start in bm.verts:
        if seen[start.index]:
            continue
        stack, part = [start], []
        seen[start.index] = True
        while stack:
            vv = stack.pop()
            part.append(vv)
            for e in vv.link_edges:
                o = e.other_vert(vv)
                if not seen[o.index]:
                    seen[o.index] = True
                    stack.append(o)
        out.append(part)
    return out


def remove_scoop_bmesh(bm, matrix_world, target=0.9, offset=0.0):
    """Cut the scoop off every island of bm (local coordinates) in place and tag
    the cap faces with the boolean face attribute CUT_ATTR.
    target: contact share 0.5-1.0. offset: world units added to the cut height.
    Returns the number of islands cut."""
    mw = np.array(matrix_world)
    inv = matrix_world.inverted()
    n_local = (inv.to_3x3().inverted().transposed() @ Vector((0.0, 0.0, 1.0))).normalized()
    layer = bm.faces.layers.bool.get(CUT_ATTR) or bm.faces.layers.bool.new(CUT_ATTR)
    cut = 0
    for part in _islands(bm):
        idx = {vv: i for i, vv in enumerate(part)}
        co = np.array([vv.co[:] for vv in part])
        wv = co @ mw[:3, :3].T + mw[:3, 3]
        edges = np.array([(idx[e.verts[0]], idx[e.verts[1]])
                          for e in {e for vv in part for e in vv.link_edges}], dtype=np.int64)
        if len(edges) < 3:
            continue
        h = contact_cut_height(wv, edges, target) + offset
        zmin, zmax = float(wv[:, 2].min()), float(wv[:, 2].max())
        if h <= 0.0 or zmin + h >= zmax:
            continue
        p = inv @ Vector((0.0, 0.0, zmin + h))
        geom = list(part) + list({e for vv in part for e in vv.link_edges}) + \
            list({f for vv in part for f in vv.link_faces})
        res = bmesh.ops.bisect_plane(bm, geom=geom, plane_co=p, plane_no=n_local, clear_inner=True, dist=1e-6)
        cut_edges = [e for e in res['geom_cut'] if isinstance(e, bmesh.types.BMEdge) and e.is_valid and e.is_boundary]
        if cut_edges:
            filled = bmesh.ops.holes_fill(bm, edges=cut_edges, sides=0)
            faces = filled['faces']
            if faces:
                tri = bmesh.ops.triangulate(bm, faces=faces, quad_method='BEAUTY', ngon_method='BEAUTY')
                for f in tri['faces']:
                    f[layer] = True
        cut += 1
    bm.normal_update()
    return cut


# ---------------------------------------------------------------------------
# Stone Damage on the cut face
# ---------------------------------------------------------------------------

def damage_cut_group():
    """A copy of the file's Stone_Damage node group whose dents go only on faces
    tagged sp_cut (the original picks every face not facing down). None if the
    file has no Stone_Damage group."""
    ng = bpy.data.node_groups.get(DAMAGE_CUT_GROUP)
    if ng is not None:
        return ng
    src = bpy.data.node_groups.get(DAMAGE_GROUP)
    if src is None:
        return None
    ng = src.copy()
    ng.name = DAMAGE_CUT_GROUP
    attr = ng.nodes.new('GeometryNodeInputNamedAttribute')
    attr.data_type = 'BOOLEAN'
    attr.inputs['Name'].default_value = CUT_ATTR
    hooked = 0
    for n in ng.nodes:
        if n.bl_idname == 'GeometryNodeGroup' and n.node_tree and 'Scatter' in n.node_tree.name and 'Selection' in n.inputs:
            ng.links.new(attr.outputs['Attribute'], n.inputs['Selection'])
            hooked += 1
    if not hooked:
        bpy.data.node_groups.remove(ng)
        return None
    return ng


def apply_damage_cut(obj, depsgraph):
    """Run Stone_Damage_Cut on obj and bake the result into its mesh."""
    ng = damage_cut_group()
    if ng is None:
        return False
    mod = obj.modifiers.new("SP_DamageCut", 'NODES')
    mod.node_group = ng
    # keep it at the top of the stack so it runs on the cut mesh only
    while obj.modifiers.find(mod.name) > 0:
        obj.modifiers.move(obj.modifiers.find(mod.name), obj.modifiers.find(mod.name) - 1)
    others = [(m, m.show_viewport) for m in obj.modifiers if m != mod]
    for m, _ in others:
        m.show_viewport = False
    depsgraph.update()
    new = bpy.data.meshes.new_from_object(obj.evaluated_get(depsgraph), preserve_all_data_layers=True, depsgraph=depsgraph)
    obj.modifiers.remove(mod)
    for m, vis in others:
        m.show_viewport = vis
    old = obj.data
    for mat in old.materials:
        new.materials.append(mat)
    new.name = old.name
    obj.data = new
    if old.users == 0:
        bpy.data.meshes.remove(old)
    return True


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

def remove_scoop_objects(context, objs, s):
    """Cut + optional damage on objs. Returns (result objects, islands cut, damaged?)."""
    offset = float(s.scoop_offset) / STL_EXPORT_SCALE      # mm -> Blender units
    target = float(s.scoop_contact) / 100.0
    results, islands = [], 0
    for obj in objs:
        src = obj.data
        if s.replace_original:
            if src.users > 1:
                obj.data = src.copy()
            target_obj, me = obj, obj.data
        else:
            me = src.copy()
            me.name = obj.name + "_NoScoop"
            target_obj = link_new_object(obj.name + "_NoScoop", me, list(obj.users_collection) or [context.scene.collection])
            target_obj.matrix_world = obj.matrix_world.copy()
        bm = bmesh.new()
        bm.from_mesh(me)
        islands += remove_scoop_bmesh(bm, obj.matrix_world, target, offset)
        bm.to_mesh(me)
        bm.free()
        me.update()
        results.append(target_obj)
    damaged = False
    if s.scoop_damage:
        dg = context.evaluated_depsgraph_get()
        damaged = all(apply_damage_cut(o, dg) for o in results) if results else False
    return results, islands, damaged


class QUICKINFILL_OT_remove_scoop(Operator):
    bl_idname = "quick_infill.remove_scoop"
    bl_label = "Remove Scoop"
    bl_description = ("Cut the support scoop off the bottom of each stone (each island of the selected meshes) "
                      "flat, where the cut reaches the Contact share of the stone's footprint, keeping every "
                      "face above it untouched. Optionally adds Stone Damage on the cut face and runs Trim Edges")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.quick_infill_settings
        objs = selected_meshes(context)
        if not objs:
            self.report({'ERROR'}, "No mesh selected.")
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        results, islands, damaged = remove_scoop_objects(context, objs, s)
        select_results(results)
        if islands == 0:
            self.report({'WARNING'}, "No scoop found on the selected mesh(es).")
            return {'FINISHED'}
        msg = f"Removed {islands} scoop(s) from {len(results)} object(s)"
        if s.scoop_damage and not damaged:
            self.report({'WARNING'}, "Stone Damage skipped: this file has no 'Stone_Damage' node group")
        if s.scoop_trim:
            r = bpy.ops.quick_infill.trim_edges('EXEC_DEFAULT')
            msg += ", Trim Edges " + ("done" if 'FINISHED' in r else "failed")
        self.report({'INFO'}, msg)
        return {'FINISHED'}


def draw_remove_scoop(col, s):
    col.separator()
    col.label(text="Remove Scoop:")
    col.prop(s, "scoop_contact", text="Contact", slider=True)
    col.prop(s, "scoop_offset", text="Offset (mm)")
    row = col.row(align=True)
    row.prop(s, "scoop_damage", text="Damage Cut", toggle=True)
    row.prop(s, "scoop_trim", text="Trim Edges", toggle=True)
    col.operator("quick_infill.remove_scoop", text="Remove Scoop", icon='MOD_BOOLEAN')


classes = (QUICKINFILL_OT_remove_scoop,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
