"""Flat polygons: 2D grow / shrink and Create Outline.

Uses the same outline method as Ashlarize's stone outlines (copied here, Quick
Infill does not import Ashlarize):

  * outline  = meshlib PlanarTriangulation.getOutline of the union of the
               triangles, projected flat (NonZero winding),
  * offset   = meshlib offsetContours with round corners,
  * cleanup  = closing +CLOSE_R / -CLOSE_R (fills thin deep slits), then
               opening -OPEN_R / +OPEN_R (trims thin bumps), then
               Douglas-Peucker with a maximum deviation of DP_DEV.

Work is done in meshlib space (Blender units x STL_EXPORT_SCALE = mm on
SystemHEX), like the voxel tools, so the Distance setting means the same thing.
Ashlarize's locked values 0.08 / 0.04 / 0.02 BU are 0.8 / 0.4 / 0.2 here (the defaults).
In the Quick Infill tools the clean-up follows the Offset Tools settings instead
(Alex 2026-11-16): closing = Distance, opening = Distance / 2, Douglas-Peucker =
Voxel Size. See clean_from_settings().

Grow / Shrink rule (Alex 2026-11-16): an object is grown or shrunk as a flat
polygon only when ALL its pieces are flat polygons. If it also holds real
volume, the flat pieces are deleted and the rest is grown or shrunk as usual.
"""

import math

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

from .meshlib_utils import get_meshlib, get_mrmeshnumpy

SCALE = 10.0           # = blender_meshlib_utils.STL_EXPORT_SCALE (BU -> meshlib units)
CLOSE_R = 0.8          # Ashlarize CLOSE_R 0.08 BU
OPEN_R = 0.4           # Ashlarize OPEN_R 0.04 BU
DP_DEV = 0.2           # Ashlarize DP_DEV 0.02 BU
FLAT_TOL = 1e-3        # a piece is flat when no vertex is farther than this from its plane (meshlib units) ...
FLAT_REL = 1e-5        # ... or this fraction of its size, whichever is larger
MIN_AREA = 1e-4        # rings smaller than this (meshlib units^2) are dropped


# --------------------------------------------------------------------------------------------
# 2D rings: lists of (k, 2) arrays in meshlib's orientation (outer rings clockwise, holes
# counter-clockwise; getOutline gives them that way and offsetContours wants them that way)
# --------------------------------------------------------------------------------------------

def _mm():
    return get_meshlib()[0]


def area(P):
    P = np.asarray(P, float)
    if len(P) < 3:
        return 0.0
    return 0.5 * float(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]))


def region_area(rings):
    """Area covered by the region (outer rings clockwise count positive)."""
    return -sum(area(r) for r in rings)


def dedupe(P, eps=1e-4):
    """Drop repeated points (consecutive, and the last one when it equals the first).
    getOutline can return a ring with a repeated point; offsetContours then has a
    zero-length edge, does not offset there, and a closing cut up to its full radius
    into the shape (Alex's corner stones lost their bottom, 2026-11-16)."""
    P = np.asarray(P, float)
    if len(P) < 2:
        return P
    step = np.linalg.norm(P - np.roll(P, 1, 0), axis=1)
    P = P[step > eps] if (step > eps).any() else P[:1]
    return P


def _contours(rings):
    mm = _mm()
    cs = mm.std_vector_std_vector_Vector2_float()
    for P in rings:
        P = dedupe(P)
        if len(P) < 3:
            continue
        c = mm.std_vector_Vector2_float()
        for x, y in P:
            c.append(mm.Vector2f(float(x), float(y)))
        c.append(mm.Vector2f(float(P[0][0]), float(P[0][1])))      # closed
        cs.append(c)
    return cs


def _rings(res):
    out = []
    for r in res:
        x = np.array([[p.x, p.y] for p in r], float)
        if len(x) > 1 and np.allclose(x[0], x[-1]):
            x = x[:-1]
        x = dedupe(x)
        if len(x) >= 3 and abs(area(x)) > MIN_AREA:
            out.append(x)
    return out


def outline(rings):
    """Union outline (NonZero winding) of any closed rings."""
    mm = _mm()
    if not rings:
        return []
    prm = mm.PlanarTriangulation.OutlineParameters()
    bp = mm.PlanarTriangulation.BaseOutlineParameters()
    bp.allowMerge = True
    bp.innerType = mm.PlanarTriangulation.WindingMode.NonZero
    prm.baseParams = bp
    return _rings(mm.PlanarTriangulation.getOutline(_contours(rings), prm))


def _boundary_cycles(T, nv):
    """Closed vertex cycles of the boundary of a triangle set (all triangles the same turn).
    Interior edges cancel; any cycle split keeps every winding number, so a NonZero outline of
    the cycles is exactly the union of the triangles, from far fewer and cleaner contours than
    the triangles themselves (getOutline of thousands of overlapping triangles failed on
    axis-aligned views)."""
    if not len(T):
        return []
    a = T.reshape(-1)
    b = T[:, [1, 2, 0]].reshape(-1)
    lo, hi = np.minimum(a, b).astype(np.int64), np.maximum(a, b).astype(np.int64)
    key = lo * nv + hi
    uk, inv = np.unique(key, return_inverse=True)
    net = np.bincount(inv, weights=np.where(a < b, 1, -1), minlength=len(uk)).astype(np.int64)
    out = {}
    for k, m in zip(uk[net != 0], net[net != 0]):
        i, j = int(k // nv), int(k % nv)
        if m < 0:
            i, j = j, i
        for _ in range(abs(int(m))):
            out.setdefault(i, []).append(j)
    cycles = []
    while out:
        s0 = next(iter(out))
        cyc, cur = [s0], s0
        while True:
            nxt = out[cur].pop()
            if not out[cur]:
                del out[cur]
            if nxt == s0:
                break
            cyc.append(nxt)
            cur = nxt
            if cur not in out:          # cannot happen for a balanced edge set; stay safe
                cyc = None
                break
        if cyc and len(cyc) >= 3:
            cycles.append(cyc)
    return cycles


def silhouette(uv, T):
    """Outline of the union of triangles T (indices into the 2D points uv), holes kept."""
    uv = np.asarray(uv, float)
    T = np.asarray(T, np.int64).reshape(-1, 3)
    if not len(T):
        return []
    P = uv[T]
    ar = 0.5 * ((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
                - (P[:, 2, 0] - P[:, 0, 0]) * (P[:, 1, 1] - P[:, 0, 1]))
    pos, neg = T[ar > 1e-12], T[ar < -1e-12][:, ::-1]         # every triangle counter-clockwise
    # The triangles facing the viewer and the ones facing away are outlined apart, then
    # joined: one getOutline over both sets of cycles came back EMPTY on a stone seen side on
    # (their silhouette edges lie on top of each other).
    parts = []
    for part in (pos, neg):
        if len(part):
            o = outline([uv[c] for c in _boundary_cycles(part, len(uv))])
            if o:
                parts.append(o)
    if len(parts) < 2:
        return parts[0] if parts else []
    a, b = (region_area(x) for x in parts)
    for rings in (parts[0] + parts[1], [np.round(r / 1e-3) * 1e-3 for r in parts[0] + parts[1]]):
        u_ = outline(rings)
        if u_ and max(a, b) * (1 - 1e-6) - 1e-6 <= region_area(u_) <= (a + b) * (1 + 1e-6) + 1e-6:
            return u_
    return parts[0] if a >= b else parts[1]                   # a closed mesh: both are the same outline


def _offset_raw(rings, r):
    mm = _mm()
    prm = mm.OffsetContoursParams()
    prm.cornerType = mm.OffsetContoursParams.CornerType.Round
    prm.type = mm.OffsetContoursParams.Type.Offset
    return _rings(mm.offsetContours(_contours(rings), float(r), prm))


def offset(rings, r):
    """Grow (r > 0) or shrink (r < 0) the region, round corners. [] when it vanishes."""
    if not rings or abs(r) < 1e-9:
        return rings
    a = region_area(rings)
    out = outline(_offset_raw(rings, r))                 # outline: tidy overlaps
    if out and (region_area(out) - a) * r < 0:           # guard: meshlib's sign convention the other way round
        out = outline(_offset_raw(rings, -r))
    return out


def close(rings, r=CLOSE_R):
    """Closing (+r then -r): thin deep slits fill, broad notches stay."""
    if r <= 0 or not rings:
        return rings
    g = offset(rings, r)
    h = offset(g, -r) if g else []
    return h or rings


def open_(rings, r=OPEN_R):
    """Opening (-r then +r): thin bumps and spikes narrower than 2r go."""
    if r <= 0 or not rings:
        return rings
    g = offset(rings, -r)
    if not g:
        return rings                                     # the whole thing is thinner than 2r: keep it
    h = offset(g, r)
    return h or rings


def _dp(pts, tol):
    """Douglas-Peucker on an open polyline; returns kept indices."""
    keep = [0, len(pts) - 1]
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        pa, pb = pts[a], pts[b]
        d = pb - pa
        n = float(np.hypot(*d)) or 1e-9
        seg = pts[a + 1:b] - pa
        dist = np.abs(seg[:, 0] * d[1] - seg[:, 1] * d[0]) / n
        i = int(np.argmax(dist))
        if dist[i] > tol:
            k = a + 1 + i
            keep.append(k)
            stack += [(a, k), (k, b)]
    return sorted(set(keep))


def simplify_ring(P, dev=DP_DEV):
    """Douglas-Peucker on a closed ring, split at its two farthest points (orientation kept)."""
    P = np.asarray(P, float)
    if len(P) < 4:
        return P
    k = int(np.argmax(np.hypot(*(P - P[0]).T)))
    a_ = _dp(P[:k + 1], dev)
    closed = np.concatenate([P[k:], P[:1]])
    b_ = [k + i for i in _dp(closed, dev)][1:-1]
    return P[sorted(set(a_) | set(b_))]


def clean_from_settings(s):
    """(closing, opening, Douglas-Peucker) from the Offset Tools: Distance, Distance / 2, Voxel Size."""
    d = float(s.distance)
    return d, 0.5 * d, float(s.shared_voxel_size)


def cleanup(rings, clean=None):
    """Ashlarize's cleanup: closing, opening, Douglas-Peucker. clean = (closing, opening, dp)."""
    close_r, open_r, dev = clean or (CLOSE_R, OPEN_R, DP_DEV)
    rings = open_(close(rings, close_r), open_r)
    out = [simplify_ring(r, dev) if dev > 0 else r for r in rings]
    return [r for r in out if len(r) >= 3 and abs(area(r)) > MIN_AREA]


# --------------------------------------------------------------------------------------------
# Planes and Blender meshes
# --------------------------------------------------------------------------------------------

def plane_of(V, normals_hint=None):
    """(centre, unit normal, flatness) of points V. The normal follows normals_hint's side."""
    c = V.mean(0)
    _, s, vt = np.linalg.svd(V - c, full_matrices=False)
    n = vt[-1]
    if normals_hint is not None and float(n @ normals_hint) < 0:
        n = -n
    dev = float(np.abs((V - c) @ n).max()) if len(V) else 0.0
    return c, n / np.linalg.norm(n), dev


def is_flat(V):
    if len(V) < 3:
        return True
    size = float(np.linalg.norm(V.max(0) - V.min(0)))
    return plane_of(V)[2] <= max(FLAT_TOL, FLAT_REL * size)


def frame(n):
    """u, v in the plane with u x v = n (counter-clockwise in (u, v) faces along n)."""
    a = np.array((1.0, 0.0, 0.0)) if abs(n[0]) < 0.9 else np.array((0.0, 1.0, 0.0))
    u = np.cross(a, n)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def read_mesh(obj, apply_modifiers=False, depsgraph=None):
    """World-space (x SCALE) vertices (n, 3), triangles (m, 3) and edges (k, 2) of a mesh object."""
    eval_obj = None
    if apply_modifiers:
        depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(depsgraph)
        me = eval_obj.to_mesh()
    else:
        me = obj.data
    try:
        co = np.empty(len(me.vertices) * 3, np.float64)
        me.vertices.foreach_get("co", co)
        mw = np.array(obj.matrix_world, np.float64)
        V = (co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]) * SCALE
        me.calc_loop_triangles()
        T = np.empty(len(me.loop_triangles) * 3, np.int32)
        me.loop_triangles.foreach_get("vertices", T)
        T = T.reshape(-1, 3)
        if np.linalg.det(mw[:3, :3]) < 0:
            T = T[:, ::-1]
        E = np.empty(len(me.edges) * 2, np.int32)
        me.edges.foreach_get("vertices", E)
        E = E.reshape(-1, 2)
    finally:
        if eval_obj is not None:
            eval_obj.to_mesh_clear()
    return V, T, E


def _closed(obj, apply_modifiers):
    """True when every edge has exactly two faces (a closed volume: no flat pieces possible)."""
    if apply_modifiers:
        return False
    me = obj.data
    if not len(me.polygons):
        return False
    ei = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("edge_index", ei)
    return bool(np.all(np.bincount(ei, minlength=len(me.edges)) == 2))


def tri_components(T, nv):
    """Connected pieces of a triangle list: per triangle, its piece label (0..k-1)."""
    if not len(T):
        return np.zeros(0, np.int64), 0
    parent = np.arange(nv)

    def find_all(p):
        while True:
            q = p[p]
            if np.array_equal(q, p):
                return p
            p = q

    # label propagation on vertices via triangle edges (vectorised union by min label)
    a = np.concatenate([T[:, 0], T[:, 1], T[:, 2]])
    b = np.concatenate([T[:, 1], T[:, 2], T[:, 0]])
    lab = parent
    while True:
        m = np.minimum(lab[a], lab[b])
        new = lab.copy()
        np.minimum.at(new, a, m)
        np.minimum.at(new, b, m)
        new = find_all(new[new])
        if np.array_equal(new, lab):
            break
        lab = new
    tl = lab[T[:, 0]]
    _, inv = np.unique(tl, return_inverse=True)
    return inv, int(inv.max()) + 1


def classify(obj, apply_modifiers=False):
    """'flat', 'mixed' or 'volume' (see the module docstring)."""
    if _closed(obj, apply_modifiers):
        return 'volume'
    V, T, E = read_mesh(obj, apply_modifiers)
    if not len(T):
        return 'flat' if len(E) and is_flat(V[np.unique(E)]) else 'volume'
    lab, k = tri_components(T, len(V))
    flat = [is_flat(V[np.unique(T[lab == i])]) for i in range(k)]
    if all(flat):
        return 'flat'
    return 'mixed' if any(flat) else 'volume'


def strip_flat(ml_mesh):
    """meshlib mesh without its flat pieces (for 'mixed' objects before a voxel grow / shrink)."""
    mm = _mm()
    mn = get_mrmeshnumpy()
    if hasattr(ml_mesh, "pack"):
        ml_mesh.pack()
    V = mn.getNumpyVerts(ml_mesh).astype(np.float64)
    T = mn.getNumpyFaces(ml_mesh.topology).astype(np.int64)
    lab, k = tri_components(T, len(V))
    keep = np.zeros(len(T), bool)
    for i in range(k):
        sel = lab == i
        if not is_flat(V[np.unique(T[sel])]):
            keep |= sel
    if keep.all():
        return ml_mesh
    T = T[keep]
    used, T2 = np.unique(T, return_inverse=True)
    return mn.meshFromFacesVerts(T2.reshape(-1, 3).astype(np.int32), V[used].astype(np.float32))


def _loops_from_edges(E):
    """Closed vertex loops from loose edges (each vertex on exactly two edges)."""
    nb = {}
    for a, b in E:
        nb.setdefault(int(a), []).append(int(b))
        nb.setdefault(int(b), []).append(int(a))
    seen, loops = set(), []
    for s in nb:
        if s in seen or len(nb[s]) != 2:
            continue
        loop, prev, cur = [s], None, s
        seen.add(s)
        while True:
            nxt = [x for x in nb[cur] if x != prev and len(nb[x]) == 2]
            if not nxt:
                loop = None
                break
            prev, cur = cur, nxt[0]
            if cur == s:
                break
            if cur in seen:
                loop = None
                break
            loop.append(cur)
            seen.add(cur)
        if loop and len(loop) >= 3:
            loops.append(loop)
    return loops


def flat_pieces(V, T, E):
    """[(centre, normal, rings2d)] per plane: coplanar pieces are merged into one region."""
    groups = []                                           # [centre, normal, u, v, [2d ring lists]]

    def group_for(c, n, size):
        tol = max(FLAT_TOL, FLAT_REL * size) * 2
        for g in groups:
            if abs(float(g[1] @ n)) > 1 - 1e-6 and abs(float((c - g[0]) @ g[1])) < tol:
                return g
        u, v = frame(n)
        g = [c, n, u, v, []]
        groups.append(g)
        return g

    def flat2d(g, P):
        return np.stack([(P - g[0]) @ g[2], (P - g[0]) @ g[3]], -1)

    if len(T):
        lab, k = tri_components(T, len(V))
        for i in range(k):
            Ti = T[lab == i]
            Pv = V[np.unique(Ti)]
            fn = np.cross(V[Ti[:, 1]] - V[Ti[:, 0]], V[Ti[:, 2]] - V[Ti[:, 0]]).sum(0)
            c, n, _ = plane_of(Pv, fn)
            g = group_for(c, n, float(np.linalg.norm(Pv.max(0) - Pv.min(0))))
            g[4].append(silhouette(flat2d(g, V), Ti))
    else:
        for loop in _loops_from_edges(E):
            Pv = V[loop]
            c, n, _ = plane_of(Pv)
            g = group_for(c, n, float(np.linalg.norm(Pv.max(0) - Pv.min(0))))
            P2 = flat2d(g, Pv)
            g[4].append([P2 if area(P2) < 0 else P2[::-1]])      # outer ring: clockwise
    out = []
    for c, n, u, v, parts in groups:
        rings = [r for p in parts for r in p]
        out.append((c, n, u, v, outline(rings) if len(parts) > 1 else rings))
    return out


def build_mesh(name, pieces, to_local=None):
    """Blender mesh of flat regions. pieces = [(centre, normal, u, v, rings2d)] in meshlib units, world.
    Rings without holes become one n-gon each; with holes, the fewest faces bmesh can make."""
    bm = bmesh.new()
    for c, n, u, v, rings in pieces:
        nn = Vector(n)
        holes = any(area(R) > 0 for R in rings)               # holes run counter-clockwise
        made = []
        edges = []
        for R in rings:
            P = (c + R[:, :1] * u + R[:, 1:2] * v) / SCALE
            vs = [bm.verts.new(p) for p in P]
            if holes:
                edges += [bm.edges.new((vs[i], vs[(i + 1) % len(vs)])) for i in range(len(vs))]
            else:
                made.append(bm.faces.new(vs))                  # one n-gon per ring
        if edges:
            res = bmesh.ops.triangle_fill(bm, use_beauty=True, use_dissolve=True, edges=edges)
            made += [f for f in res.get("geom", []) if isinstance(f, bmesh.types.BMFace)]
        for f in made:
            f.normal_update()
            if f.normal.dot(nn) < 0:
                f.normal_flip()
    if to_local is not None:
        bm.transform(to_local)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return me


# --------------------------------------------------------------------------------------------
# Grow / Shrink a flat object
# --------------------------------------------------------------------------------------------

def offset_object(obj, dist, *, replace=True, suffix="", apply_modifiers=False, clean=None):
    """Grow (dist > 0) or shrink (dist < 0) a flat-polygon object in its plane(s).
    Returns the result object, or None if everything vanished (object left unchanged)."""
    V, T, E = read_mesh(obj, apply_modifiers)
    pieces = []
    for c, n, u, v, rings in flat_pieces(V, T, E):
        rings = cleanup(offset(rings, dist), clean)
        if rings:
            pieces.append((c, n, u, v, rings))
    if not pieces:
        return None
    if replace:
        me = build_mesh(obj.name, pieces, obj.matrix_world.inverted())
        old = obj.data
        for mat in old.materials:
            me.materials.append(mat)
        obj.data = me
        if old.users == 0:
            bpy.data.meshes.remove(old)
        return obj
    me = build_mesh(obj.name + suffix, pieces, obj.matrix_world.inverted())
    new = bpy.data.objects.new(obj.name + suffix, me)
    new.matrix_world = obj.matrix_world.copy()
    for coll in (obj.users_collection or [bpy.context.scene.collection]):
        coll.objects.link(new)
    return new


# --------------------------------------------------------------------------------------------
# Create Outline: the outline seen along a view direction
# --------------------------------------------------------------------------------------------

def view_outline(V, T, d, clean=None):
    """(centre, normal, u, v, rings) of the outline of triangles (V, T) seen along unit direction d
    (towards the object), on the plane through the point nearest the viewer."""
    n = -np.asarray(d, float)
    n /= np.linalg.norm(n)
    u, v = frame(n)
    h = V @ n
    c0 = V.mean(0)
    c = c0 + n * (float(h.max()) - float(c0 @ n))         # the outer plane, facing the viewer
    P = V - c
    uv = np.stack([P @ u, P @ v], -1)
    rings = cleanup(silhouette(uv, T), clean)
    return c, n, u, v, rings


def create_outline(obj, view_dir, *, apply_modifiers=True, suffix="_Outline", clean=None):
    """New flat object: obj's outline seen along view_dir (world, towards the scene)."""
    V, T, _ = read_mesh(obj, apply_modifiers)
    if not len(T):
        return None
    c, n, u, v, rings = view_outline(V, T, view_dir, clean)
    if not rings:
        return None
    centre = Vector(c / SCALE)
    me = build_mesh(obj.name + suffix, [(c, n, u, v, rings)], Matrix.Translation(-centre))
    new = bpy.data.objects.new(obj.name + suffix, me)
    new.location = centre
    for coll in (obj.users_collection or [bpy.context.scene.collection]):
        coll.objects.link(new)
    return new
