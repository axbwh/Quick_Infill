"""
Make Manifold: turn a pile of overlapping closed meshes (stones, blank, infill,
plugs) into ONE watertight mesh that a slicer accepts, without leaving Blender.

    1. every loose shell of the selection is taken apart; pieces under 0.2 mm
       are left out, open ones are rebuilt through voxels (the Manifold solver
       refuses open meshes);
    2. on ONE temporary Blender object, Blender's Manifold boolean unites them
       all, intersects with the blank and cuts at the plate. The mesh never goes
       through meshlib between these booleans: that trip opened holes in the
       union and the solver then silently skipped the cut. This is the same
       order as Alex's "Merge_Manifold" geometry nodes. With a blank, its bottom
       is the bottom and there is no plate cut. Before that, each island that
       ends at the bottom (within SHIFT below to 2 x SHIFT above it, SHIFT =
       0.0005) is moved straight down on its own until its lowest point is
       SHIFT below the bottom; the rest stays where it is. Its bottom then gets
       one fresh cut instead of a second cut in exactly the same spot, which
       leaves degenerate slivers. For the same reason, with Regrow on and a blank, every
       island is grown by SHIFT on its own with Quick Infill's own Grow
       (voxel offset at its own resolution tag, then its Auto Decimate), so faces already cut against
       the blank's walls stick out a hair past them and are cut once, fresh;
    3. the n-gons the solver returns are split with Blender's Beauty
       triangulation; floating and inside islands are deleted (Merge_Manifold's
       Delete Islands); the few small holes the triangulation leaves are closed
       (new triangles only, nothing moves); no vertex is snapped: the bottom is
       the boolean's own cut plane; health and flatness are measured and reported.

There is no meshlib self-intersection or zero-area repair (Alex 2026-11-20: it
did not work right). What the check finds is only reported.

Why not meshlib's own union (uniteManyMeshes)? Measured on the 550-piece
tower it silently lost about 5% of the solid volume. Blender's Manifold union
kept it (131,914 of 131,960 mm3).

The meshlib part of this file has no bpy; the *_in_blender functions are the
bpy part. Sizes in meshlib space (mm for Alex's files).
"""

import numpy as np

from .meshlib_utils import get_meshlib, get_mrmeshnumpy, check_cancelled
from .offset_utils import cuda_offset, face_count, auto_decimate


class Health:
    """What is wrong with a mesh, counted."""

    def __init__(self, mesh):
        mm, _ = get_meshlib()
        topo = mesh.topology
        self.faces = topo.numValidFaces()
        ok = self.faces > 0
        self.holes = len(topo.findHoleRepresentiveEdges()) if ok else 0
        self.shells = len(mm.MeshComponents.getAllComponents(mm.MeshPart(mesh))) if ok else 0
        self.self_intersections = mm.SelfIntersections.getFaces(mesh, False).count() if ok else 0
        self.zero_faces = mm.findDegenerateFaces(mm.MeshPart(mesh)).count() if ok else 0
        self.zero_edges = mm.findShortEdges(mm.MeshPart(mesh), 1e-6).count() if ok else 0
        self.slivers = mm.findDegenerateFaces(mm.MeshPart(mesh), 1e5).count() if ok else 0
        self.multiple_edges = bool(mm.hasMultipleEdges(topo)) if ok else False
        self.volume = float(mesh.volume()) if ok else 0.0

    def clean(self):
        """Watertight and free of everything a slicer or Fusion rejects.
        (Thin sliver faces with a real area are counted but allowed.)"""
        return (self.faces > 0 and self.holes == 0 and self.self_intersections == 0 and self.zero_faces == 0
                and self.zero_edges == 0 and not self.multiple_edges and self.volume > 0)

    def problems(self):
        bits = []
        if self.holes:
            bits.append(f"{self.holes} open boundaries")
        if self.self_intersections:
            bits.append(f"{self.self_intersections} self-intersecting faces")
        if self.zero_faces:
            bits.append(f"{self.zero_faces} zero-area faces")
        if self.zero_edges:
            bits.append(f"{self.zero_edges} zero-length edges")
        if self.multiple_edges:
            bits.append("doubled edges")
        if self.faces and self.volume <= 0:
            bits.append("inside-out")
        return bits

    def text(self):
        shells = "1 shell" if self.shells == 1 else f"{self.shells} separate shells"
        if self.clean():
            return f"watertight, no self-intersections, {self.faces} faces, {shells}"
        return ", ".join(self.problems()) + f" ({self.faces} faces, {shells})"


def split_shells_with_faces(mesh):
    """Loose shells of a mesh as (separate mesh, numpy array of the source face ids it holds)."""
    mm, _ = get_meshlib()
    mn = get_mrmeshnumpy()
    nf = mesh.topology.faceSize()
    comps = mm.MeshComponents.getAllComponents(mm.MeshPart(mesh))
    if len(comps) <= 1:
        return [(mesh, np.nonzero(mn.getNumpyBitSet(mesh.topology.getValidFaces())[:nf])[0])]
    out = []
    for c in comps:
        part = mm.Mesh()
        part.addMeshPart(mm.MeshPart(mesh, c))
        out.append((part, np.nonzero(mn.getNumpyBitSet(c)[:nf])[0]))
    return out


def split_shells(mesh):
    """Loose shells of a joined mesh as separate meshes."""
    mm, _ = get_meshlib()
    comps = mm.MeshComponents.getAllComponents(mm.MeshPart(mesh))
    if len(comps) <= 1:
        return [mesh]
    out = []
    for c in comps:
        part = mm.Mesh()
        part.addMeshPart(mm.MeshPart(mesh, c))
        out.append(part)
    return out


def close_open_shell(mesh, voxel):
    """A shell with open boundaries, rebuilt through voxels so it is closed."""
    out = cuda_offset(mesh, float(voxel), 0.0)
    out.pack()
    return out


class Flatness:
    """How flat the bottom of a mesh lies on the plate.

    on_plate   area (mm2) of the faces lying exactly in the plate
    off_plate  area (mm2) of downward faces hovering within `band` of the plate
               without being in it: the part of the bottom that is NOT flat
    max_gap    how far (mm) the worst of those hovers above the plate
    """

    def __init__(self, mesh, plate_z, band=0.3):
        mn = get_mrmeshnumpy()
        v = mn.getNumpyVerts(mesh).astype(np.float64)
        f = mn.getNumpyFaces(mesh.topology)
        z = v[:, 2] - float(plate_z)
        p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        cr = np.cross(p1 - p0, p2 - p0)
        area = 0.5 * np.linalg.norm(cr, axis=1)
        zf = z[f]
        down = cr[:, 2] < -0.9 * 2.0 * np.maximum(area, 1e-30)      # normal within about 25 degrees of straight down
        on = (np.abs(zf) < 1e-6).all(axis=1)
        near = down & ~on & (zf.max(axis=1) < band) & (zf.min(axis=1) > -1e-6)
        self.on_plate = float(area[on].sum())
        self.off_plate = float(area[near].sum())
        self.max_gap = float(zf[near].max()) if near.any() else 0.0
        used = np.zeros(len(v), dtype=bool)
        used[f.ravel()] = True
        self.lowest = float(z[used].min()) if used.any() else 0.0

    def flat(self):
        return self.on_plate > 0 and self.off_plate < 0.01 and self.lowest > -1e-6

    def text(self):
        if self.on_plate <= 0:
            return "nothing lies on the plate"
        if self.flat():
            return f"bottom flat on the plate ({self.on_plate:.0f} mm2)"
        return (f"{self.on_plate:.0f} mm2 lies flat on the plate; {self.off_plate:.1f} mm2 of underside hovers "
                f"up to {self.max_gap:.2f} mm above it without reaching it")


# ---------------------------------------------------------------------------
# bpy part: the booleans, with Blender's Manifold solver
# ---------------------------------------------------------------------------

class _Scratch:
    """Temporary Blender objects for one boolean, removed again on exit."""

    def __enter__(self):
        import bpy
        self.bpy = bpy
        self.scene = bpy.context.scene
        self.coll = bpy.data.collections.new("QI_Manifold_Tmp")
        self.scene.collection.children.link(self.coll)
        self.objs, self.meshes = [], []
        return self

    def add(self, mesh_or_me, name, in_collection=True):
        from .blender_meshlib_utils import meshlib_to_blender_mesh
        bpy = self.bpy
        me = mesh_or_me if isinstance(mesh_or_me, bpy.types.Mesh) else meshlib_to_blender_mesh(mesh_or_me, name)
        ob = bpy.data.objects.new(name, me)
        (self.coll if in_collection else self.scene.collection).objects.link(ob)
        self.objs.append(ob)
        self.meshes.append(me)
        return ob

    def boolean(self, base, setup):
        """Apply one Manifold boolean to base (its mesh is replaced by the result)."""
        bpy = self.bpy
        bpy.context.view_layer.update()
        mod = base.modifiers.new("QI", 'BOOLEAN')
        mod.solver = 'MANIFOLD'
        setup(mod)
        dg = bpy.context.evaluated_depsgraph_get()
        me = bpy.data.meshes.new_from_object(base.evaluated_get(dg))
        base.modifiers.remove(mod)
        base.data = me
        self.meshes.append(me)

    def __exit__(self, *exc):
        bpy = self.bpy
        for ob in self.objs:
            try:
                bpy.data.objects.remove(ob, do_unlink=True)
            except ReferenceError:
                pass
        try:
            bpy.data.collections.remove(self.coll)
        except ReferenceError:
            pass
        bpy.context.view_layer.update()
        for me in self.meshes:
            try:
                if me.users == 0:
                    bpy.data.meshes.remove(me)
            except ReferenceError:
                pass
        return False


def _counts(ob):
    return len(ob.data.vertices), len(ob.data.polygons)


def _plate_cutter(tmp, base, z):
    """A box under Blender height z, a little wider than base. None if base is all above z."""
    bpy = tmp.bpy
    co = np.empty(len(base.data.vertices) * 3, dtype=np.float32)
    base.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    lo, hi = co.min(axis=0), co.max(axis=0)
    if lo[2] >= z:
        return None
    pad = 1.0
    x0, x1, y0, y1, zb = (float(lo[0]) - pad, float(hi[0]) + pad, float(lo[1]) - pad, float(hi[1]) + pad,
                          float(lo[2]) - pad)
    cme = bpy.data.meshes.new("QI_tmp_cutter")
    cme.from_pydata([(x0, y0, zb), (x1, y0, zb), (x1, y1, zb), (x0, y1, zb),
                     (x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)], [],
                    [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
    cme.update()
    return tmp.add(cme, "QI_tmp_cutter", in_collection=False)


SHIFT = 0.0005   # Blender units: objects sitting exactly on the bottom are moved down this much (see make_manifold)


def booleans_in_blender(shells, plate_z=None, blank=None, log=None):
    """Union of meshlib shells, then the flat cuts (intersect with `blank`, remove
    everything below plate_z), all with Blender's Manifold solver on ONE scratch
    object, the way Alex's "Merge_Manifold" geometry nodes do it. The mesh never
    leaves Blender between the booleans: a trip through meshlib in between opened
    holes in the union, and the Manifold solver then silently skipped the cut.
    Returns a meshlib mesh. Nothing is left in the file."""
    from .blender_meshlib_utils import blender_to_meshlib, STL_EXPORT_SCALE
    say = log or (lambda msg: None)
    with _Scratch() as tmp:
        base = tmp.add(shells[0], "QI_tmp_0000", in_collection=False)
        for i, m in enumerate(shells[1:], 1):
            tmp.add(m, f"QI_tmp_{i:04d}")
        if len(shells) > 1:
            before = _counts(base)

            def setup(mod):
                mod.operation = 'UNION'
                mod.operand_type = 'COLLECTION'
                mod.collection = tmp.coll
            tmp.boolean(base, setup)
            if _counts(base) == before:
                say("WARNING: Blender's Manifold union did not run (an input is not a closed solid)")
        check_cancelled()
        if blank is not None:
            other = tmp.add(blank, "QI_tmp_blank", in_collection=False)
            before = _counts(base)

            def setup_blank(mod):
                mod.operation = 'INTERSECT'
                mod.operand_type = 'OBJECT'
                mod.object = other
            tmp.boolean(base, setup_blank)
            if _counts(base) == before:
                say("WARNING: the intersection with the blank did not change anything (Blender's Manifold "
                    "solver skips it when the union is not closed)")
        if plate_z is not None and len(base.data.vertices):
            cutter = _plate_cutter(tmp, base, float(plate_z) / STL_EXPORT_SCALE)
            if cutter is not None:
                before = _counts(base)

                def setup_cut(mod):
                    mod.operation = 'DIFFERENCE'
                    mod.operand_type = 'OBJECT'
                    mod.object = cutter
                tmp.boolean(base, setup_cut)
                if _counts(base) == before:
                    say("WARNING: the cut at the plate did not run")
        # The Manifold solver hands back big n-gons; split them here with Blender's
        # "Beauty" triangulation (far fewer slivers than splitting them later).
        import bmesh
        bm = bmesh.new()
        bm.from_mesh(base.data)
        bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method='BEAUTY', ngon_method='BEAUTY')
        bm.to_mesh(base.data)
        bm.free()
        return blender_to_meshlib(base)


def fill_holes(mesh):
    """Close the few small holes left where Blender's n-gons could not be split
    cleanly into triangles. Only new triangles are added across each hole: no
    existing vertex moves, nothing is smoothed. Returns how many holes were filled."""
    mm, _ = get_meshlib()
    holes = mesh.topology.findHoleRepresentiveEdges()
    for e in holes:
        mm.fillHole(mesh, e, mm.FillHoleParams())
    return len(holes)


TINY_SHELL = 0.2       # mm: Merge_Manifold's tiny-island size (0.02 Blender units)


def drop_small_shells(shells, min_size=TINY_SHELL):
    """Shells smaller than min_size (mm) in all three directions, left out before
    the union (Merge_Manifold's "Tiny Islands"). Returns (kept, how many dropped)."""
    kept = []
    for s in shells:
        bb = s.computeBoundingBox()
        if max(bb.max.x - bb.min.x, bb.max.y - bb.min.y, bb.max.z - bb.min.z) < min_size:
            continue
        kept.append(s)
    return kept, len(shells) - len(kept)


def keep_main_body(mesh):
    """Merge_Manifold's "Delete Islands": every loose shell smaller than the biggest
    size along every axis (floating crumbs, sealed voids inside the solid) is
    deleted; a shell that reaches the full size along at least one axis stays.
    Whole shells only: nothing that is kept moves. Returns how many were deleted."""
    mm, _ = get_meshlib()
    mn = get_mrmeshnumpy()
    comps = mm.MeshComponents.getAllComponents(mm.MeshPart(mesh))
    if len(comps) <= 1:
        return 0
    v = mn.getNumpyVerts(mesh).astype(np.float64)
    f = mn.getNumpyFaces(mesh.topology)
    sizes = []
    for c in comps:
        bits = mn.getNumpyBitSet(c)[:len(f)]
        pts = v[np.unique(f[np.nonzero(bits)[0]].ravel())]
        sizes.append(pts.max(axis=0) - pts.min(axis=0))
    sizes = np.array(sizes)
    biggest = sizes.max(axis=0)
    removed = 0
    for c, sz in zip(comps, sizes):
        if (sz < biggest - 1e-6).all():
            mesh.deleteFaces(c)
            removed += 1
    if removed:
        mesh.pack()
    return removed


def grow_like_quick_infill(mesh, dist, voxel, ratio, mode):
    """Quick Infill's Grow on one object: meshlib voxel offset (cuda_offset) by dist at the
    object's voxel size, then the same Auto Decimate (mode / ratio of its resolution tag,
    or the panel's). Returns the grown mesh, or the mesh as it was if it collapsed."""
    out = cuda_offset(mesh, float(voxel), float(dist))
    if face_count(out) == 0:
        return mesh
    if mode != "OFF":
        out = auto_decimate(out, face_count(mesh), float(voxel), mode, ratio)
    return out


def make_manifold(meshes, plate_z=None, blank=None, open_shell_voxel=0.1, log=None, delete_islands=True,
                  grow_settings=None, face_tags=None, regrow=False):
    """meshes (meshlib, world space) -> (one watertight meshlib mesh, Health, Flatness or None).

    blank: closed meshlib mesh the union is intersected with (the tower blank). Its
           lowest point is the bottom; plate_z is then ignored.
    plate_z: without a blank, everything below is cut away, leaving a flat face on the plate.
    delete_islands: keep only the main body (no floating or inside shells).
    grow_settings: per mesh (voxel mm, ratio, decimate mode) for the SHIFT grow with a blank:
                   each object's resolution tag (as Quick Infill's Grow with Smart), else the panel's.
    regrow: grow every island by SHIFT before the blank cut (see below). Off: nothing is grown.
    face_tags: per mesh, the resolution tag index of each of its faces (or None): an island
               with tagged faces grows at its own tag, the most common one among its faces.
    Must run on Blender's main thread (the booleans are Blender's).
    """
    say = log or (lambda msg: None)
    if blank is not None and not blank.topology.isClosed():
        raise ValueError("the blank is not a closed mesh, so it cannot be used for the intersection")
    # With a blank, its flat bottom IS the bottom: no plate cut (a second cut in the same
    # spot only makes slivers). Without one, Cut at Plate makes the bottom.
    bottom_z = None
    if blank is not None:
        bottom_z = float(blank.computeBoundingBox().min.z)
        if plate_z is not None:
            say("the blank's bottom is the bottom: no cut at the plate")
        plate_z = None
    elif plate_z is not None:
        bottom_z = float(plate_z)
    # Everything below works island by island (each loose part of each object on its own).
    from .blender_meshlib_utils import STL_EXPORT_SCALE
    from .offset_utils import RESOLUTION_PRESETS, RESOLUTION_TAG_ORDER
    gs = grow_settings or [None] * len(meshes)
    ft = face_tags or [None] * len(meshes)
    shells = []          # (island, (voxel, ratio, mode) for its grow)
    for m, g, tags in zip(meshes, gs, ft):
        for part, faces in split_shells_with_faces(m):
            if face_count(part) == 0:
                continue
            gi = g
            if tags is not None and len(faces):
                # the island's own resolution tag: the most common tag of its faces (Smart + Islands)
                t = np.asarray(tags)[faces[faces < len(tags)]]
                t = t[(t >= 0) & (t < len(RESOLUTION_TAG_ORDER))]
                if len(t):
                    tag = RESOLUTION_TAG_ORDER[int(np.bincount(t).argmax())]
                    voxel, ratio = RESOLUTION_PRESETS[tag]
                    gi = (voxel, ratio, "VOXEL_RATIO")
            shells.append((part, gi))
    if bottom_z is not None:
        # Stones often end at the bottom already (cut there against the blank or at the plate
        # before), or a hair above or below it. Cutting again in exactly the same spot leaves
        # degenerate slivers from rounding, so each of those ISLANDS (only those; the rest stays
        # where it is) is moved straight down, on its own, until its lowest point is SHIFT below
        # the bottom: it then gets ONE fresh cut on the cut plane.
        mm, _ = get_meshlib()
        d = SHIFT * STL_EXPORT_SCALE
        moved = 0
        for part, _g in shells:
            gap = float(part.computeBoundingBox().min.z) - bottom_z
            if -d <= gap <= 2.0 * d:
                part.transform(mm.AffineXf3f.translation(mm.Vector3f(0.0, 0.0, -(gap + d))))
                moved += 1
        if moved:
            say(f"{moved} island(s) ending at the bottom were moved down to {SHIFT:g} below it so they are cut fresh")
    if regrow and blank is not None:
        # Every island grows by SHIFT on its own, with Quick Infill's own Grow (voxel offset at its
        # resolution tag, then its Auto Decimate): faces already cut against the blank (walls and
        # bottom) now stick out a hair past it and get ONE fresh cut. Island by island, so each
        # voxel grid is only as big as that island. (Not for Cut at Plate alone: a stone edge
        # lying exactly on the plate came out as doubled edges after growing.)
        grown = []
        kept_tiny, _ = drop_small_shells([sh for sh, _ in shells])
        keep = {id(x) for x in kept_tiny}
        for sh, g in shells:
            check_cancelled()
            if id(sh) not in keep:
                grown.append((sh, g))        # left out as a tiny piece just below: not worth growing
                continue
            voxel, ratio, mode = g if g is not None else (open_shell_voxel * 2.0, None, "VOXEL_RATIO")
            grown.append((grow_like_quick_infill(sh, SHIFT * STL_EXPORT_SCALE, voxel, ratio, mode), g))
        shells = grown
        say(f"{len(grown)} island(s) grown by {SHIFT:g} one by one with Quick Infill's Grow (each at its resolution tag)")
    shells = [s for s, _ in shells]
    shells, tiny = drop_small_shells(shells)
    if tiny:
        say(f"{tiny} tiny piece(s) under {TINY_SHELL:g} mm were left out")
    reopened = vanished = 0
    kept = []
    for s in shells:
        if not s.topology.isClosed():
            s = close_open_shell(s, open_shell_voxel)
            if face_count(s) == 0:
                vanished += 1            # a flat speck with no volume: nothing to unite
                continue
            reopened += 1
        kept.append(s)
    shells = kept
    if reopened:
        say(f"{reopened} open shell(s) were rebuilt through voxels ({open_shell_voxel:g} mm) so they could be united")
    if vanished:
        say(f"{vanished} open speck(s) with no volume were left out")
    if not shells:
        raise ValueError("nothing to unite")
    # the biggest shell is the base of the union
    shells.sort(key=lambda s: -abs(float(s.volume())))
    out = booleans_in_blender(shells, plate_z, blank, log=say)
    check_cancelled()
    if delete_islands:
        n = keep_main_body(out)
        if n:
            say(f"{n} floating or inside island(s) were deleted")
    n = fill_holes(out)
    if n:
        say(f"{n} small hole(s) left by splitting Blender's n-gons into triangles were closed")
    if out.topology.numValidFaces() != out.topology.faceSize():
        out.pack()
    health = Health(out)
    flat = Flatness(out, bottom_z) if bottom_z is not None else None
    return out, health, flat
