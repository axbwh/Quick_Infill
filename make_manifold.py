"""
Make Manifold: turn a pile of overlapping closed meshes (stones, blank, infill,
plugs) into ONE watertight mesh that a slicer accepts, without leaving Blender.

    1. every loose shell of the selection becomes its own clean temporary
       object (world space, no modifiers) and Blender's Manifold boolean
       unites them in one go (about a second for 550 stones). Shells that are
       not closed are rebuilt through voxels first, because the Manifold
       solver refuses open meshes;
    2. meshlib repairs locally what Blender's own check does not see but
       Fusion and slicers complain about: self-intersecting faces (they come
       from decimated infill and overlapping detail), zero-area faces,
       zero-length edges and loose specks. Nothing is re-voxelised, so the
       visible stone detail stays exactly as it was;
    3. only THEN the flat cuts are made, again with Blender's Manifold solver:
       the intersection with the blank and / or the cut at the build plate.
       Repairing first matters: a repair that smooths a patch next to a cut
       face pulls it out of flat, which is how repaired bottoms end up not
       lying flat on the plate;
    4. after the cut only a tidy-up that cannot move anything runs (zero-size
       faces left by the cut are collapsed within 0.0001 mm), vertices on the
       plate are set to its exact height, and the result is measured again:
       mesh health and how flat the bottom is.

Why not meshlib's own union (uniteManyMeshes)? Measured on the 550-piece
tower it silently lost about 5% of the solid volume. Blender's Manifold union
kept it (131,914 of 131,960 mm3).

The meshlib part of this file has no bpy; the *_in_blender functions are the
bpy part. Sizes in meshlib space (mm for Alex's files).
"""

import numpy as np

from .meshlib_utils import get_meshlib, get_mrmeshnumpy, check_cancelled
from .offset_utils import cuda_offset, face_count


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


def drop_specks(mesh, min_volume=0.5):
    """Remove loose shells smaller than min_volume (mm3): floating crumbs a
    union leaves behind, and inside-out shells (closed voids inside the solid,
    i.e. sealed cavities, which this fills). Returns how many were removed."""
    mm, _ = get_meshlib()
    comps = mm.MeshComponents.getAllComponents(mm.MeshPart(mesh))
    if len(comps) <= 1:
        return 0
    removed = 0
    for c in comps:
        if float(mesh.volume(c)) < min_volume:
            mesh.deleteFaces(c)
            removed += 1
    if removed:
        mesh.pack()
    return removed


def _fill_all_holes(mesh):
    mm, _ = get_meshlib()
    for hole in mesh.topology.findHoleRepresentiveEdges():
        mm.fillHole(mesh, hole, mm.FillHoleParams())


def flip_caps(mesh):
    """Remove flat "cap" triangles (one corner lying on the opposite edge) by
    flipping that long edge. No vertex moves, so flat faces stay flat. Returns
    how many were flipped."""
    mm, _ = get_meshlib()
    mn = get_mrmeshnumpy()
    zero = mm.findDegenerateFaces(mm.MeshPart(mesh))
    if not zero.count():
        return 0
    topo = mesh.topology
    v = mn.getNumpyVerts(mesh).astype(np.float64)
    f = mn.getNumpyFaces(topo)
    flags = mn.getNumpyBitSet(zero)
    flipped = 0
    for fi in np.nonzero(flags[:len(f)])[0]:
        a, b, c = (int(x) for x in f[fi])
        la, lb, lc = np.linalg.norm(v[b] - v[c]), np.linalg.norm(v[c] - v[a]), np.linalg.norm(v[a] - v[b])
        # the longest edge and the corner (apex) that lies on it
        if la >= lb and la >= lc:
            o, d, apex = b, c, a
        elif lb >= lc:
            o, d, apex = c, a, b
        else:
            o, d, apex = a, b, c
        if min(la, lb, lc) < 1e-7:
            continue                     # a zero-length edge, not a cap: left to the collapse step
        e = topo.findEdge(mm.VertId(o), mm.VertId(d))
        if not e.valid() or topo.isBdEdge(e):
            continue
        if not topo.left(e).valid() or not topo.right(e).valid():
            continue
        x, y = topo.dest(topo.next(e)).get(), topo.dest(topo.prev(e)).get()
        other = y if x == apex else x
        if apex not in (x, y) or other == apex:
            continue                     # the mesh changed under us (an earlier flip touched this face)
        if topo.findEdge(mm.VertId(apex), mm.VertId(other)).valid():
            continue                     # flipping would double an existing edge
        topo.flipEdge(e)
        flipped += 1
    if flipped:
        mesh.invalidateCaches()
    return flipped


def repair(mesh, max_deviation=0.01, passes=8, relax=3, max_step=0.08, subdivide=True, cut_and_fill=True):
    """Local repair, in place: self-intersections, zero-area faces, zero-length
    edges, small holes, specks. Stops as soon as the mesh checks clean.
    max_deviation: how far (mm) the surface may move where a defect is fixed;
    it is doubled per pass, up to max_step, only if defects survive.
    relax: smoothing passes on a patch that replaces self-intersecting faces
    (0 = the patch is not smoothed, so nothing next to it moves).
    subdivide: let meshlib split faces where collapsing alone cannot remove a
    sliver. cut_and_fill: allow cutting stubborn zero-area faces out and
    refilling the hole. Returns Health."""
    mm, _ = get_meshlib()
    drop_specks(mesh)
    health = Health(mesh)
    dev = float(max_deviation)
    for _ in range(passes):
        if health.clean():
            break
        check_cancelled()
        if health.multiple_edges:
            mm.fixMultipleEdges(mesh)
        if health.zero_faces or health.zero_edges or health.slivers:
            # Only the defects and two rings of faces round them may change; the
            # rest of the mesh (flat cut faces included) stays exactly as it is.
            region = mm.findDegenerateFaces(mm.MeshPart(mesh), 1e4)
            region |= mm.getIncidentFaces(mesh.topology, mm.findShortEdges(mm.MeshPart(mesh), max(dev * 0.2, 1e-6)))
            if region.count():
                mm.expand(mesh.topology, region, 2)
                p = mm.FixMeshDegeneraciesParams()
                p.maxDeviation = dev
                p.tinyEdgeLength = dev * 0.2
                p.criticalTriAspectRatio = 1e4
                p.stabilizer = 1e-6
                p.region = region
                if not subdivide:
                    p.mode = mm.FixMeshDegeneraciesParams.Mode.Decimate
                mm.fixMeshDegeneracies(mesh, p)
            # Flat "cap" triangles (one corner lying on the opposite edge) can survive
            # that. Flipping their long edge removes them without moving anything;
            # what is still left is cut out with its neighbours and the small hole
            # closed again (that last step can chamfer an edge, so it is optional).
            flip_caps(mesh)
            zero = mm.findDegenerateFaces(mm.MeshPart(mesh))
            if cut_and_fill and zero.count():
                mm.expand(mesh.topology, zero, 1)
                mesh.deleteFaces(zero)
                _fill_all_holes(mesh)
        if mm.SelfIntersections.getFaces(mesh, False).count():
            s = mm.SelfIntersections.Settings()
            s.method = mm.SelfIntersections.Settings.Method.CutAndFill
            s.relaxIterations = int(relax)
            s.maxExpand = 2
            s.touchIsIntersection = False
            mm.SelfIntersections.fix(mesh, s)
        _fill_all_holes(mesh)
        drop_specks(mesh)
        health = Health(mesh)
        dev = min(dev * 2.0, float(max_step))
    if mesh.topology.numValidFaces() != mesh.topology.faceSize():
        mesh.pack()
    return health


def _refill_stubborn(mesh):
    """Zero-area faces that neither collapsing nor flipping removed are cut out
    with neighbours and the hole is closed again, in a way that cannot bend a
    flat face: each sliver takes only neighbours that lie in ONE common plane
    (the one with the most area round it), so the hole and its fill are flat.
    A sliver lying along a sharp edge therefore never takes faces from both
    sides of that edge, which would chamfer it. Returns the number of faces cut."""
    mm, _ = get_meshlib()
    mn = get_mrmeshnumpy()
    topo = mesh.topology
    zero = mm.findDegenerateFaces(mm.MeshPart(mesh))
    if not zero.count():
        return 0
    f = mn.getNumpyFaces(topo)
    n = len(f)
    v = mn.getNumpyVerts(mesh).astype(np.float64)
    zmask = mn.getNumpyBitSet(zero)[:n]
    zmask = np.pad(zmask, (0, n - len(zmask)))
    valid = mn.getNumpyBitSet(topo.getValidFaces())[:n]
    valid = np.pad(valid, (0, n - len(valid)))
    cut = np.zeros(n, dtype=bool)
    for zi in np.nonzero(zmask)[0][:500]:
        corners = f[zi]
        ring = np.nonzero(np.isin(f, corners).any(axis=1) & valid & ~zmask)[0]
        if len(ring) == 0:
            continue
        p0, p1, p2 = v[f[ring, 0]], v[f[ring, 1]], v[f[ring, 2]]
        cr = np.cross(p1 - p0, p2 - p0)
        dbl = np.linalg.norm(cr, axis=1)
        ok = dbl > 1e-9
        if not ok.any():
            continue
        ring, cr, dbl, p0 = ring[ok], cr[ok], dbl[ok], p0[ok]
        nrm = cr / dbl[:, None]
        off = (nrm * p0).sum(axis=1)
        # the sliver must lie in the neighbour's plane
        dist = np.abs(v[corners] @ nrm.T - off[None, :]).max(axis=0)
        keep = dist < 1e-4
        if not keep.any():
            continue
        ring, nrm, off, dbl = ring[keep], nrm[keep], off[keep], dbl[keep]
        key = np.round(np.column_stack([nrm, off]) / 1e-3).astype(np.int64)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()
        area = np.bincount(inv, weights=dbl)
        best = int(np.argmax(area))
        cut[zi] = True
        cut[ring[inv == best]] = True
    if not cut.any():
        return 0
    mesh.deleteFaces(mn.faceBitSetFromBools(cut))
    for hole in topo.findHoleRepresentiveEdges():
        fp = mm.FillHoleParams()
        fp.metric = mm.getMinTriAngleMetric(mesh)    # the fill avoids needle triangles
        mm.fillHole(mesh, hole, fp)
    return int(cut.sum())


def tidy_after_cut(mesh, plate_z=None):
    """Clean-up for a mesh that was just cut flat. It removes the zero-size faces
    a cut leaves without disturbing the flat faces: collapses within 0.0001 mm,
    edge flips, and for the few faces those cannot remove a cut-and-refill that
    stays inside one plane. Nothing is smoothed. If a refill round changes the
    volume it is undone. Vertices on the plate then get its exact height.
    Returns (mesh, Health): the mesh object may be a restored copy."""
    mm, _ = get_meshlib()

    def light():
        return repair(mesh, max_deviation=1e-4, passes=4, relax=0, max_step=1e-4, subdivide=False, cut_and_fill=False)

    health = light()
    for _ in range(4):
        if health.clean():
            break
        backup = mm.copyMesh(mesh)
        before = health
        if not _refill_stubborn(mesh):
            break
        health = light()
        worse = (abs(health.volume - before.volume) > 0.01 or health.holes
                 or health.self_intersections > before.self_intersections)
        if worse:
            mesh, health = backup, before
            break
    if plate_z is not None:
        if snap_to_plate(mesh, plate_z):
            health = Health(mesh)
    return mesh, health


def snap_to_plate(mesh, plate_z, tolerance=1e-3):
    """Vertices within `tolerance` (mm) of the plate get its exact height. Returns how many moved."""
    mm, _ = get_meshlib()
    z = float(plate_z)
    v = get_mrmeshnumpy().getNumpyVerts(mesh)
    idx = np.nonzero((np.abs(v[:, 2] - z) < tolerance) & (v[:, 2] != np.float32(z)))[0]
    if len(idx) == 0:
        return 0
    pts = mesh.points.vec
    for i in idx:
        pt = pts[int(i)]
        pts[int(i)] = mm.Vector3f(pt.x, pt.y, z)
    mesh.invalidateCaches()
    return int(len(idx))


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


def union_in_blender(shells):
    """Unite meshlib meshes (each one a closed shell) with Blender's Manifold
    boolean. Returns a meshlib mesh. Nothing is left in the file."""
    from .blender_meshlib_utils import blender_to_meshlib
    with _Scratch() as tmp:
        base = tmp.add(shells[0], "QI_tmp_0000", in_collection=False)
        for i, m in enumerate(shells[1:], 1):
            tmp.add(m, f"QI_tmp_{i:04d}")
        if len(shells) > 1:
            def setup(mod):
                mod.operation = 'UNION'
                mod.operand_type = 'COLLECTION'
                mod.collection = tmp.coll
            tmp.boolean(base, setup)
        return blender_to_meshlib(base)


def cut_in_blender(mesh, plate_z=None, blank=None):
    """Flat cuts with Blender's Manifold boolean: intersect with `blank` (a
    closed meshlib mesh) and / or remove everything below plate_z. Returns a meshlib mesh."""
    from .blender_meshlib_utils import blender_to_meshlib, STL_EXPORT_SCALE
    if blank is None and plate_z is None:
        return mesh
    with _Scratch() as tmp:
        bpy = tmp.bpy
        base = tmp.add(mesh, "QI_tmp_base", in_collection=False)
        if blank is not None:
            other = tmp.add(blank, "QI_tmp_blank")

            def setup(mod):
                mod.operation = 'INTERSECT'
                mod.operand_type = 'OBJECT'
                mod.object = other
            tmp.boolean(base, setup)
        if plate_z is not None and len(base.data.vertices):
            co = np.empty(len(base.data.vertices) * 3, dtype=np.float32)
            base.data.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3)
            lo, hi = co.min(axis=0), co.max(axis=0)
            z = float(plate_z) / STL_EXPORT_SCALE
            if lo[2] < z:
                pad = 1.0
                x0, x1, y0, y1, zb = (float(lo[0]) - pad, float(hi[0]) + pad, float(lo[1]) - pad, float(hi[1]) + pad,
                                      float(lo[2]) - pad)
                cme = bpy.data.meshes.new("QI_tmp_cutter")
                cme.from_pydata([(x0, y0, zb), (x1, y0, zb), (x1, y1, zb), (x0, y1, zb),
                                 (x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)], [],
                                [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
                cme.update()
                cutter = tmp.add(cme, "QI_tmp_cutter")

                def setup_cut(mod):
                    mod.operation = 'DIFFERENCE'
                    mod.operand_type = 'OBJECT'
                    mod.object = cutter
                tmp.boolean(base, setup_cut)
        return blender_to_meshlib(base)


def make_manifold(meshes, plate_z=None, blank=None, open_shell_voxel=0.1, max_deviation=0.01, log=None,
                  repair_before_cut=True):
    """meshes (meshlib, world space) -> (one watertight meshlib mesh, Health, Flatness or None).

    blank: closed meshlib mesh the union is intersected with (the tower blank).
    plate_z: everything below is cut away, leaving a flat face on the plate.
    Must run on Blender's main thread (the booleans are Blender's).
    repair_before_cut=False is the old order (cut, then repair), kept for comparison.
    """
    say = log or (lambda msg: None)
    shells = []
    for m in meshes:
        shells.extend(split_shells(m))
    shells = [s for s in shells if face_count(s) > 0]
    if not shells:
        raise ValueError("nothing to unite")
    reopened = 0
    for i, s in enumerate(shells):
        if not s.topology.isClosed():
            shells[i] = close_open_shell(s, open_shell_voxel)
            reopened += 1
    if reopened:
        say(f"{reopened} open shell(s) were rebuilt through voxels ({open_shell_voxel:g} mm) so they could be united")
    if blank is not None and not blank.topology.isClosed():
        raise ValueError("the blank is not a closed mesh, so it cannot be used for the intersection")
    # the biggest shell is the base of the union
    shells.sort(key=lambda s: -abs(float(s.volume())))
    out = union_in_blender(shells)
    say(f"Blender union of {len(shells)} shells: " + Health(out).text())
    has_cut = plate_z is not None or blank is not None
    if repair_before_cut or not has_cut:
        health = repair(out, max_deviation)
        if has_cut:
            if not health.clean():
                say("before the cut: " + health.text())
            out = cut_in_blender(out, plate_z, blank)
            say("after the cut: " + Health(out).text())
            # After a cut nothing may be smoothed or re-meshed any more (that is what
            # pulls a flat bottom out of flat), so whatever this cannot fix is reported.
            out, health = tidy_after_cut(out, plate_z)
    else:
        out = cut_in_blender(out, plate_z, blank)
        health = repair(out, max_deviation)
    flat = Flatness(out, plate_z) if plate_z is not None else None
    return out, health, flat
