"""
Make Manifold: turn a pile of overlapping closed meshes (stones, blank, infill,
plugs) into ONE watertight mesh that a slicer accepts, without leaving Blender.

    1. every loose shell of the selection becomes its own clean temporary
       object (world space, no modifiers) and Blender's Manifold boolean
       unites them in one go (about a second for 550 stones). Shells that are
       not closed are rebuilt through voxels first, because the Manifold
       solver refuses open meshes;
    2. the flat cuts are made, again with Blender's Manifold solver: the
       intersection with the blank and / or the cut at the build plate;
    3. vertices on the plate are set to its exact height, and the result is
       measured: mesh health (holes, self-intersections, zero-area faces) and
       how flat the bottom is. Nothing is repaired: what the check finds is
       only reported.

There is no meshlib repair step any more (Alex 2026-11-20: it did not work
right). The old one fixed self-intersections (SelfIntersections.fix) and
zero-area faces (fixMeshDegeneracies, edge flips, cut and refill), and it
could move or re-mesh surface next to the defects.

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


def make_manifold(meshes, plate_z=None, blank=None, open_shell_voxel=0.1, log=None):
    """meshes (meshlib, world space) -> (one watertight meshlib mesh, Health, Flatness or None).

    blank: closed meshlib mesh the union is intersected with (the tower blank).
    plate_z: everything below is cut away, leaving a flat face on the plate.
    Must run on Blender's main thread (the booleans are Blender's).
    """
    say = log or (lambda msg: None)
    shells = []
    for m in meshes:
        shells.extend(split_shells(m))
    shells = [s for s in shells if face_count(s) > 0]
    if not shells:
        raise ValueError("nothing to unite")
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
    if blank is not None and not blank.topology.isClosed():
        raise ValueError("the blank is not a closed mesh, so it cannot be used for the intersection")
    # the biggest shell is the base of the union
    shells.sort(key=lambda s: -abs(float(s.volume())))
    out = union_in_blender(shells)
    check_cancelled()
    if plate_z is not None or blank is not None:
        out = cut_in_blender(out, plate_z, blank)
        if plate_z is not None:
            snap_to_plate(out, plate_z)
    if out.topology.numValidFaces() != out.topology.faceSize():
        out.pack()
    health = Health(out)
    flat = Flatness(out, plate_z) if plate_z is not None else None
    return out, health, flat
