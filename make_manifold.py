"""
Make Manifold: turn a pile of overlapping closed meshes (stones, blank, infill,
plugs) into ONE watertight mesh that a slicer accepts, without leaving Blender.

    1. every loose shell of the selection becomes its own clean temporary
       object (world space, no modifiers) and Blender's Manifold boolean
       unites them in one go (about a second for 550 stones). Shells that are
       not closed are rebuilt through voxels first, because the Manifold
       solver refuses open meshes;
    2. optional cut at the build plate (also Manifold);
    3. meshlib then repairs locally what Blender's own check does not see but
       Fusion and slicers complain about: self-intersecting faces (they come
       from decimated infill and overlapping detail), zero-area faces,
       zero-length edges and loose specks. Nothing is re-voxelised, so the
       visible stone detail stays exactly as it was;
    4. the result is measured again and the numbers are reported.

Why not meshlib's own union (uniteManyMeshes)? Measured on the 550-piece
tower it silently lost about 5% of the solid volume. Blender's Manifold union
kept it (131,914 of 131,960 mm3).

The meshlib part of this file has no bpy; union_in_blender() is the bpy part.
Sizes in meshlib space (mm for Alex's files).
"""

from .meshlib_utils import get_meshlib, check_cancelled
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


def repair(mesh, max_deviation=0.01, passes=8):
    """Local repair, in place: self-intersections, zero-area faces, zero-length
    edges, small holes, specks. Stops as soon as the mesh checks clean.
    max_deviation: how far (mm) the surface may move where a defect is fixed;
    it is raised step by step only if defects survive. Returns Health."""
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
            p = mm.FixMeshDegeneraciesParams()
            p.maxDeviation = dev
            p.tinyEdgeLength = dev * 0.2
            p.criticalTriAspectRatio = 1e4
            p.stabilizer = 1e-6
            mm.fixMeshDegeneracies(mesh, p)
            # Flat "cap" triangles (one corner lying on the opposite edge) can survive
            # that; cut them out with their neighbours and close the small hole again.
            zero = mm.findDegenerateFaces(mm.MeshPart(mesh))
            if zero.count():
                mm.expand(mesh.topology, zero, 1)
                mesh.deleteFaces(zero)
                _fill_all_holes(mesh)
        if mm.SelfIntersections.getFaces(mesh, False).count():
            s = mm.SelfIntersections.Settings()
            s.method = mm.SelfIntersections.Settings.Method.CutAndFill
            s.relaxIterations = 3
            s.maxExpand = 2
            s.touchIsIntersection = False
            mm.SelfIntersections.fix(mesh, s)
        _fill_all_holes(mesh)
        drop_specks(mesh)
        health = Health(mesh)
        dev = min(dev * 2.0, 0.08)
    if mesh.topology.numValidFaces() != mesh.topology.faceSize():
        mesh.pack()
    return health


# ---------------------------------------------------------------------------
# bpy part: the union itself, with Blender's Manifold solver
# ---------------------------------------------------------------------------

def union_in_blender(shells, plate_z=None):
    """Unite meshlib meshes (each one a closed shell) with Blender's Manifold
    boolean and optionally cut at plate_z (meshlib space). Returns a meshlib mesh.

    Works on temporary objects in a temporary collection that is never linked
    to the user's collections; everything is removed again.
    """
    import bpy
    from .blender_meshlib_utils import meshlib_to_blender_mesh, blender_to_meshlib, STL_EXPORT_SCALE

    scene = bpy.context.scene
    coll = bpy.data.collections.new("QI_Manifold_Tmp")
    scene.collection.children.link(coll)
    tmp_objs, tmp_meshes = [], []
    try:
        for i, m in enumerate(shells):
            me = meshlib_to_blender_mesh(m, f"QI_tmp_{i:04d}")
            ob = bpy.data.objects.new(me.name, me)
            (scene.collection if i == 0 else coll).objects.link(ob)
            tmp_objs.append(ob)
            tmp_meshes.append(me)
        base = tmp_objs[0]
        bpy.context.view_layer.update()

        def apply(mod_setup):
            mod = base.modifiers.new("QI", 'BOOLEAN')
            mod.solver = 'MANIFOLD'
            mod_setup(mod)
            dg = bpy.context.evaluated_depsgraph_get()
            me = bpy.data.meshes.new_from_object(base.evaluated_get(dg))
            base.modifiers.remove(mod)
            old = base.data
            base.data = me
            tmp_meshes.append(me)
            return old

        if len(shells) > 1:
            def setup_union(mod):
                mod.operation = 'UNION'
                mod.operand_type = 'COLLECTION'
                mod.collection = coll
            apply(setup_union)

        if plate_z is not None:
            lo = [min(v[i] for v in base.bound_box) for i in range(3)]
            hi = [max(v[i] for v in base.bound_box) for i in range(3)]
            z = float(plate_z) / STL_EXPORT_SCALE
            if lo[2] < z:
                pad = 1.0
                depth = (z - lo[2]) + pad
                cme = bpy.data.meshes.new("QI_tmp_cutter")
                x0, x1, y0, y1 = lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad
                verts = [(x0, y0, z - depth), (x1, y0, z - depth), (x1, y1, z - depth), (x0, y1, z - depth),
                         (x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
                faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
                cme.from_pydata(verts, [], faces)
                cme.update()
                cutter = bpy.data.objects.new("QI_tmp_cutter", cme)
                coll.objects.link(cutter)
                tmp_objs.append(cutter)
                tmp_meshes.append(cme)
                bpy.context.view_layer.update()

                def setup_cut(mod):
                    mod.operation = 'DIFFERENCE'
                    mod.operand_type = 'OBJECT'
                    mod.object = cutter
                apply(setup_cut)

        return blender_to_meshlib(base)
    finally:
        for ob in tmp_objs:
            try:
                bpy.data.objects.remove(ob, do_unlink=True)
            except ReferenceError:
                pass
        try:
            bpy.data.collections.remove(coll)
        except ReferenceError:
            pass
        bpy.context.view_layer.update()
        for me in tmp_meshes:
            try:
                if me.users == 0:
                    bpy.data.meshes.remove(me)
            except ReferenceError:
                pass


def make_manifold(meshes, plate_z=None, open_shell_voxel=0.1, max_deviation=0.01, log=None):
    """meshes (meshlib, world space) -> (one watertight meshlib mesh, Health).
    Must run on Blender's main thread (the union is a Blender boolean)."""
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
    # the biggest shell is the base of the union
    shells.sort(key=lambda s: -abs(float(s.volume())))
    out = union_in_blender(shells, plate_z)
    before = Health(out)
    say(f"Blender union of {len(shells)} shells: " + before.text())
    health = repair(out, max_deviation)
    return out, health
