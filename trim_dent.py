"""Trim Edges, Chip method: stone damage meshes knocked off the sharp edges and corners.

Alex 2026-11-18: the flat chip (trim_chip.fast_chip) did not look right. This method places the
stone damage meshes (Ashlarize's Dent_S library: small rough dents, Z up = out of the stone) on the
stone's sharp convex edges and corners, each one picked, turned and sized at random, subtracts them
all with one meshlib boolean, and then runs Trim Thin (trim_round.trim_thin) so nothing too thin to
print is left by the bites.

meshlib used (MESHLIB FIRST): Mesh.findCreaseEdges (the sharp edges), filterCreaseEdges (drop short
crease runs: sculpt noise), Mesh.dihedralAngle (convex or not), Mesh.normal (corner direction),
uniteManyMeshes (all cutters into one solid: it merges the ones that don't touch first), boolean
DifferenceAB. Checked and not used: there is no meshlib "scatter instances on edges" function, so
the points along the edges are drawn here with numpy; Blender's boolean modifier is not used (this
runs in worker processes, no bpy).

Pure meshlib + numpy: runs in Blender and in Quick Infill's worker processes. Lengths are in the
mesh's units; `unit` = mesh units per Blender unit (Quick Infill works at x10). The dent arrays are
in Blender units.
"""
import math
import time

import numpy as np

try:
    from .meshlib_utils import get_meshlib
except ImportError:                    # loaded by path (tests)
    get_meshlib = None


def _ml():
    if get_meshlib is not None:
        mm, _ = get_meshlib()
    else:
        from meshlib import mrmeshpy as mm
    from meshlib import mrmeshnumpy as mn
    return mm, mn


CREASE_DEG = 25.0      # an edge is sharp when its faces turn at least this much (a bevel of two 25 deg facets still counts)
MIN_RUN = 0.08         # BU: crease runs shorter than this in all are sculpt noise, not stone edges
AMOUNT = 2.5           # chips per BU of sharp edge (the "Amount" slider), corners come on top
CORNER_CHANCE = 0.75   # a corner (three or more sharp edges, or two meeting at a turn) gets a chip this often
CORNER_TURN = 45.0     # deg: two sharp edges at a vertex that turn more than this make a corner
CORNER_SCALE = 1.25    # corner chips are this much bigger
SCALE_RANGE = (0.6, 1.3)   # each chip's size, x the Size slider, drawn in this range ...
AXIS_JITTER = 0.15         # ... and stretched up to this much per axis
APART = 0.5            # chips are kept at least this x their width apart (width of an unscaled Dent_S: ~0.14 BU)
DENT_WIDTH = 0.14      # BU
LEAN = (0.3, 0.7)      # an edge chip's Z leans between the two faces' normals at random: it bites more into one face
TILT = 15.0            # deg: and is tipped at random by up to this much
SINK = 0.25            # x the chip's bite: chips sit up to this much higher or lower on the edge
BITE = 0.035           # BU: how far an unscaled Dent_S reaches below its origin (measured: 0.015 to 0.04)


def _seed_of(V):
    """A seed from the shape itself: the same stone gets the same chips, in Blender and in a worker."""
    h = np.round(np.asarray(V, np.float64), 3)
    return int(abs(float(h.sum())) * 1000.0 + len(h) * 7919) % (2 ** 31)


def _normals(V, F):
    """Unit face normals and area-weighted vertex normals."""
    c = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fn = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-20)
    vn = np.zeros_like(V)
    for j in range(3):
        np.add.at(vn, F[:, j], c)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-20
    return fn, vn


def sharp_edges(m, unit=1.0, cut_faces=None):
    """The stone's sharp convex edges: (p0 (k,3), p1 (k,3), n_left (k,3), n_right (k,3), v0 (k,), v1 (k,)),
    plus the vertex normals. cut_faces: per face, True on a cut face: then only edges with a cut face
    on exactly one side."""
    mm, mn = _ml()
    V = np.asarray(mn.getNumpyVerts(m), np.float64)
    F = np.asarray(mn.getNumpyFaces(m.topology), np.int64)
    fn, vn = _normals(V, F)
    cr = m.findCreaseEdges(math.radians(CREASE_DEG))
    try:
        mm.filterCreaseEdges(m, cr, float(MIN_RUN * unit))
    except Exception:
        pass
    bits = np.asarray(mn.getNumpyBitSet(cr), bool)
    topo = m.topology
    rows = []
    for i in np.flatnonzero(bits):
        e = mm.EdgeId(2 * int(i))
        fl, fr = topo.left(e), topo.right(e)
        if not fl.valid() or not fr.valid():
            continue
        if m.dihedralAngle(mm.UndirectedEdgeId(int(i))) <= 0:     # concave: a groove, not an edge
            continue
        a, b = int(fl.get()), int(fr.get())
        if cut_faces is not None:
            ca = bool(cut_faces[a]) if a < len(cut_faces) else False
            cb = bool(cut_faces[b]) if b < len(cut_faces) else False
            if ca == cb:
                continue
        rows.append((int(topo.org(e).get()), int(topo.dest(e).get()), a, b))
    if not rows:
        z = np.zeros((0, 3))
        return z, z, z, z, np.zeros(0, int), np.zeros(0, int), V, vn
    R = np.array(rows, np.int64)
    return V[R[:, 0]], V[R[:, 1]], fn[R[:, 2]], fn[R[:, 3]], R[:, 0], R[:, 1], V, vn


def corners(V, vn, p0, p1, v0, v1):
    """Corner vertices of the sharp edges: [(point, outward direction)]."""
    at = {}
    for k in range(len(v0)):
        d = p1[k] - p0[k]
        d = d / (np.linalg.norm(d) + 1e-12)
        at.setdefault(int(v0[k]), []).append(d)
        at.setdefault(int(v1[k]), []).append(-d)
    cos_turn = math.cos(math.radians(180.0 - CORNER_TURN))
    out = []
    for v, ds in at.items():
        if len(ds) >= 3 or (len(ds) == 2 and float(ds[0] @ ds[1]) > cos_turn):
            out.append((V[v].copy(), vn[v].copy()))
    return out


def _frame(z, rng):
    """Columns x, y, z: z (normalised) with a random spin about it."""
    z = z / (np.linalg.norm(z) + 1e-12)
    x = np.cross(z, (0.0, 0.0, 1.0) if abs(z[2]) < 0.9 else (1.0, 0.0, 0.0))
    x /= np.linalg.norm(x) + 1e-12
    y = np.cross(z, x)
    sp = rng.uniform(0, 2 * math.pi)
    c, s = math.cos(sp), math.sin(sp)
    return np.stack([x * c + y * s, -x * s + y * c, z], 1)


def _tilt(z, rng, deg):
    """z tipped by up to deg in a random direction."""
    B = _frame(z, rng)
    a = math.radians(rng.uniform(0, deg))
    return B[:, 2] * math.cos(a) + B[:, 0] * math.sin(a)


def place(m, dents, size=1.0, amount=AMOUNT, unit=1.0, seed=0, cut_faces=None, info=None):
    """The chip cutters for mesh m: a list of (V (n,3) float32, F (k,3) int32) in the mesh's units."""
    info = {} if info is None else info
    rng = np.random.default_rng(seed)
    p0, p1, nl, nr, v0, v1, V, vn = sharp_edges(m, unit, cut_faces)
    info["sharp_edges"] = int(len(p0))
    if not len(p0) or not dents:
        return []
    lens = np.linalg.norm(p1 - p0, axis=1)
    L = float(lens.sum()) / unit
    info["edge_length"] = L
    picks = []                                         # (point, z, scale)
    for c, n in corners(V, vn, p0, p1, v0, v1):
        if rng.random() < CORNER_CHANCE:
            picks.append((c, _tilt(n, rng, TILT), CORNER_SCALE))
    info["corners"] = len(picks)
    n_edge = int(rng.poisson(L * float(amount)))
    pr = lens / lens.sum()
    for _ in range(n_edge):
        sc = rng.uniform(*SCALE_RANGE)
        for _try in range(12):
            k = int(rng.choice(len(lens), p=pr))
            t = rng.random()
            pnt = p0[k] * (1 - t) + p1[k] * t
            gap = APART * DENT_WIDTH * float(size) * unit * sc
            if all(np.linalg.norm(pnt - q[0]) >= gap for q in picks):
                break
        else:
            continue
        w = rng.uniform(*LEAN)
        z = nl[k] * w + nr[k] * (1 - w)
        picks.append((pnt, _tilt(z, rng, TILT), sc))
    out = []
    for pnt, z, sc in picks:
        z = z / (np.linalg.norm(z) + 1e-12)
        B = _frame(z, rng)
        s = float(size) * sc * (1.0 + rng.uniform(-AXIS_JITTER, AXIS_JITTER, 3))
        dv, dt = dents[int(rng.integers(0, len(dents)))]
        sink = rng.uniform(-SINK, SINK) * BITE * float(size) * sc * unit
        V = ((np.asarray(dv, float) * s) @ B.T) * unit + pnt + z * sink
        out.append((V.astype(np.float32), np.asarray(dt, np.int32)))
    info["chips"] = len(out)
    return out


def _cutter(mm, mn, cut):
    V, F = cut
    return mn.meshFromFacesVerts(np.ascontiguousarray(F, np.int32), np.ascontiguousarray(V, np.float32))


def _holes(m):
    return len(m.topology.findHoleRepresentiveEdges())


def dent_chip(mesh, dents, size=1.0, amount=AMOUNT, unit=1.0, seed=None, cut_faces=None, info=None):
    """mesh with the damage chips cut off its sharp edges and corners (a new mesh; the input is kept).
    dents: [(V (n,3), F (k,3))] in Blender units, Z out of the stone. size: x the dents' own size."""
    mm, mn = _ml()
    info = {} if info is None else info
    t0 = time.perf_counter()
    m = mm.Mesh(mesh)
    m.pack()
    if seed is None:
        seed = _seed_of(mn.getNumpyVerts(m))
    cuts = place(m, dents, size, amount, unit, seed, cut_faces, info)
    info["place_s"] = time.perf_counter() - t0
    if not cuts:
        info["seconds"] = time.perf_counter() - t0
        return m
    holes0 = _holes(m)
    cms = [_cutter(mm, mn, c) for c in cuts]
    cms = [c for c in cms if c.topology.isClosed()] or cms
    w = None
    try:                                   # all cutters as one solid, one boolean
        vec = mm.std_vector_Mesh_const_ptr()
        for c in cms:
            vec.append(c)
        u = mm.uniteManyMeshes(vec)
        r = mm.boolean(mm.Mesh(m), u, mm.BooleanOperation.DifferenceAB)
        if r.valid() and _holes(r.mesh) <= holes0 and r.mesh.topology.numValidFaces():
            w = r.mesh
            info["method"] = "united"
    except Exception as e:
        info["unite_error"] = repr(e)
    if w is None:                          # one by one; a chip that fails is left out
        w, ok = m, 0
        for c in cms:
            try:
                r = mm.boolean(mm.Mesh(w), mm.Mesh(c), mm.BooleanOperation.DifferenceAB)
            except Exception:
                continue
            if r.valid() and _holes(r.mesh) <= holes0 and r.mesh.topology.numValidFaces():
                w, ok = r.mesh, ok + 1
        info["method"] = "one by one"
        info["chips_cut"] = ok
    info["seconds"] = time.perf_counter() - t0
    return w


def dent_trim(mesh, res, dents, size=1.0, amount=AMOUNT, unit=1.0, cut_faces=None, info=None):
    """The Chip method of Trim Edges, the way the operator runs it: the chips, then Trim Thin.
    (mesh, None); raises MeshCollapsedError when Trim Thin leaves nothing."""
    from .trim_round import trim_thin
    chipped = dent_chip(mesh, dents, size, amount, unit, cut_faces=cut_faces, info=info)
    return trim_thin(chipped, res)
