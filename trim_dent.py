"""Trim Edges, Chip method: the stone worked over the way Alex does it by hand with Ashlarize's tools.

Alex 2026-11-23 ("change it so it instead simulates using the carve brush on sharp edges and corners, and the
stone damage on completely flat faces such as those resulting from a boolean", his Chip_example.blend): the
sharp convex edges get short chisel strokes (Ashlarize Carve's wedge: a V with 30-degree walls and a narrow
floor, its ends lifting out), each leaning at random between the edge's two faces, with gaps between them;
corners get a stroke run out past them; the flat faces a boolean left get the big stone damage dents (Dent_L)
scattered over them, as the Stone Damage brush drops them. All cut with one meshlib boolean, then Trim Thin.

Before 2026-11-23 (`place`, kept: the dents' frames and the sharp edges are shared):

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
AMOUNT = 8.0           # the "Amount" slider: the share of the long edges chiselled is AMOUNT / (AMOUNT + 1), 8: 89 %
#                        (Alex's example covers its main edges end to end)
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


def sharp_edges(m, unit=1.0, cut_faces=None, faces=False):
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
        e = np.zeros(0, int)
        return (z, z, z, z, e, e, V, vn) + ((e, e) if faces else ())
    R = np.array(rows, np.int64)
    out = (V[R[:, 0]], V[R[:, 1]], fn[R[:, 2]], fn[R[:, 3]], R[:, 0], R[:, 1], V, vn)
    return out + ((R[:, 2], R[:, 3]) if faces else ())


CONCAVE_CLEAR = 0.35   # x the chisel (was 1.0: too much of the XL's joint outlines went): no stroke this close to a concave sharp edge that crosses its line (an inside corner,
#                        where two mouldings meet: Alex 2026-11-23 "avoid chipping concave corners / edges") ...
CONCAVE_ALONG = 0.7    # ... crosses: its direction's |cos| to the stroke's under this. A groove running alongside the
#                        arris (an XL arch profile's mouldings, under a chisel width from every edge) does not count:
#                        with those counted the XL lost a third of its strokes (Alex 24 Nov: "it was better before")
CONCAVE_CORNER = 0.25  # x the chisel: a corner this close to a concave edge sits on it: no chip


def concave_points(m, unit=1.0, step=None):
    """Points along the stone's concave sharp edges (inside corners), at most `step` apart, and each one's edge
    direction: (k, 3), (k, 3)."""
    mm, mn = _ml()
    V = np.asarray(mn.getNumpyVerts(m), np.float64)
    cr = m.findCreaseEdges(math.radians(CREASE_DEG))
    bits = np.asarray(mn.getNumpyBitSet(cr), bool)
    topo = m.topology
    out, dirs = [], []
    step = step or 0.25 * CHISEL * unit
    for i in np.flatnonzero(bits):
        e = mm.EdgeId(2 * int(i))
        if not topo.left(e).valid() or not topo.right(e).valid():
            continue
        if m.dihedralAngle(mm.UndirectedEdgeId(int(i))) > 0:      # convex: an edge to chip, not a corner to keep
            continue
        a, b = V[int(topo.org(e).get())], V[int(topo.dest(e).get())]
        ln = float(np.linalg.norm(b - a))
        k = max(1, int(math.ceil(ln / step)))
        out += [a + (b - a) * (j / k) for j in range(k + 1)]
        dirs += [(b - a) / (ln + 1e-12)] * (k + 1)
    return np.array(out).reshape(-1, 3), np.array(dirs).reshape(-1, 3)


def _clear_of(pts, C, lim, along=None, D=None):
    """Per point of pts: farther than lim from every point of C. along (a unit direction) with D (each point of C's own
    direction): only the points of C whose direction crosses `along` count (|cos| under CONCAVE_ALONG)."""
    if not len(C):
        return np.ones(len(pts), bool)
    if along is not None and D is not None:
        C = C[np.abs(D @ np.asarray(along, np.float64)) < CONCAVE_ALONG]
        if not len(C):
            return np.ones(len(pts), bool)
    P = np.asarray(pts, np.float64)
    ok = np.ones(len(P), bool)
    for a in range(0, len(P), 64):
        q = P[a:a + 64]
        ok[a:a + 64] = ((q[:, None, :] - C[None, :, :]) ** 2).sum(-1).min(1) > lim * lim
    return ok


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


# --- Carve strokes on the sharp edges (Alex 2026-11-23) ---------------------------------------------
CHISEL = 0.066         # BU: the chisel's opening at Size 1 (Carve's Standard preset, 0.66 mm)
WIDTH_RANGE = (0.7, 1.3)    # each stroke's chisel, x Size, drawn in this range
WALL_DEG = 30.0        # Carve's walls lean this far from the groove's axis ...
FLOOR_SHARE = 0.6      # ... its floor this share of the opening (wider than Carve's 0.3: drawn along an edge it
#                        leaves a flat chamfer, as in Alex's example, not a dark slot) ...
ABOVE = 0.6            # ... and the cutter runs on straight up this far above the surface, x the opening
STEP_SHARE = 0.35      # the stroke's rings this far apart, x the opening
# (measured 2026-11-23 on Alex's hand strokes, Tna_Stone_006.001, sections every 0.02 BU along its top front edge: the
# cut 0.011 - 0.021 BU deep along the arris' bisector, steady within a stroke, stepping ~30 % from one stroke to the next
# every 0.2 - 0.4 BU, the whole edge covered; its corner chips 0.086 and 0.136 deep, one flat facet each)
STROKE_LEN = (5.0 * 0.7, 5.0 * 1.3)   # a stroke's length, x its opening: 5 +- 30 % (Alex: "vary by 30% the depth/length
#                                       of entire strokes")
STROKE_GAP = (-0.3, 0.3)   # x the opening between one stroke and the next along a run (they run on into each other)
STROKE_DEPTH = 0.25        # how deep a stroke cuts in from the arris, x its opening (0.016 BU at 0.066) ...
STROKE_DEPTH_VAR = (-0.45, 0.30)  # ... this much shallower / deeper, the whole stroke alike (Alex 23 Nov: "the maximum
#                                  depth was good at 30% the minimum depth is what should get more variation")
STROKE_BEND = 0.12         # a stroke is one gentle bend: its middle off the edge's line by up to this x its opening,
#                            sideways, and up to STROKE_BEND_UP x its depth in and out
STROKE_BEND_UP = (-0.40, 0.25)    # (the same way: the deepest swell as before, thinner places thinner)
RUN_MIN = 4.0          # x the opening: a run shorter than this gets no stroke, nor its corners (the facets of a
#                        rough, sculpted side: Alex's example leaves those as they are)
RUN_MIN_FLAT = 1.2     # ... but round a big flat face (a joint face's moulded outline: short steps) down to this, and
#                        its corners always get a stroke or a chip (Alex 23 Nov: "more damage on these corners")
EASE = 0.8             # its ends lift out of the stone over this much, x the opening (Carve's ends)
RUN_TURN = 35.0        # deg: a run of sharp edges is split where it turns more than this (a corner)
STROKE_SINK = (-0.2, 0.05)    # x the depth: a stroke sits this much deeper (-) or shallower along its axis
STROKE_TILT = 8.0      # deg: and its axis is tipped at random by up to this much
CORNER_STROKE = (0.8, 1.6)   # a corner's stroke: this long (x the opening) ...
CORNER_PAST = 0.4      # ... starting this far past the corner
# Alex 2026-11-23 round 2: "more randomness and jitter to them, varying the width and depth as they go along, and hit
# raised corners harder"
JIT_WIDTH = 0.0        # (along a stroke nothing wanders: Alex wants single strokes, varied as a whole)
JIT_DEPTH = 0.0
JIT_SINK = 0.0
JIT_WAVE = (1.2, 4.0)  # over waves this long (x the opening), two of them added
RAISED_EDGES = 3       # a corner where this many sharp convex edges meet stands out: a raised corner ...
RAISED_DEPTH = (4.0, 7.0)   # ... knocked off with one flat chip this many times a stroke's depth deep (Alex's example:
#                             0.086 and 0.136 against 0.011 - 0.021 on the edge) ...
RAISED_DEPTH_FLAT = (2.0, 3.5)  # ... a joint face's corner with one long edge only (a moulded profile's steps): half that
CORNER_THICK_CUT = 0.5  # a thick stone's corner chips this much shallower (Alex 23 Nov: "Corner chipping on large stones max
#                        depth should be reduced 50%"), a thin one's as they are, in proportion between
RAISED_CAP = 1.5       # ... at most this x the depth the stone there allows (was x 4 - 7 too: Alex 23 Nov "cut is too deep"
#                        on a small arch stone's end)
RAISED_TILT = 15.0     # ... its face tipped at random by up to this much
# sizes from the part's own detail (Alex: "automatically work out how much material to remove based on the level of
# detail of each part, replacing the resolution tagging"): meshlib's ray thickness (how much stone stands behind each
# point, measured straight in) at the edge sets that stroke's chisel; the part's thinnest tenth sets Trim Thin's voxel
AUTO_SHARE = 0.2       # a stroke's opening: this share of the stone's thickness at its edge ...
AUTO_RANGE = (0.012, 0.09)   # ... BU, held in here (0.12 - 0.9 mm), then x the Size slider
# Alex 2026-11-23: "the width of the stroke on higher detail model should not change much, rather, the depth should
# vary [not] to cut off too much": the chisel stays Carve's width; the detail holds how deep it may go
# Alex 2026-11-23 (marked picture: "more damage on the top and bottom (large flat face) and less on the mostly vertical
# ones with a lot of detail"; a plain block "not enough chipping and stone damage"): an edge on a big flat face is
# chiselled in full (only the stone behind it holds the depth); an edge in the detail gets fewer, shallower strokes
BIG_FLAT = 0.008       # BU^2: a flat face this big counts as a big flat face (an arch stone's joint face) ...
BIG_FLAT_SHAPE = 0.1   # ... if it is not a long strip: its area at least this x its length squared (the long narrow
#                        strip down an arch stone's back is not: its edges chewed the stone end to end)
DETAIL_KEEP = 0.15     # an edge that does not border a big flat face keeps at most this share of its strokes (Alex:
#                        "less on the mostly vertical ones with a lot of detail": an arch stone's moulded sides), and
DETAIL_KEEP_MIN = 0.05 # an edge in the detail keeps a stroke this often at least: (the detail's depth / a full stroke's)
#                        squared (a moulding's fillets: about 1 stroke in 10; 0.15 linear still chewed the arch stones)
DEPTH_SPACING = 0.25   # a stroke cuts in at most this share of the distance to the next sharp edge (a moulding's
#                        fillets stay) ...
DEPTH_THICK = 0.1      # ... and at most this share of the stone's thickness behind its edge (a cusp tip or a leaf only
#                        gets touched: no hole, no sliver)
SPACING_SHARE = 0.6    # (no longer sets the width) the distance to the next sharp edge (a moulding's fillets lie a
#                        fraction of a mm apart: a chisel sized from the stone behind them cut through the whole profile)
RES_PCT = 50          # (the median thickness: the 10th percentile read the rays from edge vertices, near 0, and sent a
#                       plain 0.09 BU base plate to MINI; thin detail is kept by the deviation test anyway)
RES_SHARE = 0.25       # Trim Thin's voxel at most this share of the part's thin features (10th percentile thickness):
#                        it takes off what is thinner than two voxels
REMESH_TOL = 0.004     # BU (0.04 mm): Auto Detail's remesh keeps 99 % of the part's surface this close (the remesh
#                        alone: the decimation after it has its own bound, DEC_TOL; 0.02 mm sent plain stones to Fine /
#                        Mini and 8x their faces, measured on the Live Pillar row 2026-11-23)
DEC_TOL = 0.01         # BU (0.1 mm; was 0.005, Alex 23 Nov "not decimated enough"): Auto Detail decimates each island as far as this error allows, no face target:
#                        flat faces go down to a few triangles, carved detail keeps what its shape needs (Alex 2026-11-23:
#                        "replace quick infill's resolution and auto-decimation so it knows automatically how much each
#                        part needs"; the presets' face ratios took a leafy abacus's leaves)
DEC_VOX = 1.5          # ... or 1.5 x (was 2: Alex 23 Nov "slightly too much decimation" on a curved arch soffit) the island's own remesh voxel if more (STONE 0.015 BU, FINE 0.006: Alex 23 Nov "not
#                        decimated enough" at 0.005 everywhere; the error goes with the detail the island was found to need)
DEC_ASPECT = 20.0      # the longest / shortest side a decimated triangle may have (meshlib's default 20 keeps big flat
#                        faces in many small triangles: Alex 23 Nov "not decimated enough")


def decimate_flat(m, err):
    """meshlib decimation as far as err (the mesh's units) allows, no face target; long thin triangles allowed up to
    DEC_ASPECT so flat faces go down to a few. Above the parallel size, done in parts and again on one thread if
    the parts' seams fold (as offset_utils.decimate_mesh)."""
    mm, _mn = _ml()
    from .offset_utils import face_count as _fc, PARALLEL_DECIMATE_FACES
    s = mm.DecimateSettings()
    s.maxDeletedFaces = max(0, _fc(m) - 1)
    s.maxError = float(err)
    s.maxTriangleAspectRatio = float(DEC_ASPECT)
    s.packMesh = True
    par = _fc(m) > PARALLEL_DECIMATE_FACES
    backup = mm.Mesh(m) if par else None
    if par:
        s.subdivideParts = 16
    mm.decimateMesh(m, s)
    if par and len(mm.findSelfCollidingTriangles(mm.MeshPart(m))) > 0:
        s.subdivideParts = 1
        mm.decimateMesh(backup, s)
        return backup
    return m
DETAIL_TOL = 0.005      # BU (0.05 mm, the finest detail Alex works at): a resolution keeps a part's detail when,
#                        remeshed and decimated at it, its surface stays this close (99 % of its points). (Every part
#                        is 0.03 - 0.07 mm off at Stone from the voxels alone; 0.02 mm sent plain stones to Mini.)


def depth_of(W):
    """The groove's depth for opening W: Carve's V (30-degree walls down to a floor FLOOR_SHARE wide)."""
    return (0.5 * W - 0.5 * FLOOR_SHARE * W) / math.tan(math.radians(WALL_DEG))


def runs(p0, p1, v0, v1, nl, nr):
    """The sharp edges chained into runs: [(points (n,3), face normals left (n-1,3), right (n-1,3))], split where a
    run turns more than RUN_TURN (each piece nearly straight, its ends the corners)."""
    adj = {}
    pos = {}
    for k in range(len(v0)):
        adj.setdefault(int(v0[k]), []).append((k, int(v1[k])))
        adj.setdefault(int(v1[k]), []).append((k, int(v0[k])))
        pos[int(v0[k])] = p0[k]
        pos[int(v1[k])] = p1[k]
    used = np.zeros(len(v0), bool)
    chains = []

    def walk(start):
        pts, L, R, ids = [pos[start]], [], [], [start]
        v = start
        while True:
            nxt = [(k, w) for k, w in adj[v] if not used[k]]
            if not nxt or (len(adj[v]) > 2 and v != start):
                break
            k, w = nxt[0]
            used[k] = True
            pts.append(pos[w])
            ids.append(w)
            L.append(nl[k])
            R.append(nr[k])
            v = w
            if v == start:
                break
        return pts, L, R, ids

    ends = [v for v, a in adj.items() if len(a) != 2]
    for v in ends + list(adj.keys()):
        while any(not used[k] for k, _w in adj[v]):
            pts, L, R, ids = walk(v)
            if len(pts) >= 2:
                chains.append((np.array(pts), np.array(L), np.array(R), np.array(ids)))
    pieces = []
    cos_t = math.cos(math.radians(RUN_TURN))
    for P, L, R, I in chains:
        cut = [0]
        for i in range(1, len(P) - 1):
            a, b = P[i] - P[i - 1], P[i + 1] - P[i]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 1e-12 and nb > 1e-12 and float(a @ b) / (na * nb) < cos_t:
                cut.append(i)
        cut.append(len(P) - 1)
        for a, b in zip(cut[:-1], cut[1:]):
            if b > a:
                pieces.append((P[a:b + 1], L[a:b], R[a:b], I[a:b + 1]))
    return pieces


def _along(P):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])


def _at(P, s, t):
    """The point at arc length t along P (s its arc lengths), straight on past the ends, and the segment index."""
    if t < 0.0:
        d = P[1] - P[0]
        return P[0] + d / (np.linalg.norm(d) + 1e-12) * t, 0
    if t > s[-1]:
        d = P[-1] - P[-2]
        return P[-1] + d / (np.linalg.norm(d) + 1e-12) * (t - s[-1]), len(P) - 2
    k = int(min(np.searchsorted(s, t, side="right") - 1, len(P) - 2))
    f = (t - s[k]) / max(s[k + 1] - s[k], 1e-12)
    return P[k] * (1 - f) + P[k + 1] * f, k


def _waves(rng, W):
    """A smooth wobble along a stroke: f(s) in about -1..1, two waves of JIT_WAVE x W, random phases."""
    ks = [2.0 * math.pi / (rng.uniform(*JIT_WAVE) * W) for _ in range(2)]
    ph = [rng.uniform(0, 2 * math.pi) for _ in range(2)]
    return lambda s: 0.6 * math.sin(ks[0] * s + ph[0]) + 0.4 * math.sin(ks[1] * s + ph[1])


def stroke_cutter(pts, up, W, rng=None):
    """Carve's wedge swept along pts (straight-ish), `up` out of the stone: a closed (V, F) solid. The opening W at
    the surface, 30-degree walls down to a floor FLOOR_SHARE wide, the walls run on ABOVE x W over the surface; its
    two ends lift out over EASE x W (the groove fades out as a chisel stroke does). rng: the opening, the depth and the
    stroke's height wander along it (JIT_*), as a hand-held chisel does."""
    P = np.asarray(pts, np.float64)
    if rng is not None:
        fw, fd, fs = _waves(rng, W), _waves(rng, W), _waves(rng, W)
    else:
        fw = fd = fs = (lambda _s: 0.0)
    s = _along(P)
    L = float(s[-1])
    D = depth_of(W)
    f = 0.5 * FLOOR_SHARE * W
    a = 0.5 * W
    H = ABOVE * W
    slope = (a - f) / D                     # across per unit of height
    rings = []
    for i, p in enumerate(P):
        e = min(s[i], L - s[i]) / max(EASE * W, 1e-12)
        k = min(1.0, max(0.12, e))
        k = k * k * (3 - 2 * k)
        d = D * k * (1.0 + JIT_DEPTH * fd(s[i]))       # (the floor rises to the surface at the ends)
        t = P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)]
        t = t / (np.linalg.norm(t) + 1e-12)
        u = up - t * float(up @ t)
        u = u / (np.linalg.norm(u) + 1e-12)
        x = np.cross(t, u)
        fx = (f * k + 1e-3 * W) * (1.0 + JIT_WIDTH * fw(s[i]))
        q = p + u * (JIT_SINK * D * k * fs(s[i]))
        xs = [(-fx, -d), (fx, -d), (fx + slope * (d + H), H), (-fx - slope * (d + H), H)]
        rings.append([q + x * cx + u * cy for cx, cy in xs])
    R = np.array(rings)                      # (n, 4, 3)
    n = len(R)
    V = R.reshape(-1, 3)
    F = []
    for i in range(n - 1):
        for j in range(4):
            a0, a1 = i * 4 + j, i * 4 + (j + 1) % 4
            b0, b1 = a0 + 4, a1 + 4
            F += [(a0, b0, b1), (a0, b1, a1)]
    F += [(0, 1, 2), (0, 2, 3)]               # (the caps wound against the sides: one closed solid)
    e0 = (n - 1) * 4
    F += [(e0, e0 + 2, e0 + 1), (e0, e0 + 3, e0 + 2)]
    F = np.array(F, np.int64)
    vol = float(np.einsum('ij,ij->i', V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum())
    if vol < 0:
        F = F[:, ::-1]
    return V.astype(np.float32), F.astype(np.int32)


CHAMFER_REACH = (1.5, 6.0)  # x the depth: how far across the chamfer plane reaches either side (from the arris' angle)
CHAMFER_EASE = (1.0 / 3.0, 4.0)  # the cut fades in / out over this share of the stroke, at most this x its depth (was
#                                  0.25 / 3: Alex 23 Nov "make the exit/entry angle shallower by 25 percent")
CHAMFER_ABOVE = 2.0    # x the depth: how far out past the arris the cutter stands


def chamfer_cutter(pts, up, d, turn, rng):
    """One stroke as Carve's Ctrl (Flatten) cut across an arris (Alex 2026-11-23: "I created my cuts holding ctrl":
    a flat chamfer, never a groove with a floor that leaves a ledge): everything past a plane `d` in from the arris,
    square to `up` (out of the stone, between its two faces), over the stroke; the depth eases in and out over its
    ends (EASE x its length share) and swells or thins once along it (STROKE_BEND_UP); turn: the arris' angle in deg,
    how far across the plane must reach. A closed (V, F) solid."""
    P = np.asarray(pts, np.float64)
    s = _along(P)
    L = float(s[-1])
    X = d * float(np.clip(1.3 / math.tan(math.radians(max(10.0, min(170.0, turn)) / 2.0)), *CHAMFER_REACH))
    H = CHAMFER_ABOVE * d + 1e-3 * d
    lift = rng.uniform(*STROKE_BEND_UP)
    ease = max(min(CHAMFER_EASE[0] * L, CHAMFER_EASE[1] * d), 1e-9)
    rings = []
    for i, p in enumerate(P):
        e = min(s[i], L - s[i]) / ease
        k = min(1.0, max(0.05, e))
        k = k * k * (3 - 2 * k)
        dk = d * k * (1.0 + lift * math.sin(math.pi * s[i] / max(L, 1e-12)))
        t = P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)]
        t = t / (np.linalg.norm(t) + 1e-12)
        u = up - t * float(up @ t)
        u = u / (np.linalg.norm(u) + 1e-12)
        x = np.cross(t, u)
        xs = [(-X, -dk), (X, -dk), (X, H), (-X, H)]
        rings.append([p + x * cx + u * cy for cx, cy in xs])
    R = np.array(rings)
    n = len(R)
    V = R.reshape(-1, 3)
    F = []
    for i in range(n - 1):
        for j in range(4):
            a0, a1 = i * 4 + j, i * 4 + (j + 1) % 4
            b0, b1 = a0 + 4, a1 + 4
            F += [(a0, b0, b1), (a0, b1, a1)]
    F += [(0, 1, 2), (0, 2, 3)]
    e0 = (n - 1) * 4
    F += [(e0, e0 + 2, e0 + 1), (e0, e0 + 3, e0 + 2)]
    F = np.array(F, np.int64)
    vol = float(np.einsum('ij,ij->i', V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum())
    if vol < 0:
        F = F[:, ::-1]
    return V.astype(np.float32), F.astype(np.int32)


def stroke_depth(W, rng, cap=None, scale=1.0):
    """A stroke's depth: STROKE_DEPTH x W x scale, varied by STROKE_DEPTH_VAR for the whole stroke, held by cap."""
    d = STROKE_DEPTH * W * float(scale) * (1.0 + rng.uniform(*STROKE_DEPTH_VAR))
    if cap is not None and np.isfinite(cap):
        d = min(d, float(cap) * (1.0 + rng.uniform(STROKE_DEPTH_VAR[0], 0.0)))
    return d


def coverage(amount):
    """The share of each run the strokes go over, from the Amount slider: 2.5 (the default) about 70 %."""
    a = max(0.0, float(amount))
    return a / (a + 1.0)


def edge_thickness(m):
    """meshlib's ray thickness at each vertex (how much stone stands behind it, straight in), in the mesh's units."""
    mm, _mn = _ml()
    t = np.array(mm.computeRayThicknessAtVertices(m).vec, float)
    return np.where(np.isfinite(t) & (t > 0), t, np.nan)


def thickness_in(m, p, inward, unit=1.0):
    """How much stone stands behind point p, straight in along `inward` (one ray, cast from just inside p so it does not
    hit the faces at p: meshlib's ray thickness AT an edge's own vertices reads near 0). inf when it finds nothing."""
    mm, _mn = _ml()
    d = np.asarray(inward, np.float64)
    d = d / (np.linalg.norm(d) + 1e-12)
    o = np.asarray(p, np.float64) + d * (1e-3 * unit)
    hit = mm.rayMeshIntersect(mm.MeshPart(m), mm.Line3f(mm.Vector3f(*map(float, o)), mm.Vector3f(*map(float, d))))
    if not hit:
        return np.inf
    t = float(hit.distanceAlongLine)
    try:                                     # (p out past a rounded arris: the first hit is going in; measure from there)
        fn = m.normal(hit.proj.face)
        if float(fn.x * d[0] + fn.y * d[1] + fn.z * d[2]) < 0.0:
            o2 = o + d * (t + 1e-3 * unit)
            h2 = mm.rayMeshIntersect(mm.MeshPart(m), mm.Line3f(mm.Vector3f(*map(float, o2)),
                                                                 mm.Vector3f(*map(float, d))))
            return float(h2.distanceAlongLine) + 1e-3 * unit if h2 else np.inf
    except Exception:
        pass
    return t + 1e-3 * unit


def auto_width(th, unit, size):
    """A stroke's opening from the stone's thickness at its edge (AUTO_SHARE, held in AUTO_RANGE), x Size."""
    w = AUTO_SHARE * float(th) if np.isfinite(th) else CHISEL * unit
    return float(np.clip(w, AUTO_RANGE[0] * unit, AUTO_RANGE[1] * unit)) * float(size)


def run_spacing(pieces, cap):
    """Per run: how far the other sharp edges are from it (the median over its points of the nearest point of another
    run; near the corners it shares with them that is small, so the median, not the least), at most cap. Looked up in a
    grid of cap-sized cells, so only nearby edges are compared (the leafy parts have thousands)."""
    pts = [np.asarray(P, np.float64) for P, _l, _r, _i in pieces]
    allp = np.concatenate(pts)
    lab = np.concatenate([np.full(len(P), r) for r, P in enumerate(pts)])
    # (runs that share a vertex are one edge going on round a bend, or the edges of one corner: not "the next edge")
    near = [set() for _ in pieces]
    owner = {}
    for r, (_P, _l, _r, I) in enumerate(pieces):
        for v in (int(I[0]), int(I[-1])):
            owner.setdefault(v, []).append(r)
    for rs in owner.values():
        for a in rs:
            near[a].update(rs)
    h = max(float(cap), 1e-9)
    key = np.floor(allp / h).astype(np.int64)
    cells = {}
    for i, k in enumerate(map(tuple, key)):
        cells.setdefault(k, []).append(i)
    cells = {k: np.array(v) for k, v in cells.items()}
    offs = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
    out = np.full(len(pts), float(cap))
    start = 0
    for r, P in enumerate(pts):
        ks = {tuple(k) for k in key[start:start + len(P)]}
        start += len(P)
        cand = [cells[(k[0] + a, k[1] + b, k[2] + c)] for k in ks for a, b, c in offs
                if (k[0] + a, k[1] + b, k[2] + c) in cells]
        if not cand:
            continue
        cand = np.unique(np.concatenate(cand))
        cand = cand[~np.isin(lab[cand], list(near[r] | {r}))]
        if not len(cand):
            continue
        Q = allp[cand]
        d = np.empty(len(P))
        for a in range(0, len(P), 256):
            q = P[a:a + 256]
            d[a:a + 256] = np.sqrt(((q[:, None, :] - Q[None, :, :]) ** 2).sum(-1)).min(1)
        out[r] = min(float(cap), float(np.median(d)))
    return out


def bent(pts, up, W, rng, cap=None, scale=1.0):
    """A stroke's line, set so it cuts STROKE_DEPTH x W in (+- STROKE_DEPTH_VAR, the whole stroke alike) and bent once,
    gently (one arc, its middle up to STROKE_BEND x W sideways and STROKE_BEND_UP x its depth in or out)."""
    P = np.asarray(pts, np.float64)
    d = STROKE_DEPTH * W * float(scale) * (1.0 + rng.uniform(*STROKE_DEPTH_VAR))
    if cap is not None and np.isfinite(cap):                 # (held by the detail there, still varied +-30 %)
        d = min(d, float(cap) * (1.0 + rng.uniform(STROKE_DEPTH_VAR[0], 0.0)))
    t = P[-1] - P[0]
    t = t / (np.linalg.norm(t) + 1e-12)
    x = np.cross(t, up)
    x = x / (np.linalg.norm(x) + 1e-12)
    s = _along(P)
    u = s / max(s[-1], 1e-12)
    arc = np.sin(np.pi * u)[:, None]
    side = rng.uniform(-STROKE_BEND, STROKE_BEND) * W
    lift = rng.uniform(*STROKE_BEND_UP) * d
    return P + up * (depth_of(W) - d) + arc * (x * side + up * lift)


def corner_chip(c, n, depth, rng, sides=16):
    """A raised corner knocked off with one flat chip (Alex's example, the red circle): a cone whose flat base lies
    `depth` in from the corner across its normal n (tipped by up to RAISED_TILT), CHIP_RADIUS x depth wide, its point
    CHIP_HEIGHT x depth above the base. A sharp corner loses one flat facet; on a blunt corner the narrowing sides only
    take a shallow round dimple (a square slab there cut square steps into the stone). (V, F)."""
    z = _tilt(np.asarray(n, np.float64), rng, RAISED_TILT)
    z = z / (np.linalg.norm(z) + 1e-12)
    B = _frame(z, rng)
    o = np.asarray(c, np.float64) - z * depth
    a = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    ring = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], 1) * (CHIP_RADIUS * depth)
    V = np.vstack([ring @ B.T + o, o[None], (o + z * (CHIP_HEIGHT * depth))[None]])
    cb, ct = sides, sides + 1
    F = []
    for i in range(sides):
        j = (i + 1) % sides
        F.append((cb, j, i))                    # (the flat base, facing into the stone)
        F.append((i, j, ct))                    # (the sides up to the point)
    F = np.array(F, np.int64)
    vol = float(np.einsum('ij,ij->i', V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum())
    if vol < 0:
        F = F[:, ::-1]
    return V.astype(np.float32), F.astype(np.int32)


RAISED_SHARP = 105.0   # deg: a raised corner's edges meet at angles averaging under this (a cube's corner: 90); a blunt
#                        corner gets no chip (a round scoop there, not a chip)
CHIP_VOLUME = 3.0      # a corner chip may take at most this x what a square corner's chip takes (0.87 depth^3): stone
#                        corners are blunter than a cube's (2.0 let 1 corner in 15 through on Alex's Tna_Stone_006)
CHIP_RADIUS = 1.6      # the chip's flat face this x its depth wide (a square corner's facet: 1.41 x)
CHIP_HEIGHT = 3.0      # its cone's point this x its depth above the face


def standing(dirs):
    """The corner's edges (unit directions away from it) meet sharply enough to stand out: their angles to each other
    average under RAISED_SHARP."""
    a = [math.degrees(math.acos(max(-1.0, min(1.0, float(dirs[i] @ dirs[j])))))
         for i in range(len(dirs)) for j in range(i + 1, len(dirs))]
    return bool(a) and float(np.mean(a)) < RAISED_SHARP


def chip_too_big(m, chip, depth):
    """The flat chip would take more than CHIP_VOLUME x a square corner's chip out of the stone (the stone rises round
    that corner, or its normal points along a face): meshlib boolean Intersection, its volume."""
    mm, mn = _ml()
    try:
        c = mn.meshFromFacesVerts(np.ascontiguousarray(chip[1]), np.ascontiguousarray(chip[0]))
        r = mm.boolean(m, c, mm.BooleanOperation.Intersection)
        if not r.valid():
            return True
        return abs(r.mesh.volume()) > CHIP_VOLUME * 0.87 * depth ** 3
    except Exception:
        return True


FAMILY_DEG = 20.0      # deg: a flat face beside another flat face this close in direction is a facet of a curved side
#                        (an arch stone's faceted soffit), not a big flat face ...
FAMILY_MIN = 3.0       # ... turned more than this (parallel ones are a step, not a curve) ...
FAMILY_SIZE = 3        # ... in a chain of at least this many (a joint face in two planes is still a joint face)
ARRIS_GAP = 1.5        # x the chisel: two big flat faces whose outlines come this close meet in an arris (round a
#                        rounded or bevelled block edge: no sharp edge there at all, the fillet's steps are under 25 deg)
ARRIS_ANGLE = (45.0, 135.0)  # deg between their directions
ARRIS_STEP = 1.0       # x the gap: an outline point at least this often along the line, on both faces (rim vertices of
#                        a straight edge can be far apart: the line is sampled, the outline's own points are not)


def _rim_verts(F):
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    u, c = np.unique(e, axis=0, return_counts=True)
    return np.unique(u[c == 1])


def facet_family(V, F, regs, gap, groups=False):
    """Per flat region: True if it lies beside (outlines within `gap`) another flat region turned under FAMILY_DEG
    from it: one facet of a faceted curved side."""
    rv = [V[_rim_verts(F[reg])] for reg, _n, _A in regs]
    par = list(range(len(regs)))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for a in range(len(regs)):
        for b in range(a + 1, len(regs)):
            if not (FAMILY_MIN < _angle(np.asarray(regs[a][1]), np.asarray(regs[b][1])) < FAMILY_DEG):
                continue
            A_, B_ = rv[a], rv[b]
            if np.any(A_.min(0) - gap > B_.max(0)) or np.any(B_.min(0) - gap > A_.max(0)):
                continue
            if np.sqrt(((A_[:, None, :] - B_[None, :, :]) ** 2).sum(-1)).min() < gap:   # (side by side)
                par[find(a)] = find(b)
    size = {}
    for i in range(len(regs)):
        size[find(i)] = size.get(find(i), 0) + 1
    flags = [size[find(i)] >= FAMILY_SIZE for i in range(len(regs))]
    if groups:
        g = {}
        for i in range(len(regs)):
            if flags[i]:
                g.setdefault(find(i), []).append(i)
        return list(g.values())
    return flags


def arrises(V, F, regs, gap, nv):
    """The arrises where two big flat faces meet (their planes' line, over the stretch where both outlines run within
    `gap` of it), as runs [(points, normals left, right, ids)]: a block's edge whether modelled sharp, bevelled or
    rounded. ids are made-up vertex numbers past the mesh's own (nv), two per arris (-1 between)."""
    rims, pl = [], []
    for reg, n, _A in regs:
        Fr = F[reg]
        e = np.sort(np.concatenate([Fr[:, [0, 1]], Fr[:, [1, 2]], Fr[:, [2, 0]]]), axis=1)
        u, c = np.unique(e, axis=0, return_counts=True)
        rim = u[c == 1]                              # (the outline's edges, sampled: a straight one has two ends only)
        pts = [_sample(V[[a, b]], 0.5 * gap) for a, b in rim]
        n = np.asarray(n, np.float64)
        n = n / (np.linalg.norm(n) + 1e-12)
        rims.append(np.concatenate(pts) if pts else np.zeros((0, 3)))
        pl.append((n, float(np.mean(V[F[reg]].reshape(-1, 3) @ n))))
    out = []
    lo, hi = ARRIS_ANGLE
    for i in range(len(regs)):
        for j in range(i + 1, len(regs)):
            ni, di = pl[i]
            nj, dj = pl[j]
            if not (lo <= _angle(ni, nj) <= hi):
                continue
            Ri, Rj = rims[i], rims[j]
            if not len(Ri) or not len(Rj):
                continue
            if np.any(Ri.min(0) - gap > Rj.max(0)) or np.any(Rj.min(0) - gap > Ri.max(0)):
                continue
            d = np.cross(ni, nj)
            d /= np.linalg.norm(d) + 1e-12
            p0 = np.linalg.solve(np.array([ni, nj, d]), np.array([di, dj, 0.0]))   # (on both planes)
            near = []
            for R in (Ri, Rj):
                q = R - p0
                off = q - np.outer(q @ d, d)          # (from the line, across it)
                near.append(np.sort((q @ d)[np.linalg.norm(off, axis=1) < gap]))
            if not len(near[0]) or not len(near[1]):
                continue
            t0 = max(float(near[0][0]), float(near[1][0]))
            t1 = min(float(near[0][-1]), float(near[1][-1]))
            if t1 - t0 < gap:
                continue
            # (only where both outlines follow the line: over a moulding's notch the line runs through air, and a stroke
            # along it would cut into the moulding)
            step = 0.5 * gap
            ts = np.arange(t0, t1 + 1e-9, step)
            cov = np.ones(len(ts), bool)
            for nr_ in near:
                k_ = np.clip(np.searchsorted(nr_, ts), 1, len(nr_) - 1)
                dist = np.minimum(np.abs(nr_[k_] - ts), np.abs(nr_[k_ - 1] - ts))
                cov &= dist <= ARRIS_STEP * gap
            runs_ = []
            a_ = None
            for i_, c_ in enumerate(list(cov) + [False]):
                if c_ and a_ is None:
                    a_ = i_
                elif not c_ and a_ is not None:
                    runs_.append((ts[a_], ts[i_ - 1]))
                    a_ = None
            for s0, s1 in runs_:
                if s1 - s0 < gap:
                    continue
                k = max(2, int(math.ceil((s1 - s0) / step)) + 1)
                P = p0 + np.outer(np.linspace(s0, s1, k), d)
                ids = np.array([nv + 2 * len(out)] + [-1] * (k - 2) + [nv + 2 * len(out) + 1])
                out.append((P, np.tile(ni, (k - 1, 1)), np.tile(nj, (k - 1, 1)), ids))
    return out
BAND_GAP = 1.0         # x the chisel: parallel sharp edges this close (over half their length) are one bevelled arris
#                        (a block's 90-degree edge modelled as three 30-degree ones 0.03 - 0.1 BU apart), cut as one
BAND_PAR = 0.9         # (parallel: directions this close, |cos|)
BAND_MAX = 110.0       # deg: not if the band would turn more than this (a bevelled block edge turns about 90; a rounded
#                        bead on a moulding chained its facets round past 160 and was cut as a knife edge, leaving a step
#                        and a sliver: Alex 23 Nov)
HARD_ANGLE = (30.0, 90.0)  # deg an edge turns (bevels joined): from the first to the second it is hit harder ...
HARD_DEPTH = (0.6, 1.0)    # ... its strokes this x deeper (Alex 23 Nov: "edges around 90 degree should be hit the hardest")
HARD_KEEP = (0.5, 1.0)     # ... and this x more often
CORNER_JOIN = 1.0      # x the chisel: run ends this close together are one corner (a bevelled block's corner: three
#                        bands ending round a little rounded patch, no vertex where they meet)


def _sample(P, step):
    """Points along P at most `step` apart (a straight edge has only its two ends)."""
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        k = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / step)))
        out += [a + (b - a) * (i / k) for i in range(1, k + 1)]
    return np.array(out)


def _angle(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b)))))


def bands(pieces, on_flat, gap, nv=None):
    """Bevelled arrises joined: runs that lie parallel within `gap` of each other over at least half the shorter one,
    and turn no more than BAND_MAX together, are one band, cut once along its longest run with the band's outer face
    normals. Returns (pieces, on_flat, angle per piece in degrees, how many runs were joined, per piece: a bevelled or
    rounded arris (joined, or an arris between flat faces: made-up vertex ids from nv on))."""
    n = len(pieces)
    if n == 0:
        return pieces, on_flat, [], 0, []
    S = [_sample(np.asarray(P, np.float64), 0.5 * gap) for P, _l, _r, _i in pieces]
    D = []
    for P, _l, _r, _i in pieces:
        d = np.asarray(P[-1], np.float64) - np.asarray(P[0], np.float64)
        D.append(d / (np.linalg.norm(d) + 1e-12))
    nrm = []
    for _P, L, R, _i in pieces:
        a = np.asarray(L, np.float64).mean(0)
        b = np.asarray(R, np.float64).mean(0)
        nrm.append((a / (np.linalg.norm(a) + 1e-12), b / (np.linalg.norm(b) + 1e-12)))
    allp = np.concatenate(S)
    lab = np.concatenate([np.full(len(s), r) for r, s in enumerate(S)])
    key = np.floor(allp / gap).astype(np.int64)
    cells = {}
    for i, k in enumerate(map(tuple, key)):
        cells.setdefault(k, []).append(i)
    par = list(range(n))
    gnorm = [list(nrm[r]) for r in range(n)]  # (each band's face normals so far)

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    offs = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
    start = 0
    for r in range(n):
        ks = {tuple(k) for k in key[start:start + len(S[r])]}
        start += len(S[r])
        cand = set()
        for k in ks:
            for a, b, c in offs:
                cand.update(lab[cells.get((k[0] + a, k[1] + b, k[2] + c), [])].tolist())
        for q in cand:
            if q <= r or find(q) == find(r) or abs(float(D[r] @ D[q])) < BAND_PAR:
                continue
            A, B = (S[r], S[q]) if len(S[r]) <= len(S[q]) else (S[q], S[r])
            dd = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min(1)
            if float(np.mean(dd < gap)) < 0.5:
                continue
            ra, rq = find(r), find(q)
            ns = gnorm[ra] + gnorm[rq]               # (the whole band as it would be: a rounded bead's facets, each
            N_ = np.array(ns)                        # step small, chain round to a knife's turn)
            if math.degrees(math.acos(max(-1.0, min(1.0, float((N_ @ N_.T).min()))))) > BAND_MAX:
                continue
            par[rq] = ra
            gnorm[ra] = ns
    groups = {}
    for r in range(n):
        groups.setdefault(find(r), []).append(r)
    out, flat, ang, bev = [], [], [], []
    joined = 0
    for g in groups.values():
        ns = [x for r in g for x in nrm[r]]
        best = (0.0, ns[0], ns[-1])
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                a = _angle(ns[i], ns[j])
                if a > best[0]:
                    best = (a, ns[i], ns[j])
        rep = max(g, key=lambda r: float(_along(pieces[r][0])[-1]))
        P, L, R, I = pieces[rep]
        if len(g) > 1:
            joined += len(g) - 1
            L = np.tile(best[1], (len(L), 1))
            R = np.tile(best[2], (len(R), 1))
        out.append((P, L, R, I))
        flat.append(any(on_flat[r] for r in g))
        ang.append(best[0])
        bev.append(len(g) > 1 or (nv is not None and any(int(pieces[r][3][0]) >= nv for r in g)))
    return out, flat, ang, joined, bev


SIZE_DEPTH = ((0.2, 1.0), (0.4, 1.25))  # (BU median thickness, x depth): a stone this thick or less keeps the depth, one
#                                         this thick or more cuts 25 % deeper, between in proportion (Alex 23 Nov: the
#                                         thin arch profiles "are good but the XL profile needs ... 25% (relative) more
#                                         depth"; thin ones 0.17 - 0.19 BU, the XL 0.41)


THIN_KEEP = 0.75           # x: every stroke of a thin stone kept this often, a thick one's always (Alex 23 Nov: the S arch
#                            ring, seven stones each chipped at both ends, looked busy: "lower it by 25% (relative)")
THIN_DETAIL_KEEP = 0.3     # x: a thin stone's edges in its detail (not along a flat or joint face) keep this share of
#                            their strokes, a thick one all (Alex 23 Nov: "reduce the frequency of cuts on the detailed
#                            parts of the thin profile 70% (relative)")


def thick_share(k_size):
    """0 for a thin stone, 1 for a thick one (from its SIZE_DEPTH factor)."""
    (_t0, k0), (_t1, k1) = SIZE_DEPTH
    return float(np.clip((k_size - k0) / max(k1 - k0, 1e-9), 0.0, 1.0))


def size_depth(th):
    """The depth factor for a stone of median thickness th (BU; None: 1)."""
    if th is None or not np.isfinite(th):
        return 1.0
    (t0, k0), (t1, k1) = SIZE_DEPTH
    return float(k0 + (k1 - k0) * np.clip((th - t0) / (t1 - t0), 0.0, 1.0))


SQUARE_SPAN = 30.0     # deg: an arris within this of 90 deg keeps its strokes in the detail too, fully at 90 (Alex 23 Nov:
#                        "more chipping on the thin arch's 90 degree edges": they border its curved sides, not a big flat
#                        face, and kept 15 %)
KNIFE_DEG = 120.0      # deg: an arris turning this much or more is a knife edge: every stroke, at the normal depth, the
#                        thickness behind it not holding it back (Alex 23 Nov: "knife thin edges should likely get trimmed
#                        off at normal depth")


def square(angle):
    """1 at a 90-degree arris, down to 0 SQUARE_SPAN either side."""
    return float(np.clip(1.0 - abs(float(angle) - 90.0) / SQUARE_SPAN, 0.0, 1.0))


def _hard(angle, rng_):
    """HARD_DEPTH / HARD_KEEP at an edge turning `angle` degrees."""
    h = float(np.clip((angle - HARD_ANGLE[0]) / (HARD_ANGLE[1] - HARD_ANGLE[0]), 0.0, 1.0))
    return rng_[0] + (rng_[1] - rng_[0]) * h


def place_strokes(m, size=1.0, amount=AMOUNT, unit=1.0, seed=0, cut_faces=None, info=None, auto=False):
    """The chisel strokes for mesh m's sharp edges and corners: [(V, F)] in the mesh's units. Each run of sharp edges
    is walked from end to end in strokes of STROKE_LEN with STROKE_GAP between, each one kept coverage(amount) of the
    time; corners: a stroke run out past them CORNER_CHANCE of the time (not with Amount 0); raised corners (RAISED_EDGES
    sharp edges) always, harder, RAISED_STROKES of them. auto: each stroke's chisel from the stone's thickness at its edge
    (auto_width), else CHISEL x size."""
    info = {} if info is None else info
    rng = np.random.default_rng(seed)
    p0, p1, nl, nr, v0, v1, V, vn, fa, fb = sharp_edges(m, unit, cut_faces, faces=True)
    info["sharp_edges"] = int(len(p0))
    pieces = runs(p0, p1, v0, v1, nl, nr) if len(p0) else []
    # (which runs lie along a big flat face: its sharp edges have one)
    _mm, _mn = _ml()
    VV = np.asarray(_mn.getNumpyVerts(m), np.float64)
    FF = np.asarray(_mn.getNumpyFaces(m.topology), np.int64)
    regs, _ar = flat_all(VV, FF, unit, info)                 # (the regions down to BIG_FLAT)
    bigf = np.zeros(len(FF), bool)
    big = []
    curved = facet_family(VV, FF, regs, ARRIS_GAP * CHISEL * float(size) * unit)
    for i, (reg, _n, A) in enumerate(regs):
        ext = float(np.ptp(VV[np.unique(FF[reg])], axis=0).max())
        if A >= BIG_FLAT * unit * unit and A >= BIG_FLAT_SHAPE * ext * ext and not curved[i]:
            bigf[reg] = True
            big.append(i)
    W0_ = CHISEL * float(size) * unit
    ar_pieces = arrises(VV, FF, [regs[i] for i in big], ARRIS_GAP * W0_, len(VV))
    info["arrises"] = len(ar_pieces)
    eflat = {}
    for k in range(len(v0)):
        eflat[(min(int(v0[k]), int(v1[k])), max(int(v0[k]), int(v1[k])))] = bool(
            (fa[k] < len(bigf) and bigf[fa[k]]) or (fb[k] < len(bigf) and bigf[fb[k]]))
    on_flat = []
    for _P, _l, _r, I in pieces:
        fl = [eflat.get((min(int(a), int(b)), max(int(a), int(b))), False) for a, b in zip(I[:-1], I[1:])]
        on_flat.append(bool(fl) and float(np.mean(fl)) >= 0.5)
    pieces = list(pieces) + ar_pieces          # (the arrises between big flat faces: the band their bevel's runs join)
    on_flat += [True] * len(ar_pieces)
    info["runs_on_big_flat"] = int(sum(on_flat))
    vflat = set(np.unique(FF[bigf]).tolist()) if bigf.any() else set()
    pieces, on_flat, angs, nj, bevel = bands(pieces, on_flat, BAND_GAP * CHISEL * float(size) * unit, len(VV))
    info["bevel_runs_joined"] = nj
    lens = np.array([float(_along(P)[-1]) for P, _l, _r, _i in pieces])
    TH = edge_thickness(m) if auto else None
    # (a thick stone's cuts a little deeper: the stone's median thickness, from SIZE_DEPTH)
    th_med = float(np.nanmedian(TH)) / unit if TH is not None and np.isfinite(TH).any() else None
    k_size = size_depth(th_med)
    info["size_depth"] = round(k_size, 3)
    k_corner = 1.0 - CORNER_THICK_CUT * thick_share(k_size)   # (a big stone's corner chips: shallower)
    SP = run_spacing(pieces, AUTO_RANGE[1] * unit / SPACING_SHARE) if auto else None
    vrun = {}                                # vertex -> the runs that end there (the corners' chisel)
    for r, (_P, _l, _r, I) in enumerate(pieces):
        for v in (int(I[0]), int(I[-1])):
            vrun.setdefault(v, []).append(r)
    if not len(pieces) or lens.sum() <= 0:
        return []
    total = float(lens.sum()) / unit
    info["edge_length"] = total
    out = []
    keep = coverage(amount)
    jobs = []                                # (run, start, end, chisel)
    W0 = CHISEL * float(size) * unit
    caps = []                                # per run: how deep its strokes may go (auto: the detail there)

    def depth_cap(TH_, SP_, r_, I_):
        if TH_ is None:
            return np.inf
        P_, L_, R_, _I = pieces[r_]
        k_ = len(L_) // 2                      # (the stone behind the run's middle, along its bisector)
        th = thickness_in(m, 0.5 * (P_[k_] + P_[k_ + 1]), -(L_[k_] + R_[k_]), unit)
        if on_flat[r_]:                        # (along a big flat face: the stone behind it only)
            return DEPTH_THICK * float(th)
        return min(DEPTH_SPACING * float(SP_[r_]), DEPTH_THICK * float(th))
    widths = []
    long_ends = []                           # (the ends of the runs long enough: the corners that get a stroke)
    end_run = []                             # (which run each of those ends is)
    for r, (P, Ln, Rn, I) in enumerate(pieces):
        L = lens[r]
        caps.append(np.inf if angs[r] >= KNIFE_DEG else depth_cap(TH, SP, r, I))   # (a knife edge: cut off at the
        #                                       normal depth, not held by the little stone behind it)
        if L < (RUN_MIN_FLAT if on_flat[r] else RUN_MIN) * W0:
            continue
        widths.append(W0 / unit)
        long_ends += [P[0], P[-1]]
        end_run += [r, r]
        t = -rng.uniform(0.0, 0.5) * W0      # (a stroke may start just past the run's end: over the corner)
        while t < L:
            W = W0 * rng.uniform(*WIDTH_RANGE)
            ln = rng.uniform(*STROKE_LEN) * W
            # (in the detail, fewer strokes: kept as often as the detail lets a stroke go deep)
            k_keep = keep if on_flat[r] else keep * DETAIL_KEEP * (float(np.clip(
                (caps[r] / (STROKE_DEPTH * W)) ** 2, DETAIL_KEEP_MIN, 1.0)) if np.isfinite(caps[r]) else 1.0)
            k_keep = min(1.0, k_keep * _hard(angs[r], HARD_KEEP))     # (a 90-degree arris: every stroke)
            k_keep = max(k_keep, keep * square(angs[r]))   # (a square arris in the detail too: the thin arch's long edges)
            if angs[r] >= KNIFE_DEG:
                k_keep = keep                              # (a knife edge: trimmed all along)
            if not on_flat[r]:                             # (the detail of a thin stone: fewer cuts)
                k_keep *= THIN_DETAIL_KEEP + (1.0 - THIN_DETAIL_KEEP) * thick_share(k_size)
            k_keep *= THIN_KEEP + (1.0 - THIN_KEEP) * thick_share(k_size)     # (and a thin stone's strokes overall)
            if rng.random() < k_keep:
                jobs.append((r, t, min(t + ln, L + 0.4 * W), W))
            t += ln + rng.uniform(*STROKE_GAP) * W
    CC, CD = concave_points(m, unit)             # (the inside corners: kept clear)
    n_conc = 0
    for r, t0, t1, W in jobs:
        P, Ln, Rn, _I = pieces[r]
        s = _along(P)
        if t1 - t0 < 1.0 * W:
            continue
        k = max(2, int(math.ceil((t1 - t0) / (STEP_SHARE * W))) + 1)
        got = [_at(P, s, t) for t in np.linspace(t0, t1, k)]
        pts = np.array([g[0] for g in got])
        if len(CC):                                # (not into an inside corner: the longest stretch clear of them)
            dir_ = pts[-1] - pts[0]
            ok_ = _clear_of(pts, CC, CONCAVE_CLEAR * W, dir_ / (np.linalg.norm(dir_) + 1e-12), CD)
            if not ok_.all():
                best, a_ = (0, 0), None
                for i_, o_ in enumerate(list(ok_) + [False]):
                    if o_ and a_ is None:
                        a_ = i_
                    elif not o_ and a_ is not None:
                        if i_ - a_ > best[1] - best[0]:
                            best = (a_, i_)
                        a_ = None
                if best[1] - best[0] < 2 or float(np.linalg.norm(pts[best[1] - 1] - pts[best[0]])) < 1.0 * W:
                    n_conc += 1
                    continue
                got, pts = got[best[0]:best[1]], pts[best[0]:best[1]]
                n_conc += 1
        km = got[len(got) // 2][1]
        w = rng.uniform(*LEAN)
        up = _tilt(Ln[km] * w + Rn[km] * (1 - w), rng, STROKE_TILT)
        up /= np.linalg.norm(up) + 1e-12
        out.append(chamfer_cutter(pts, up, stroke_depth(W, rng, caps[r], _hard(angs[r], HARD_DEPTH) * k_size), angs[r], rng))
    info["strokes"] = len(out)
    info["concave_trims"] = n_conc
    info["runs_dbg"] = [(round(float(lens[r]) / unit, 3), round(float(angs[r])), bool(on_flat[r]),
                         round(float(caps[r]) / unit, 4) if np.isfinite(caps[r]) else None,
                         sum(1 for j in jobs if j[0] == r)) for r in range(len(pieces))]
    info["chisel_bu"] = (round(float(np.min(widths)), 4), round(float(np.median(widths)), 4),
                         round(float(np.max(widths)), 4)) if widths else None
    cf = [c_ / unit for c_ in caps if np.isfinite(c_)]
    info["depth_cap_bu"] = (round(float(np.min(cf)), 4), round(float(np.median(cf)), 4),
                            round(float(np.max(cf)), 4)) if cf else None
    nc = nr_ = 0
    E = np.array(long_ends).reshape(-1, 3)
    at = {}                                  # vertex -> its sharp edges (along each, from the vertex)
    for k in range(len(v0)):
        at.setdefault(int(v0[k]), []).append((p1[k] - p0[k], nl[k], nr[k]))
        at.setdefault(int(v1[k]), []).append((p0[k] - p1[k], nl[k], nr[k]))
    cos_turn = math.cos(math.radians(180.0 - CORNER_TURN))
    chipped_at = []
    for vi, ds in at.items():
        c, n = V[vi], vn[vi]
        un = [d / (np.linalg.norm(d) + 1e-12) for d, _a, _b in ds]
        # (a raised corner: three edges or more, meeting sharply, at least two of them long main edges ending here: a
        # block's corner, not a facet's edge meeting a long edge part way along it)
        n_long = int((np.linalg.norm(E - c, axis=1) < 1e-6 * max(unit, 1.0)).sum()) if len(E) else 0
        fv = vi in vflat                     # (a corner of a big flat face: a joint face's corner, hit harder)
        raised = len(ds) >= RAISED_EDGES and standing(un) and n_long >= (1 if fv else 2)
        if not raised and not (len(ds) == 2 and float(un[0] @ un[1]) > cos_turn):
            continue
        if keep <= 0.0 or not len(E) or float(np.linalg.norm(E - c, axis=1).min()) > 1e-6 * max(unit, 1.0):
            continue
        if not raised and not fv and rng.random() >= CORNER_CHANCE:
            continue
        if len(CC) and not _clear_of(c[None], CC, CONCAVE_CORNER * CHISEL * float(size) * unit)[0]:
            continue                             # (an inside corner meets this one: left alone)
        Wc = CHISEL * float(size) * unit
        cap = np.inf
        if TH is not None:
            cap = DEPTH_THICK * thickness_in(m, c, -n, unit)
        if SP is not None and vi in vrun:
            cap = min(cap, DEPTH_SPACING * min(float(SP[r]) for r in vrun[vi]))
        if raised:                           # (one flat chip off the corner, 4 - 7 x a stroke's depth deep)
            rd = RAISED_DEPTH if n_long >= 2 else RAISED_DEPTH_FLAT
            dc = min(rng.uniform(*rd) * STROKE_DEPTH * Wc * k_size * k_corner, RAISED_CAP * cap)
            chip = corner_chip(c, n, dc, rng)
            if chip_too_big(m, chip, dc):    # (the corner is not a clean, standing corner: no chip)
                continue
            out.append(chip)
            chipped_at.append(c)
            nc += 1
            nr_ += 1
            continue
        order = rng.permutation(len(ds))[:1]
        for j in order:
            d, a_, b_ = ds[int(j)]
            d = d / (np.linalg.norm(d) + 1e-12)
            W = Wc * rng.uniform(*WIDTH_RANGE)
            ln = rng.uniform(*CORNER_STROKE) * W
            k = max(2, int(math.ceil((ln + CORNER_PAST * W) / (STEP_SHARE * W))) + 1)
            pts = c + np.outer(np.linspace(-CORNER_PAST * W, ln, k), d)
            w = rng.uniform(*LEAN)
            up = _tilt(0.5 * (a_ * w + b_ * (1 - w)) + 0.5 * n, rng, STROKE_TILT)
            up /= np.linalg.norm(up) + 1e-12
            out.append(chamfer_cutter(pts, up, stroke_depth(W, rng, cap, k_size), _angle(a_, b_), rng))
            nc += 1
    # (bevelled corners: three bands or more ending close together round a little rounded patch, no vertex they share)
    nb = 0
    if keep > 0.0 and len(E) >= 3:
        Wc = CHISEL * float(size) * unit
        jn = CORNER_JOIN * Wc
        par = list(range(len(E)))

        def find(a):
            while par[a] != a:
                par[a] = par[par[a]]
                a = par[a]
            return a
        dd = np.sqrt(((E[:, None, :] - E[None, :, :]) ** 2).sum(-1))
        for i, j in zip(*np.nonzero(np.triu(dd < jn, 1))):
            par[find(int(i))] = find(int(j))
        grp = {}
        for i in range(len(E)):
            grp.setdefault(find(i), []).append(i)
        for g in grp.values():
            rs = sorted({end_run[i] for i in g})
            if len(rs) < RAISED_EDGES or sum(bevel[r] for r in rs) < 2:   # (a bevelled block's corner, not
                continue                                                  # a moulding's steps)
            c = E[g].mean(0)
            if any(float(np.linalg.norm(c - q)) < 1.5 * Wc for q in chipped_at):
                continue
            if len(CC) and not _clear_of(c[None], CC, CONCAVE_CORNER * Wc)[0]:
                continue
            try:                             # (the ends meet out on the stone, not over a notch in it)
                pr = _mm.findProjection(_mm.Vector3f(*map(float, c)), _mm.MeshPart(m))
                if math.sqrt(float(pr.distSq)) > jn:
                    continue
            except Exception:
                pass
            # (the edges' directions away from the corner: they must stand out as a corner does)
            un = []
            for r in rs:
                P = pieces[r][0]
                a_, b_ = (P[0], P[-1]) if np.linalg.norm(P[0] - c) < np.linalg.norm(P[-1] - c) else (P[-1], P[0])
                d = np.asarray(b_ - a_, np.float64)
                un.append(d / (np.linalg.norm(d) + 1e-12))
            if not standing(un):
                continue
            n = np.zeros(3)
            for r in rs:
                Lr, Rr = pieces[r][1], pieces[r][2]
                q = np.asarray(Lr[0] + Rr[0], np.float64)
                n += q / (np.linalg.norm(q) + 1e-12)
            n /= np.linalg.norm(n) + 1e-12
            cap = DEPTH_THICK * thickness_in(m, c, -n, unit) if TH is not None else np.inf
            dc = min(rng.uniform(*RAISED_DEPTH) * STROKE_DEPTH * Wc * k_size * k_corner, RAISED_CAP * cap)
            # (the corner itself lies out past the bevels: the chip's base set from there)
            corner = c + n * float(np.mean([np.linalg.norm(E[i] - c) for i in g]))
            chip = corner_chip(corner, n, dc, rng)
            if chip_too_big(m, chip, dc):
                continue
            out.append(chip)
            chipped_at.append(c)
            nb += 1
    info["corner_strokes"] = nc
    info["raised_corners"] = nr_ + nb
    info["bevelled_corners"] = nb
    return out


def _deviation(m, r, n_max=20000):
    """How far the part's surface points lie from r (99th percentile, the mesh's units)."""
    mm, mn = _ml()
    V = np.asarray(mn.getNumpyVerts(m), np.float32)
    if len(V) > n_max:
        V = V[np.random.default_rng(0).choice(len(V), n_max, replace=False)]
    vc = mn.fromNumpyArray(np.ascontiguousarray(V))
    if not isinstance(vc, mm.VertCoords):
        tmp = mm.VertCoords()
        tmp.vec = vc
        vc = tmp
    d = np.abs(np.array(mm.findSignedDistances(r, vc).vec, float))
    return float(np.percentile(d, 99)) if len(d) else 0.0


def detail_tag(m, unit=1.0, info=None):
    """The resolution a part needs, from its own detail instead of a tag (Alex 2026-11-23: "automatically work out
    how much material to remove based on the level of detail of each part, replacing the resolution tagging"): the
    coarsest preset (STONE, DETAIL, FINE, MINI) that
      * keeps its surface: remeshed and decimated the way every tool's result is at that preset, it stays within
        DETAIL_TOL of the part (a carved capital needs a fine one, a plain block does not), and
      * keeps its thin features through Trim Thin: the voxel at most RES_SHARE of the 10th percentile thickness.
    Else the finest. (tag, voxel) in the mesh's units."""
    from .offset_utils import RESOLUTION_PRESETS, cuda_offset, auto_decimate, face_count
    info = {} if info is None else info
    th = edge_thickness(m)
    th = th[np.isfinite(th)]
    need = RES_SHARE * float(np.percentile(th, RES_PCT)) if len(th) else RESOLUTION_PRESETS['STONE'][0]
    order = sorted(((v[0], v[1], k) for k, v in RESOLUTION_PRESETS.items() if k in ('STONE', 'DETAIL', 'FINE', 'MINI')),
                   reverse=True)
    f0 = face_count(m)
    devs = {}
    ok = [(vox, ratio, k) for vox, ratio, k in order if vox <= need] or [order[-1]]
    for vox, ratio, k in order:
        if (vox, ratio, k) not in ok:
            devs[k] = None
    got = []
    for vox, ratio, k in ok:
        try:
            r = cuda_offset(m, vox, 0.0)
            dev = _deviation(m, r)
        except Exception:
            continue
        devs[k] = round(dev / unit, 4)
        got.append((dev, vox, k))
        if dev <= REMESH_TOL * unit:
            info["detail"] = devs
            return k, vox
    info["detail"] = devs
    # (off by as much at every resolution: something in the mesh itself, inner pieces or open caps, not detail a finer
    # voxel would keep: the thickness rule alone)
    if len(got) >= 2 and got[-1][0] >= 0.8 * got[0][0]:
        return ok[0][2], ok[0][0]
    return ok[-1][2], ok[-1][0]


# --- Stone Damage dents on the flat faces (Alex 2026-11-23) ------------------------------------------
FLAT_DEG = 2.0         # deg: faces this close in direction ...
FLAT_TOL = 0.0005      # BU: ... and this close to one plane make one flat face
FLAT_AREA = 0.02       # BU^2: a flat face at least this big gets dents (a boolean's cut, not a stone's facet)
DENT_AMOUNT = 16.0     # dents per BU^2 of flat face (the "Dents" slider)
FLAT_OUTLINE = 0       # (Alex 2026-11-23: a plain block's big faces get stone damage too; was 16: only a boolean's
#                        jagged cut) a flat face's outline has at least this many points: a boolean's cut through a rough stone
#                        (jagged), not a flat facet the stone was modelled with (Alex's example leaves its top alone)
DENT_SIZE = 0.5        # x the dent meshes' own size (the "Dent Size" slider)
DENT_RANGE = (0.6, 1.3)
DENT_EDGE = 1.0        # x a dent's half width: its centre is kept this far in from the flat face's outline (its whole
#                        footprint inside the face: at 0.25 a dent by the rim broke through the edge in a round scoop)
DENT_BITE = (0.1, 0.2)     # x its width: how deep a dent bites into the flat face (the damage meshes only reach
#                            0.03 BU under their origin: laid on the face as they are, most only scraped it)


def edge_pairs(F):
    """The mesh's edges and the face pairs across them, built with numpy (a dict per edge took a third of Chip's time):
    (edges (k, 2) vertex ids low, high; how many faces each has (k,); pairs (j, 2): the two faces of each edge with
    exactly two, in the order the edges first appear going through the faces; the edge of each pair (j,))."""
    n = len(F)
    he = F[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)             # (half edge 3 i + j)
    lo, hi = he.min(1), he.max(1)
    key = lo.astype(np.int64) * (int(F.max()) + 1 if n else 1) + hi
    order = np.argsort(key, kind='stable')                         # (the same edge's half edges together, in order)
    ks = key[order]
    first = np.concatenate([[True], ks[1:] != ks[:-1]]) if n else np.zeros(0, bool)
    start = np.flatnonzero(first)
    count = np.diff(np.concatenate([start, [len(ks)]]))
    edges = np.stack([lo[order[start]], hi[order[start]]], 1)
    two = np.flatnonzero(count == 2)
    h0, h1 = order[start[two]], order[start[two] + 1]
    seen = np.argsort(h0, kind='stable')                           # (as the edges first appear)
    pairs = np.stack([h0[seen] // 3, h1[seen] // 3], 1)
    return edges, count, pairs, two[seen]


def flat_regions(V, F, unit=1.0, min_area=None):
    """The flat faces (coplanar edge-connected regions) of at least min_area (FLAT_AREA), BU^2: ([(face indices,
    normal, area)], face areas). (Passed in, never set on the module: the operator chips stones on several threads at
    once, and a module setting changed by one changed the other's dents: Chip gave a different result run to run.)
    The regions grow from the biggest face down, each face tried against the region's own mean normal and its seed's
    plane; plain float arithmetic per face (numpy calls on three numbers were two thirds of Chip's time)."""
    min_area = FLAT_AREA if min_area is None else min_area
    c = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    ar = 0.5 * np.linalg.norm(c, axis=1)
    nr = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-30)
    nb = [[] for _ in range(len(F))]
    for f0, f1 in edge_pairs(F)[2].tolist():
        nb[f0].append(f1)
        nb[f1].append(f0)
    cosa = math.cos(math.radians(FLAT_DEG))
    tol = FLAT_TOL * unit
    lab = [-1] * len(F)
    nrl, arl, FV = nr.tolist(), ar.tolist(), V[F].tolist()
    sqrt = math.sqrt
    out = []
    for seed in np.argsort(-ar).tolist():
        if lab[seed] >= 0:
            continue
        lab[seed] = seed
        reg = [seed]
        q = [seed]
        n0, a0 = nrl[seed], arl[seed]
        ax, ay, az = n0[0] * a0, n0[1] * a0, n0[2] * a0
        (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = FV[seed]
        px, py, pz = (x0 + x1 + x2) / 3.0, (y0 + y1 + y2) / 3.0, (z0 + z1 + z2) / 3.0
        while q:
            f = q.pop()
            ln = sqrt(ax * ax + ay * ay + az * az) + 1e-30
            mx, my, mz = ax / ln, ay / ln, az / ln
            for h in nb[f]:
                if lab[h] >= 0:
                    continue
                nh = nrl[h]
                if nh[0] * mx + nh[1] * my + nh[2] * mz < cosa:
                    continue
                far = False
                for vx, vy, vz in FV[h]:
                    if abs((vx - px) * mx + (vy - py) * my + (vz - pz) * mz) > tol:
                        far = True
                        break
                if far:
                    continue
                lab[h] = seed
                reg.append(h)
                q.append(h)
                ah = arl[h]
                ax, ay, az = ax + nh[0] * ah, ay + nh[1] * ah, az + nh[2] * ah
        if len(reg) == 1 and a0 < min_area * unit * unit:
            continue
        A = float(ar[reg].sum())
        if A >= min_area * unit * unit:
            acc = np.array([ax, ay, az])
            out.append((np.array(reg), acc / (np.linalg.norm(acc) + 1e-30), A))
    return out, ar


def flat_all(V, F, unit, info):
    """flat_regions down to the smallest size any step asks for, found once per stone and kept in info for the next
    step (the strokes want them down to BIG_FLAT, the dents down to FLAT_AREA: the same list, cut off higher)."""
    got = info.get("_flat") if info is not None else None
    if got is None or got[0].shape != F.shape or not np.array_equal(got[0], F):
        got = (F,) + flat_regions(V, F, unit, min(FLAT_AREA, BIG_FLAT))
        if info is not None:
            info["_flat"] = got
    return got[1], got[2]


COVER_GAP = 0.04       # BU: a spot with another stone's surface this close (a joint: 0.02, and a little) is covered by it
COVER_SHARE = 0.5      # ... or within this share of the dent's half width more: a dent half under its neighbour
COVER_MARGIN = 0.3     # BU: a neighbour's faces this far round a stone's box are kept as its cover


def crop_near(V, F, lo, hi):
    """The faces of a neighbour (V, F) with a vertex inside the box lo..hi: (V, F) packed, or None if none."""
    V = np.asarray(V)
    F = np.asarray(F)
    if not len(F):
        return None
    inside = np.all((V >= lo) & (V <= hi), axis=1)
    keep = inside[F].any(1)
    if not keep.any():
        return None
    f = F[keep]
    used = np.unique(f)
    remap = np.full(len(V), -1, np.int64)
    remap[used] = np.arange(len(used))
    return np.ascontiguousarray(V[used], np.float32), np.ascontiguousarray(remap[f], np.int32)


def cover_of(m, others, unit=1.0):
    """What covers the stone m: the parts of the other stones (meshlib meshes or (V, F)) near its box, as [(V, F)].
    (Alex 2026-11-23: dents on a foot's top faces lay hidden under the plinth: "skip hidden faces".)"""
    mm, mn = _ml()
    V = np.asarray(mn.getNumpyVerts(m), np.float64)
    if not len(V):
        return []
    lo, hi = V.min(0) - COVER_MARGIN * unit, V.max(0) + COVER_MARGIN * unit
    out = []
    for o in others:
        if isinstance(o, tuple):
            Vo, Fo = o
        else:
            b = o.computeBoundingBox()
            if (b.min.x > hi[0] or b.max.x < lo[0] or b.min.y > hi[1] or b.max.y < lo[1]
                    or b.min.z > hi[2] or b.max.z < lo[2]):
                continue
            Vo, Fo = mn.getNumpyVerts(o), mn.getNumpyFaces(o.topology)
        got = crop_near(Vo, Fo, lo, hi)
        if got is not None:
            out.append(got)
    return out


GROUND_DOWN = 0.5      # a face turned down this much (its normal's z under minus this) ...
GROUND_REACH = 0.3     # BU: ... within this of the ground (the lowest point of the stones worked together) rests on it


def covered_fn(cover, unit=1.0, ground=None):
    """covered(p, half, n) -> True where the point p (face normal n) cannot be seen: another stone lies over it (its
    surface within COVER_GAP, or within COVER_SHARE of the dent's half width more), or it is a bottom face resting on the
    ground (Alex 2026-11-23: "bottom faces should count as obscured": turned down, within GROUND_REACH of `ground`,
    the lowest z of the stones worked together). cover: [(V, F)] (cover_of)."""
    def grounded(p, n):
        return (ground is not None and n is not None and float(n[2]) < -GROUND_DOWN
                and float(p[2]) <= float(ground) + GROUND_REACH * unit)
    if not cover:
        return lambda p, half, n=None: grounded(p, n)
    mm, mn = _ml()
    # (the meshes are held by the test itself: a MeshPart only points at its mesh, and one freed under it crashed Blender)
    meshes = [mn.meshFromFacesVerts(np.ascontiguousarray(F_, np.int32), np.ascontiguousarray(V_, np.float32))
              for V_, F_ in cover]

    def covered(p, half, n=None, meshes=meshes):
        if grounded(p, n):
            return True
        lim = COVER_GAP * unit + COVER_SHARE * half
        v = mm.Vector3f(float(p[0]), float(p[1]), float(p[2]))
        for m_ in meshes:
            if float(mm.findProjection(v, mm.MeshPart(m_)).distSq) < lim * lim:
                return True
        return False
    return covered


def place_dents(m, dents, size=DENT_SIZE, amount=DENT_AMOUNT, unit=1.0, seed=0, cut_faces=None, info=None, cover=None,
                ground=None):
    """Stone Damage's dents on the flat faces of m: [(V, F)] in the mesh's units. dents: [(V, F)] in Blender units,
    Z out of the stone. cut_faces: only the flat faces marked cut (Only Cut Edges)."""
    info = {} if info is None else info
    if not dents or amount <= 0:
        return []
    mm, mn = _ml()
    rng = np.random.default_rng(seed + 1)
    V = np.asarray(mn.getNumpyVerts(m), np.float64)
    F = np.asarray(mn.getNumpyFaces(m.topology), np.int64)
    regs, ar = flat_all(V, F, unit, info)
    regs = [r for r in regs if r[2] >= FLAT_AREA * unit * unit]
    info["flat_faces"] = len(regs)
    covered = covered_fn(cover, unit, ground)
    under = covered_fn(cover, unit)              # (under another stone only: what sends the damage to the sides)
    info["covered_skips"] = 0
    out = []
    for reg, n, A in regs:
        if cut_faces is not None:
            cf = np.asarray(cut_faces, bool)
            if not np.any(cf[reg[reg < len(cf)]]):
                continue
        ec = {}                              # (the flat face's outline: the dents kept in from it)
        for i in reg:
            a, b, c = F[i]
            for e in ((a, b), (b, c), (c, a)):
                kk = (min(e), max(e))
                ec[kk] = ec.get(kk, 0) + 1
        rim = np.array([k for k, c in ec.items() if c == 1], np.int64).reshape(-1, 2)
        if len(np.unique(rim)) < FLAT_OUTLINE:
            continue
        k_n = int(rng.poisson(A / (unit * unit) * float(amount)))
        pw = ar[reg] / ar[reg].sum()
        k_n = max(k_n, 1 if rng.random() < A / (unit * unit) * float(amount) else 0)
        for _ in range(k_n):
            dv, dt = dents[int(rng.integers(0, len(dents)))]
            dv = np.asarray(dv, np.float64)
            sc = float(size) * rng.uniform(*DENT_RANGE)
            half = 0.5 * max(np.ptp(dv[:, 0]), np.ptp(dv[:, 1])) * sc * unit
            for _try in range(24):
                if _try and _try % 8 == 0:         # (a small face: a smaller dent, so it fits inside)
                    sc *= 0.7
                    half *= 0.7
                f = int(reg[rng.choice(len(reg), p=pw)])
                r1, r2 = rng.random(), rng.random()
                if r1 + r2 > 1:
                    r1, r2 = 1 - r1, 1 - r2
                p = V[F[f, 0]] + r1 * (V[F[f, 1]] - V[F[f, 0]]) + r2 * (V[F[f, 2]] - V[F[f, 0]])
                if not len(rim):
                    break
                A_, B_ = V[rim[:, 0]], V[rim[:, 1]]
                d = B_ - A_
                t = np.clip(((p - A_) * d).sum(1) / np.maximum((d * d).sum(1), 1e-24), 0, 1)
                if float(np.linalg.norm(A_ + d * t[:, None] - p, axis=1).min()) >= DENT_EDGE * half:
                    if covered(p, half, n):          # (under the next stone, or on the ground: not here)
                        info["covered_skips"] += 1 if under(p, half) else 0
                        continue
                    break
            else:
                continue
            B = _frame(n, rng)
            s3 = sc * (1.0 + rng.uniform(-AXIS_JITTER, AXIS_JITTER, 3))
            own = max(-float(dv[:, 2].min()) * s3[2] * unit, 0.0)       # (how far it reaches under its origin)
            bite = rng.uniform(*DENT_BITE) * 2.0 * half
            Vd = ((dv * s3) @ B.T) * unit + p - n * max(bite - own, 0.0)
            out.append((Vd.astype(np.float32), np.asarray(dt, np.int32)))
    flat = np.zeros(len(F), bool)
    for reg, _n, _A in regs:
        flat[reg] = True
    # (a stone whose flat faces lie under its neighbours takes its damage on its sides: always, not by chance)
    out += curved_dents(V, F, dents, size, unit, rng, info, flat, covered, always=info["covered_skips"] > 0)
    info["dents"] = len(out)
    return out


CURVED_CHANCE = 1.0 / 3.0  # (Alex 2026-11-23, an arch stone's curved soffit: "about a third of such arc stones should get
CURVED_CHANCE_THICK = 0.30  # ... plus this on a thick stone (Alex 23 Nov: "increase the frequency of the XL profile getting
#                              damaged 30% (absolute)"), in proportion between
CURVED_COUNT = (1, 3)      # one to three stone damage on the flat surface ... large damage first, then small")
CURVED_SIZES = (1.3, 0.8, 0.5)  # x the Dent Size, the first, second and third (each +- 15 %)
CURVED_MIN_AREA = 0.03    # BU^2: a curved face this big or more can take a dent
CURVED_ONE_WAY = 0.2       # its normals nearly in one plane (smallest / largest spread): curved one way, as a soffit, a
#                            moulding's channel or a shaft (Alex 23 Nov: "XL profile and the pillars should get some stone
#                            damage"); a leaf or a capital's carving curves both ways and is left alone
CURVED_SHRINKS = 4         # a dent that does not fit is tried this many times smaller (x 0.75: down to a third, a foot's
#                            narrow moulded band), then left out
CURVED_STONE_AREA = 0.5    # BU^2: only on a stone with this much surface (leaf pieces have under 0.4, stones over 0.6)
CURVED_PROBE = 0.08        # BU: the stone is looked at this far round a dent, however small the dent: straight there (a
#                            moulding, whatever its profile does), not a leaf
CURVED_LOCAL_MAX = 50.0   # deg: under one dent the stone turns at most this much from where the dent sits
CURVED_DEG = 20.0          # deg: the curved side grown from its facets over edges turning less than this


def curved_surface(F, fn, seeds):
    """The faces of the smooth (curved) side the seed faces lie on: grown over edges turning under CURVED_DEG."""
    adj = {}
    for i, (a, b, c) in enumerate(F):
        for e in ((a, b), (b, c), (c, a)):
            adj.setdefault((min(e), max(e)), []).append(i)
    nb = [[] for _ in range(len(F))]
    for fs in adj.values():
        if len(fs) == 2:
            nb[fs[0]].append(fs[1])
            nb[fs[1]].append(fs[0])
    cmin = math.cos(math.radians(CURVED_DEG))
    seen = np.zeros(len(F), bool)
    seen[seeds] = True
    q = list(seeds)
    while q:
        i = q.pop()
        for j in nb[i]:
            if not seen[j] and float(fn[i] @ fn[j]) > cmin:
                seen[j] = True
                q.append(j)
    return np.flatnonzero(seen)


def smooth_patches(F, fn):
    """Per face, the smooth patch it lies on: faces joined over edges turning under CURVED_DEG (a moulding's curved
    channel, a shaft's round side, an arch's soffit), cut apart by the sharp edges. (labels, the edges (k, 2) between
    two patches or with other than two faces: the patches' outlines)."""
    edges, count, pairs, pe = edge_pairs(F)
    par = list(range(len(F)))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    cmin = math.cos(math.radians(CURVED_DEG))
    smooth = (fn[pairs[:, 0]] * fn[pairs[:, 1]]).sum(1) > cmin if len(pairs) else np.zeros(0, bool)
    for f0, f1 in pairs[smooth].tolist():
        ra, rb = find(f0), find(f1)
        if ra != rb:
            par[ra] = rb
    lab = np.array([find(i) for i in range(len(F))])
    rim = np.ones(len(edges), bool)
    if len(pairs):
        rim[pe] = lab[pairs[:, 0]] != lab[pairs[:, 1]]
    return lab, edges[rim].astype(np.int64)


def curved_dents(V, F, dents, size, unit, rng, info, flat=None, covered=None, always=False):
    """Stone damage on the stone's smooth curved faces (an arch's soffit, a moulding's channels, a pillar's shaft):
    on CURVED_CHANCE of the stones (more on thick ones), one to three dents, the big one first, each kept inside the
    curved face it lands on: [(V, F)] in the mesh's units. flat: per face, True on the flat faces that get their own
    dents."""
    info["curved_dents"] = 0
    if not dents:
        return []
    chance = CURVED_CHANCE + CURVED_CHANCE_THICK * thick_share(info.get("size_depth", 1.0))
    if rng.random() >= chance and not always:
        return []
    c = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    ar = 0.5 * np.linalg.norm(c, axis=1)
    if float(ar.sum()) < CURVED_STONE_AREA * unit * unit:
        return []                                    # (a small piece: a leaf of a leafy abacus, a loose bead)
    fn = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-30)
    lab, rim = smooth_patches(F, fn)
    # (where a dent may land: the smooth faces big enough, the flat faces' own dents apart; whether the stone under one
    # dent is flat or curved one way is looked at there, under its own footprint: a foot's moulded sides run round its
    # rounded corners as one patch, curved both ways as a whole, flat or one way under any one dent)
    ok = np.zeros(len(F), bool)
    for L in np.unique(lab):
        fs = np.flatnonzero(lab == L)
        if float(ar[fs].sum()) >= CURVED_MIN_AREA * unit * unit:
            ok[fs] = True
    if flat is not None:
        ok &= ~flat
    faces = np.flatnonzero(ok)
    info["curved_faces"] = int(len(faces))
    if not len(faces) or float(ar[faces].sum()) <= 0.0:
        return []
    cen = V[F].mean(1)
    cos_max = math.cos(math.radians(CURVED_LOCAL_MAX))
    pw = ar[faces] / ar[faces].sum()
    out = []
    placed = []
    for k in range(int(rng.integers(CURVED_COUNT[0], CURVED_COUNT[1] + 1))):
        dv, dt = dents[int(rng.integers(0, len(dents)))]
        dv = np.asarray(dv, np.float64)
        sc = float(size) * CURVED_SIZES[k] * rng.uniform(0.85, 1.15)
        half = 0.5 * max(np.ptp(dv[:, 0]), np.ptp(dv[:, 1])) * sc * unit
        for _try in range(8 * (CURVED_SHRINKS + 1)):
            if _try and _try % 8 == 0:               # (a smaller one, CURVED_SHRINKS times at most: a dent shrunk
                sc *= 0.75                           # until a leaf looks flat under it is not stone damage)
                half *= 0.75
            f = int(faces[rng.choice(len(faces), p=pw)])
            r1, r2 = rng.random(), rng.random()
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            p = V[F[f, 0]] + r1 * (V[F[f, 1]] - V[F[f, 0]]) + r2 * (V[F[f, 2]] - V[F[f, 0]])
            if any(float(np.linalg.norm(p - q)) < (h + half) * 0.8 for q, h in placed):
                continue                             # (not on top of the one before)
            if len(rim):
                A_, B_ = V[rim[:, 0]], V[rim[:, 1]]
                d = B_ - A_
                t = np.clip(((p - A_) * d).sum(1) / np.maximum((d * d).sum(1), 1e-24), 0, 1)
                if float(np.linalg.norm(A_ + d * t[:, None] - p, axis=1).min()) < DENT_EDGE * half:
                    continue
            d2 = ((cen - p) ** 2).sum(1)
            near = np.flatnonzero(d2 < half * half)                  # (the stone under this dent)
            if len(near) and float((fn[near] @ fn[f]).min()) < cos_max:
                continue                             # (it turns away under the dent: an edge, a thin part's far side)
            probe = np.flatnonzero(d2 < max(half, CURVED_PROBE * unit) ** 2)    # (and round it, whatever its size)
            if len(probe) > 2:
                sv = np.linalg.svd(fn[probe] * np.sqrt(ar[probe])[:, None], compute_uv=False)
                if sv[2] > CURVED_ONE_WAY * sv[0]:
                    continue                         # (curved both ways there: a leaf, carving, a rounded corner)
            if covered is not None and covered(p, half, fn[f]):
                info["covered_skips"] = info.get("covered_skips", 0) + 1
                continue                             # (under the next stone)
            break
        else:
            continue
        n = fn[f]
        B = _frame(n, rng)
        s3 = sc * (1.0 + rng.uniform(-AXIS_JITTER, AXIS_JITTER, 3))
        own = max(-float(dv[:, 2].min()) * s3[2] * unit, 0.0)
        bite = rng.uniform(*DENT_BITE) * 2.0 * half
        Vd = ((dv * s3) @ B.T) * unit + p - n * max(bite - own, 0.0)
        out.append((Vd.astype(np.float32), np.asarray(dt, np.int32)))
        placed.append((p, half))
    info["curved_dents"] = len(out)
    return out


def _cutter(mm, mn, cut):
    V, F = cut
    return mn.meshFromFacesVerts(np.ascontiguousarray(F, np.int32), np.ascontiguousarray(V, np.float32))


def _holes(m):
    return len(m.topology.findHoleRepresentiveEdges())


def dent_chip(mesh, dents, size=1.0, amount=AMOUNT, unit=1.0, seed=None, cut_faces=None, info=None,
              face_dents=None, dent_size=DENT_SIZE, dent_amount=DENT_AMOUNT, auto=False, cover=None, ground=None):
    """mesh worked over (a new mesh; the input is kept): Carve's chisel strokes on its sharp edges and corners
    (size: x Carve's Standard chisel, amount: strokes per BU of edge), Stone Damage's dents on its flat faces
    (face_dents: [(V, F)] in Blender units, Z out of the stone; dent_size: x their own size; dent_amount: per
    BU^2). dents (the old chip meshes on the edges) is no longer used: kept for old callers."""
    mm, mn = _ml()
    info = {} if info is None else info
    t0 = time.perf_counter()
    m = mm.Mesh(mesh)
    m.pack()
    if seed is None:
        seed = _seed_of(mn.getNumpyVerts(m))
    cuts = place_strokes(m, size, amount, unit, seed, cut_faces, info, auto)
    cuts += place_dents(m, face_dents or [], dent_size, dent_amount, unit, seed, cut_faces, info, cover, ground)
    info.pop("_flat", None)
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


def islands_of(mesh):
    """The mesh's islands (its loose parts) as [(meshlib mesh, face indices in the mesh)]; one island: the mesh as it is
    (Alex 2026-11-23: "make sure each island gets chipped individually")."""
    mm, mn = _ml()
    m = mm.Mesh(mesh)
    m.pack()
    F = np.asarray(mn.getNumpyFaces(m.topology), np.int64)
    try:
        comps = list(mm.MeshComponents.getAllComponents(mm.MeshPart(m)))
    except Exception:
        comps = []
    if len(comps) <= 1:
        return [(m, np.arange(len(F)))]
    V = np.asarray(mn.getNumpyVerts(m), np.float32)
    out = []
    for bs in comps:
        idx = np.nonzero(np.asarray(mn.getNumpyBitSet(bs), bool)[:len(F)])[0]
        if not len(idx):
            continue
        f = F[idx]
        used = np.unique(f)
        remap = np.full(len(V), -1, np.int64)
        remap[used] = np.arange(len(used))
        out.append((mn.meshFromFacesVerts(remap[f].astype(np.int32), np.ascontiguousarray(V[used])), idx))
    return out or [(m, np.arange(len(F)))]


def _joined(meshes):
    mm, mn = _ml()
    if len(meshes) == 1:
        return meshes[0]
    Vs, Fs, n = [], [], 0
    for r in meshes:
        V = np.asarray(mn.getNumpyVerts(r), np.float32)
        F = np.asarray(mn.getNumpyFaces(r.topology), np.int64)
        F = F[(F >= 0).all(1)]
        Vs.append(V)
        Fs.append(F + n)
        n += len(V)
    return mn.meshFromFacesVerts(np.concatenate(Fs).astype(np.int32), np.concatenate(Vs))


def dent_trim(mesh, res, dents, size=1.0, amount=AMOUNT, unit=1.0, cut_faces=None, info=None,
              face_dents=None, dent_size=DENT_SIZE, dent_amount=DENT_AMOUNT, auto=False, cover=None, ground=None):
    """The Chip method of Trim Edges, the way the operator runs it, island by island (cover: the other stones of the
    run near this one, [(V, F)] or meshes: no dent where one of them, or another island of this mesh, lies over it): each loose part gets its own
    strokes and dents, its own detail (auto: detail_tag) and its own Trim Thin, then they are put back together side by
    side (nothing merged). (mesh, tag): tag None, or with auto the finest resolution any island needed (the object is
    tagged with it in place of its own tag). Raises MeshCollapsedError when Trim Thin leaves nothing of any island."""
    from .trim_round import trim_thin
    from .offset_utils import MeshCollapsedError
    info = {} if info is None else info
    parts, tags, ftags = [], [], []
    isl = islands_of(mesh)
    info["islands"] = len(isl)
    if ground is None:                       # (worked alone: its own lowest point is what it stands on)
        _mm, _mn = _ml()
        _v = _mn.getNumpyVerts(mesh)
        ground = float(_v[:, 2].min()) if len(_v) else None
    for k, (m, idx) in enumerate(isl):
        ii = {}
        r = res
        tag = None
        if auto:
            tag, r = detail_tag(m, unit, ii)
        cf = None
        if cut_faces is not None:
            cf = np.asarray(cut_faces, bool)
            cf = cf[idx[idx < len(cf)]] if len(cf) else None
        cov = cover_of(m, list(cover or []) + [o for j, (o, _i) in enumerate(isl) if j != k], unit)
        chipped = dent_chip(m, dents, size, amount, unit, cut_faces=cf, info=ii, face_dents=face_dents,
                            dent_size=dent_size, dent_amount=dent_amount, auto=auto, cover=cov, ground=ground)
        try:
            got = trim_thin(chipped, r)[0]
            if auto:                         # (as far as DEC_TOL allows: no face target)
                from .offset_utils import face_count as _fc
                f_before = _fc(got)
                got = decimate_flat(got, max(DEC_TOL * unit, DEC_VOX * r))
                ii["faces"] = (f_before, _fc(got))
            parts.append(got)
            if tag is not None:
                tags.append((r, tag, len(idx)))
                ftags.append((tag, got))
        except MeshCollapsedError:
            ii["collapsed"] = True
        ii["tag"] = tag
        info.setdefault("per_island", []).append(ii)
        for key in ("strokes", "corner_strokes", "raised_corners", "dents", "covered_skips"):
            info[key] = info.get(key, 0) + int(ii.get(key) or 0)
    if not parts:
        raise MeshCollapsedError("Trim Thin left nothing")
    # (one tag for the object, for now: the one its biggest island needed; a small thin island trimmed at its own finer
    # voxel above, but it no longer sends the whole object to Mini)
    tag = max(tags, key=lambda t_: t_[2])[1] if tags else None
    info["tag"] = tag
    if auto and ftags and len(ftags) == len(parts):
        # (each island keeps its own tag, face by face; already decimated: the pipeline leaves it)
        try:
            from .offset_utils import RESOLUTION_TAG_ORDER
            from .offset_utils import face_count as _fc
            info["face_tags"] = np.concatenate([np.full(_fc(g), RESOLUTION_TAG_ORDER.index(t_), np.int32)
                                                for t_, g in ftags])
        except Exception:
            info["face_tags"] = None
        info["decimated"] = True
    return _joined(parts), tag
