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


# --- Carve strokes on the sharp edges (Alex 2026-11-23) ---------------------------------------------
CHISEL = 0.066         # BU: the chisel's opening at Size 1 (Carve's Standard preset, 0.66 mm)
WIDTH_RANGE = (0.7, 1.3)    # each stroke's chisel, x Size, drawn in this range
WALL_DEG = 30.0        # Carve's walls lean this far from the groove's axis ...
FLOOR_SHARE = 0.6      # ... its floor this share of the opening (wider than Carve's 0.3: drawn along an edge it
#                        leaves a flat chamfer, as in Alex's example, not a dark slot) ...
ABOVE = 0.6            # ... and the cutter runs on straight up this far above the surface, x the opening
STEP_SHARE = 0.35      # the stroke's rings this far apart, x the opening
STROKE_LEN = (3.0, 10.0)   # a stroke's length, x its opening (Alex's example: long strips along the main edges)
STROKE_GAP = (0.2, 1.2)    # x the opening between one stroke and the next along a run
RUN_MIN = 4.0          # x the opening: a run shorter than this gets no stroke, nor its corners (the facets of a
#                        rough, sculpted side: Alex's example leaves those as they are)
EASE = 0.8             # its ends lift out of the stone over this much, x the opening (Carve's ends)
RUN_TURN = 35.0        # deg: a run of sharp edges is split where it turns more than this (a corner)
STROKE_SINK = (-0.2, 0.05)    # x the depth: a stroke sits this much deeper (-) or shallower along its axis
STROKE_TILT = 8.0      # deg: and its axis is tipped at random by up to this much
CORNER_STROKE = (0.8, 1.6)   # a corner's stroke: this long (x the opening) ...
CORNER_PAST = 0.4      # ... starting this far past the corner


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
        pts, L, R = [pos[start]], [], []
        v = start
        while True:
            nxt = [(k, w) for k, w in adj[v] if not used[k]]
            if not nxt or (len(adj[v]) > 2 and v != start):
                break
            k, w = nxt[0]
            used[k] = True
            pts.append(pos[w])
            L.append(nl[k])
            R.append(nr[k])
            v = w
            if v == start:
                break
        return pts, L, R

    ends = [v for v, a in adj.items() if len(a) != 2]
    for v in ends + list(adj.keys()):
        while any(not used[k] for k, _w in adj[v]):
            pts, L, R = walk(v)
            if len(pts) >= 2:
                chains.append((np.array(pts), np.array(L), np.array(R)))
    pieces = []
    cos_t = math.cos(math.radians(RUN_TURN))
    for P, L, R in chains:
        cut = [0]
        for i in range(1, len(P) - 1):
            a, b = P[i] - P[i - 1], P[i + 1] - P[i]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 1e-12 and nb > 1e-12 and float(a @ b) / (na * nb) < cos_t:
                cut.append(i)
        cut.append(len(P) - 1)
        for a, b in zip(cut[:-1], cut[1:]):
            if b > a:
                pieces.append((P[a:b + 1], L[a:b], R[a:b]))
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


def stroke_cutter(pts, up, W):
    """Carve's wedge swept along pts (straight-ish), `up` out of the stone: a closed (V, F) solid. The opening W at
    the surface, 30-degree walls down to a floor FLOOR_SHARE wide, the walls run on ABOVE x W over the surface; its
    two ends lift out over EASE x W (the groove fades out as a chisel stroke does)."""
    P = np.asarray(pts, np.float64)
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
        d = D * k                            # (the floor rises to the surface at the ends)
        t = P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)]
        t = t / (np.linalg.norm(t) + 1e-12)
        u = up - t * float(up @ t)
        u = u / (np.linalg.norm(u) + 1e-12)
        x = np.cross(t, u)
        fx = f * k + 1e-3 * W
        xs = [(-fx, -d), (fx, -d), (fx + slope * (d + H), H), (-fx - slope * (d + H), H)]
        rings.append([p + x * cx + u * cy for cx, cy in xs])
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


def coverage(amount):
    """The share of each run the strokes go over, from the Amount slider: 2.5 (the default) about 70 %."""
    a = max(0.0, float(amount))
    return a / (a + 1.0)


def place_strokes(m, size=1.0, amount=AMOUNT, unit=1.0, seed=0, cut_faces=None, info=None):
    """The chisel strokes for mesh m's sharp edges and corners: [(V, F)] in the mesh's units. Each run of sharp edges
    is walked from end to end in strokes of STROKE_LEN with STROKE_GAP between, each one kept coverage(amount) of the
    time; corners: a stroke run out past them CORNER_CHANCE of the time (not with Amount 0)."""
    info = {} if info is None else info
    rng = np.random.default_rng(seed)
    p0, p1, nl, nr, v0, v1, V, vn = sharp_edges(m, unit, cut_faces)
    info["sharp_edges"] = int(len(p0))
    if not len(p0):
        return []
    pieces = runs(p0, p1, v0, v1, nl, nr)
    lens = np.array([float(_along(P)[-1]) for P, _l, _r in pieces])
    if not len(pieces) or lens.sum() <= 0:
        return []
    total = float(lens.sum()) / unit
    info["edge_length"] = total
    out = []
    keep = coverage(amount)
    jobs = []                                # (run, start, end, chisel)
    W0 = CHISEL * float(size) * unit
    long_ends = []                           # (the ends of the runs long enough: the corners that get a stroke)
    for r, (P, Ln, Rn) in enumerate(pieces):
        L = lens[r]
        if L < RUN_MIN * W0:
            continue
        long_ends += [P[0], P[-1]]
        t = -rng.uniform(0.0, 0.5) * W0      # (a stroke may start just past the run's end: over the corner)
        while t < L:
            W = W0 * rng.uniform(*WIDTH_RANGE)
            ln = rng.uniform(*STROKE_LEN) * W
            if rng.random() < keep:
                jobs.append((r, t, min(t + ln, L + 0.4 * W), W))
            t += ln + rng.uniform(*STROKE_GAP) * W
    for r, t0, t1, W in jobs:
        P, Ln, Rn = pieces[r]
        s = _along(P)
        if t1 - t0 < 1.0 * W:
            continue
        k = max(2, int(math.ceil((t1 - t0) / (STEP_SHARE * W))) + 1)
        got = [_at(P, s, t) for t in np.linspace(t0, t1, k)]
        pts = np.array([g[0] for g in got])
        km = got[len(got) // 2][1]
        w = rng.uniform(*LEAN)
        up = _tilt(Ln[km] * w + Rn[km] * (1 - w), rng, STROKE_TILT)
        up /= np.linalg.norm(up) + 1e-12
        pts = pts + up * rng.uniform(*STROKE_SINK) * depth_of(W)
        out.append(stroke_cutter(pts, up, W))
    info["strokes"] = len(out)
    nc = 0
    E = np.array(long_ends).reshape(-1, 3)
    for c, n in corners(V, vn, p0, p1, v0, v1):
        if keep <= 0.0 or not len(E) or float(np.linalg.norm(E - c, axis=1).min()) > 1e-6 * max(unit, 1.0):
            continue
        if rng.random() >= CORNER_CHANCE:
            continue
        W = CHISEL * float(size) * rng.uniform(*WIDTH_RANGE) * unit
        ds = []                              # (along one of the sharp edges at the corner, from past it inwards)
        for k in range(len(v0)):
            if np.linalg.norm(p0[k] - c) < 1e-9:
                ds.append((p1[k] - p0[k], nl[k], nr[k]))
            elif np.linalg.norm(p1[k] - c) < 1e-9:
                ds.append((p0[k] - p1[k], nl[k], nr[k]))
        if not ds:
            continue
        d, a_, b_ = ds[int(rng.integers(0, len(ds)))]
        d = d / (np.linalg.norm(d) + 1e-12)
        ln = rng.uniform(*CORNER_STROKE) * W
        k = max(2, int(math.ceil((ln + CORNER_PAST * W) / (STEP_SHARE * W))) + 1)
        pts = c + np.outer(np.linspace(-CORNER_PAST * W, ln, k), d)
        w = rng.uniform(*LEAN)
        up = _tilt(0.5 * (a_ * w + b_ * (1 - w)) + 0.5 * n, rng, STROKE_TILT)
        up /= np.linalg.norm(up) + 1e-12
        pts = pts + up * rng.uniform(*STROKE_SINK) * depth_of(W)
        out.append(stroke_cutter(pts, up, W))
        nc += 1
    info["corner_strokes"] = nc
    return out


# --- Stone Damage dents on the flat faces (Alex 2026-11-23) ------------------------------------------
FLAT_DEG = 2.0         # deg: faces this close in direction ...
FLAT_TOL = 0.0005      # BU: ... and this close to one plane make one flat face
FLAT_AREA = 0.02       # BU^2: a flat face at least this big gets dents (a boolean's cut, not a stone's facet)
DENT_AMOUNT = 16.0     # dents per BU^2 of flat face (the "Dents" slider)
FLAT_OUTLINE = 16      # a flat face's outline has at least this many points: a boolean's cut through a rough stone
#                        (jagged), not a flat facet the stone was modelled with (Alex's example leaves its top alone)
DENT_SIZE = 0.5        # x the dent meshes' own size (the "Dent Size" slider)
DENT_RANGE = (0.6, 1.3)
DENT_EDGE = 0.25       # x a dent's width: dents are kept this far in from the flat face's outline at least
DENT_BITE = (0.1, 0.2)     # x its width: how deep a dent bites into the flat face (the damage meshes only reach
#                            0.03 BU under their origin: laid on the face as they are, most only scraped it)


def flat_regions(V, F, unit=1.0):
    """The flat faces (coplanar edge-connected regions) of at least FLAT_AREA: ([(face indices, normal, area)],
    face areas)."""
    c = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    ar = 0.5 * np.linalg.norm(c, axis=1)
    nr = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-30)
    emap = {}
    for i, (a, b, d) in enumerate(F):
        for e in ((a, b), (b, d), (d, a)):
            emap.setdefault((min(e), max(e)), []).append(i)
    nb = [[] for _ in range(len(F))]
    for fs in emap.values():
        if len(fs) == 2:
            nb[fs[0]].append(fs[1])
            nb[fs[1]].append(fs[0])
    cosa = math.cos(math.radians(FLAT_DEG))
    tol = FLAT_TOL * unit
    lab = np.full(len(F), -1)
    out = []
    for seed in np.argsort(-ar):
        if lab[seed] >= 0:
            continue
        lab[seed] = seed
        reg = [seed]
        q = [seed]
        acc = nr[seed] * ar[seed]
        p0 = V[F[seed]].mean(0)
        while q:
            f = q.pop()
            m = acc / (np.linalg.norm(acc) + 1e-30)
            for h in nb[f]:
                if lab[h] < 0 and nr[h] @ m >= cosa and np.abs((V[F[h]] - p0) @ m).max() <= tol:
                    lab[h] = seed
                    reg.append(h)
                    q.append(h)
                    acc = acc + nr[h] * ar[h]
        A = float(ar[reg].sum())
        if A >= FLAT_AREA * unit * unit:
            out.append((np.array(reg), acc / (np.linalg.norm(acc) + 1e-30), A))
    return out, ar


def place_dents(m, dents, size=DENT_SIZE, amount=DENT_AMOUNT, unit=1.0, seed=0, cut_faces=None, info=None):
    """Stone Damage's dents on the flat faces of m: [(V, F)] in the mesh's units. dents: [(V, F)] in Blender units,
    Z out of the stone. cut_faces: only the flat faces marked cut (Only Cut Edges)."""
    info = {} if info is None else info
    if not dents or amount <= 0:
        return []
    mm, mn = _ml()
    rng = np.random.default_rng(seed + 1)
    V = np.asarray(mn.getNumpyVerts(m), np.float64)
    F = np.asarray(mn.getNumpyFaces(m.topology), np.int64)
    regs, ar = flat_regions(V, F, unit)
    info["flat_faces"] = len(regs)
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
        for _ in range(k_n):
            dv, dt = dents[int(rng.integers(0, len(dents)))]
            dv = np.asarray(dv, np.float64)
            sc = float(size) * rng.uniform(*DENT_RANGE)
            half = 0.5 * max(np.ptp(dv[:, 0]), np.ptp(dv[:, 1])) * sc * unit
            for _try in range(10):
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
                    break
            else:
                continue
            B = _frame(n, rng)
            s3 = sc * (1.0 + rng.uniform(-AXIS_JITTER, AXIS_JITTER, 3))
            own = max(-float(dv[:, 2].min()) * s3[2] * unit, 0.0)       # (how far it reaches under its origin)
            bite = rng.uniform(*DENT_BITE) * 2.0 * half
            Vd = ((dv * s3) @ B.T) * unit + p - n * max(bite - own, 0.0)
            out.append((Vd.astype(np.float32), np.asarray(dt, np.int32)))
    info["dents"] = len(out)
    return out


def _cutter(mm, mn, cut):
    V, F = cut
    return mn.meshFromFacesVerts(np.ascontiguousarray(F, np.int32), np.ascontiguousarray(V, np.float32))


def _holes(m):
    return len(m.topology.findHoleRepresentiveEdges())


def dent_chip(mesh, dents, size=1.0, amount=AMOUNT, unit=1.0, seed=None, cut_faces=None, info=None,
              face_dents=None, dent_size=DENT_SIZE, dent_amount=DENT_AMOUNT):
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
    cuts = place_strokes(m, size, amount, unit, seed, cut_faces, info)
    cuts += place_dents(m, face_dents or [], dent_size, dent_amount, unit, seed, cut_faces, info)
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


def dent_trim(mesh, res, dents, size=1.0, amount=AMOUNT, unit=1.0, cut_faces=None, info=None,
              face_dents=None, dent_size=DENT_SIZE, dent_amount=DENT_AMOUNT):
    """The Chip method of Trim Edges, the way the operator runs it: the strokes and dents, then Trim Thin.
    (mesh, None); raises MeshCollapsedError when Trim Thin leaves nothing."""
    from .trim_round import trim_thin
    chipped = dent_chip(mesh, dents, size, amount, unit, cut_faces=cut_faces, info=info, face_dents=face_dents,
                        dent_size=dent_size, dent_amount=dent_amount)
    return trim_thin(chipped, res)
