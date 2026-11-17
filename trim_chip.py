"""Flat-chip Trim Edges (prototype, no UI yet).

Today's Trim Edges rounds the thin edges of a stone with a voxel opening (grow / shrink / regrow /
intersect). This module chips them FLAT instead, like a chip knocked off a stone (Alex 2026-11-17):

1. the mesh is re-meshed finely where it may be thin (meshlib subdivideMesh, with a region);
2. the thickness at every vertex is read (meshlib computeInSphereThicknessAtVertices: diameter of the
   largest ball inside the stone touching that vertex);
3. the faces thinner than `thin` form the chip regions (cleaned with meshlib expand / shrink);
4. each region is cut out and the hole refilled with flat triangles spanning its rim (meshlib fillHole,
   or buildCylinderBetweenTwoHoles for a band that runs all the way round), with no new vertices and no
   smoothing, so the chip is a ruled, flat facet from one face of the edge to the other. A strip that
   wraps round a corner comes out as several flat facets;
5. a chip whose new faces would stand outside the original stone is left out (checked with meshlib
   findSignedDistances), the rest is decimated back on its flat parts only (meshlib decimateMesh, tiny
   max error) so the face count stays near the original.

The "Thin" slider (`size`, 0.02 to 0.04 BU, default 0.03): faces thinner than 1.2 x size (in-sphere
thickness) are chipped. Measured on 14 stones (trim-edges-flat-chip): 0.03 chips about the same area as
today's Stone-preset Trim Edges, 0.02 lighter, 0.04 heavier. Sharp edges get a straight flat chord, not
today's deep rounding, so less volume goes than today on 30 to 60 degree wedges.

Only cut edges (switch): chip only near a cut face. Cut faces come from the caller (Ashlarize's sp_cut
face attribute) or, when none are given, from detect_cut_faces: coplanar regions with a long outline
(the plane of a boolean cuts many edges of the stone), unlike the few-sided flat facets of the stones.

Units: everything is in the mesh's units; `unit` = how many mesh units make one Blender unit (Quick
Infill works at x10), and every length argument is given in Blender units.
"""
import math
import time
import os
_DBG = bool(os.environ.get("CHIPDBG"))

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


THICK_PER_SIZE = 1.2  # faces thinner than this x Thin are chipped (Thin 0.03 = today's look, see docstring)
SIZE = 0.03            # BU, default of the "Thin" slider (range 0.02 to 0.04): chip face width at a 90 deg edge
CUT_REACH = 2.0        # only-cut mode: chip within CUT_REACH x thin of a cut face's rim
MAX_OUTSIDE = 0.002    # BU: a chip face standing further outside the original stone is refused
MIN_CHIP = 0.25        # a chip region smaller than MIN_CHIP x thin^2 (area) is ignored
PIECE = 6.0          # long strips are cut into pieces this many times the thickness long
MAX_SHARE = 0.3        # a chip region larger than this share of the stone's area is refused (thin all over)
TILT = 8.0           # deg: chip planes lean up to this much either way along the edge (today: ~10 deg)
BOOL_DEPTH = 0.5    # boolean method: chip plane at this share of the outline's heights
MAX_CUT = 1.5       # boolean method: a chip goes at most this many x thickness deep
BOX_PAD = 2.0       # boolean method: the cutter reaches this many x thickness past the chip
PLANE_DEPTH = 0.85   # the chip plane sits at this share of its outline's heights (from inside); with the
                     # outline fixed, a lower plane folds the faces next to it (tested 0.15 / 0.5 / 0.85 / 1.0)
METRICS = ("minarea", "universal", "plane", "circum")   # meshlib fill metrics tried in turn
DETECT_ANGLE = 2.0     # deg: cut face detection, normal tolerance
DETECT_TOL = 0.0005    # BU: cut face detection, distance off the region's plane
DETECT_AREA = 0.005    # BU^2: least area of a cut face
DETECT_OUTLINE = 16    # least number of outline vertices of a cut face
DECIMATE_ERROR = 0.0002  # BU: decimation after the chip only merges faces this close to flat


def _arrays(mm, mn, m):
    if m.topology.numValidFaces() != m.topology.faceSize() or m.topology.numValidVerts() != m.topology.vertSize():
        m.pack()
    return np.array(mn.getNumpyVerts(m), float), np.array(mn.getNumpyFaces(m.topology), np.int64)


def _face_areas(V, F):
    c = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    a = np.linalg.norm(c, axis=1)
    return a / 2, c / (a[:, None] + 1e-30)


def _bits(mn, mask, kind):
    mask = np.ascontiguousarray(mask, dtype=bool)
    return mn.faceBitSetFromBools(mask) if kind == "f" else mn.vertBitSetFromBools(mask)


def _np_bits(mn, bs, n):
    a = np.array(mn.getNumpyBitSet(bs), bool)
    out = np.zeros(n, bool)
    out[:min(n, a.size)] = a[:n]
    return out


def detect_cut_faces(mesh, unit=1.0):
    """Faces of `mesh` that look like boolean cut faces: coplanar edge-connected regions of at least
    DETECT_AREA with at least DETECT_OUTLINE outline vertices. Returns a bool array over faces."""
    mm, mn = _ml()
    V, F = _arrays(mm, mn, mesh)
    ar, nr = _face_areas(V, F)
    emap = {}
    for i, (a, b, c) in enumerate(F):
        for e in ((a, b), (b, c), (c, a)):
            emap.setdefault((min(e), max(e)), []).append(i)
    nb = [[] for _ in range(len(F))]
    for fs in emap.values():
        if len(fs) == 2:
            nb[fs[0]].append(fs[1]); nb[fs[1]].append(fs[0])
    cosa = math.cos(math.radians(DETECT_ANGLE))
    tol = DETECT_TOL * unit
    lab = np.full(len(F), -1)
    out = np.zeros(len(F), bool)
    for seed in np.argsort(-ar):
        if lab[seed] >= 0:
            continue
        lab[seed] = seed
        reg = [seed]; q = [seed]; acc = nr[seed] * ar[seed]; p0 = V[F[seed]].mean(0)
        while q:
            f = q.pop()
            m = acc / (np.linalg.norm(acc) + 1e-30)
            for h in nb[f]:
                if lab[h] < 0 and nr[h] @ m >= cosa and np.abs((V[F[h]] - p0) @ m).max() <= tol:
                    lab[h] = seed; reg.append(h); q.append(h); acc = acc + nr[h] * ar[h]
        if ar[reg].sum() < DETECT_AREA * unit * unit:
            continue
        ec = {}
        for i in reg:
            a, b, c = F[i]
            for e in ((a, b), (b, c), (c, a)):
                k = (min(e), max(e)); ec[k] = ec.get(k, 0) + 1
        if len({x for k, c in ec.items() if c == 1 for x in k}) >= DETECT_OUTLINE:
            out[reg] = True
    return out


def _subdivide(mm, mn, m, region_mask, edge):
    ss = mm.SubdivideSettings()
    ss.maxEdgeLen = float(edge)
    ss.maxEdgeSplits = 3_000_000
    ss.maxDeviationAfterFlip = 0.0
    ss.maxAngleChangeAfterFlip = 0.0      # no edge flips: on a rough stone they can fold faces through
    ss.criticalAspectRatioFlip = 1e9      # each other (215 crossings on one test stone)
    if region_mask is not None:
        reg = _bits(mn, region_mask, "f")
        ss.region = reg
    mm.subdivideMesh(m, ss)


def _thickness(mm, m, radius):
    st = mm.InSphereSearchSettings()
    st.maxRadius = float(radius)
    return np.array(mm.computeInSphereThicknessAtVertices(m, st).vec, float)


def _components(F, mask):
    """Vertex-connected groups of the faces in mask (two chips never meet at a lone vertex)."""
    idx = np.nonzero(mask)[0]
    emap = {}
    for f in idx:
        for v in F[f]:
            emap.setdefault(int(v), []).append(int(f))
    parent = {int(f): int(f) for f in idx}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for fs in emap.values():
        ra = find(fs[0])
        for f2 in fs[1:]:
            rb = find(f2)
            if ra != rb:
                parent[rb] = ra
    groups = {}
    for f in idx:
        groups.setdefault(find(int(f)), []).append(int(f))
    return [np.array(g) for g in groups.values()]


def _loops_ok(mm, topo, bs):
    """Outline loops of a region and whether they are clean (no vertex visited twice: deleting the
    region would then leave one hole per loop)."""
    loops = [list(lp) for lp in mm.findLeftBoundary(topo, bs)]
    seen = set()
    for lp in loops:
        for e in lp:
            v = topo.org(e).get()
            if v in seen:
                return loops, False
            seen.add(v)
    return loops, True


def _clean_regions(mm, mn, m, F, ar, mask, min_area, max_area, info):
    """Turn the thin faces into clean chip regions: gaps closed, small islands of thick faces taken in,
    pinched outlines grown until every region is a disc (one loop) or a band (two loops)."""
    topo = m.topology
    reg = _bits(mn, mask, "f")
    mm.expand(topo, reg, 2)
    mm.shrink(topo, reg, 2)
    mask = _np_bits(mn, reg, len(F)) | mask
    # islands of thick faces inside a region
    rest = _bits(mn, ~mask, "f")
    for comp in mm.MeshComponents.getAllComponents(mm.MeshPart(m, rest)):
        cm = _np_bits(mn, comp, len(F))
        if ar[cm].sum() < min_area * 8:
            mask |= cm
    out = []
    for g in _components(F, mask):
        if ar[g].sum() < min_area:
            continue
        cm = np.zeros(len(F), bool); cm[g] = True
        ok = False
        for _ in range(5):
            bs = _bits(mn, cm, "f")
            loops, clean = _loops_ok(mm, topo, bs)
            if clean and len(loops) in (1, 2):
                ok = True
                break
            mm.expand(topo, bs, 1)
            cm = _np_bits(mn, bs, len(F))
        if not ok:
            info["refused_shape"] = info.get("refused_shape", 0) + 1
            continue
        if ar[cm].sum() > max_area:
            info["refused_large"] = info.get("refused_large", 0) + 1
            continue
        out.append(np.nonzero(cm)[0])
    # grown regions may now overlap or touch: keep them apart (drop the later one)
    used = np.zeros(len(F) + 1, bool)
    vused = set()
    final = []
    for g in out:
        vs = set(np.unique(F[g]).tolist())
        if used[g].any() or vs & vused:
            info["refused_touching"] = info.get("refused_touching", 0) + 1
            continue
        used[g] = True; vused |= vs
        final.append(g)
    return final


def _edge_components(F, idx):
    """Edge-connected groups among the faces idx."""
    emap = {}
    for f in idx:
        a, b, c = (int(v) for v in F[f])
        for e in ((a, b), (b, c), (c, a)):
            emap.setdefault((min(e), max(e)), []).append(int(f))
    parent = {int(f): int(f) for f in idx}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for fs in emap.values():
        if len(fs) == 2:
            ra, rb = find(fs[0]), find(fs[1])
            if ra != rb:
                parent[rb] = ra
    groups = {}
    for f in idx:
        groups.setdefault(find(int(f)), []).append(int(f))
    return [np.array(g) for g in groups.values()]


def _strips(mm, mn, m, F, ar, mask, T, info):
    """Split the thin faces into strips, one per pair of neighbouring stone faces.

    The thin edges of a stone join into one network round all its faces, which can't be cut out and
    refilled as one hole. The faces left between the thin edges (patches) are found, each thin face is
    given the two patches nearest to it along the surface (meshlib computeSurfaceDistances), and each
    edge-connected group of thin faces with the same two patches is one strip: the chip of one edge."""
    topo = m.topology
    nv = int(topo.vertSize())
    reg = _bits(mn, mask, "f")
    mm.expand(topo, reg, 2)
    mm.shrink(topo, reg, 2)
    mask = _np_bits(mn, reg, len(F)) | mask
    rest = _bits(mn, ~mask, "f")
    patches = []
    for comp in mm.MeshComponents.getAllComponents(mm.MeshPart(m, rest)):
        cm = _np_bits(mn, comp, len(F))
        if ar[cm].sum() < 2.0 * T * T:
            mask |= cm                         # small island of thick faces: part of the chip
        else:
            patches.append(cm)
    idx = np.nonzero(mask)[0]
    info["patches"] = len(patches)
    if not len(idx) or len(patches) < 1:
        return []
    in_reg = np.zeros(nv, bool); in_reg[F[idx].ravel()] = True
    D = np.full((len(patches), len(idx)), np.inf)
    for k, cm in enumerate(patches):
        pv = np.zeros(nv, bool); pv[F[cm].ravel()] = True
        start = pv & in_reg
        if not start.any():
            continue
        d = np.array(mm.computeSurfaceDistances(m, _bits(mn, start, "v"), float(8 * T)).vec, float)[:nv]
        d[d > 1e30] = np.inf
        D[k] = d[F[idx]].sum(1)
    if len(patches) == 1:
        lab = np.zeros(len(idx), np.int64)
    else:
        two = np.argsort(D, axis=0)[:2]
        lo, hi = np.minimum(two[0], two[1]), np.maximum(two[0], two[1])
        lab = lo * len(patches) + hi
    strips = []
    V = np.array(mn.getNumpyVerts(m), float)
    for L in np.unique(lab):
        for g in _edge_components(F, idx[lab == L]):
            if ar[g].sum() < MIN_CHIP * T * T:
                continue
            # long strips are cut into pieces of about PIECE x T along their length: each piece is
            # one flat facet, so a chip follows a curved or wavy edge in several flats
            c = V[F[g]].mean(1)
            cc = c - c.mean(0)
            ax = np.linalg.eigh(cc.T @ cc)[1][:, 2]
            s = cc @ ax
            n = max(1, int(round(np.ptp(s) / (PIECE * T))))
            bins = np.minimum(((s - s.min()) / (np.ptp(s) + 1e-12) * n).astype(int), n - 1)
            for b in range(n):
                for h in _edge_components(F, g[bins == b]):
                    if ar[h].sum() >= 0.2 * MIN_CHIP * T * T:
                        strips.append(h)
    strips.sort(key=lambda g: -ar[g].sum())
    return strips


def _cut_rim_distance(mm, mn, m, cut_mask, reach):
    """Surface distance of every vertex to the rim of the cut faces (meshlib computeSurfaceDistances)."""
    V, F = _arrays(mm, mn, m)
    nv = len(V)
    on_cut = np.zeros(nv, bool); on_nat = np.zeros(nv, bool)
    on_cut[F[cut_mask].ravel()] = True
    on_nat[F[~cut_mask].ravel()] = True
    rim = on_cut & on_nat
    if not rim.any():
        return np.full(nv, np.inf)
    d = np.array(mm.computeSurfaceDistances(m, _bits(mn, rim, "v"), float(reach) * 1.5).vec, float)[:nv]
    d[d > 1e30] = np.inf
    return d


def _map_cut(mm, mn, src, src_cut, dst):
    """Cut flags of dst's faces, from the src face each face centre lies on (meshlib findProjection)."""
    V, F = _arrays(mm, mn, dst)
    cen = V[F].mean(1)
    out = np.zeros(len(F), bool)
    mp = mm.MeshPart(src)
    for i, c in enumerate(cen):
        pr = mm.findProjection(mm.Vector3f(*c), mp)
        out[i] = src_cut[pr.proj.face.get()]
    return out


def _fill_component(mm, mn, m, loops, metric_name):
    """Refill the hole(s) left by one removed region; returns a bool array of the new faces
    (meshlib reuses the ids of deleted faces, so they are found by comparing valid faces)."""
    before = np.array(mn.getNumpyBitSet(m.topology.getValidFaces()), bool)
    if len(loops) == 2:
        p = mm.StitchHolesParams()
        p.metric = _metric(mm, m, loops[0][0], metric_name)
        mm.buildCylinderBetweenTwoHoles(m, loops[0][0], loops[1][0], p)
    else:
        for lp in loops:
            p = mm.FillHoleParams()
            p.metric = _metric(mm, m, lp[0], metric_name)
            p.multipleEdgesResolveMode = mm.FillHoleParams.MultipleEdgesResolveMode.Strong
            mm.fillHole(m, lp[0], p)
    after = np.array(mn.getNumpyBitSet(m.topology.getValidFaces()), bool)
    before = np.pad(before, (0, max(0, after.size - before.size)))[:after.size]
    return after & ~before


def _metric(mm, m, e, name):
    if name == "plane":
        return mm.getPlaneNormalizedFillMetric(m, e)
    if name == "universal":
        return mm.getUniversalMetric(m)
    if name == "circum":
        return mm.getCircumscribedMetric(m)
    return mm.getMinAreaMetric(m)


def flat_chip(mesh, size=SIZE, *, unit=1.0, only_cut=False, cut_faces=None, detect=True, metric="minarea",
              decimate=True, method="boolean", tilt=TILT, seed=0, info=None):
    """Return a new meshlib mesh with the thin edges of `mesh` chipped flat.

    size: the "Thin" slider, 0.02 to 0.04 BU. Faces thinner than 1.2 x size (meshlib in-sphere thickness)
    are chipped. unit: mesh units per BU.
    only_cut: chip only within CUT_REACH x thin of a cut face's rim. cut_faces: bool array over the
    input's faces (None: detect them when `detect`). info: dict filled with numbers for reports."""
    mm, mn = _ml()
    info = {} if info is None else info
    t0 = time.perf_counter()
    tm = info.setdefault("timings", {})
    _last = [t0]

    def lap(name):
        now = time.perf_counter()
        tm[name] = tm.get(name, 0.0) + now - _last[0]
        _last[0] = now
    T = THICK_PER_SIZE * float(size) * unit
    info["thickness"] = T / unit
    src = mm.Mesh(mesh)
    src.pack()
    m = mm.Mesh(src)
    V, F = _arrays(mm, mn, m)
    area0 = float(m.area())
    holes0 = len(src.topology.findHoleRepresentiveEdges())
    info["source_holes"] = holes0
    info["source_self_intersections"] = len(mm.findSelfCollidingTriangles(mm.MeshPart(src)))

    lap("setup")
    cut0 = None
    if only_cut:
        if cut_faces is not None:
            cut0 = np.asarray(cut_faces, bool)[:len(F)]
            info["cut_source"] = "marked"
        elif detect:
            cut0 = detect_cut_faces(src, unit)
            info["cut_source"] = "detected"
        info["cut_area"] = float(_face_areas(V, F)[0][cut0].sum()) / unit ** 2 if cut0 is not None else 0.0
        if cut0 is None or not cut0.any():
            info.update(chips=0, reason="no cut faces", seconds=time.perf_counter() - t0)
            return src

    lap("cut_faces")
    # 0. sliver triangles (booleans leave many) collapsed first, shape kept within 0.00001 BU: split
    #    finely, they read as crossing faces and blocked chips next to them
    fp = mm.FixMeshDegeneraciesParams()
    fp.mode = mm.FixMeshDegeneraciesParams.Mode.Decimate
    fp.maxDeviation = 1e-5 * unit
    fp.tinyEdgeLength = 1e-5 * unit
    fp.criticalTriAspectRatio = 1e4
    try:
        mm.fixMeshDegeneracies(m, fp)
        m.pack()
    except Exception as e:
        info["degeneracy_fix_error"] = repr(e)

    try:
        if len(mm.findMultipleEdges(m.topology)):
            mm.fixMultipleEdges(m)         # a doubled edge (sliver collapse can leave one) blocks every boolean
            m.pack()
    except Exception as e:
        info["multiple_edge_fix_error"] = repr(e)
    lap("degeneracies")
    base = mm.Mesh(m)                  # coarse (slivers collapsed): the boolean method cuts this one
    multi0 = len(mm.findMultipleEdges(base.topology))

    # 1. fine mesh where the stone may be thin: coarse read first, then subdivide that region to thin / 5
    _subdivide(mm, mn, m, None, 2.0 * T)
    lap("subdivide_1")
    th = _thickness(mm, m, 1.5 * T)
    lap("thickness_1")
    V, F = _arrays(mm, mn, m)
    near = (th[F] < 2.5 * T).any(1)
    reg = _bits(mn, near, "f")
    mm.expand(m.topology, reg, 2)
    _subdivide(mm, mn, m, _np_bits(mn, reg, len(F)), T / 5.0)
    lap("subdivide_2")
    V, F = _arrays(mm, mn, m)
    info["verts_fine"] = len(V)
    info["fine_self_intersections"] = len(mm.findSelfCollidingTriangles(mm.MeshPart(m)))

    lap("fine_check")
    # 2. thickness at each vertex
    th = _thickness(mm, m, T)
    lap("thickness_2")
    ar, nr = _face_areas(V, F)

    # 3. chip regions: faces whose three vertices are all thinner than T, small gaps closed
    thin_f = (th[F] < T).all(1)
    if only_cut:
        cutm = _map_cut(mm, mn, src, cut0, m)
        dist = _cut_rim_distance(mm, mn, m, cutm, CUT_REACH * T)
        thin_f &= (dist[F] <= CUT_REACH * T).all(1)
    lap("regions")
    strips = _strips(mm, mn, m, F, ar, thin_f, T, info)

    lap("strips")
    srcx = info["source_self_intersections"]
    Vm, Fm = _arrays_nopack(mn, m)

    nodec = [False]

    def build_plane(skip):
        # 4. each piece is chipped by ONE plane: fitted to the piece's outline, tilted a little at random
        #    along the edge (today's chips lean ~10 deg), set into the stone at the PLANE_DEPTH share of
        #    the outline; every vertex of the piece standing above it is moved straight down onto it.
        #    Vertices only move inward and the topology stays as it is: the result stays closed and
        #    inside the original, and each chip face is exactly flat.
        V2 = Vm.copy()
        moved_f = np.zeros(len(Fm), bool)
        accepted, chip_area = [], 0.0
        nrmV = np.zeros_like(Vm)
        fnm = np.cross(Vm[Fm[:, 1]] - Vm[Fm[:, 0]], Vm[Fm[:, 2]] - Vm[Fm[:, 0]])
        np.add.at(nrmV, Fm.ravel(), np.repeat(fnm, 3, axis=0))
        for si, g in enumerate(strips):
            if si in skip:
                continue
            vi = np.unique(Fm[g].ravel())
            ec = {}
            for f in g:
                a_, b_, c_ = (int(x) for x in Fm[f])
                for e in ((a_, b_), (b_, c_), (c_, a_)):
                    k = (min(e), max(e)); ec[k] = ec.get(k, 0) + 1
            bv = np.array(sorted({x for k, c in ec.items() if c == 1 for x in k}))
            if len(bv) < 3:
                continue
            P = V2[bv]
            c0 = P.mean(0)
            ev, U = np.linalg.eigh(np.cov((P - c0).T))
            ax = U[:, 2]                       # along the edge
            out = nrmV[vi].sum(0)              # outward, about the edge's bisector
            n = out - (out @ ax) * ax          # the chip plane holds the edge direction
            if np.linalg.norm(n) < 1e-12:
                continue
            n = n / np.linalg.norm(n)
            if tilt > 0:
                rng = np.random.default_rng(int(abs(c0 * 1e4).sum()) % (2 ** 31) + seed)
                ang = math.radians(rng.uniform(-tilt, tilt))
                n = n * math.cos(ang) + np.cross(ax, n) * math.sin(ang)
            lev = float(np.percentile(P @ n, 100 * PLANE_DEPTH))
            vi = np.setdiff1d(vi, bv)          # the outline stays put: faces beside the chip never fold
            h = V2[vi] @ n - lev
            up = h > 0
            if not up.any():
                continue
            V2[vi[up]] -= np.outer(h[up], n)
            if _DBG:
                _w = mn.meshFromFacesVerts(np.ascontiguousarray(Fm, np.int32), np.ascontiguousarray(V2, np.float32))
                _x = len(mm.findSelfCollidingTriangles(mm.MeshPart(_w)))
                info.setdefault("dbg_pieces", []).append(dict(si=si, x=_x, nv=int(len(vi)), nup=int(up.sum()),
                    hmax=float(h.max() / unit), nb=int(len(bv)), area=float(ar[g].sum() / unit ** 2),
                    ang_out=float(np.degrees(np.arccos(np.clip(n @ out / np.linalg.norm(out), -1, 1))))))
            moved_f[g] = True
            accepted.append(si)
            chip_area += float(ar[g].sum())
        w = mn.meshFromFacesVerts(np.ascontiguousarray(Fm, np.int32), np.ascontiguousarray(V2, np.float32))
        chip_mask = moved_f
        n_ok = len(accepted)
        info.update(chips=n_ok, strips=len(strips), refused={}, chip_area=chip_area / unit ** 2)
        # back to few faces where nothing changed (flat parts only); a decimation that makes the mesh
        # cut itself is retried tighter, then skipped
        if decimate and not nodec[0]:
            base = len(mm.findSelfCollidingTriangles(mm.MeshPart(w)))
            keep_out = None
            if chip_mask.any():
                nfs = int(w.topology.faceSize())
                cb = _bits(mn, np.pad(chip_mask, (0, max(0, nfs - chip_mask.size)))[:nfs], "f")
                mm.expand(w.topology, cb, 3)
                keep_out = _np_bits(mn, cb, nfs)
            for err, away in ((DECIMATE_ERROR, False), (DECIMATE_ERROR, True), (DECIMATE_ERROR / 10, True)):
                if away and keep_out is None:
                    continue
                d = mm.Mesh(w)
                ds = mm.DecimateSettings()
                ds.maxError = err * unit
                ds.maxEdgeLen = 1e9
                ds.maxTriangleAspectRatio = 1e4
                ds.criticalTriAspectRatio = 1e5
                ds.stabilizer = 0.0
                ds.optimizeVertexPos = False      # vertices stay where they are: chip edges stay crisp
                ds.packMesh = True
                if away:
                    nfs = int(d.topology.faceSize())
                    valid = _np_bits(mn, d.topology.getValidFaces(), nfs)
                    ds.region = _bits(mn, valid & ~np.pad(keep_out, (0, max(0, nfs - keep_out.size)))[:nfs], "f")
                mm.decimateMesh(d, ds)
                if len(mm.findSelfCollidingTriangles(mm.MeshPart(d))) <= base:
                    w = d
                    info["decimate_error"] = err
                    info["decimate_mode"] = "away from chips" if away else "all"
                    break
            else:
                info["decimate_error"] = None
        return w, accepted

    def build_fill(skip):
        # 4. cut out and refill flat, one strip at a time on a copy; a strip is kept only if the result
        #    has no new hole, no new self-intersection and no new face standing outside the original
        w = mm.Mesh(m)
        holes = len(w.topology.findHoleRepresentiveEdges())
        selfx0 = len(mm.findSelfCollidingTriangles(mm.MeshPart(w)))
        n_ok, chip_area, why = 0, 0.0, {}
        chip_mask = np.zeros(0, bool)
        accepted = []
        for si, g in enumerate(strips):
            if si in skip:
                continue
            mask = np.zeros(len(F), bool); mask[g] = True
            bs = _bits(mn, mask, "f")
            loops, clean = _loops_ok(mm, w.topology, bs)
            if not clean or len(loops) not in (1, 2):
                why["shape"] = why.get("shape", 0) + 1
                continue
            ok = None
            for met in (metric,) + tuple(x for x in METRICS if x != metric):
                w2 = mm.Mesh(w)
                w2.deleteFaces(bs)
                try:
                    nf = _fill_component(mm, mn, w2, loops, met)
                except Exception as e:
                    ok = "fill_error"; continue
                if len(w2.topology.findHoleRepresentiveEdges()) > holes:
                    ok = "hole_left"; continue
                W, WF = _arrays_nopack(mn, w2)
                fm = np.zeros(len(WF), bool)
                fm[:min(len(WF), nf.size)] = nf[:len(WF)]
                if not fm.any():
                    ok = "no_faces"; continue
                tri = W[WF[fm]]
                pts = np.concatenate([tri.mean(1), (tri[:, 0] + tri[:, 1]) / 2, (tri[:, 1] + tri[:, 2]) / 2,
                                      (tri[:, 2] + tri[:, 0]) / 2])
                vc = mn.fromNumpyArray(np.ascontiguousarray(pts, np.float32))
                if not isinstance(vc, mm.VertCoords):
                    tmp = mm.VertCoords(); tmp.vec = vc; vc = tmp
                if float(np.array(mm.findSignedDistances(src, vc).vec, float).max()) > MAX_OUTSIDE * unit:
                    ok = "outside"; continue
                if len(mm.findSelfCollidingTriangles(mm.MeshPart(w2))) > selfx0:
                    ok = "self_intersection"; continue
                ok = True
                if met != metric:
                    why["used_metric_" + met] = why.get("metric_" + met, 0) + 1
                break
            if ok is not True:
                why[ok] = why.get(ok, 0) + 1
                continue
            w = w2
            chip_mask = np.pad(chip_mask, (0, max(0, len(fm) - chip_mask.size)))
            chip_mask[:len(fm)] |= fm
            n_ok += 1
            accepted.append(si)
            chip_area += float(ar[g].sum())
        info.update(chips=n_ok, strips=len(strips), refused=why, chip_area=chip_area / unit ** 2)
        # back to few faces where nothing changed (flat parts only); a decimation that makes the mesh
        # cut itself is retried tighter, then skipped
        if decimate and not nodec[0]:
            base = len(mm.findSelfCollidingTriangles(mm.MeshPart(w)))
            keep_out = None
            if chip_mask.any():
                nfs = int(w.topology.faceSize())
                cb = _bits(mn, np.pad(chip_mask, (0, max(0, nfs - chip_mask.size)))[:nfs], "f")
                mm.expand(w.topology, cb, 3)
                keep_out = _np_bits(mn, cb, nfs)
            for err, away in ((DECIMATE_ERROR, False), (DECIMATE_ERROR, True), (DECIMATE_ERROR / 10, True)):
                if away and keep_out is None:
                    continue
                d = mm.Mesh(w)
                ds = mm.DecimateSettings()
                ds.maxError = err * unit
                ds.maxEdgeLen = 1e9
                ds.maxTriangleAspectRatio = 1e4
                ds.criticalTriAspectRatio = 1e5
                ds.stabilizer = 0.0
                ds.optimizeVertexPos = False      # vertices stay where they are: chip edges stay crisp
                ds.packMesh = True
                if away:
                    nfs = int(d.topology.faceSize())
                    valid = _np_bits(mn, d.topology.getValidFaces(), nfs)
                    ds.region = _bits(mn, valid & ~np.pad(keep_out, (0, max(0, nfs - keep_out.size)))[:nfs], "f")
                mm.decimateMesh(d, ds)
                if len(mm.findSelfCollidingTriangles(mm.MeshPart(d))) <= base:
                    w = d
                    info["decimate_error"] = err
                    info["decimate_mode"] = "away from chips" if away else "all"
                    break
            else:
                info["decimate_error"] = None
        return w, accepted

    CUBE = np.array([(0, 2, 3), (0, 3, 1), (4, 5, 7), (4, 7, 6), (0, 1, 5), (0, 5, 4),
                     (2, 6, 7), (2, 7, 3), (0, 4, 6), (0, 6, 2), (1, 3, 7), (1, 7, 5)], np.int32)

    def build_boolean(skip):
        # 4. each piece is chipped by ONE plane, as a real boolean: a box whose bottom face is the chip
        #    plane is subtracted from the stone (meshlib boolean, DifferenceAB). The box reaches well
        #    past the chip across the edge, so the plane runs out where it meets the two faces: the chip
        #    edge is the exact line where plane and stone meet (no re-mesh staircase). Its ends are square
        #    to the edge at the piece's ends, its top just above the chip. Worked on the coarse mesh
        #    (the fine one only chooses the piece and its plane); nothing can stand outside the stone.
        w = mm.Mesh(base)
        nrmV = np.zeros_like(Vm)
        fnm = np.cross(Vm[Fm[:, 1]] - Vm[Fm[:, 0]], Vm[Fm[:, 2]] - Vm[Fm[:, 0]])
        np.add.at(nrmV, Fm.ravel(), np.repeat(fnm, 3, axis=0))
        accepted, chip_area, why = [], 0.0, {}
        for si, g in enumerate(strips):
            if si in skip:
                continue
            vi = np.unique(Fm[g].ravel())
            ec = {}
            for f in g:
                a_, b_, c_ = (int(x) for x in Fm[f])
                for e in ((a_, b_), (b_, c_), (c_, a_)):
                    k = (min(e), max(e)); ec[k] = ec.get(k, 0) + 1
            bv = np.array(sorted({x for k, c in ec.items() if c == 1 for x in k}))
            if len(bv) < 3:
                continue
            P = Vm[bv]
            c0 = P.mean(0)
            ev, U = np.linalg.eigh(np.cov((P - c0).T))
            ax = U[:, 2]
            out = nrmV[vi].sum(0)
            n = out - (out @ ax) * ax
            if np.linalg.norm(n) < 1e-12:
                continue
            n = n / np.linalg.norm(n)
            if tilt > 0:
                rng = np.random.default_rng(int(abs(c0 * 1e4).sum()) % (2 ** 31) + seed)
                ang = math.radians(rng.uniform(-tilt, tilt))
                n = n * math.cos(ang) + np.cross(ax, n) * math.sin(ang)
            u = np.cross(n, ax)
            lev = float(np.percentile(P @ n, 100 * BOOL_DEPTH))
            R = Vm[vi]
            lev = max(lev, float((R @ n).max()) - MAX_CUT * T)   # a sharp wedge is chipped, not sliced off
            hs = R @ n - lev
            if hs.max() <= 0:
                continue
            s_ = R @ ax; u_ = R @ u
            pad = BOX_PAD * T
            lo = np.array([s_.min(), u_.min() - pad, lev])
            hi = np.array([s_.max(), u_.max() + pad, lev + hs.max() + pad])
            corners = []
            for k in (0, 1):
                for j in (0, 1):
                    for i in (0, 1):
                        x = (lo[0], hi[0])[i]; y = (lo[1], hi[1])[j]; z = (lo[2], hi[2])[k]
                        corners.append(ax * x + u * y + n * z)
            box = mn.meshFromFacesVerts(CUBE, np.ascontiguousarray(np.array(corners), np.float32))
            if float(box.volume()) < 0:
                box = mn.meshFromFacesVerts(np.ascontiguousarray(CUBE[:, ::-1]), np.ascontiguousarray(np.array(corners), np.float32))
            try:
                res = mm.boolean(w, box, mm.BooleanOperation.DifferenceAB)
            except Exception:
                why["boolean_error"] = why.get("boolean_error", 0) + 1
                continue
            if not res.valid():
                why["boolean_invalid"] = why.get("boolean_invalid", 0) + 1
                continue
            r = res.mesh
            if len(r.topology.findHoleRepresentiveEdges()) > holes0 or r.topology.numValidFaces() == 0:
                why["boolean_open"] = why.get("boolean_open", 0) + 1
                continue
            if len(mm.findMultipleEdges(r.topology)) > multi0:
                why["boolean_double_edge"] = why.get("boolean_double_edge", 0) + 1
                continue
            w = r
            accepted.append(si)
            chip_area += float(ar[g].sum())
        w.pack()
        info.update(chips=len(accepted), strips=len(strips), refused=why, chip_area=chip_area / unit ** 2)
        return w, accepted

    build = {"plane": build_plane, "boolean": build_boolean}.get(method, build_fill)

    # 6. the result must not cut itself more than the input did: strips next to any new crossing are
    #    left out and the result is built again
    skip = set()
    for _pass in range(4):
        lap("before_build")
        w, accepted = build(skip)
        lap("build")
        pairs = mm.findSelfCollidingTriangles(mm.MeshPart(w))
        lap("crossing_check")
        if len(pairs) <= srcx or _pass == 3:
            break
        W, WF = _arrays_nopack(mn, w)
        bad_pts = []
        for p in pairs:
            for f in (_fid(p.aFace), _fid(p.bFace)):
                bad_pts.append(W[WF[f]].mean(0))
        bad_pts = np.array(bad_pts)
        before = len(skip)
        for si in accepted:
            sv = Vm[np.unique(Fm[strips[si]].ravel())]
            dd = np.sqrt(((sv[:, None, :] - bad_pts[None, :, :]) ** 2).sum(-1)).min()
            if dd < 3 * T:
                skip.add(si)
        if len(skip) == before:
            if nodec[0]:
                break
            nodec[0] = True               # crossings away from every chip: from the decimation
    info["self_intersection_passes"] = _pass + 1
    info["decimated"] = decimate and not nodec[0]
    info["dropped_for_crossing"] = len(skip)
    info["final_self_intersections"] = len(pairs)
    w.pack()
    lap("finish")
    info["seconds"] = time.perf_counter() - t0
    return w


def _fid(x):
    x = x() if callable(x) else x
    return x.get()


def _arrays_nopack(mn, m):
    return np.array(mn.getNumpyVerts(m), float), np.array(mn.getNumpyFaces(m.topology), np.int64)


def check_manifold(mesh):
    """(closed, multiple edges, self-intersecting face pairs)."""
    mm, _ = _ml()
    holes = len(mesh.topology.findHoleRepresentiveEdges())
    multi = len(mm.findMultipleEdges(mesh.topology))
    selfx = len(mm.findSelfCollidingTriangles(mm.MeshPart(mesh)))
    return holes == 0 and multi == 0 and selfx == 0, dict(holes=holes, multiple_edges=multi, self_intersections=selfx)


# ---------------------------------------------------------------------------------------------------
# Fast variant (speed gate, Alex 2026-11-17: "can we do this faster than trim edges?")
#
# No re-mesh and no thickness field. The stones are low-poly: their sharp edges ARE mesh edges, so
# each edge's inside angle comes straight from its two face normals. An edge whose inside angle is
# under ALPHA_MAX is chipped by one flat plane across it, as deep as the thickness rule asks for that
# angle (a wedge of angle a is thinner than T within T / (2 tan(a/2)) of its edge: the plane through
# those two lines lies T cos^2(a/2) / (2 sin(a/2)) under the edge), capped at MAX_CUT x T. Edges are
# chained into pieces along the edge; the chip boxes are subtracted in a few booleans (boxes whose
# bounds don't overlap go in together).

ALPHA_MAX = 140.0    # deg: edges with a smaller inside angle get a chip (the depth rule keeps obtuse ones shallow)
POST_TOL = 1e-4     # fast variant: sliver clean-up after the booleans keeps the shape within this (BU)
LIP_TRIM = False     # fast variant: cut thin lips back along straight lines (_lip_boxes). OFF: on the 30 deg sliver it
                     # took 26.6 % (today 10.4 %), and on the 0102 lip most of its booleans were refused
LIP_SEGMENTS = 3     # ... in at most this many straight segments per lip
MERGE_ANGLE = 12.0   # fast variant: neighbouring chips whose planes differ less than this become one cut
LIP_MERGE_ANGLE = 35.0  # ... along a thin lip (stone under the edge thinner than LIP_THICK x T)
LIP_THICK = 1.5
MERGE_STRAIGHT = 0.5  # ... if the joined run stays within this x chip width of a straight line
SHORT_CUT = 2.0      # fast variant: a piece with two free ends shorter than this x chip width gets no chip
JOINT_OVERLAP = 0.5  # fast variant: at a joint the two cutters run this x chip width past the shared vertex
RAMP = 0.6           # fast variant: chip ends ramp to zero depth over this x the chip width
TAPER = 1.0          # fast variant: the chip ends slant out by this x the box height (chips fade out)
FACE_CAP = 0.5       # fast variant: a chip is at most this x sqrt(area of the larger face) deep: bumps (two small faces) get small chips
THICK_CAP = 0.15     # fast variant: a chip is at most this share of the stone's thickness under it
FAST_MAX_CUT = 1.0          # fast variant: a chip goes at most this many x thickness deep
FAST_THICK_PER_SIZE = 3.0   # fast variant: thickness = this x Thin (Thin 0.03 = today's chipped area, measured)
CHAIN_TURN = 35.0    # deg: neighbouring edges are one piece while they turn less than this
FAST_PIECE = 10.0    # pieces are at most this many x thickness long
SIDE_TOL = 0.1       # fast variant: the plane is lifted only where the stone beside the band stands this x h above it
SIDE_LIFT = 0.5      # fast variant: ... and by at most this share of the chip depth
MIN_PIECE = 1.5      # fast variant: pieces shorter than this x their depth get no chip (they'd only nick)
DEPTH_K = 1.6        # fast variant: chip depth = this x the depth rule below (1.6: median 0.94x today's volume over the 14 test stones)
DEPTH_RULE = "plane"  # fast variant: "wedge" = k x Thin / (2 tan(a/2)); "plane" = k x T cos^2(a/2) / (2 sin(a/2))
LIP_UNDER = 3.0      # fast variant: material under an edge thinner than this x Thin is a thin lip: it gets chipped harder
LIP_FRAC = 0.9       # fast variant: a lip at most Thin thick loses this share of it (the whole lip, less a hair)
LIP_KEEP_ANGLE = True  # fast variant: on a lip the thickness cap is off: the angle depth (or the lip rule, if deeper) applies
UNDER_BOGUS = "skip"   # fast variant: what such a reading gets: "angle" (the angle depth) or "skip" (no chip)
FACE_FREE_DEG = 90.0  # fast variant: edges sharper than this (inside angle, deg) are not face-capped (knife lips on small faces)
CHAIN_DANGLE = 25.0  # deg: a piece only takes edges whose inside angle is within this of its first edge's (a sharp run is not averaged away)
PIECE_ANGLE = "mean"  # fast variant: a piece's angle: "mean" (length-weighted) or "min" (its sharpest edge)
UNDER_MIN = 0.15     # fast variant: a thickness reading under this x Thin is the ray leaving through a neighbour face: ignored


def chip_depth(a_half, T, face_size, thick_under):
    """How deep (along the chip normal, mesh units) the chip of one piece goes. The ONE place the
    depth is decided, so it can be tuned or replaced on its own.

    a_half: half the edge's inside angle (radians); T: thickness in mesh units (FAST_THICK_PER_SIZE x
    Thin); face_size: sqrt of the area of the larger of the edge's two faces; thick_under: the stone's
    thickness straight under the edge along the chip normal (None if unknown).
    The chip goes down to where the wedge is Thin (= T / FAST_THICK_PER_SIZE) thick:
    DEPTH_K x Thin / (2 tan(a/2)), i.e. 0.015 at 90 deg and 0.026 at 60 deg for Thin 0.03 (the earlier
    T cos^2(a/2) / (2 sin(a/2)) went about 2x deeper at 90 deg and over-chipped the jamb). Capped at
    FAST_MAX_CUT x T, at FACE_CAP x face_size (bumps between two small faces get small chips) and at
    THICK_CAP x thick_under (thick stones are not sliced).
    Thin lips are never sent to zero: where the material under the edge is under LIP_UNDER x Thin,
    the chip is at least the lip rule's depth: LIP_FRAC x thick_under for a lip up to Thin thick (the
    whole lip), falling off linearly to 0 at LIP_UNDER x Thin. A reading under UNDER_MIN x Thin
    is a ray that left through a neighbour face, not a thickness: it is ignored."""
    thin = T / FAST_THICK_PER_SIZE
    if DEPTH_RULE == "plane":
        s = max(math.sin(a_half), 1e-6)
        d = DEPTH_K * T * math.cos(a_half) ** 2 / (2 * s)
    else:
        d = DEPTH_K * thin / (2 * max(math.tan(a_half), 1e-6))
    thin_lip = thick_under is not None and UNDER_MIN * thin <= thick_under < LIP_UNDER * thin
    if 2.0 * math.degrees(a_half) < FACE_FREE_DEG or thin_lip:
        h = min(d, FAST_MAX_CUT * T)              # sharp edges and thin lips: the face-size cap is off
    else:
        h = min(d, FAST_MAX_CUT * T, FACE_CAP * face_size)
    if thick_under is None:
        return h
    if thick_under < UNDER_MIN * thin:          # a knife lip (sharp edge) or a ray that left through a neighbour
        sharp = 2.0 * math.degrees(a_half) < FACE_FREE_DEG
        return h if (sharp or UNDER_BOGUS == "angle") else 0.0
    if thick_under >= LIP_UNDER * thin or not LIP_KEEP_ANGLE:
        h = min(h, THICK_CAP * thick_under)
    if thick_under < LIP_UNDER * thin:
        ramp = 1.0 if thick_under <= thin else (LIP_UNDER * thin - thick_under) / ((LIP_UNDER - 1.0) * thin)
        h = max(h, LIP_FRAC * min(thick_under, thin) * ramp)
    return min(h, FAST_MAX_CUT * T)


def _column_prism(ax, u, n, uc, half, prof, top, ztop):
    """Closed mesh (V, F) of a cutter: across the edge (u) from uc - half to uc + half; along it (ax)
    the stations of prof = [(s, depth)], bottom at top - depth (along n), top face at ztop."""
    m = len(prof)
    V = []
    for s, d in prof:
        zb = top - d
        for y in (uc - half, uc + half):
            V.append(ax * s + u * y + n * zb)
            V.append(ax * s + u * y + n * ztop)
    V = np.array(V)
    # vertex index: station i, side j (0: u-, 1: u+), level k (0 bottom, 1 top)
    idx = lambda i, j, k: 4 * i + 2 * j + k
    F = []
    for i in range(m - 1):
        a0, a1 = i, i + 1
        F += [(idx(a0, 0, 0), idx(a1, 0, 0), idx(a1, 1, 0)), (idx(a0, 0, 0), idx(a1, 1, 0), idx(a0, 1, 0))]   # bottom
        F += [(idx(a0, 0, 1), idx(a0, 1, 1), idx(a1, 1, 1)), (idx(a0, 0, 1), idx(a1, 1, 1), idx(a1, 0, 1))]   # top
        F += [(idx(a0, 0, 0), idx(a0, 0, 1), idx(a1, 0, 1)), (idx(a0, 0, 0), idx(a1, 0, 1), idx(a1, 0, 0))]   # side u-
        F += [(idx(a0, 1, 0), idx(a1, 1, 0), idx(a1, 1, 1)), (idx(a0, 1, 0), idx(a1, 1, 1), idx(a0, 1, 1))]   # side u+
    for i, rev in ((0, False), (m - 1, True)):                                                            # ends
        q = [(idx(i, 0, 0), idx(i, 1, 0), idx(i, 1, 1)), (idx(i, 0, 0), idx(i, 1, 1), idx(i, 0, 1))]
        F += [f[::-1] for f in q] if rev else q
    F = np.array(F, np.int32)
    # outward orientation: flip all faces if the signed volume comes out negative
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    if np.einsum("ij,ij->i", a, np.cross(b, c)).sum() < 0:
        F = F[:, ::-1].copy()
    return V, F


def _lip_boxes(mm, mn, m, T, info):
    """Cutters (V, F) that cut thin lips back. A lip = a connected region of faces whose three
    vertices are thinner than T (meshlib computeRayThicknessAtVertices). Its plate plane comes from
    its face normals (top and underside point opposite ways); its inner boundary (where the plate
    reaches T) is fitted with a straight line, split at real corners into up to LIP_SEGMENTS lines;
    each cutter takes away the part of the plate beyond its line, square to the plate."""
    V, F = _arrays(mm, mn, m)
    th = np.array(mm.computeRayThicknessAtVertices(m).vec, float)[:len(V)]
    thin_v = th < T
    thin_f = thin_v[F].all(1)
    ar, nr = _face_areas(V, F)
    out = []
    regions = _edge_components(F, np.nonzero(thin_f)[0]) if thin_f.any() else []
    for g in regions:
        if ar[g].sum() < T * T:
            continue
        # plate normal: the dominant axis of the face normals (top and underside are opposite)
        M = (nr[g] * ar[g, None]).T @ nr[g]
        pn = np.linalg.eigh(M)[1][:, 2]
        # inner boundary: vertices of the region's outline (edges used once inside the region)
        ec = {}
        for f in g:
            a, b, c = (int(x) for x in F[f])
            for e in ((a, b), (b, c), (c, a)):
                k = (min(e), max(e)); ec[k] = ec.get(k, 0) + 1
        bv = np.array(sorted({x for k, c_ in ec.items() if c_ == 1 for x in k}))
        if len(bv) < 3:
            continue
        rv = np.unique(F[g].ravel())
        R = V[rv]
        B = V[bv] - np.outer(V[bv] @ pn, pn)                    # boundary, flattened onto the plate
        cR = (R - np.outer(R @ pn, pn)).mean(0)
        segs = _split_line(B, T, LIP_SEGMENTS)
        for S in segs:
            c0 = S.mean(0)
            d = np.linalg.eigh(np.cov((S - c0).T) + 1e-12 * np.eye(3))[1][:, 2]
            d = d - (d @ pn) * pn; d /= (np.linalg.norm(d) + 1e-12)
            o = np.cross(pn, d)
            if (cR - c0) @ o < 0:                               # o points from the line towards the rim
                o = -o
            line = float((S @ o).max())                         # the thickest side of the boundary
            ro = R @ o
            beyond = R[ro > line]
            if len(beyond) == 0:
                continue
            sd = S @ d
            lo_d, hi_d = float(sd.min()) - 0.5 * T, float(sd.max()) + 0.5 * T
            far = float((beyond @ o).max()) + T
            pz = R @ pn
            uc = float(pz.mean()); half = float(np.ptp(pz)) / 2 + T
            prof = [(lo_d, far - line), (hi_d, far - line)]
            out.append(_column_prism(d, pn, o, uc, half, prof, far, far))
    info["lip_regions"] = len(regions)
    return out


def _split_line(B, T, most):
    """Split boundary points into up to `most` runs, each close to a straight line (within T/2)."""
    def dev(S):
        c0 = S.mean(0)
        d = np.linalg.eigh(np.cov((S - c0).T) + 1e-12 * np.eye(3))[1][:, 2]
        off = (S - c0) - np.outer((S - c0) @ d, d)
        return float(np.linalg.norm(off, axis=1).max()), d, c0
    runs = [B]
    while len(runs) < most:
        worst = max(range(len(runs)), key=lambda i: dev(runs[i])[0] if len(runs[i]) >= 4 else -1)
        e, d, c0 = dev(runs[worst]) if len(runs[worst]) >= 4 else (0.0, None, None)
        if e <= 0.5 * T:
            break
        s = (runs[worst] - c0) @ d
        cut = np.median(s)
        a, b = runs[worst][s <= cut], runs[worst][s > cut]
        if len(a) < 2 or len(b) < 2:
            break
        runs = runs[:worst] + [a, b] + runs[worst + 1:]
    return runs


def fast_chip(mesh, size=SIZE, *, unit=1.0, only_cut=False, cut_faces=None, detect=True, tilt=TILT, seed=0,
              info=None, lip_round=False):
    mm, mn = _ml()
    info = {} if info is None else info
    tm = info.setdefault("timings", {})
    t0 = time.perf_counter()
    last = [t0]

    def lap(name):
        now = time.perf_counter(); tm[name] = tm.get(name, 0.0) + now - last[0]; last[0] = now

    T = FAST_THICK_PER_SIZE * float(size) * unit
    info["thickness"] = T / unit
    m = mm.Mesh(mesh)
    m.pack()
    src0 = mm.Mesh(m)
    selfx_in = len(mm.findSelfCollidingTriangles(mm.MeshPart(src0)))
    fp = mm.FixMeshDegeneraciesParams()           # booleans next to sliver triangles leave crossings
    fp.mode = mm.FixMeshDegeneraciesParams.Mode.Decimate
    fp.maxDeviation = 1e-5 * unit
    fp.tinyEdgeLength = 1e-5 * unit
    fp.criticalTriAspectRatio = 1e4
    try:
        mm.fixMeshDegeneracies(m, fp)
        if len(mm.findMultipleEdges(m.topology)):
            mm.fixMultipleEdges(m)
        m.pack()
    except Exception as e:
        info["degeneracy_fix_error"] = repr(e)
    selfx0 = len(mm.findSelfCollidingTriangles(mm.MeshPart(m)))
    if selfx0 > selfx_in:                     # the clean-up made crossings of its own: work on the input
        m = mm.Mesh(src0); selfx0 = selfx_in
        info["degeneracy_fix_reverted"] = True
        if len(mm.findMultipleEdges(m.topology)):   # but still mend doubled edges (Blender booleans leave some)
            mm2 = mm.Mesh(m)
            try:
                mm.fixMultipleEdges(mm2)
                mm2.pack()
                if len(mm.findSelfCollidingTriangles(mm.MeshPart(mm2))) <= selfx0:
                    m = mm2
                    info["multiple_edges_fixed"] = True
            except Exception as e:
                info["multiple_edge_fix_error"] = repr(e)
    holes0 = len(m.topology.findHoleRepresentiveEdges())
    V0src, F0src = _arrays(mm, mn, src0)
    V, F = _arrays(mm, mn, m)
    cut = None
    if only_cut:
        if cut_faces is not None:
            cut = np.asarray(cut_faces, bool)[:len(F0src)]
            info["cut_source"] = "marked"
        elif detect:
            cut = detect_cut_faces(src0, unit)
            info["cut_source"] = "detected"
        if cut is None or not cut.any():
            info.update(chips=0, reason="no cut faces", seconds=time.perf_counter() - t0)
            return m
    lap("setup")

    # edges and their two faces
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    fid = np.tile(np.arange(len(F)), 3)
    key = np.minimum(E[:, 0], E[:, 1]) * (len(V) + 1) + np.maximum(E[:, 0], E[:, 1])
    order = np.argsort(key, kind="stable")
    ks = key[order]
    same = np.nonzero(ks[1:] == ks[:-1])[0]
    f1, f2 = fid[order[same]], fid[order[same + 1]]
    ev = E[order[same]]
    ar, nr = _face_areas(V, F)
    cosn = np.einsum("ij,ij->i", nr[f1], nr[f2])
    cen = V[F].mean(1)
    convex = np.einsum("ij,ij->i", cen[f2] - cen[f1], nr[f1]) < 0
    alpha = 180.0 - np.degrees(np.arccos(np.clip(cosn, -1, 1)))
    sharp = convex & (alpha < ALPHA_MAX)
    if only_cut:
        # cut flags of the (sliver-cleaned) faces next to candidate edges, from the input face each
        # one lies on (meshlib findProjection)
        cand = np.nonzero(sharp)[0]
        need = np.unique(np.concatenate([f1[cand], f2[cand]]))
        flag = np.zeros(len(F), bool)
        mp = mm.MeshPart(src0)
        for f in need:
            c = cen[f]
            flag[f] = cut[mm.findProjection(mm.Vector3f(*c), mp).proj.face.get()]
        sharp &= flag[f1] ^ flag[f2]
    idx = np.nonzero(sharp)[0]
    a2 = np.radians(alpha[idx]) / 2
    lap("sharp_edges")

    # chain edges into pieces
    bis = nr[f1[idx]] + nr[f2[idx]]
    bis /= np.linalg.norm(bis, axis=1, keepdims=True) + 1e-12
    p0, p1 = V[ev[idx, 0]], V[ev[idx, 1]]
    dirs = p1 - p0
    lens = np.linalg.norm(dirs, axis=1)
    dirs = dirs / (lens[:, None] + 1e-12)
    at = {}
    for k, (a, b) in enumerate(ev[idx]):
        at.setdefault(int(a), []).append(k); at.setdefault(int(b), []).append(k)
    used = np.zeros(len(idx), bool)
    cos_turn = math.cos(math.radians(CHAIN_TURN))
    pieces = []
    for k in np.argsort(-lens):
        if used[k]:
            continue
        used[k] = True
        piece = [k]; length = lens[k]
        for end in (int(ev[idx[k], 0]), int(ev[idx[k], 1])):
            cur, v = k, end
            while length < FAST_PIECE * T:
                nxt = [j for j in at.get(v, []) if not used[j] and abs(dirs[j] @ dirs[cur]) >= cos_turn
                       and bis[j] @ bis[cur] >= cos_turn and abs(alpha[idx[j]] - alpha[idx[k]]) < CHAIN_DANGLE]
                if not nxt:
                    break
                j = max(nxt, key=lambda q: abs(dirs[q] @ dirs[cur]))   # through junctions: the straightest
                used[j] = True; piece.append(j); length += lens[j]
                a, b = int(ev[idx[j], 0]), int(ev[idx[j], 1])
                v = b if a == v else a
                cur = j
        pieces.append(piece)
    lap("chain")

    # one cutter per piece: a box in the piece's frame whose bottom is the chip plane, only as wide as
    # the chip band (so it can't bite bumps beside the edge), barely above the edge, with its two ends
    # slanted (TAPER) so the chip fades out along the edge instead of stopping at a wall; pieces
    # overlap by those slants. The depth is capped by the stone's own thickness under the edge
    # (meshlib rayMeshIntersect, inward along the chip normal) and by the size of the smaller face.
    mp_m = mm.MeshPart(m)
    fsize = np.sqrt(np.maximum(ar[f1[idx]], ar[f2[idx]]))
    # end vertices of each piece (a vertex used once by its edges); an end shared with another piece
    # is a joint: the two chips overlap there instead of each ramping out (that left a peg between)
    ends_of = []
    end_count = {}
    for pc in pieces:
        cnt = {}
        for k in pc:
            for v in ev[idx[k]]:
                cnt[int(v)] = cnt.get(int(v), 0) + 1
        e = [v for v, c_ in cnt.items() if c_ == 1]
        ends_of.append(e)
        for v in e:
            end_count[v] = end_count.get(v, 0) + 1
    boxes = []
    planned = []
    for pi, pc in enumerate(pieces):
        P = np.concatenate([p0[pc], p1[pc]])
        ax = dirs[pc[0]] if len(pc) == 1 else np.linalg.eigh(np.cov((P - P.mean(0)).T))[1][:, 2]
        n = (bis[pc] * lens[pc, None]).sum(0)
        n = n - (n @ ax) * ax
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        c0 = P.mean(0)
        if tilt > 0:
            rng = np.random.default_rng(int(abs(c0 * 1e4).sum()) % (2 ** 31) + seed)
            ang = math.radians(rng.uniform(-tilt, tilt))
            n = n * math.cos(ang) + np.cross(ax, n) * math.sin(ang)
        u = np.cross(n, ax)
        w8 = lens[pc] / lens[pc].sum()
        a_half = float((a2[pc] * w8).sum()) if PIECE_ANGLE == "mean" else float(a2[pc].min())
        thick_under = None
        try:
            hit = mm.rayMeshIntersect(mp_m, mm.Line3f(mm.Vector3f(*(c0 - n * 1e-4 * unit)), mm.Vector3f(*(-n))))
            if hit:
                dl = hit.distanceAlongLine
                thick_under = float(dl() if callable(dl) else dl)
        except Exception:
            pass
        h = chip_depth(a_half, T, float((fsize[pc] * w8).sum()), thick_under)
        dbg = None
        if "debug" in info:              # caller asked for per-piece numbers (diagnosis)
            dbg = dict(edges=[[int(x) for x in ev[idx[k]]] for k in pc], alpha=[float(alpha[idx[k]]) for k in pc],
                       a_half_deg=math.degrees(a_half), face_size=float((fsize[pc] * w8).sum()) / unit,
                       thick_under=None if thick_under is None else thick_under / unit, T=T / unit,
                       depth=h / unit, length=float(lens[pc].sum()) / unit)
            info["debug"].append(dbg)
        if h < 0.05 * T:
            if dbg is not None:
                dbg["skip"] = "depth < 0.05 T"
            continue
        top = float((P @ n).max())
        lev = top - h
        s_ = P @ ax
        if float(s_.max() - s_.min()) < MIN_PIECE * max(h, 1e-9):
            if dbg is not None:
                dbg["skip"] = "too short"
            continue                      # too short to show a chip: it would only leave a nick
        # the chip plane must run out inside its band: where the stone just outside the band stands
        # above the plane (the rim of a pit, a bump), the cutter's side would leave a wall. Rays
        # straight down just outside both sides (meshlib rayMeshIntersect) raise the plane to it.
        half0 = h * math.tan(a_half) * 1.15 + 0.05 * T
        uc0 = float((P @ u).mean())
        for sf in (0.3, 0.5, 0.7):
            sx = float(s_.min() + sf * (s_.max() - s_.min()))
            for side in (-1.0, 1.0):
                o = ax * sx + u * (uc0 + side * half0 * 1.05) + n * (top + T)
                try:
                    hit2 = mm.rayMeshIntersect(mp_m, mm.Line3f(mm.Vector3f(*o), mm.Vector3f(*(-n))))
                except Exception:
                    hit2 = None
                if hit2:
                    d2 = hit2.distanceAlongLine
                    d2 = float(d2() if callable(d2) else d2)
                    zh = top + T - d2
                    if zh > lev + SIDE_TOL * h:
                        lev = min(max(lev, zh), top - (1.0 - SIDE_LIFT) * h)
        h = top - lev
        if dbg is not None:
            dbg["depth_after_side_clamp"] = h / unit
        if h < 0.05 * T:
            if dbg is not None:
                dbg["skip"] = "side clamp"
            continue
        half = h * math.tan(a_half) * 1.15 + 0.05 * T
        uc = uc0
        ztop = top + 0.15 * T
        length = float(s_.max() - s_.min())
        width = max(2.0 * h * math.tan(a_half), h)            # chip width across the edge
        planned.append(dict(pi=pi, ax=ax, u=u, n=n, smin=float(s_.min()), smax=float(s_.max()), uc=uc,
                            half=half, lev=lev, top=top, ztop=ztop, length=length, width=width, dbg=dbg,
                            ends=list(ends_of[pi]), P=P, a_half=a_half,
                            thin=thick_under is not None and thick_under < LIP_THICK * T, h0=top - lev))
        if dbg is not None:
            dbg.update(thin=planned[-1]["thin"], n=[round(float(x), 3) for x in n], centre=[round(float(x), 3) for x in c0])

    # merge neighbouring chips that should be one cut: two planned pieces meeting at a joint whose chip
    # planes are close (MERGE_ANGLE; LIP_MERGE_ANGLE along a thin lip) and whose joined run is still
    # nearly straight (no point further than MERGE_STRAIGHT x chip width off its line) become ONE
    # piece with one plane fitted to the whole run, so a lip or a wavy edge is cut back in one flat
    # face instead of a sawtooth of small chips
    cos_m, cos_l = math.cos(math.radians(MERGE_ANGLE)), math.cos(math.radians(LIP_MERGE_ANGLE))
    changed = True
    while changed and len(planned) > 1:
        changed = False
        at_v = {}
        for p in planned:
            for v in p["ends"]:
                at_v.setdefault(v, []).append(p)
        for v, ps in at_v.items():
            if len(ps) != 2:
                continue
            a, b = ps
            lim = cos_l if (a["thin"] and b["thin"]) else cos_m
            if float(a["n"] @ b["n"]) < lim:
                continue
            P = np.concatenate([a["P"], b["P"]])
            c0 = P.mean(0)
            axm = np.linalg.eigh(np.cov((P - c0).T))[1][:, 2]
            off = (P - c0) - np.outer((P - c0) @ axm, axm)
            W = max(a["width"], b["width"])
            if float(np.linalg.norm(off, axis=1).max()) > MERGE_STRAIGHT * W:
                continue
            la, lb = a["length"], b["length"]
            nm = a["n"] * la + b["n"] * lb
            nm = nm - (nm @ axm) * axm
            nm /= (np.linalg.norm(nm) + 1e-12)
            um = np.cross(nm, axm)
            top_m = float((P @ nm).max())
            h_m = (a["h0"] * la + b["h0"] * lb) / (la + lb)
            s_m = P @ axm
            ends = [x for x in a["ends"] + b["ends"] if x != v]
            merged = dict(pi=a["pi"], ax=axm, u=um, n=nm, smin=float(s_m.min()), smax=float(s_m.max()),
                          uc=float((P @ um).mean()), half=max(a["half"], b["half"]), lev=top_m - h_m, top=top_m,
                          ztop=top_m + 0.15 * T, length=float(s_m.max() - s_m.min()), width=W, dbg=a["dbg"],
                          ends=ends, P=P, a_half=(a["a_half"] * la + b["a_half"] * lb) / (la + lb),
                          thin=a["thin"] and b["thin"], h0=h_m)
            if merged["dbg"] is not None:
                merged["dbg"]["merged"] = merged["dbg"].get("merged", 1) + 1
            planned = [q for q in planned if q is not a and q is not b] + [merged]
            changed = True
            break
    info["pieces_after_merge"] = len(planned)

    # second pass: a piece's end is a joint only where another piece that is really cut ends too; a
    # piece with two free ends shorter than SHORT_CUT chip widths is dropped (it would only nick the
    # edge), and the joints are worked out again without it
    for _round in range(3):
        cut_ends = {}
        for p in planned:
            for v in p["ends"]:
                cut_ends[v] = cut_ends.get(v, 0) + 1
        keep = []
        for p in planned:
            joint = [False, False]                              # (low s end, high s end)
            for v in p["ends"]:
                if cut_ends.get(v, 0) >= 2:
                    sv = float(V[v] @ p["ax"])
                    joint[0 if abs(sv - p["smin"]) <= abs(sv - p["smax"]) else 1] = True
            p["joint"] = joint
            if not any(joint) and p["length"] < SHORT_CUT * p["width"]:
                if p["dbg"] is not None:
                    p["dbg"]["skip"] = "short free piece"
                continue
            keep.append(p)
        if len(keep) == len(planned):
            break
        planned = keep
    # depth of every planned chip, and which chips meet at each joint vertex
    at_vertex = {}
    for p in planned:
        p["h"] = p["top"] - p["lev"]
        for v in p["ends"]:
            at_vertex.setdefault(v, []).append(p)
    for p in planned:
        # one cutter per chip: a prism across the edge whose bottom follows the chip's depth along the
        # edge. Middle: the chip plane (depth h). Free end: the depth ramps to 0 over about one chip
        # width (RAMP), stopping at the piece's own end vertex. Joint: over about one chip width the
        # depth moves to the mean of the chips meeting there, reached at the shared vertex, and the
        # cutter runs JOINT_OVERLAP chip widths past it on the same slope, so neighbouring chips of
        # different depth meet without a step.
        h, top, W = p["h"], p["top"], p["width"]
        ramp = min(RAMP * W, 0.3 * p["length"])
        blend = min(W, 0.45 * p["length"])
        overlap = JOINT_OVERLAP * W
        ax, u, n, joint = p["ax"], p["u"], p["n"], p["joint"]
        prof = []                               # (s, depth) along the edge, s increasing
        for side, s_end, inward in ((0, p["smin"], 1.0), (1, p["smax"], -1.0)):
            if joint[side]:
                vs = [v for v in p["ends"]
                      if abs(float(V[v] @ ax) - s_end) <= abs(float(V[v] @ ax) - (p["smax"] if side == 0 else p["smin"]))]
                others = [q for v in vs for q in at_vertex.get(v, []) if q is not p]
                if others and vs:
                    # one shared point for every chip meeting here: the joint vertex, lowered by the
                    # mean depth along the mean chip normal; each chip's bottom passes through it,
                    # so neighbouring chips meet on the edge line even when their planes tilt apart
                    grp = [p] + others
                    hbar = sum(q["h"] for q in grp) / len(grp)
                    nb = sum(q["n"] for q in grp); nb = nb / (np.linalg.norm(nb) + 1e-12)
                    Pj = V[vs[0]] - nb * hbar
                    hm = max(top - float(Pj @ n), 0.0)
                else:
                    hm = h
                slope = (h - hm) / max(blend, 1e-12)            # depth change per unit towards the end
                past = max(hm - slope * overlap, 0.0)
                pts = [(s_end - inward * overlap, past), (s_end, hm), (s_end + inward * blend, h)]
            else:
                pts = [(s_end, 0.0), (s_end + inward * ramp, h)]
            prof.append(pts if side == 0 else pts[::-1])
        prof = prof[0] + prof[1]
        prof = [pt for i_, pt in enumerate(prof) if i_ == 0 or pt[0] > prof[i_ - 1][0] + 1e-9 * max(1.0, abs(pt[0]))]
        if p["dbg"] is not None:
            p["dbg"].update(joint=joint, width=W / unit, kept=True, profile=[(float(s), float(d) / unit) for s, d in prof])
        boxes.append(_column_prism(ax, u, n, p["uc"], p["half"], prof, top, p["ztop"]))
    info.update(sharp_edges=int(len(idx)), pieces=len(boxes))
    lap("boxes")

    # batched booleans: boxes whose bounds don't overlap are subtracted together
    CUBE = np.array([(0, 2, 3), (0, 3, 1), (4, 5, 7), (4, 7, 6), (0, 1, 5), (0, 5, 4),
                     (2, 6, 7), (2, 7, 3), (0, 4, 6), (0, 6, 2), (1, 3, 7), (1, 7, 5)], np.int32)
    bb = [(c[0].min(0), c[0].max(0)) for c in boxes]
    groups = []
    for i in range(len(boxes)):
        for g in groups:
            if all(np.any(bb[i][0] > bb[j][1]) or np.any(bb[j][0] > bb[i][1]) for j in g):
                g.append(i); break
        else:
            groups.append([i])

    def box_mesh(ids):
        Vs, Fs, off = [], [], 0
        for i in ids:
            Vb_, Fb_ = boxes[i]
            Vs.append(Vb_); Fs.append(Fb_ + off); off += len(Vb_)
        Vb = np.concatenate(Vs).astype(np.float32)
        Fb = np.concatenate(Fs).astype(np.int32)
        return mn.meshFromFacesVerts(np.ascontiguousarray(Fb), np.ascontiguousarray(Vb))

    multi0 = len(mm.findMultipleEdges(m.topology))
    w, done, why = m, 0, {}

    def apply(w, ids):
        try:
            r = mm.boolean(w, box_mesh(ids), mm.BooleanOperation.DifferenceAB)
        except Exception:
            return None
        if not r.valid():
            return None
        r = r.mesh
        if len(r.topology.findHoleRepresentiveEdges()) > holes0 or len(mm.findMultipleEdges(r.topology)) > multi0:
            return None
        if len(mm.findSelfCollidingTriangles(mm.MeshPart(r))) > selfx0:
            return None
        return r

    for g in groups:
        r = apply(w, g)
        if r is not None:
            w = r; done += len(g); continue
        for i in g:                                   # the group failed: box by box
            r = apply(w, [i])
            if r is None:
                why["refused"] = why.get("refused", 0) + 1
            else:
                w = r; done += 1
    info.update(chips=done, boolean_groups=len(groups), refused=why)
    lap("booleans")
    # thin lips (a plate thinner than the chip thickness: knife edges the per-edge chips can't reach
    # cleanly) are cut back square to the plate along one straight line per lip segment, so the new
    # lip edge is a straight line at about the chip thickness instead of a sawtooth of small chips
    lips = _lip_boxes(mm, mn, m, T, info) if LIP_TRIM else []
    for lb in lips:
        boxes.append(lb)
        r = apply(w, [len(boxes) - 1])
        if r is None:
            why["lip_refused"] = why.get("lip_refused", 0) + 1
        else:
            w = r; done += 1
            info["lips_cut"] = info.get("lips_cut", 0) + 1
    lap("lips")
    # the booleans leave near-flat sliver triangles; rounded to float32 at another scale (Quick Infill
    # works at x10 and scales back) they can cross. Collapse them, shape kept within POST_TOL.
    if done:
        c = mm.Mesh(w)
        pp = mm.FixMeshDegeneraciesParams()
        pp.mode = mm.FixMeshDegeneraciesParams.Mode.Decimate
        pp.maxDeviation = POST_TOL * unit
        pp.tinyEdgeLength = POST_TOL * unit
        pp.criticalTriAspectRatio = 1e3
        try:
            mm.fixMeshDegeneracies(c, pp)
            c.pack()
            if (len(c.topology.findHoleRepresentiveEdges()) <= holes0
                    and len(mm.findMultipleEdges(c.topology)) <= multi0
                    and len(mm.findSelfCollidingTriangles(mm.MeshPart(c))) <= selfx0):
                w = c
        except Exception as e:
            info["post_clean_error"] = repr(e)
    lap("post_clean")
    if lip_round:
        w = round_lips(w, size, unit, info)
        lap("lip_round")
    w.pack()
    info["seconds"] = time.perf_counter() - t0
    return w


# ---------------------------------------------------------------------------------------------------
# Lip hybrid (prototype, off by default): flat chips everywhere, today's Round trim only where sharp,
# thin points are left (knife lips the edge-by-edge chips can't cut cleanly).

LIP_ROUND_ANGLE = 60.0    # deg: a convex edge sharper than this, with both ends thinner than T, is a seed
LIP_ROUND_MARGIN = 3.0    # the Round box reaches this many x Thin past the seeds
ROUND_STONE = dict(voxel=0.0075, distance=0.07, x=0.25)   # BU: Quick Infill's Stone preset


def _offset(mesh, voxel, dist):
    """Quick Infill's voxel offset (GPU winding number when there is one)."""
    try:
        from .offset_utils import cuda_offset
        return cuda_offset(mesh, voxel, dist)
    except ImportError:
        pass
    mm, _ = _ml()
    p = mm.GeneralOffsetParameters()
    p.voxelSize = float(voxel)
    p.signDetectionMode = mm.SignDetectionMode.HoleWindingRule
    try:
        import meshlib.mrcudapy as mc
        p.fwn = mc.FastWindingNumber(mesh)
    except Exception:
        pass
    return mm.generalOffsetMesh(mp=mesh, offset=float(dist), params=p)


def _box(lo, hi):
    ax, u, n = np.eye(3)
    prof = [(float(lo[0]), float(hi[2] - lo[2])), (float(hi[0]), float(hi[2] - lo[2]))]
    return _column_prism(ax, u, n, float(lo[1] + hi[1]) / 2, float(hi[1] - lo[1]) / 2, prof, float(hi[2]), float(hi[2]))


def sharp_thin_points(m, T, angle=LIP_ROUND_ANGLE):
    """Midpoints of convex edges sharper than `angle` whose two ends are thinner than T (meshlib
    computeRayThicknessAtVertices): the sharp, fragile points."""
    mm, mn = _ml()
    V, F = _arrays(mm, mn, mm.Mesh(m))
    th = np.array(mm.computeRayThicknessAtVertices(m).vec, float)[:len(V)]
    ar, nr = _face_areas(V, F)
    cen = V[F].mean(1)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    fid = np.tile(np.arange(len(F)), 3)
    key = np.minimum(E[:, 0], E[:, 1]) * (len(V) + 1) + np.maximum(E[:, 0], E[:, 1])
    order = np.argsort(key, kind="stable"); ks = key[order]
    same = np.nonzero(ks[1:] == ks[:-1])[0]
    f1, f2 = fid[order[same]], fid[order[same + 1]]; ev = E[order[same]]
    convex = np.einsum("ij,ij->i", cen[f2] - cen[f1], nr[f1]) < 0
    inside = 180.0 - np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", nr[f1], nr[f2]), -1, 1)))
    pick = convex & (inside < angle) & (th[ev[:, 0]] < T) & (th[ev[:, 1]] < T)
    return (V[ev[pick, 0]] + V[ev[pick, 1]]) / 2


def _splice_round(w, lo, hi, unit, base, info):
    """Today's Round trim (Stone preset) on the box lo..hi of mesh w only, spliced in:
    cutter = box - Round(piece), result = w - cutter (piece cropped with a 4d margin so its own cut
    faces never round inside the box). The splice's slivers are collapsed (POST_TOL); refused (None)
    if the result opens, doubles an edge or crosses itself more than `base` = (holes, multi, selfx)."""
    mm, mn = _ml()
    vox, d, x = ROUND_STONE["voxel"] * unit, ROUND_STONE["distance"] * unit, ROUND_STONE["x"]
    ob = _box(lo - 4 * d, hi + 4 * d)
    piece = mm.boolean(w, mn.meshFromFacesVerts(ob[1], ob[0].astype(np.float32)), mm.BooleanOperation.Intersection)
    if not piece.valid():
        return None
    piece = piece.mesh
    r = _offset(_offset(_offset(piece, vox, 2 * d), vox, -3 * d), vox, (1 + x) * d)
    r = mm.boolean(r, piece, mm.BooleanOperation.Intersection)
    if not r.valid():
        return None
    ib = _box(lo, hi)
    cut = mm.boolean(mn.meshFromFacesVerts(ib[1], ib[0].astype(np.float32)), r.mesh, mm.BooleanOperation.DifferenceAB)
    if not cut.valid():
        return None
    res = mm.boolean(w, cut.mesh, mm.BooleanOperation.DifferenceAB)
    if not res.valid():
        return None
    res = res.mesh
    pp = mm.FixMeshDegeneraciesParams()
    pp.mode = mm.FixMeshDegeneraciesParams.Mode.Decimate
    pp.maxDeviation = POST_TOL * unit
    pp.tinyEdgeLength = POST_TOL * unit
    pp.criticalTriAspectRatio = 1e3
    try:
        c = mm.Mesh(res); mm.fixMeshDegeneracies(c, pp); c.pack(); res = c
    except Exception:
        pass
    holes0, multi0, selfx0 = base
    if (len(res.topology.findHoleRepresentiveEdges()) > holes0 or len(mm.findMultipleEdges(res.topology)) > multi0
            or len(mm.findSelfCollidingTriangles(mm.MeshPart(res))) > selfx0):
        info["round_refused"] = info.get("round_refused", 0) + 1
        return None
    return res


def _clusters(points, reach):
    """Groups of points, each point within `reach` of another of its group; returns their boxes
    (lo, hi) grown by reach, overlapping boxes merged."""
    clusters = []
    for p in points:
        for c in clusters:
            if np.min(np.linalg.norm(np.array(c) - p, axis=1)) < reach:
                c.append(p); break
        else:
            clusters.append([p])
    boxes = [[np.array(c).min(0) - reach, np.array(c).max(0) + reach] for c in clusters]
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if np.all(a[0] <= b[1]) and np.all(b[0] <= a[1]):
                    boxes[i] = [np.minimum(a[0], b[0]), np.maximum(a[1], b[1])]
                    del boxes[j]; merged = True; break
            if merged:
                break
    return boxes


def _base(w):
    mm, _ = _ml()
    return (len(w.topology.findHoleRepresentiveEdges()), len(mm.findMultipleEdges(w.topology)),
            len(mm.findSelfCollidingTriangles(mm.MeshPart(w))))


def round_lips(w, size, unit=1.0, info=None):
    """Today's Round trim only in small boxes round the clusters of sharp thin points left on the
    chipped stone w (see _splice_round)."""
    info = {} if info is None else info
    t0 = time.perf_counter()
    T = FAST_THICK_PER_SIZE * float(size) * unit
    seeds = sharp_thin_points(w, T)
    info["lip_seeds"] = int(len(seeds))
    boxes = _clusters(seeds, LIP_ROUND_MARGIN * float(size) * unit) if len(seeds) else []
    done = 0
    for lo, hi in boxes:
        try:
            r = _splice_round(w, lo, hi, unit, _base(w), info)
        except Exception as ex:
            info.setdefault("lip_round_errors", []).append(repr(ex)); r = None
        if r is not None:
            w = r; done += 1
    clusters = boxes
    info.update(lip_boxes=len(clusters), lip_rounded=done, lip_round_seconds=time.perf_counter() - t0)
    return w


LOCAL_ALPHA = 110.0       # deg: Local Round rounds every convex edge sharper than this (today's Round: < ~106)


def local_round(mesh, size=SIZE, unit=1.0, target_faces=None, info=None):
    """'Local Round' (Alex 2026-11-17: "they don't try too hard to make it flat ... just make it round
    and decimate it to the same amount doing trim edges would be"): today's Round trim (Stone preset)
    only in boxes round the stone's sharp edges (no flat planes), spliced in, then the whole stone
    decimated (meshlib decimateMesh) to target_faces (today's Trim Edges face count)."""
    mm, mn = _ml()
    info = {} if info is None else info
    t0 = time.perf_counter()
    w = mm.Mesh(mesh); w.pack()
    V, F = _arrays(mm, mn, w)
    ar, nr = _face_areas(V, F)
    cen = V[F].mean(1)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    fid = np.tile(np.arange(len(F)), 3)
    key = np.minimum(E[:, 0], E[:, 1]) * (len(V) + 1) + np.maximum(E[:, 0], E[:, 1])
    order = np.argsort(key, kind="stable"); ks = key[order]
    same = np.nonzero(ks[1:] == ks[:-1])[0]
    f1, f2 = fid[order[same]], fid[order[same + 1]]; ev = E[order[same]]
    convex = np.einsum("ij,ij->i", cen[f2] - cen[f1], nr[f1]) < 0
    inside = 180.0 - np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", nr[f1], nr[f2]), -1, 1)))
    pick = convex & (inside < LOCAL_ALPHA)
    seeds = (V[ev[pick, 0]] + V[ev[pick, 1]]) / 2
    boxes = _clusters(seeds, LIP_ROUND_MARGIN * float(size) * unit) if len(seeds) else []
    done = 0
    for lo, hi in boxes:
        try:
            r = _splice_round(w, lo, hi, unit, _base(w), info)
        except Exception as ex:
            info.setdefault("round_errors", []).append(repr(ex)); r = None
        if r is not None:
            w = r; done += 1
    info.update(round_boxes=len(boxes), rounded=done)
    if target_faces:
        cur = int(w.topology.numValidFaces())
        if cur > target_faces:
            ds = mm.DecimateSettings()
            ds.maxDeletedFaces = cur - int(target_faces)
            ds.maxError = 1e9
            ds.packMesh = True
            mm.decimateMesh(w, ds)
    w.pack()
    info["seconds"] = time.perf_counter() - t0
    return w


LOCAL_EDGE_ANGLE = 125.0  # deg: Local Round rounds round every convex edge sharper than this
LOCAL_MAX_ANGLE = 140.0   # deg: Local Round looks at convex edges up to this inside angle ...
LOCAL_THIN = 2.0          # ... whose stone is thinner than this x d there (today's Round changes nothing thicker)
LOCAL_PAD = 2.0           # each run's box is padded by this x d
LOCAL_RUN = 10.0          # runs longer than this x d are split into boxes of about that length


def _boxes_mesh(boxes):
    Vs, Fs, off = [], [], 0
    for lo, hi in boxes:
        Vb, Fb = _box(lo, hi)
        Vs.append(Vb); Fs.append(Fb + off); off += len(Vb)
    mm, mn = _ml()
    return mn.meshFromFacesVerts(np.ascontiguousarray(np.concatenate(Fs), np.int32),
                                 np.ascontiguousarray(np.concatenate(Vs), np.float32))


def _splice_round_group(w, boxes, unit, base, info):
    """_splice_round for several boxes at once (their 4d-padded crops must not overlap): one crop,
    one Round (Stone preset), one splice."""
    mm, mn = _ml()
    vox, d, x = ROUND_STONE["voxel"] * unit, ROUND_STONE["distance"] * unit, ROUND_STONE["x"]
    outer = [(lo - 4 * d, hi + 4 * d) for lo, hi in boxes]
    piece = mm.boolean(w, _boxes_mesh(outer), mm.BooleanOperation.Intersection)
    if not piece.valid():
        return None
    piece = piece.mesh
    r = _offset(_offset(_offset(piece, vox, 2 * d), vox, -3 * d), vox, (1 + x) * d)
    r = mm.boolean(r, piece, mm.BooleanOperation.Intersection)
    if not r.valid():
        return None
    cut = mm.boolean(_boxes_mesh(boxes), r.mesh, mm.BooleanOperation.DifferenceAB)
    if not cut.valid():
        return None
    res = mm.boolean(w, cut.mesh, mm.BooleanOperation.DifferenceAB)
    if not res.valid():
        return None
    res = res.mesh
    pp = mm.FixMeshDegeneraciesParams()
    pp.mode = mm.FixMeshDegeneraciesParams.Mode.Decimate
    pp.maxDeviation = POST_TOL * unit
    pp.tinyEdgeLength = POST_TOL * unit
    pp.criticalTriAspectRatio = 1e3
    try:
        c = mm.Mesh(res); mm.fixMeshDegeneracies(c, pp); c.pack(); res = c
    except Exception:
        pass
    holes0, multi0, selfx0 = base
    if (len(res.topology.findHoleRepresentiveEdges()) > holes0 or len(mm.findMultipleEdges(res.topology)) > multi0
            or len(mm.findSelfCollidingTriangles(mm.MeshPart(res))) > selfx0):
        info["round_refused"] = info.get("round_refused", 0) + 1
        return None
    return res


def local_round2(mesh, size=SIZE, unit=1.0, target_faces=None, info=None):
    """Local Round, round 9: today's Round trim only round the edges it would change, then decimated
    to today's face count. Seeds follow today's effective rule (measured on 14 stones: nothing thicker
    than about 2d changes): convex edges up to LOCAL_MAX_ANGLE whose stone (meshlib ray thickness) is
    thinner than LOCAL_THIN x d at either end, plus edges whose two ends are thinner than d (plates,
    lips). One box per edge run (runs split every LOCAL_RUN x d), padded LOCAL_PAD x d; boxes whose
    crops don't overlap are rounded in one go."""
    mm, mn = _ml()
    info = {} if info is None else info
    t0 = time.perf_counter()
    d = ROUND_STONE["distance"] * unit
    w = mm.Mesh(mesh); w.pack()
    V, F = _arrays(mm, mn, w)
    th = np.array(mm.computeRayThicknessAtVertices(w).vec, float)[:len(V)]
    ar, nr = _face_areas(V, F)
    cen = V[F].mean(1)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    fid = np.tile(np.arange(len(F)), 3)
    key = np.minimum(E[:, 0], E[:, 1]) * (len(V) + 1) + np.maximum(E[:, 0], E[:, 1])
    order = np.argsort(key, kind="stable"); ks = key[order]
    same = np.nonzero(ks[1:] == ks[:-1])[0]
    f1, f2 = fid[order[same]], fid[order[same + 1]]; ev = E[order[same]]
    convex = np.einsum("ij,ij->i", cen[f2] - cen[f1], nr[f1]) < 0
    inside = 180.0 - np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", nr[f1], nr[f2]), -1, 1)))
    tmin = np.minimum(th[ev[:, 0]], th[ev[:, 1]]); tmax = np.maximum(th[ev[:, 0]], th[ev[:, 1]])
    pick = (convex & (inside < LOCAL_EDGE_ANGLE)) | (tmax < d)
    se = ev[pick]
    info["round_seed_edges"] = int(len(se))
    t1 = time.perf_counter()
    # runs: edges joined through shared vertices
    parent = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a, b in se:
        a, b = int(a), int(b)
        parent.setdefault(a, a); parent.setdefault(b, b)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    runs = {}
    for v in parent:
        runs.setdefault(find(v), []).append(v)
    boxes = []
    pad = LOCAL_PAD * d
    for vs in runs.values():
        P = V[vs]
        stack = [P]
        while stack:
            Q = stack.pop()
            if len(Q) > 2 and np.linalg.norm(Q.max(0) - Q.min(0)) > LOCAL_RUN * d:
                c0 = Q.mean(0)
                ax = np.linalg.eigh(np.cov((Q - c0).T) + 1e-12 * np.eye(3))[1][:, 2]
                s = (Q - c0) @ ax
                stack += [Q[s <= np.median(s)], Q[s > np.median(s)]]
            else:
                boxes.append((Q.min(0) - pad, Q.max(0) + pad))
    # boxes whose 4d-padded crops overlap are merged (one region each); the regions are then disjoint
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if np.all(a[0] - 4 * d <= b[1] + 4 * d) and np.all(b[0] - 4 * d <= a[1] + 4 * d):
                    boxes[i] = (np.minimum(a[0], b[0]), np.maximum(a[1], b[1]))
                    del boxes[j]; merged = True; break
            if merged:
                break
    groups = [[i] for i in range(len(boxes))]
    info.update(round_boxes=len(boxes), round_groups=len(groups))
    done = 0
    for g in groups:
        try:
            r = _splice_round_group(w, [boxes[i] for i in g], unit, _base(w), info)
        except Exception as ex:
            info.setdefault("round_errors", []).append(repr(ex)); r = None
        if r is not None:
            w = r; done += 1
    info["rounded_groups"] = done
    t2 = time.perf_counter()
    if target_faces:
        cur = int(w.topology.numValidFaces())
        if cur > target_faces:
            ds = mm.DecimateSettings()
            ds.maxDeletedFaces = cur - int(target_faces)
            ds.maxError = 1e9
            ds.packMesh = True
            mm.decimateMesh(w, ds)
    w.pack()
    info["timings"] = dict(seeds=t1 - t0, round=t2 - t1, decimate=time.perf_counter() - t2)
    info["seconds"] = time.perf_counter() - t0
    return w


# ---------------------------------------------------------------------------------------------------
# Lean Round: today's Round Trim Edges (grow 2d, shrink 3d, regrow (1+x)d, intersect with the stone),
# whole stone, with fewer steps. Today's voxel intersect re-meshes the decimated offset back to
# ~130k faces and a second decimate brings it down again; here the decimated offset is intersected
# with the stone by an exact meshlib boolean (small output), then decimated once to today's face
# count (today's ratio x the faces the voxel steps would give). Stone -> Detail fallback kept.

ROUND_DETAIL = dict(voxel=0.005, distance=0.03, x=0.4)     # BU: Quick Infill's Detail preset
ROUND_RATIO = dict(STONE=0.0175, DETAIL=0.03)
STONE_KEEPS = 2.0 / 3.0
ROUND_DECIMATE_PARTS = 8   # parallel decimation parts for the big offset mesh (1: off)


def _round_once(mm, mesh, pre, density, unit, tm):
    vox, d, x = pre["voxel"] * unit, pre["distance"] * unit, pre["x"]
    t = time.perf_counter()
    w = _offset(mesh, vox, 2 * d)
    w = _offset(w, vox, -3 * d)
    w = _offset(w, vox, (1 + x) * d)
    tm["offsets"] = tm.get("offsets", 0) + time.perf_counter() - t
    n_vox = int(w.topology.numValidFaces())
    if n_vox == 0:
        return None, 0
    t = time.perf_counter()
    # the density decimate of today's trim (faces per unit of bounding-box area, in Quick Infill units)
    bb = w.computeBoundingBox()
    lo = np.array([bb.min.x, bb.min.y, bb.min.z]); hi = np.array([bb.max.x, bb.max.y, bb.max.z])
    dx, dy, dz = ((hi - lo) * (10.0 / unit)).tolist()
    target = int(min(n_vox, max(20.0, 2.0 * (dx * dy + dy * dz + dz * dx) * density)))
    if target < n_vox:
        diag = float(np.linalg.norm(hi - lo))
        red = max(0.0, 1.0 - target / max(1, n_vox))
        ds = mm.DecimateSettings()
        ds.maxDeletedFaces = n_vox - target
        ds.maxError = max(vox * 10.0, diag * (0.01 + 0.49 * red))
        ds.packMesh = True
        if ROUND_DECIMATE_PARTS > 1:
            ds.subdivideParts = ROUND_DECIMATE_PARTS   # meshlib decimates the parts in parallel
        mm.decimateMesh(w, ds)
    tm["decimate"] = tm.get("decimate", 0) + time.perf_counter() - t
    t = time.perf_counter()
    r = mm.boolean(w, mesh, mm.BooleanOperation.Intersection)
    tm["intersect"] = tm.get("intersect", 0) + time.perf_counter() - t
    if not r.valid():
        return None, n_vox
    return r.mesh, n_vox


def lean_round(mesh, unit=1.0, density=1.0, info=None, parts=None):
    """Today's Round Trim Edges result with fewer steps (see above). unit: mesh units per BU.
    The big decimate runs in parallel parts; if that leaves the result crossing itself (or open)
    more than the input, it is done again in one part."""
    mm, mn = _ml()
    info = {} if info is None else info
    global ROUND_DECIMATE_PARTS
    keep = ROUND_DECIMATE_PARTS
    if parts is not None:
        ROUND_DECIMATE_PARTS = parts
    try:
        out = _lean_round(mesh, unit, density, info)
    finally:
        ROUND_DECIMATE_PARTS = keep
    if (parts is None and keep > 1 and out is not None):
        src = mm.Mesh(mesh)
        bad = (len(out.topology.findHoleRepresentiveEdges()) > len(src.topology.findHoleRepresentiveEdges())
               or len(mm.findSelfCollidingTriangles(mm.MeshPart(out))) > len(mm.findSelfCollidingTriangles(mm.MeshPart(src))))
        if bad:
            t1 = info.get("seconds", 0.0)
            info2 = {}
            out = lean_round(mesh, unit, density, info2, parts=1)
            info2["seconds"] = info2.get("seconds", 0.0) + t1
            info2["redone_serial"] = True
            info.clear(); info.update(info2)
    return out


def _lean_round(mesh, unit, density, info):
    mm, mn = _ml()
    info = {} if info is None else info
    tm = info.setdefault("timings", {})
    t0 = time.perf_counter()
    src = mm.Mesh(mesh); src.pack()
    v0 = abs(float(src.volume()))
    out, n_vox = _round_once(mm, src, ROUND_STONE, density, unit, tm)
    tag = "STONE"
    share = abs(float(out.volume())) / v0 if (out is not None and v0 > 0) else 0.0
    if share < STONE_KEEPS:
        fine, n2 = _round_once(mm, src, ROUND_DETAIL, density, unit, tm)
        if fine is not None and abs(float(fine.volume())) / v0 > share:
            out, n_vox, tag = fine, n2, "DETAIL"
    if out is None:
        info["seconds"] = time.perf_counter() - t0
        return src
    t = time.perf_counter()
    target = int(ROUND_RATIO[tag] * n_vox * 0.96)            # today's output: ratio x its voxel intersect faces
    cur = int(out.topology.numValidFaces())
    if target < cur:
        ds = mm.DecimateSettings()
        ds.maxDeletedFaces = cur - target
        ds.maxError = (ROUND_STONE if tag == "STONE" else ROUND_DETAIL)["voxel"] * unit * 10.0
        ds.packMesh = True
        mm.decimateMesh(out, ds)
    tm["decimate_final"] = time.perf_counter() - t
    out.pack()
    info.update(tag=tag, faces=int(out.topology.numValidFaces()), target_faces=target, seconds=time.perf_counter() - t0)
    return out
