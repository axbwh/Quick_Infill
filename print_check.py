"""
Print Check: looks at a model the way a resin slicer does, layer by layer from
the build plate up, and reports what would go wrong in the print.

    sealed cavities   empty space with no way out at all (the build plate
                      counts as a wall). Resin stays trapped inside for good.
    suction cups      empty space that is closed while its layers print and
                      only opens higher up (a pit that opens away from the
                      plate). Reported with its depth.
    islands           material that starts in mid-air: a region of a layer
                      that touches nothing printed below it.
    ledges            flat overhangs sticking out further than `ledge` past
                      everything in the layer below.
    plate contact     how much of the footprint of the bottom strip really
                      touches the plate.

Everything works on one voxel grid (solid / empty) made by meshlib from the
winding number, so open, overlapping and self-intersecting meshes are fine.
Sizes are in meshlib space (mm for Alex's files: Blender units x 10).

The analysis itself is plain numpy (no bpy), so it can run on a worker thread.

Where the method comes from
---------------------------
The layer-by-layer test is the one resin slicers use, and it was checked
against the published sources:

  * UVtools (open source, sn4k3/UVtools, IssueManager.cs) finds "resin traps"
    and "suction cups" in two passes over the sliced layers: bottom-up it
    keeps a map of air and marks every closed hollow of a layer that does not
    overlap air from the layer below; top-down it re-tests those against air
    from above. Open above = suction cup, closed both ways = resin trap. It
    drops traps that touch the first layer ("Not a trap if on plate"), and it
    needs a minimum number of overlapping pixels before it calls a hollow
    drained. Here: `sealed` = its resin trap, `cups` = its suction cup,
    `plate pockets` = the on-plate case it drops, `drain_gap` = its
    minimum-overlap rule done in 3D on the distance field.
  * Manners and Smalley, US patent 5,945,058 (3D Systems, 1999), is the oldest
    description: holes of layer N are intersected with the trapped region of
    layer N-1, tracked layer to layer, and openings narrower than about
    0.5 to 2 mm are treated as closed because resin does not flow through.
  * Tedia and Williams, "Manufacturability analysis tool for additive
    manufacturing using voxel-based geometric modeling" (Solid Freeform
    Fabrication Symposium 2016), does the same for voids that trap powder,
    resin or support material by filling a voxel model from outside.
  * CHITUBOX's own manual only says its Cavity Detection finds "enclosed
    pockets or voids" that cause "vacuum effects": sealed + cups here.

The difference from UVtools is the grid: it reads the printer's pixels
(0.02 to 0.05 mm), this reads voxels of `voxel` size (0.2 mm by default).
"""

import numpy as np

from .meshlib_utils import get_meshlib, get_mrmeshnumpy, CUDA_LOCK, check_cancelled


# ---------------------------------------------------------------------------
# Voxel grid
# ---------------------------------------------------------------------------

class Grid:
    """solid[z, y, x] (bool) plus where it sits in space.

    Layer z covers plate + z*voxel .. plate + (z+1)*voxel. One empty voxel of
    padding surrounds the model sideways and on top, so "outside" always exists.
    """

    def __init__(self, solid, origin, voxel):
        self.solid = solid
        self.origin = np.asarray(origin, dtype=np.float64)  # corner of voxel (0, 0, 0): x, y, z
        self.voxel = float(voxel)

    @property
    def shape(self):
        return self.solid.shape

    def volume(self, mask):
        return float(np.count_nonzero(mask)) * self.voxel ** 3

    def area(self, mask2d):
        return float(np.count_nonzero(mask2d)) * self.voxel ** 2


def merged_mesh(meshes):
    mm, _ = get_meshlib()
    if not isinstance(meshes, (list, tuple)):
        return meshes
    if len(meshes) == 1:
        return meshes[0]
    out = mm.Mesh()
    for m in meshes:
        out.addMesh(m)
    return out


def voxelize(meshes, voxel, plate_z=None, max_voxels=400e6, slab_voxels=24e6, top_z=None, with_distance=None):
    """Solid/empty grid of the model from the plate up. Returns a Grid.

    plate_z: height of the build plate (default: the lowest point of the model).
    Anything below the plate is ignored, as the cut against the plate removes it.
    top_z: stop at this height (to look at the bottom of a model only).
    with_distance: also keep the signed distance (float32, negative inside), exact
    up to this distance; the Grid then has .dist.
    """
    mm, mc = get_meshlib()
    mn = get_mrmeshnumpy()
    mesh = merged_mesh(meshes)
    box = mesh.computeBoundingBox()   # vertices in use only
    lo = np.array([box.min.x, box.min.y, box.min.z], dtype=np.float64)
    hi = np.array([box.max.x, box.max.y, box.max.z], dtype=np.float64)
    if plate_z is None:
        plate_z = float(lo[2])
    h = float(voxel)
    pad = 1
    x0, y0 = lo[0] - pad * h, lo[1] - pad * h
    nx = int(np.ceil((hi[0] - lo[0]) / h)) + 2 * pad
    ny = int(np.ceil((hi[1] - lo[1]) / h)) + 2 * pad
    top = float(hi[2]) if top_z is None else min(float(hi[2]), float(top_z))
    nz = max(1, int(np.ceil((top - plate_z) / h))) + (pad if top_z is None else 0)
    if float(nx) * ny * nz > max_voxels:
        raise ValueError(f"Print Check: {nx}x{ny}x{nz} voxels is too many at voxel size {h:g}. Use a larger voxel.")

    solid = np.zeros((nz, ny, nx), dtype=bool)
    dist = np.zeros((nz, ny, nx), dtype=np.float32) if with_distance else None
    reach = max(h, float(with_distance)) if with_distance else h
    per_slab = max(1, int(slab_voxels // (nx * ny)))
    with CUDA_LOCK:
        fwn = mc.FastWindingNumber(mesh)
        z = 0
        while z < nz:
            check_cancelled()
            n = min(per_slab, nz - z)
            p = mm.MeshToDistanceVolumeParams()
            p.vol.origin = mm.Vector3f(float(x0), float(y0), float(plate_z + z * h))
            p.vol.voxelSize = mm.Vector3f(h, h, h)
            p.vol.dimensions = mm.Vector3i(nx, ny, n)
            p.dist.signMode = mm.SignDetectionMode.HoleWindingRule
            # Only the sign is needed: distances are not searched further than one voxel,
            # and beyond that meshlib returns an approximate value with the correct sign.
            p.dist.maxDistSq = float(reach * reach)
            p.dist.nullOutsideMinMax = False
            p.fwn = fwn
            vol = mm.meshToDistanceVolume(mm.MeshPart(mesh), p)
            a = mn.getNumpy3Darray(vol)  # [x, y, z]
            solid[z:z + n] = (a < 0.0).transpose(2, 1, 0)
            if dist is not None:
                dist[z:z + n] = a.transpose(2, 1, 0)
            del a, vol
            z += n
    g = Grid(solid, (x0, y0, plate_z), h)
    g.dist = dist
    return g


# ---------------------------------------------------------------------------
# 2D flood fill without scipy
# ---------------------------------------------------------------------------

def _sweep(seed, mask, axis, reverse):
    """Spread seed along one direction of one axis, inside runs of mask."""
    if reverse:
        seed, mask = np.flip(seed, axis), np.flip(mask, axis)
    n = mask.shape[axis]
    shape = [1, 1]
    shape[axis] = n
    idx = np.arange(1, n + 1, dtype=np.int32).reshape(shape)
    last_seed = np.maximum.accumulate(np.where(seed, idx, 0), axis=axis)
    last_wall = np.maximum.accumulate(np.where(mask, 0, idx), axis=axis)
    out = mask & (last_seed > last_wall)
    return np.flip(out, axis) if reverse else out


def flood2d(seed, mask):
    """Everything in mask connected (4 neighbours) to seed. Both bool [y, x]."""
    cur = seed & mask
    count = int(np.count_nonzero(cur))
    if count == 0 or count == int(np.count_nonzero(mask)):
        return cur
    while True:
        for axis in (1, 0):
            cur = _sweep(cur, mask, axis, False)
            cur = _sweep(cur, mask, axis, True)
        new_count = int(np.count_nonzero(cur))
        if new_count == count:
            return cur
        count = new_count


def flood3d(seed, mask):
    """Everything in mask connected (6 neighbours) to seed. Both bool [z, y, x]."""
    nz = mask.shape[0]
    out = np.zeros_like(mask)
    stack = []
    for z in range(nz):
        s = seed[z] & mask[z]
        if s.any():
            add = flood2d(s | out[z], mask[z])
            out[z] = add
            stack.append((z, add))
    while stack:
        z, added = stack.pop()
        for z2 in (z - 1, z + 1):
            if z2 < 0 or z2 >= nz:
                continue
            room = mask[z2] & ~out[z2]
            s = added & room
            if not s.any():
                continue
            more = flood2d(s, room)
            out[z2] |= more
            stack.append((z2, more))
    return out


def outside_mask(empty):
    """The part of the empty space connected to the border of the grid."""
    seed = np.zeros_like(empty)
    seed[0] = seed[-1] = True
    seed[:, 0] = seed[:, -1] = True
    seed[:, :, 0] = seed[:, :, -1] = True
    return flood3d(seed, empty)


def dilate2d(a, n=1):
    """Grow a bool [y, x] mask by n voxels (8 neighbours per step)."""
    out = a
    for _ in range(int(n)):
        b = out.copy()
        b[1:, :] |= out[:-1, :]
        b[:-1, :] |= out[1:, :]
        c = b.copy()
        c[:, 1:] |= b[:, :-1]
        c[:, :-1] |= b[:, 1:]
        out = c
    return out


def dilate3d(a, n=1):
    """Grow a bool [z, y, x] mask by n voxels (6 neighbours per step)."""
    out = a
    for _ in range(int(n)):
        b = out.copy()
        b[1:] |= out[:-1]
        b[:-1] |= out[1:]
        b[:, 1:] |= out[:, :-1]
        b[:, :-1] |= out[:, 1:]
        b[:, :, 1:] |= out[:, :, :-1]
        b[:, :, :-1] |= out[:, :, 1:]
        out = b
    return out


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def find_trapped(grid, solid=None):
    """Per empty voxel, the layer at which it first connects to the outside.

    solid: use this solid mask instead of grid.solid (e.g. the model grown by
    half the drain gap, so gaps too narrow to drain count as closed).

    Returns spill[z, y, x] (int32): -1 = solid, NEVER = sealed for good,
    otherwise the layer index L >= z where it opens. spill == z means the voxel
    is open as soon as it prints (not a cup).
    """
    solid = grid.solid if solid is None else solid
    nz, ny, nx = solid.shape
    never = np.iinfo(np.int32).max
    spill = np.where(solid, -1, never).astype(np.int32)
    vent = np.zeros_like(solid)
    border = np.zeros((ny, nx), dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True

    for level in range(nz):
        check_cancelled()
        empty = ~solid[level]
        seed = empty & (border | vent[level - 1]) if level else empty & border
        new = flood2d(seed, empty)
        if not new.any():
            continue
        vent[level] = new
        spill[level][new] = level
        # The freshly opened region may reach pockets below it (and, through
        # them, back up): follow those, still at this level.
        stack = [(level, new)]
        while stack:
            z, added = stack.pop()
            for z2 in (z - 1, z + 1):
                if z2 < 0 or z2 > level:
                    continue
                room = ~solid[z2] & ~vent[z2]
                s = added & room
                if not s.any():
                    continue
                more = flood2d(s, room)
                vent[z2] |= more
                spill[z2][more] = level
                stack.append((z2, more))
    return spill


NEVER = np.iinfo(np.int32).max


def find_unsupported(grid, reach=1):
    """(islands, unsupported): bool grids.

    islands: the first layer of every region that starts in mid-air.
    unsupported: all material printed before such a region joins supported
    material (the part that would droop or fail).
    reach: how many voxels sideways the layer below still counts as support.
    """
    solid = grid.solid
    nz = solid.shape[0]
    islands = np.zeros_like(solid)
    unsupported = np.zeros_like(solid)
    sup = solid[0].copy()        # everything in the first layer sits on the plate
    below = solid[0]
    for z in range(1, nz):
        check_cancelled()
        layer = solid[z]
        if not layer.any():
            sup = layer
            below = layer
            continue
        # Islands: no contact with ANY material in the layer below.
        ok = flood2d(layer & dilate2d(below, reach), layer)
        islands[z] = layer & ~ok
        # Unsupported: no contact with SUPPORTED material in the layer below.
        ok2 = flood2d(layer & dilate2d(sup, reach), layer)
        unsupported[z] = layer & ~ok2
        sup = ok2
        below = layer
    return islands, unsupported


def find_ledges(grid, ledge_voxels):
    """Material sticking out more than ledge_voxels past the layer below."""
    solid = grid.solid
    out = np.zeros_like(solid)
    for z in range(1, solid.shape[0]):
        if solid[z].any():
            out[z] = solid[z] & ~dilate2d(solid[z - 1], ledge_voxels)
    return out


class Report:
    """Numbers (mm, mm2, mm3) and the voxel masks behind them."""

    def __init__(self):
        self.voxel = 0.0
        self.grid = None
        self.masks = {}
        self.sealed_volume = 0.0          # sealed for good: no opening at all, not even at the plate
        self.sealed_count = 0
        self.pocket_volume = 0.0          # sealed only by the build plate: opens when the print comes off it
        self.pocket_count = 0
        self.cup_volume = 0.0
        self.cup_max_depth = 0.0
        self.island_count = 0
        self.island_area = 0.0
        self.unsupported_volume = 0.0
        self.ledge_area = 0.0
        self.contact_area = 0.0
        self.footprint_area = 0.0
        self.seconds = {}

    @property
    def contact_percent(self):
        return 100.0 * self.contact_area / self.footprint_area if self.footprint_area > 0 else 0.0

    def ok(self):
        return (self.sealed_count == 0 and self.pocket_count == 0 and self.island_count == 0
                and self.contact_percent >= 99.5)

    def summary(self):
        return (f"sealed cavities {self.sealed_count} ({self.sealed_volume:.2f} mm3), "
                f"plate pockets {self.pocket_count} ({self.pocket_volume:.2f} mm3), "
                f"cups {self.cup_volume:.1f} mm3 (deepest {self.cup_max_depth:.1f} mm), "
                f"islands {self.island_count} ({self.island_area:.1f} mm2, {self.unsupported_volume:.1f} mm3 unsupported), "
                f"ledges {self.ledge_area:.1f} mm2, "
                f"plate contact {self.contact_percent:.1f}% of {self.footprint_area:.0f} mm2")


def blob_volumes(mask, voxel):
    """Volume of every separate blob in a bool grid (measured on its surface mesh,
    so a single voxel reads as about a sixth of its cube). Largest first."""
    m = mask_to_mesh(mask, (0, 0, 0), voxel)
    if m is None:
        return []
    mm, _ = get_meshlib()
    comps = mm.MeshComponents.getAllComponents(mm.MeshPart(m))
    return sorted((abs(float(m.volume(c))) for c in comps), reverse=True)


def count_blobs(mask, voxel=1.0, min_volume=0.0):
    """Number of separate blobs with at least min_volume."""
    return sum(1 for v in blob_volumes(mask, voxel) if v >= min_volume)


def sealed_mask(grid, drain_gap=0.0):
    """(sealed, spill): sealed = empty voxels with no way out.

    drain_gap > 0: openings narrower than this count as closed, because resin
    does not drain through them (needs a grid made with with_distance >= drain_gap / 2).
    """
    if drain_gap > 0 and getattr(grid, "dist", None) is not None:
        grown = grid.dist < (drain_gap * 0.5)
        spill = find_trapped(grid, grown)
        core = spill == NEVER
        # back out to the real walls: the grown skin around a sealed core is sealed too
        n = int(np.ceil(drain_gap * 0.5 / grid.voxel)) + 1
        sealed = dilate3d(core, n) & ~grid.solid & grown | core
        sealed &= ~grid.solid
        return sealed, np.where(grid.solid, -1, np.where(sealed, NEVER, np.minimum(spill, NEVER - 1)))
    spill = find_trapped(grid)
    return spill == NEVER, spill


def check(meshes, voxel=0.2, plate_z=None, min_cup_depth=1.0, ledge=1.0, strip=4.0, min_cavity=0.02,
          min_island=0.1, drain_gap=0.0, grid=None):
    """Run every analysis. Returns a Report.

    voxel          voxel size (also the layer height used).
    plate_z        build plate height; default = lowest point of the model.
    min_cup_depth  cups shallower than this are ignored.
    ledge          overhang length that counts as a ledge.
    strip          height of the bottom strip whose footprint must touch the plate.
    min_cavity     sealed cavities smaller than this (mm3) are not counted (voxel noise);
                   the reported volume still includes them.
    min_island     islands smaller than this (mm2) are ignored (voxel noise).
    drain_gap      openings narrower than this count as closed (0 = only truly sealed).
    """
    import time
    r = Report()
    t0 = time.perf_counter()
    g = grid if grid is not None else voxelize(meshes, voxel, plate_z, with_distance=(drain_gap * 0.5 + voxel) if drain_gap > 0 else None)
    r.grid, r.voxel = g, g.voxel
    h = g.voxel
    r.seconds["voxelize"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    sealed, spill = sealed_mask(g, drain_gap)
    zz = np.arange(g.shape[0], dtype=np.int32).reshape(-1, 1, 1)
    depth = np.where((spill >= 0) & ~sealed & (spill < NEVER - 1), spill - zz, 0)
    cups = depth >= max(1, int(round(min_cup_depth / h)))
    r.seconds["trapped"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    # Sealed space that reaches the first layer is closed only by the build plate:
    # it opens when the print is taken off (UVtools does not report it at all).
    # It still matters here, because it means the bottom is not flat on the plate.
    seed = np.zeros_like(sealed)
    seed[0] = sealed[0]
    pockets = flood3d(seed, sealed) if seed.any() else seed
    sealed = sealed & ~pockets
    r.sealed_volume = g.volume(sealed)
    r.sealed_count = count_blobs(sealed, h, min_cavity)
    r.pocket_volume = g.volume(pockets)
    r.pocket_count = count_blobs(pockets, h, min_cavity)
    r.cup_volume = g.volume(cups)
    r.cup_max_depth = float(depth.max()) * h if cups.any() else 0.0

    islands, unsupported = find_unsupported(g)
    islands2 = _drop_small_layers(islands, int(np.ceil(min_island / h ** 2)))
    r.island_count = count_blobs(islands2, h)
    r.island_area = float(np.count_nonzero(islands2)) * h * h
    r.unsupported_volume = g.volume(unsupported)
    ledges = find_ledges(g, max(1, int(round(ledge / h))))
    r.ledge_area = float(np.count_nonzero(ledges)) * h * h

    k = max(1, min(g.shape[0], int(round(strip / h))))
    contact = g.solid[0]
    footprint = g.solid[:k].any(axis=0)
    r.contact_area, r.footprint_area = g.area(contact), g.area(footprint)
    missing = np.zeros_like(g.solid)
    missing[0] = footprint & ~contact
    r.seconds["analyse"] = time.perf_counter() - t0

    r.masks = {"sealed": sealed, "plate_pockets": pockets, "cups": cups, "islands": islands2, "unsupported": unsupported,
               "ledges": ledges, "no_contact": missing}
    return r


def _drop_small_layers(mask, min_pixels):
    """Per layer, drop the island pixels when the layer holds fewer than min_pixels."""
    if min_pixels <= 1:
        return mask
    out = mask.copy()
    counts = mask.reshape(mask.shape[0], -1).sum(axis=1)
    out[counts < min_pixels] = False
    return out


# ---------------------------------------------------------------------------
# Masks back to meshes (to show the problem areas in Blender)
# ---------------------------------------------------------------------------

def mask_to_mesh(mask, origin, voxel):
    """Surface mesh around the True voxels of mask[z, y, x], or None if empty."""
    if not mask.any():
        return None
    mm, _ = get_meshlib()
    mn = get_mrmeshnumpy()
    zs, ys, xs = np.nonzero(mask)
    z0, z1, y0, y1, x0, x1 = zs.min(), zs.max() + 1, ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = np.zeros((x1 - x0 + 2, y1 - y0 + 2, z1 - z0 + 2), dtype=np.float32)
    sub[1:-1, 1:-1, 1:-1] = mask[z0:z1, y0:y1, x0:x1].transpose(2, 1, 0)
    vol = mn.simpleVolumeFrom3Darray(sub)
    vol.voxelSize = mm.Vector3f(float(voxel), float(voxel), float(voxel))
    p = mm.MarchingCubesParams()
    p.iso = 0.5
    p.lessInside = False
    p.origin = mm.Vector3f(float(origin[0] + (x0 - 1) * voxel), float(origin[1] + (y0 - 1) * voxel),
                           float(origin[2] + (z0 - 1) * voxel))
    mesh = mm.marchingCubes(vol, p)
    return mesh if mesh.topology.numValidFaces() else None


def seal_plugs(meshes, voxel=0.2, plate_z=None, drain_gap=0.0, overlap=2, report=None):
    """Mesh that fills every sealed cavity of the model (None if there is none).

    The plugs reach `overlap` voxels into the surrounding walls, so the union
    with the model leaves no sliver. Sealed cavities cannot be seen from
    outside, so filling them changes nothing visible.
    """
    g = voxelize(meshes, voxel, plate_z, with_distance=(drain_gap * 0.5 + voxel) if drain_gap > 0 else None)
    sealed, _ = sealed_mask(g, drain_gap)
    if report is not None:
        report["sealed_volume"] = g.volume(sealed)
    if not sealed.any():
        return None
    # A plug can itself close off a crumb of space beside it (its own surface
    # narrows the slit it sits in). Look again with the plugs in place, a few
    # times, until nothing new turns up.
    fill = sealed
    solid0 = g.solid
    for _ in range(4):
        g.solid = solid0 | dilate3d(fill, 1)
        if getattr(g, "dist", None) is not None:
            g.dist = np.where(g.solid & ~solid0, -g.voxel, g.dist)
        more, _ = sealed_mask(g, drain_gap)
        more &= ~fill & ~solid0
        if not more.any():
            break
        fill = fill | more
    g.solid = solid0
    if report is not None:
        report["sealed_volume"] = g.volume(fill)
    fill = dilate3d(fill, overlap) & (solid0 | fill)   # never grow a plug out of the model
    return mask_to_mesh(fill, g.origin, g.voxel)
