"""
Fix Undercuts from several directions on ONE voxel grid.

meshlib's FixUndercuts.fix works on one direction and turns the mesh into
voxels and back each time; combining N directions with voxel booleans adds two
more conversions per pair. On a wall face that is about 1.6 s per direction.

Here the mesh becomes a signed distance grid once. Fixing undercuts for a
direction u is then "every point takes the lowest distance found along the ray
towards u" (anything that has material above it along u becomes material),
which is a running minimum along the grid after shearing it so the rays line
up with a grid axis. Directions are intersected with a plain maximum, and the
grid is meshed once at the end. Same principle as meshlib (it also runs a
minimum down the voxel columns), without the repeated conversions.

Pure meshlib + numpy, no bpy.
"""

import numpy as np

from .meshlib_utils import get_meshlib, get_mrmeshnumpy, CUDA_LOCK, check_cancelled

BAND = 4.0          # distances are exact up to this many voxels from the surface
PAD = 3             # empty voxels kept around the model


def distance_grid(mesh, voxel, pad=PAD, band=BAND, slab_voxels=24e6, bounds=None):
    """(phi[z, y, x] float32, origin xyz of voxel (0,0,0)'s centre). Negative inside.
    bounds: (lo, hi) the grid must cover (default: the mesh's own box)."""
    mm, mc = get_meshlib()
    mn = get_mrmeshnumpy()
    # computeBoundingBox looks at the vertices in use only (a mesh fresh from a
    # boolean still carries the unused points of its inputs)
    box = mesh.computeBoundingBox()
    lo = np.array([box.min.x, box.min.y, box.min.z], dtype=np.float64)
    hi = np.array([box.max.x, box.max.y, box.max.z], dtype=np.float64)
    if bounds is not None:
        lo = np.minimum(lo, np.asarray(bounds[0], dtype=np.float64))
        hi = np.maximum(hi, np.asarray(bounds[1], dtype=np.float64))
    h = float(voxel)
    org = lo - pad * h
    nx, ny, nz = (int(np.ceil((hi[i] - lo[i]) / h)) + 2 * pad + 1 for i in range(3))
    phi = np.empty((nz, ny, nx), dtype=np.float32)
    per_slab = max(1, int(slab_voxels // (nx * ny)))
    reach = band * h
    with CUDA_LOCK:
        fwn = mc.FastWindingNumber(mesh)
        z = 0
        while z < nz:
            check_cancelled()
            n = min(per_slab, nz - z)
            p = mm.MeshToDistanceVolumeParams()
            # meshlib samples voxel centres at origin + (i + 0.5) * voxel
            p.vol.origin = mm.Vector3f(float(org[0] - 0.5 * h), float(org[1] - 0.5 * h), float(org[2] + (z - 0.5) * h))
            p.vol.voxelSize = mm.Vector3f(h, h, h)
            p.vol.dimensions = mm.Vector3i(nx, ny, n)
            p.dist.signMode = mm.SignDetectionMode.HoleWindingRule
            p.dist.maxDistSq = float(reach * reach)
            p.dist.nullOutsideMinMax = False
            p.fwn = fwn
            vol = mm.meshToDistanceVolume(mm.MeshPart(mesh), p)
            phi[z:z + n] = mn.getNumpy3Darray(vol).transpose(2, 1, 0)
            del vol
            z += n
    np.clip(phi, -reach, reach, out=phi)
    return phi, org


def fill_enclosed(phi, voxel, gap=1.5):
    """Make every pocket that has no way out solid (in place).

    Openings narrower than about `gap` voxels count as closed. meshlib's own
    undercut fix does the same as a side effect of how OpenVDB finds the inside
    of overlapping shells, and the wall recipe relies on it: the pockets behind
    the stones fill up here.
    """
    from .print_check import outside_mask, dilate3d
    h = float(voxel)
    empty = phi > 0.0
    wide = phi > gap * h
    outside = outside_mask(wide)
    for _ in range(int(np.ceil(gap)) + 1):      # back out to the real surface
        outside = dilate3d(outside, 1) & empty
    enclosed = empty & ~outside
    if enclosed.any():
        phi[enclosed] = -h
    return phi


def grid_to_mesh(phi, origin, voxel):
    """Surface (phi = 0) of a grid as a meshlib mesh. The outer voxel layer is
    treated as empty so the mesh is always closed."""
    mm, _ = get_meshlib()
    mn = get_mrmeshnumpy()
    h = float(voxel)
    a = np.ascontiguousarray(phi.transpose(2, 1, 0), dtype=np.float32)  # [x, y, z]
    a[0, :, :] = a[-1, :, :] = h
    a[:, 0, :] = a[:, -1, :] = h
    a[:, :, 0] = a[:, :, -1] = h
    vol = mn.simpleVolumeFrom3Darray(a)
    vol.voxelSize = mm.Vector3f(h, h, h)
    p = mm.MarchingCubesParams()
    p.iso = 0.0
    p.lessInside = True
    p.origin = mm.Vector3f(float(origin[0] - 0.5 * h), float(origin[1] - 0.5 * h), float(origin[2] - 0.5 * h))
    return mm.marchingCubes(vol, p)


# ---------------------------------------------------------------------------
# Sheared running minimum
# ---------------------------------------------------------------------------

def _shift_into(dst, src, d, axis, fill):
    """dst[i] = src[i + d] along `axis` of a 2D array (d real, linear interpolation)."""
    i0 = int(np.floor(d))
    f = float(d - i0)
    n_dst, n_src = dst.shape[axis], src.shape[axis]
    dst[...] = fill
    # destination range whose two source samples (i + i0, i + i0 + 1) both exist
    a = max(0, -i0)
    b = min(n_dst, n_src - i0 - 1)
    if b <= a:
        return
    if axis == 0:
        s0, s1 = src[a + i0:b + i0], src[a + i0 + 1:b + i0 + 1]
        dst[a:b] = s0 if f == 0.0 else s0 * (1.0 - f) + s1 * f
    else:
        s0, s1 = src[:, a + i0:b + i0], src[:, a + i0 + 1:b + i0 + 1]
        dst[:, a:b] = s0 if f == 0.0 else s0 * (1.0 - f) + s1 * f


def sweep_min(phi, up, fill):
    """phi'(p) = min over t >= 0 of phi(p + t * up). up = (x, y, z) in grid axes."""
    u = np.array([up[2], up[1], up[0]], dtype=np.float64)   # array axis order: z, y, x
    mag = np.abs(u)
    # principal axis: the largest component; on a tie the axis with the fewest layers
    best = [k for k in range(3) if mag[k] >= mag.max() * 0.999]
    k = min(best, key=lambda i: phi.shape[i])
    others = [i for i in range(3) if i != k]
    a = np.moveaxis(phi, [k] + others, [0, 1, 2])            # [N, P, Q]
    if u[k] < 0:
        a = a[::-1]
    sp, sq = u[others[0]] / mag[k], u[others[1]] / mag[k]    # sideways voxels per layer going up the ray
    n, p_, q_ = a.shape

    if abs(sp) < 1e-9 and abs(sq) < 1e-9:
        out = np.minimum.accumulate(a[::-1], axis=0)[::-1]
    else:
        pad_p, pad_q = int(np.ceil((n - 1) * abs(sp))) + 1, int(np.ceil((n - 1) * abs(sq))) + 1
        off_p = pad_p if sp > 0 else 0      # sheared index p' = p - j*sp + off  ->  source p = p' + j*sp - off
        off_q = pad_q if sq > 0 else 0
        sheared = np.empty((n, p_ + pad_p, q_ + pad_q), dtype=np.float32)
        tmp = np.empty((p_ + pad_p, q_), dtype=np.float32)
        for j in range(n):
            _shift_into(tmp, a[j], j * sp - off_p, 0, fill)
            _shift_into(sheared[j], tmp, j * sq - off_q, 1, fill)
        check_cancelled()
        sheared = np.minimum.accumulate(sheared[::-1], axis=0)[::-1]
        out = np.empty_like(a)
        tmp2 = np.empty((p_, q_ + pad_q), dtype=np.float32)
        for j in range(n):
            _shift_into(tmp2, sheared[j], off_p - j * sp, 0, fill)
            _shift_into(out[j], tmp2, off_q - j * sq, 1, fill)
        # the fill only ever adds material; keep the untouched grid exact
        np.minimum(out, a, out=out)
    if u[k] < 0:
        out = out[::-1]
    return np.moveaxis(out, [0, 1, 2], [k] + others)


def _shrink_tops(phi, up, amount, angle_deg, voxel):
    """Auto Shrink on the grid: move the surface in by `amount` where it faces
    within angle_deg of up (what Shrink Amount / Shrink Angle do on the mesh)."""
    gz, gy, gx = np.gradient(phi, voxel)
    dot = gx * up[0] + gy * up[1] + gz * up[2]
    norm = np.sqrt(gx * gx + gy * gy + gz * gz)
    mask = dot >= np.cos(np.radians(angle_deg)) * np.maximum(norm, 1e-6)
    mask &= norm > 0.3            # flat (clamped) regions have no direction
    return np.where(mask, phi + amount, phi)


def fill_reach(points, up_vectors):
    """How far the fill of fix_undercuts_grid can reach: (lo, hi, floors).

    The fill for one direction is the model's shadow along -up, down to the
    model's lowest point along up (floors[i], where meshlib's own fix stops as
    well). Several directions are intersected, so the result lies in the
    intersection of those shadows: a convex block whose corners are found here
    from its faces. With the eight side directions that block is the wedge
    under the model, and the grid has to hold all of it. (Until 2026-11-13 the
    grid was the model's box plus three voxels, which cut the wedge off just
    under the model: "the scoop doesn't go as far as it can anymore".)
    """
    P = np.asarray(points, dtype=np.float64)
    if len(P) > 20000:
        P = P[:: len(P) // 20000 + 1]
    normals, limits, floors = [], [], []
    for up in up_vectors:
        u = np.asarray(up, dtype=np.float64)
        u = u / np.linalg.norm(u)
        d = P @ u
        floors.append(float(d.min()))
        normals += [u, -u]
        limits += [float(d.max()), -float(d.min())]
        for e in np.eye(3):
            n = np.cross(e, u)
            ln = np.linalg.norm(n)
            if ln < 1e-6:
                continue
            n = n / ln
            d = P @ n
            normals += [n, -n]
            limits += [float(d.max()), -float(d.min())]
    N = np.array(normals)
    c = np.array(limits)
    lo, hi = P.min(0), P.max(0)
    m = len(N)
    if m >= 3:
        idx = np.arange(m)
        I, J, K = np.meshgrid(idx, idx, idx, indexing='ij')
        sel = (I < J) & (J < K)
        ii, jj, kk = I[sel], J[sel], K[sel]
        A = np.stack([N[ii], N[jj], N[kk]], 1)                  # (t, 3, 3)
        b = np.stack([c[ii], c[jj], c[kk]], 1)
        det = np.linalg.det(A)
        ok = np.abs(det) > 1e-6
        if ok.any():
            X = np.linalg.solve(A[ok], b[ok][:, :, None])[:, :, 0]
            span = float(np.linalg.norm(hi - lo)) + 1e-9
            inside = (X @ N.T <= c[None, :] + 1e-6 * span).all(1)
            if inside.any():
                lo = np.minimum(lo, X[inside].min(0))
                hi = np.maximum(hi, X[inside].max(0))
    return lo, hi, floors


def _clip_below(f, up, floor, origin, voxel, fill):
    """Empty every voxel lower than `floor` along `up` (in place)."""
    h = float(voxel)
    nz, ny, nx = f.shape
    xy = (origin[1] + np.arange(ny) * h)[:, None] * up[1] + (origin[0] + np.arange(nx) * h)[None, :] * up[0]
    for z in range(nz):
        low = xy + (origin[2] + z * h) * up[2] < floor
        if low.any():
            f[z][low] = fill


def fix_undercuts_grid(phi, up_vectors, voxel, shrink_amount=0.0, shrink_angle=70.0, origin=None, floors=None):
    """Undercut fix for every up vector, intersected. Returns a new grid.
    origin + floors (from fill_reach): each direction's fill stops at the
    model's lowest point along it, as meshlib's own fix does."""
    fill = float(BAND * voxel)
    result = None
    for n_up, up in enumerate(up_vectors):
        check_cancelled()
        up = np.asarray(up, dtype=np.float64)
        up = up / np.linalg.norm(up)
        f = sweep_min(phi, up, fill)
        if origin is not None and floors is not None:
            _clip_below(f, up, floors[n_up] - 2.0 * float(voxel), origin, voxel, fill)
        if shrink_amount > 0:
            f = _shrink_tops(f, up, float(shrink_amount), float(shrink_angle), float(voxel))
        result = f if result is None else np.maximum(result, f, out=result)
    return result


def fix_undercuts_mesh(mesh, up_vectors, voxel, shrink_amount=0.0, shrink_angle=70.0, enclosed_gap=1.5):
    """Mesh in, mesh out: undercuts fixed for every up vector and intersected.

    enclosed_gap: pockets closed off by openings narrower than this many voxels
    are filled first (None = leave them). 1.5 matches what meshlib's own
    undercut fix does (measured on a wall of 150 stones: surfaces agree within
    one voxel on 99% of the points).
    """
    mn = get_mrmeshnumpy()
    pts = mn.getNumpyVerts(mesh)
    box = mesh.computeBoundingBox()
    blo = np.array([box.min.x, box.min.y, box.min.z]); bhi = np.array([box.max.x, box.max.y, box.max.z])
    pts = pts[((pts >= blo - 1e-6) & (pts <= bhi + 1e-6)).all(1)]      # (not the unused points of a boolean's inputs)
    lo, hi, floors = fill_reach(pts, up_vectors)
    phi, org = distance_grid(mesh, voxel, bounds=(lo, hi))
    if enclosed_gap is not None:
        fill_enclosed(phi, voxel, enclosed_gap)
    out = fix_undercuts_grid(phi, up_vectors, voxel, shrink_amount, shrink_angle, origin=org, floors=floors)
    return grid_to_mesh(out, org, voxel)
