"""Make Manifold test, through the real operator. Background Blender only; nothing is saved.

    blender -b --python tests/make_manifold.py
    blender -b <file>.blend --python tests/make_manifold.py -- <collection> [<blank>]

Without arguments it builds overlapping spheres and cubes sunk below Z = 0, one of them carrying a
flat open speck (two triangles, no volume), and a blank box to intersect with. With a collection it
uses those objects instead (and the blank, if named).

Runs twice: with the blank (its bottom is the bottom, no plate cut) and without (Cut at Plate, Z = 0).
Checks: one closed object comes out (no open or non-manifold edges), the bottom is ONE flat plane at
the blank's bottom / the plate, nothing pokes out of the blank, the originals are unchanged, and the
meshlib repair is gone.
Prints PASS / FAIL lines, counts and the time.
"""
import bpy, sys, os, time, bmesh, addon_utils

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if not hasattr(bpy.types.Scene, "quick_infill_settings"):
    mod = next(m.__name__ for m in addon_utils.modules() if m.__name__.lower().endswith("quick_infill"))
    addon_utils.enable(mod, default_set=False)
# the Quick Infill that is registered (the operator's own module), whatever its package is called
ops_mod = next(m for n, m in list(sys.modules.items())
               if n.endswith(".print_check_ops") and getattr(m, "QUICKINFILL_OT_make_manifold", None) is not None
               and getattr(m.QUICKINFILL_OT_make_manifold, "is_registered", False))
MF = ops_mod.make_manifold
print("TESTING", MF.__file__)

sc = bpy.context.scene
s = sc.quick_infill_settings
fails = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


def world_co(ob):
    return [ob.matrix_world @ v.co for v in ob.data.vertices]


def snapshot(ob):
    return (len(ob.data.vertices), len(ob.data.polygons), tuple(round(c, 6) for v in ob.data.vertices[:20] for c in v.co))


def bad_edges(me):
    bm = bmesh.new()
    bm.from_mesh(me)
    n = sum(1 for e in bm.edges if not e.is_manifold)
    bm.free()
    return n


if args:
    coll = bpy.data.collections[args[0]]
    blank = bpy.data.objects[args[1]] if len(args) > 1 else None
    objs = [o for o in coll.all_objects if o.type == 'MESH' and o != blank]
else:
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    objs = []
    for i, (x, z) in enumerate([(0.0, 0.1), (0.9, 0.3), (1.8, 0.0)]):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.7, location=(x, 0, z))
        objs.append(bpy.context.object)
        bpy.ops.mesh.primitive_cube_add(size=0.8, location=(x + 0.45, 0.2, z + 0.5))
        objs.append(bpy.context.object)
    # a flat speck with no volume inside one object (it vanishes when rebuilt through voxels)
    me = objs[0].data
    bm = bmesh.new()
    bm.from_mesh(me)
    a, b, c, d = (bm.verts.new(p) for p in ((0, 0, 0.9), (0.004, 0, 0.9), (0.004, 0.001, 0.9), (0, 0.001, 0.9)))
    bm.faces.new((a, b, c))
    bm.faces.new((a, c, d))
    bm.to_mesh(me)
    bm.free()
    # a CLOSED flat speck as one more object: two triangles back to back, 0.3 mm long (over the tiny-piece
    # size). Blender's Manifold solver refuses it, and used to skip the whole union with it: only the
    # biggest shell came out (Alex 2026-11-23, a Print Ready support carrying such specks)
    sme = bpy.data.meshes.new("Speck")
    sme.from_pydata([(0, 0, 0.2), (0.03, 0, 0.2), (0.03, 0.005, 0.21)], [], [(0, 1, 2), (0, 2, 1)])
    speck = bpy.data.objects.new("Speck", sme)
    sc.collection.objects.link(speck)
    objs.append(speck)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.9, 0, 0.4), scale=(1.6, 0.55, 1.2))
    blank = bpy.context.object
    blank.name = "Blank"

if args:
    # some of the objects may sit in hidden collections: show them in a scratch collection so they can be selected
    tmpc = bpy.data.collections.new("MM_test_sel")
    sc.collection.children.link(tmpc)
    for o in objs:
        if o.name not in tmpc.objects:
            tmpc.objects.link(o)
        o.hide_set(False)
        o.hide_viewport = False
        o.hide_select = False
    bpy.context.view_layer.update()
before = {o.name: snapshot(o) for o in objs}
in_min = min(v.z for o in objs for v in world_co(o))
check(not hasattr(MF, "repair") and not hasattr(MF, "tidy_after_cut"), "the meshlib repair is gone from make_manifold")

def run(blank, regrow=False):
    print(("== with the blank" + (", Regrow" if regrow else "")) if blank is not None else "== no blank: Cut at Plate")
    s.manifold_regrow = regrow
    s.manifold_cut_plate = True
    s.check_plate = 'Z0'
    s.manifold_blank = blank
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    t0 = time.time()
    res = bpy.ops.quick_infill.make_manifold()
    dt = time.time() - t0
    check(res == {'FINISHED'}, f"the operator finished ({res})")
    if not args:
        ob_ = bpy.context.active_object
        xs = [p.x for p in world_co(ob_)] if ob_ is not None and ob_.name.endswith(tuple("_Solid.0123456789")) else [0.0]
        check(max(xs) - min(xs) > 1.45, f"every object is in the result, not just the biggest ({max(xs) - min(xs):.2f} BU wide)")
    out = bpy.context.view_layer.objects.active
    check(out is not None and "_Solid" in out.name, f"result object {out.name if out else None}")
    if out is not None and "_Solid" in out.name:
        check([c.name for c in out.users_collection] == ["STLs"], f"result in the STLs collection {[c.name for c in out.users_collection]}")
        me = out.data
        nb = bad_edges(me)
        check(nb == 0, f"closed: {nb} open / non-manifold edges")
        co = world_co(out)
        zmin = min(v.z for v in co)
        # with a blank its bottom is the bottom (no plate cut); without one, Z = 0 if the model reaches below it
        bottom = min(v.z for v in world_co(blank)) if blank is not None else (0.0 if in_min < 0.0 else None)
        cut = bottom is not None and in_min < bottom
        if cut:
            check(abs(zmin - bottom) < 1e-6 and all(abs(v.z - bottom) < 1e-6 or v.z > bottom for v in co),
                  f"bottom exactly at {bottom:.6f} (lowest z {zmin:.9f})")
        else:
            check(abs(zmin - in_min) < 1e-3, f"nothing below the bottom to cut: bottom stays where it was ({zmin:.4f})")
        if blank is not None:
            bco = world_co(blank)
            lo = [min(v[i] for v in bco) for i in range(3)]
            hi = [max(v[i] for v in bco) for i in range(3)]
            out_lo = [min(v[i] for v in co) for i in range(3)]
            out_hi = [max(v[i] for v in co) for i in range(3)]
            tol = 1e-4
            inside = all(out_lo[i] >= lo[i] - tol and out_hi[i] <= hi[i] + tol for i in range(3)) if not args else True
            check(inside, "the result stays inside the blank's box")
        flat_area = 0.0 if cut else 1.0
        for p in me.polygons:
            if p.normal.z < -0.999 and all(abs((out.matrix_world @ me.vertices[v].co).z - zmin) < 1e-6 for v in p.vertices):
                flat_area += p.area
        check(flat_area > 0, f"a flat face lies on the bottom ({flat_area:.4f} BU2)")
        band = 2.0 * MF.SHIFT       # the union is moved down SHIFT before the cut: nothing may be left just above the bottom
        # the cut plane itself must be exact: no vertex a rounding step off it (stones whose
        # underside really stops just short of the bottom are counted below, not failed)
        off = sum(1 for v in co if cut and 0.0 < v.z - zmin <= 1e-6)
        check(off == 0, f"flat bottom: no vertex a rounding step (<= 1e-6) above the cut plane ({off})")
        if cut:
            d = [v.z - zmin for v in co if v.z - zmin < band]
            bins = [(0, 0), (0, 1e-6), (1e-6, 1e-5), (1e-5, 1e-4), (1e-4, band)]
            print("INFO heights above the bottom (BU): exactly 0: %d, " % sum(1 for x in d if x == 0)
                  + ", ".join("%g-%g: %d" % (lo, hi, sum(1 for x in d if lo < x <= hi)) for lo, hi in bins[1:]))
            bad = [p for p in me.polygons if p.normal.z < -0.999
                   and min((out.matrix_world @ me.vertices[v].co).z for v in p.vertices) - zmin < 1e-4
                   and max((out.matrix_world @ me.vertices[v].co).z for v in p.vertices) - zmin > 0]
            print("INFO downward faces touching the bottom that are not exactly in it: %d, worst tilt %.2e BU, area %.6f BU2"
                  % (len(bad), max([max((out.matrix_world @ me.vertices[v].co).z for v in p.vertices) - zmin for p in bad] or [0]),
                     sum(p.area for p in bad)))
        near = sum(1 for v in co if cut and bottom + band <= v.z < bottom + 0.03)
        print(f"INFO {near} vertices end 0.001-0.03 above the bottom without reaching it (stones that stop just short)")
        from mathutils import Vector
        tilted = sum(1 for p in me.polygons if cut and all((out.matrix_world @ me.vertices[v].co).z == zmin for v in p.vertices)
                     and p.normal.z > -0.999 and p.area > 1e-8)   # (smaller: zero-size slivers, reported only)
        for p in me.polygons:
            if cut and all((out.matrix_world @ me.vertices[v].co).z == zmin for v in p.vertices) and p.normal.z > -0.999                     and p.area > 1e-12:
                print("INFO odd face in the bottom plane: normal %s area %.3e" % (tuple(round(x, 3) for x in p.normal), p.area))
        check(tilted == 0, f"no face lying in the bottom plane points anywhere but straight down ({tilted})")
        print("COUNTS verts %d faces %d seconds %.2f" % (len(me.vertices), len(me.polygons), dt))


run(blank)
if not os.environ.get("MM_SKIP_REGROW"):
    run(blank, regrow=True)
run(None)
check(all(snapshot(o) == before[o.name] for o in objs), "the originals are unchanged")

print("RESULT", "FAIL %d" % len(fails) if fails else "ALL PASS")
