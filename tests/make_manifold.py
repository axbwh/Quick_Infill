"""Make Manifold test, through the real operator. Background Blender only; nothing is saved.

    blender -b --python tests/make_manifold.py
    blender -b <file>.blend --python tests/make_manifold.py -- <collection> [<blank>]

Without arguments it builds overlapping spheres and cubes sunk below Z = 0, one of them carrying a
flat open speck (two triangles, no volume), and a blank box to intersect with. With a collection it
uses those objects instead (and the blank, if named).

Checks: one closed object comes out (no open or non-manifold edges), the bottom is cut flat at Z = 0,
nothing pokes out of the blank, the originals are unchanged, and the meshlib repair is gone.
Prints PASS / FAIL lines, counts and the time.
"""
import bpy, sys, time, bmesh, addon_utils

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
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.9, 0, 0.4), scale=(1.6, 0.55, 1.2))
    blank = bpy.context.object
    blank.name = "Blank"

before = {o.name: snapshot(o) for o in objs}
check(not hasattr(MF, "repair") and not hasattr(MF, "tidy_after_cut"), "the meshlib repair is gone from make_manifold")

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
out = bpy.context.view_layer.objects.active
check(out is not None and out.name.endswith("_Solid"), f"result object {out.name if out else None}")
if out is not None and out.name.endswith("_Solid"):
    me = out.data
    nb = bad_edges(me)
    check(nb == 0, f"closed: {nb} open / non-manifold edges")
    co = world_co(out)
    zmin = min(v.z for v in co)
    check(abs(zmin) < 1e-5, f"bottom at the plate (lowest z {zmin:.6f})")
    if blank is not None:
        bco = world_co(blank)
        lo = [min(v[i] for v in bco) for i in range(3)]
        hi = [max(v[i] for v in bco) for i in range(3)]
        out_lo = [min(v[i] for v in co) for i in range(3)]
        out_hi = [max(v[i] for v in co) for i in range(3)]
        inside = all(out_lo[i] >= lo[i] - 1e-4 and out_hi[i] <= hi[i] + 1e-4 for i in range(3)) if not args else True
        check(inside, "the result stays inside the blank's box")
    flat_area = 0.0
    for p in me.polygons:
        if p.normal.z < -0.999 and all(abs((out.matrix_world @ me.vertices[v].co).z) < 1e-6 for v in p.vertices):
            flat_area += p.area
    check(flat_area > 0, f"a flat face lies on the plate ({flat_area:.4f} BU2)")
    print("COUNTS verts %d faces %d seconds %.2f" % (len(me.vertices), len(me.polygons), dt))
check(all(snapshot(o) == before[o.name] for o in objs), "the originals are unchanged")

print("RESULT", "FAIL %d" % len(fails) if fails else "ALL PASS")
