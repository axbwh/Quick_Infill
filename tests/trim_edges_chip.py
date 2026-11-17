"""Trim Edges test: the Flat Chip method through the real operator, then Only Cut Edges, then the old
Round (voxel) method. Background Blender only, on copies; nothing is saved.

    blender -b <file with sp_cut stones>.blend --python tests/trim_edges_chip.py -- <stone> [<stone> ...]

Checks: each result is closed and manifold, inside the original, changed (Flat Chip), fewer or equal
changes with Only Cut Edges, and the old method still runs. Prints PASS / FAIL lines and times.
"""
import bpy, sys, time, importlib, addon_utils
import numpy as np

names = sys.argv[sys.argv.index("--") + 1:]
mod = next(m.__name__ for m in addon_utils.modules() if m.__name__.endswith("Quick_Infill"))
if mod not in bpy.context.preferences.addons:
    addon_utils.enable(mod, default_set=False)
from meshlib import mrmeshpy as mm
from meshlib import mrmeshnumpy as mn

sc = bpy.context.scene
s = sc.quick_infill_settings
s.replace_original, s.process_islands, s.apply_modifiers_on_export, s.ask_above_seconds = True, False, False, 0.0
fails = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


def ml_of(ob):
    me = ob.data; me.calc_loop_triangles()
    co = np.array([ob.matrix_world @ v.co for v in me.vertices], np.float32)
    tr = np.array([t.vertices[:] for t in me.loop_triangles], np.int32)
    return mn.meshFromFacesVerts(tr, co)


def run(method, only_cut=False):
    s.trim_edges_method = method
    s.trim_edges_only_cut = only_cut
    copies = []
    for n in names:
        src = bpy.data.objects[n]
        cp = src.copy(); cp.data = src.data.copy(); sc.collection.objects.link(cp); copies.append((src, cp))
    for o in bpy.context.selected_objects:
        o.select_set(False)
    for _, cp in copies:
        cp.select_set(True)
    bpy.context.view_layer.objects.active = copies[0][1]
    t0 = time.perf_counter()
    r = bpy.ops.quick_infill.trim_edges('EXEC_DEFAULT')
    dt = time.perf_counter() - t0
    check('FINISHED' in r, "%s only_cut=%s: operator finished (%.3f s for %d stones)" % (method, only_cut, dt, len(copies)))
    out = []
    for src, cp in copies:
        O, T = ml_of(src), ml_of(cp)
        vO, vT = abs(O.volume()), abs(T.volume())
        holes = len(T.topology.findHoleRepresentiveEdges())
        selfx = len(mm.findSelfCollidingTriangles(mm.MeshPart(T)))
        lost = 100 * (vO - vT) / vO
        out.append(lost)
        check(holes == 0 and selfx == 0, "%s %s: closed and not self-crossing (holes %d, crossings %d)"
              % (method, src.name, holes, selfx))
        if method == "CHIP":
            pts = mn.getNumpyVerts(T)
            vc = mn.fromNumpyArray(np.ascontiguousarray(pts, np.float32))
            if not isinstance(vc, mm.VertCoords):
                tmp = mm.VertCoords(); tmp.vec = vc; vc = tmp
            outside = float(np.array(mm.findSignedDistances(O, vc).vec).max())
            check(outside < 1e-3, "%s %s: inside the original (max %.5f)" % (method, src.name, outside))
            check(lost >= -0.01, "%s %s: volume lost %.2f %% (never gains)" % (method, src.name, lost))
        bpy.data.objects.remove(cp)
    return out


chip = run("CHIP")
check(any(v > 0.001 for v in chip), "Flat Chip chipped something (%s)" % ", ".join("%.2f" % v for v in chip))
cut = run("CHIP", only_cut=True)
check(all(c <= a + 0.05 for c, a in zip(cut, chip)), "Only Cut Edges takes no more than all edges")
run("VOXEL")
print("RESULT", "FAIL %d" % len(fails) if fails else "ALL PASS")
