"""Trim Edges test: Chip (chisel strokes on the sharp edges, dents on the flat faces, Trim Thin), Only Cut Edges, Round, through the real
operator. Background Blender only, on copies; nothing is saved.

    blender -b <file with sp_cut stones>.blend --python tests/trim_edges_chip.py -- <stone> [<stone> ...]

Checks: each result is closed and manifold, inside the original, changed (Chip), fewer or equal
changes with Only Cut Edges; Round (old) is gone and an old file opens on Round. Prints PASS / FAIL lines and times.
"""
import bpy, sys, time, importlib, addon_utils
import numpy as np

names = sys.argv[sys.argv.index("--") + 1:]
if not hasattr(bpy.types.Scene, "quick_infill_settings"):   # (a copy loaded before this test is used as is)
    mod = next(m.__name__ for m in addon_utils.modules() if m.__name__.endswith("Quick_Infill"))
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


def sharp_thin(m, angle=45.0):
    """Convex edges with an inside angle under `angle` whose two ends are thinner (meshlib ray
    thickness) than 0.12 BU (the old flat chip's thickness): the sharp, fragile points a print would lose."""
    m = mm.Mesh(m); m.pack()
    V = np.array(mn.getNumpyVerts(m), float); F = np.array(mn.getNumpyFaces(m.topology), np.int64)
    th = np.array(mm.computeRayThicknessAtVertices(m).vec, float)[:len(V)]
    T = 0.12
    c = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    nr = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-30)
    cen = V[F].mean(1)
    faces = {}
    for i, (a, b, c_) in enumerate(F):
        for e in ((a, b), (b, c_), (c_, a)):
            faces.setdefault((min(e), max(e)), []).append(i)
    n = 0
    for (a, b), fs in faces.items():
        if len(fs) != 2:
            continue
        f1, f2 = fs
        if (cen[f2] - cen[f1]) @ nr[f1] >= 0:      # not convex
            continue
        inside = 180.0 - np.degrees(np.arccos(np.clip(nr[f1] @ nr[f2], -1, 1)))
        if inside < angle and th[a] < T and th[b] < T:
            n += 1
    return n


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
        # sharp thin points left: convex edges sharper than 45 deg whose ends are thinner than the chip
        # thickness (0.12). Reported, before -> after (the old Round way for comparison).
        print("SHARP %s only_cut=%s %s: %d -> %d" % (method, only_cut, src.name, sharp_thin(O), sharp_thin(T)))
        if method in ("CHIP", "ROUND"):
            pts = mn.getNumpyVerts(T)
            vc = mn.fromNumpyArray(np.ascontiguousarray(pts, np.float32))
            if not isinstance(vc, mm.VertCoords):
                tmp = mm.VertCoords(); tmp.vec = vc; vc = tmp
            outside = float(np.array(mm.findSignedDistances(O, vc).vec).max())
            # (Chip ends with Trim Thin, a voxel shrink and regrow: within half a Stone voxel)
            tol = 0.004 if method == "CHIP" else 1e-3
            check(outside < tol, "%s %s: inside the original (max %.5f)" % (method, src.name, outside))
            check(lost >= -0.05 if method == "CHIP" else lost >= -0.01, "%s %s: volume lost %.2f %% (never gains)" % (method, src.name, lost))
        bpy.data.objects.remove(cp)
    return out


chip = run("CHIP")
check(any(v > 0.1 for v in chip), "Chip chipped something (%s)" % ", ".join("%.2f" % v for v in chip))
cut = run("CHIP", only_cut=True)
# (the cut faces' edges are chiselled in full either way; the strokes are drawn at random in another order, so the
# totals wander a little: 2.02 vs 1.87 % on Tna_Stone_006, 2026-11-23)
check(all(c <= a + 0.3 for c, a in zip(cut, chip)), "Only Cut Edges takes no more than all edges (%s vs %s %%)"
      % (["%.2f" % c for c in cut], ["%.2f" % a for a in chip]))
lean = run("ROUND")
for n, a in zip(names, lean):
    print("ROUND %s: volume lost %.2f %%" % (n, a))
check([e.identifier for e in s.bl_rna.properties["trim_edges_method"].enum_items] == ["ROUND", "CHIP"],
      "Trim Edges has Round and Chip only (Round (old) taken out)")
check(abs(s.bl_rna.properties["trim_edges_chip_size"].default - 1.0) < 1e-6
      and abs(s.bl_rna.properties["trim_edges_chip_amount"].default - 8.0) < 1e-6
      and abs(s.bl_rna.properties["trim_edges_chip_dent_size"].default - 0.5) < 1e-6
      and abs(s.bl_rna.properties["trim_edges_chip_dents"].default - 16.0) < 1e-6,
      "Chip Size 1 (Carve's Standard chisel), Amount 8, Dent Size 0.5, Dents 16 by default")
tops = importlib.import_module(type(s).__module__.rsplit(".", 1)[0] + ".tools_operators")
check(len(tops.chip_meshes(bpy.context)) > 0, "Dent Meshes found (%d damage meshes, Dent_L)" % len(tops.chip_meshes(bpy.context)))
# Alex 2026-11-23: chisel strokes on the long sharp edges and the corners, dents on the flat faces
td = importlib.import_module(type(s).__module__.rsplit(".", 1)[0] + ".trim_dent")
info = {}
td.dent_chip(ml_of(bpy.data.objects[names[0]]), [], 1.0, 2.5, 1.0, info=info, face_dents=tops.chip_meshes(bpy.context),
             dent_size=0.5, dent_amount=16.0)
check(info.get("strokes", 0) > 0 and info.get("corner_strokes", 0) > 0 and info.get("dents", 0) > 0,
      "Chip on %s: %d chisel strokes on the edges, %d on corners, %d dents on %d flat faces"
      % (names[0], info.get("strokes", 0), info.get("corner_strokes", 0), info.get("dents", 0), info.get("flat_faces", 0)))
# Alex 2026-11-23 "make sure each island gets chipped individually": two stones in one mesh, each chipped on its own
if len(names) >= 2:
    A, B = ml_of(bpy.data.objects[names[0]]), ml_of(bpy.data.objects[names[1]])
    VA, FA = mn.getNumpyVerts(A) * 10.0, mn.getNumpyFaces(A.topology)          # (x10: Quick Infill's units)
    VB, FB = (mn.getNumpyVerts(B) + np.array([5.0, 0, 0], np.float32)) * 10.0, mn.getNumpyFaces(B.topology)
    both = mn.meshFromFacesVerts(np.concatenate([FA, FB + len(VA)]).astype(np.int32), np.concatenate([VA, VB]))
    info2 = {}
    out2, tag2 = td.dent_trim(both, 0.075, [], 1.0, 2.5, 10.0, info=info2, face_dents=tops.chip_meshes(bpy.context),
                              auto=True)
    per = info2.get("per_island", [])
    n_out = len(mm.MeshComponents.getAllComponents(mm.MeshPart(out2)))
    check(info2.get("islands") == 2 and len(per) == 2 and all(p_.get("strokes", 0) > 0 for p_ in per)
          and n_out >= 2 and out2.topology.findHoleRepresentiveEdges().__len__() == 0,
          "two islands in one mesh: each chipped on its own (%s strokes, tags %s), %d pieces after, closed"
          % ([p_.get("strokes") for p_ in per], [p_.get("tag") for p_ in per], n_out))
s.trim_edges_chip_dents = 0.0
nodent = run("CHIP")
check(sum(nodent) < sum(chip), "Dents 0: the edges only, less taken (%.2f vs %.2f %%)" % (sum(nodent), sum(chip)))
s.trim_edges_chip_dents = 16.0
# the same stones get the same chips each time (the seed comes from the stone's shape)
again = run("CHIP")
check(all(abs(x - y) < 1e-6 for x, y in zip(chip, again)), "Chip gives the same result twice")
s.trim_edges_chip_amount = 0.0
none = run("CHIP")
check(sum(none) < sum(chip), "Amount 0 (no strokes, the dents only) takes less than the default (%.2f vs %.2f %%)" % (sum(none), sum(chip)))
s.trim_edges_chip_amount = 8.0
s["trim_edges_method"] = 2                      # a file saved with Round (old)
settings_mod = sys.modules[type(s).__module__]
settings_mod._old_round_to_round()
check(s.trim_edges_method == "ROUND", "a file saved with Round (old) opens on Round")
print("RESULT", "FAIL %d" % len(fails) if fails else "ALL PASS")
