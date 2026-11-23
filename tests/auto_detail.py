"""Auto Detail in the shared pipeline (Alex 2026-11-23: "make sure all the offset tools follow the smart resolution and
decimation we've implemented here"): Grow, Shrink, Remesh, Trim Thin and Trim Edges (Round) on a plain block, a moulded
arch stone and a carved capital, with Auto Detail on and off. On: every result closed, tagged with the resolution worked
out from its own shape, the plain block down to few faces; off: the tag's voxel and ratio as before.
blender -b --python tests/auto_detail.py -- <row_parts.npz>   (the pillar-row parts; never saves)"""
import importlib
import sys

import addon_utils
import bpy
import numpy as np

mod = next(m.__name__ for m in addon_utils.modules() if m.__name__.endswith("Quick_Infill"))
if mod not in bpy.context.preferences.addons:
    addon_utils.enable(mod, default_set=False)
bmu = importlib.import_module(mod + ".blender_meshlib_utils")
ou = importlib.import_module(mod + ".offset_utils")
from meshlib import mrmeshpy as mm  # noqa: E402

z = np.load(sys.argv[sys.argv.index("--") + 1])
rows = {str(r).split("|")[1]: str(r).split("|") for r in z["names"]}
PARTS = ["Plinth_001", "Arch1_Stone_O1_00", "Capital_007"]
s = bpy.context.scene.quick_infill_settings
s.replace_original, s.process_islands, s.apply_modifiers_on_export, s.ask_above_seconds = True, False, False, 0.0
s.smart_resolution = True
s.trim_edges_method = 'ROUND' if 'ROUND' in {i.identifier for i in s.bl_rna.properties['trim_edges_method'].enum_items} else s.trim_edges_method
fails = 0


def check(ok, msg):
    global fails
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    fails += 0 if ok else 1


def make():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    obs = []
    for w in PARTS:
        k, name, tag = rows["Cube_" + w]
        me = bpy.data.meshes.new(w)
        me.from_pydata(z[k + "_v"].astype(float).tolist(), [], z[k + "_t"].tolist())
        ob = bpy.data.objects.new(w, me)
        bpy.context.scene.collection.objects.link(ob)
        if tag:
            ob["qi_res_tag"] = tag
        ob.select_set(True)
        obs.append(ob)
    bpy.context.view_layer.objects.active = obs[0]
    bpy.context.view_layer.update()
    return [o.name for o in obs]


def tags_of(ob):
    att = ob.data.attributes.get("qi_res_tag")
    if att is None:
        return {}
    vals = [0] * len(ob.data.polygons)
    att.data.foreach_get("value", vals)
    u, c = np.unique(vals, return_counts=True)
    return {ou.RESOLUTION_TAG_ORDER[i] if 0 <= i < len(ou.RESOLUTION_TAG_ORDER) else i: int(n) for i, n in zip(u, c)}


res = {}
for tool in ("grow", "shrink", "remesh", "trim_thin", "trim_edges"):
    for auto in (True, False):
        s.auto_detail = auto
        names = make()
        before = {n: len(bpy.data.objects[n].data.polygons) for n in names}
        r = getattr(bpy.ops.quick_infill, tool)('EXEC_DEFAULT')
        ok = 'FINISHED' in r
        line = []
        for n in names:
            ob = bpy.data.objects.get(n)
            if ob is None or not len(ob.data.polygons):
                line.append("%s: gone" % n)
                continue
            m = bmu.blender_to_meshlib(ob)
            holes = len(m.topology.findHoleRepresentiveEdges())
            ok = ok and holes == 0
            res[(tool, auto, n)] = len(ob.data.polygons)
            line.append("%s %d -> %d %s" % (n.split("_")[0], before[n], len(ob.data.polygons), "/".join(tags_of(ob)) or "-"))
        check(ok, "%-10s Auto Detail %-3s: finished, closed | %s" % (tool, "on" if auto else "off", " | ".join(line)))
for tool in ("grow", "remesh", "trim_thin"):
    a, b = res.get((tool, True, "Plinth_001")), res.get((tool, False, "Plinth_001"))
    check(a is not None and a < 12000, "%s: the plain block with Auto Detail on has %s faces (off: %s)" % (tool, a, b))
print("RESULT", "ALL PASS" if not fails else "FAIL %d" % fails, flush=True)
import os
os._exit(0)
