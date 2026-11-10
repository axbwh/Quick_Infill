"""
Operators for Print Check and Seal Cavities (see print_check.py for the analysis).

The selected mesh objects are read together as ONE model standing on the build
plate, the way they will be printed (stones + blank + infill selected together).
"""

import bpy
from bpy.types import Operator

from .meshlib_utils import JobCancelled
from .blender_meshlib_utils import (
    blender_to_meshlib, meshlib_to_blender_mesh, link_new_object, remove_objects, select_results, STL_EXPORT_SCALE,
)
from .job_runner import ModalJobMixin, JobSpec
from . import print_check, make_manifold
from .offset_utils import decimate_mesh, face_count

COLLECTION = "Print Check"
MARKER_MAX_FACES = 200_000
# name, mask key, object colour
MARKERS = (
    ("PC_Sealed", "sealed", (1.0, 0.05, 0.05, 1.0)),
    ("PC_Cups", "cups", (1.0, 0.45, 0.0, 1.0)),
    ("PC_Islands", "islands", (1.0, 0.9, 0.0, 1.0)),
    ("PC_Ledges", "ledges", (0.6, 0.2, 1.0, 1.0)),
    ("PC_NoContact", "no_contact", (0.0, 0.6, 1.0, 1.0)),
)


def _model_objects(context):
    return [o for o in context.selected_objects if o.type == 'MESH' and not o.name.startswith("PC_")]


def _plate_z(s, meshes):
    """Build plate height in meshlib space."""
    if s.check_plate == 'Z0':
        return 0.0
    return None  # lowest point of the model


def _marker_collection(scene):
    coll = bpy.data.collections.get(COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(COLLECTION)
    if coll.name not in scene.collection.children:
        try:
            scene.collection.children.link(coll)
        except RuntimeError:
            pass
    return coll


def _clear_markers():
    remove_objects([o for o in bpy.data.objects if o.name.split(".")[0] in {m[0] for m in MARKERS}])


class QUICKINFILL_OT_print_check(ModalJobMixin, Operator):
    bl_idname = "quick_infill.print_check"
    bl_label = "Print Check"
    bl_description = ("Check the selected objects together as one print standing on the build plate: sealed cavities "
                      "(trapped resin), suction cups, islands, ledges and plate contact. Problem areas are added as "
                      "coloured objects in the 'Print Check' collection")
    bl_options = {'REGISTER', 'UNDO'}

    def job_setup(self, context):
        s = context.scene.quick_infill_settings
        objs = _model_objects(context)
        if not objs:
            self.report({'ERROR'}, "No mesh selected.")
            return None
        apply_mods = s.apply_modifiers_on_export
        voxel, gap = float(s.check_voxel), float(s.check_drain_gap)
        state = {"meshes": [], "report": None, "marker_meshes": []}

        def prepare():
            for o in objs:
                state["meshes"].append(blender_to_meshlib(o, apply_mods))

        def compute(control):
            rep = print_check.check(state["meshes"], voxel=voxel, plate_z=_plate_z(s, state["meshes"]), drain_gap=gap)
            state["report"] = rep
            for name, key, colour in MARKERS:
                m = print_check.mask_to_mesh(rep.masks[key], rep.grid.origin, rep.grid.voxel)
                if m is not None:
                    if face_count(m) > MARKER_MAX_FACES:      # a marker only has to show where
                        decimate_mesh(m, target_face_count=MARKER_MAX_FACES, max_error=rep.grid.voxel * 2.0)
                    state["marker_meshes"].append((name, colour, m))
            rep.masks = {}       # free the grids
            rep.grid = None
            if control is not None:
                control.item_done()

        def finish():
            rep = state["report"]
            scene = bpy.context.scene
            _clear_markers()
            coll = _marker_collection(scene)
            for name, colour, m in state["marker_meshes"]:
                ob = link_new_object(name, meshlib_to_blender_mesh(m, name), [coll])
                ob.color = colour
                ob.show_in_front = True
                ob.hide_select = False
            bpy.context.view_layer.update()
            st = scene.quick_infill_settings
            st.check_result_cavities = f"Sealed cavities: {rep.sealed_count}  ({rep.sealed_volume:.2f} mm³)"
            st.check_result_cups = f"Suction cups: {rep.cup_volume:.1f} mm³, deepest {rep.cup_max_depth:.1f} mm"
            st.check_result_islands = (f"Islands: {rep.island_count}  ({rep.island_area:.1f} mm²), "
                                       f"ledges {rep.ledge_area:.1f} mm²")
            st.check_result_contact = f"Plate contact: {rep.contact_percent:.1f}% of {rep.footprint_area:.0f} mm²"
            level = {'INFO'} if rep.ok() else {'WARNING'}
            self.report(level, "Print Check: " + rep.summary())
            return {'FINISHED'}

        return JobSpec(prepare, compute, finish, items=1)


class QUICKINFILL_OT_seal_cavities(ModalJobMixin, Operator):
    bl_idname = "quick_infill.seal_cavities"
    bl_label = "Seal Cavities"
    bl_description = ("Fill every sealed cavity of the selected objects (read together as one print on the build "
                      "plate) with a plug object. Sealed cavities cannot be seen from outside, so nothing visible changes")
    bl_options = {'REGISTER', 'UNDO'}

    def job_setup(self, context):
        s = context.scene.quick_infill_settings
        objs = _model_objects(context)
        if not objs:
            self.report({'ERROR'}, "No mesh selected.")
            return None
        active = context.active_object if context.active_object in objs else objs[0]
        apply_mods = s.apply_modifiers_on_export
        voxel, gap = float(s.check_voxel), float(s.check_drain_gap)
        state = {"meshes": [], "plugs": None, "info": {}}

        def prepare():
            for o in objs:
                state["meshes"].append(blender_to_meshlib(o, apply_mods))

        def compute(control):
            state["plugs"] = print_check.seal_plugs(state["meshes"], voxel=voxel, plate_z=_plate_z(s, state["meshes"]),
                                                    drain_gap=gap, report=state["info"])
            if control is not None:
                control.item_done()

        def finish():
            plugs = state["plugs"]
            if plugs is None:
                self.report({'INFO'}, "Seal Cavities: no sealed cavity found.")
                return {'FINISHED'}
            name = active.name + "_Plugs"
            try:
                colls = list(active.users_collection) or [bpy.context.scene.collection]
            except ReferenceError:
                colls = [bpy.context.scene.collection]
            ob = link_new_object(name, meshlib_to_blender_mesh(plugs, name), colls)
            bpy.context.view_layer.update()
            select_results([ob])
            self.report({'INFO'}, f"Seal Cavities: filled {state['info'].get('sealed_volume', 0.0):.2f} mm³ "
                                  f"of sealed space. Created '{ob.name}'")
            return {'FINISHED'}

        return JobSpec(prepare, compute, finish, items=1)


class QUICKINFILL_OT_make_manifold(Operator):
    bl_idname = "quick_infill.make_manifold"
    bl_label = "Make Manifold"
    bl_description = ("Unite the selected objects (stones, infill, plugs) into ONE watertight object with Blender's "
                      "Manifold solver, repair self-intersections and zero-size faces with meshlib, then make the flat "
                      "cuts last (Intersect With blank, Cut at Plate) so they stay flat. The originals are kept; the "
                      "result is '<active>_Solid'")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return any(o.type == 'MESH' for o in context.selected_objects)

    def execute(self, context):
        s = context.scene.quick_infill_settings
        blank_obj = s.manifold_blank if (s.manifold_blank is not None and s.manifold_blank.type == 'MESH') else None
        objs = [o for o in _model_objects(context) if o != blank_obj]
        if not objs:
            self.report({'ERROR'}, "No mesh selected (apart from the blank).")
            return {'CANCELLED'}
        active = context.active_object if context.active_object in objs else objs[0]
        name = active.name + "_Solid"
        colls = list(active.users_collection) or [context.scene.collection]
        notes = []
        try:
            meshes = [blender_to_meshlib(o, s.apply_modifiers_on_export) for o in objs]
            # the blank is read as you see it: with its modifiers
            blank = blender_to_meshlib(blank_obj, True) if blank_obj is not None else None
            # with "Lowest Point" the plate already touches the model: nothing to cut
            plate = 0.0 if (s.manifold_cut_plate and s.check_plate == 'Z0') else None
            out, health, flat = make_manifold.make_manifold(meshes, plate_z=plate, blank=blank,
                                                            open_shell_voxel=float(s.check_voxel) * 0.5,
                                                            log=notes.append)
        except JobCancelled:
            self.report({'WARNING'}, "Make Manifold cancelled. Nothing was changed.")
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Make Manifold failed: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
        ob = link_new_object(name, meshlib_to_blender_mesh(out, name), colls)
        context.view_layer.update()
        select_results([ob])
        for n in notes:
            self.report({'INFO'}, "Make Manifold: " + n)
        if flat is not None:
            self.report({'INFO'} if flat.flat() else {'WARNING'}, "Make Manifold: " + flat.text())
        if health.clean():
            self.report({'INFO'}, f"Make Manifold: '{ob.name}' is {health.text()}")
        else:
            self.report({'WARNING'}, f"Make Manifold: '{ob.name}' still has {health.text()}")
        return {'FINISHED'}


def draw_print_check(layout, context):
    s = context.scene.quick_infill_settings
    col = layout.column(align=True)
    row = col.row(align=True)
    row.operator("quick_infill.print_check", text="Print Check", icon='VIEWZOOM')
    row.operator("quick_infill.seal_cavities", text="Seal Cavities", icon='MOD_SOLIDIFY')
    col.separator(factor=0.5)
    col.prop(s, "check_voxel", text="Check Voxel (mm)")
    col.prop(s, "check_drain_gap", text="Min Drain Gap (mm)")
    col.row(align=True).prop(s, "check_plate", expand=True)
    if s.check_result_cavities:
        box = col.box().column(align=True)
        for attr in ("check_result_cavities", "check_result_cups", "check_result_islands", "check_result_contact"):
            box.label(text=getattr(s, attr))
    col.separator()
    row = col.row(align=True)
    row.operator("quick_infill.make_manifold", text="Make Manifold", icon='MESH_CUBE')
    row.prop(s, "manifold_cut_plate", text="Cut at Plate", toggle=True)
    col.prop(s, "manifold_blank", text="Intersect With")


classes = (
    QUICKINFILL_OT_print_check,
    QUICKINFILL_OT_seal_cavities,
    QUICKINFILL_OT_make_manifold,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
