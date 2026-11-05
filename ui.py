"""Quick Infill sidebar panel, preset operators and resolution-tag operators."""

import bpy
from bpy.types import Panel, Operator
from bpy.props import FloatProperty, EnumProperty

from . import tools_panel, support_tools
from .offset_utils import OFFSET_PRESETS, INFILL_PRESETS, RESOLUTION_PRESETS, RESOLUTION_TAG_ORDER, nearest_resolution_tag
from .settings import get_active_offset_preset, get_active_building_preset


class QUICKINFILL_OT_voxel_preset(Operator):
    bl_idname = "quick_infill.voxel_preset"
    bl_label = "Voxel Preset"
    bl_description = "Set voxel size for Offset Tools and Support Tools"
    bl_options = {'REGISTER', 'UNDO'}

    size: FloatProperty(name="Size", default=0.1)  # type: ignore

    def execute(self, context):
        context.scene.quick_infill_settings.shared_voxel_size = self.size
        return {'FINISHED'}


class QUICKINFILL_OT_offset_preset(Operator):
    bl_idname = "quick_infill.offset_preset"
    bl_label = "Offset Preset"
    bl_description = "Set offset distance, voxel size, trim edge x factor and decimation ratio"
    bl_options = {'REGISTER', 'UNDO'}

    preset_name: EnumProperty(
        name="Preset Name",
        items=[(k, k.title(), f"{k.title()} preset") for k in OFFSET_PRESETS],
        default="STONE",
    )

    def execute(self, context):
        s = context.scene.quick_infill_settings
        p = OFFSET_PRESETS[self.preset_name]
        s.shared_voxel_size = p["voxel"]
        s.distance = p["distance"]
        s.trim_edges_x = p["x"]
        s.decimation_ratio = p["ratio"]
        s.active_offset_preset = self.preset_name
        return {'FINISHED'}


class QUICKINFILL_OT_infill_preset(Operator):
    bl_idname = "quick_infill.infill_preset"
    bl_label = "Infill Preset"
    bl_description = "Set the Cavity Infill settings"

    preset_name: EnumProperty(
        name="Preset Name",
        items=[(k, k.replace('_', ' ').title(), "") for k in INFILL_PRESETS],
        default="BUILDING_ACCURATE",
    )

    def execute(self, context):
        s = context.scene.quick_infill_settings
        for k, v in INFILL_PRESETS[self.preset_name].items():
            setattr(s, k, v)
        s.active_preset = self.preset_name
        return {'FINISHED'}


def _make_infill_preset_op(idname, label, description, preset):
    """Keep the old per-preset operator ids working (keymaps, scripts)."""
    def execute(self, context):
        return bpy.ops.quick_infill.infill_preset(preset_name=preset)
    return type(f"QUICKINFILL_OT_preset_{idname}", (Operator,), {
        "bl_idname": f"quick_infill.preset_{idname}", "bl_label": label, "bl_description": description, "execute": execute,
    })


QUICKINFILL_OT_preset_building_fast = _make_infill_preset_op("building_fast", "Fast", "Fast settings for buildings", "BUILDING_FAST")
QUICKINFILL_OT_preset_building_accurate = _make_infill_preset_op("building_accurate", "Accurate", "Accurate settings for buildings", "BUILDING_ACCURATE")
QUICKINFILL_OT_preset_mini_large_holes = _make_infill_preset_op("mini_large_holes", "Large Holes", "Large holes settings for miniatures", "MINI_LARGE_HOLES")
QUICKINFILL_OT_preset_mini_accurate = _make_infill_preset_op("mini_accurate", "Accurate", "Accurate settings for miniatures", "MINI_ACCURATE")


class QUICKINFILL_OT_select_by_resolution_tag(Operator):
    bl_idname = "quick_infill.select_by_resolution_tag"
    bl_label = "Select Same Resolution"
    bl_description = "Select currently visible objects (or mesh islands in Edit Mode) tagged with the current voxel size preset"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from .blender_meshlib_utils import read_object_resolution_tag, RES_TAG_ATTR

        target_tag = nearest_resolution_tag(context.scene.quick_infill_settings.shared_voxel_size)
        target_index = RESOLUTION_TAG_ORDER.index(target_tag)

        if context.mode == 'EDIT_MESH':
            import bmesh
            edit_objs = [o for o in context.objects_in_mode if o.type == 'MESH']
            if not edit_objs:
                self.report({'ERROR'}, "No mesh in edit mode.")
                return {'CANCELLED'}
            matched, untagged = 0, []
            for obj in edit_objs:
                bm = bmesh.from_edit_mesh(obj.data)
                layer = bm.faces.layers.int.get(RES_TAG_ATTR)
                if layer is None:
                    untagged.append(obj.name)
                    continue
                for f in bm.faces:
                    hit = f[layer] == target_index
                    f.select = hit
                    matched += hit
                bm.select_flush_mode()
                bmesh.update_edit_mesh(obj.data)
            if matched == 0:
                self.report({'INFO'}, f"No faces tagged '{target_tag}'.")
            elif untagged:
                self.report({'INFO'}, f"Selected {matched} face(s) tagged '{target_tag}'. No tags on: " + ", ".join(f"'{n}'" for n in untagged))
            else:
                self.report({'INFO'}, f"Selected {matched} face(s) tagged '{target_tag}'.")
            return {'FINISHED'}

        # Object mode: only visible objects are touched; hidden ones keep their selection state.
        matched, last = 0, None
        for obj in context.view_layer.objects:
            if obj.type != 'MESH' or not obj.visible_get():
                continue
            hit = read_object_resolution_tag(obj) == target_tag
            obj.select_set(hit)
            if hit:
                matched += 1
                last = obj
        if last is not None:
            context.view_layer.objects.active = last
        self.report({'INFO'}, f"Selected {matched} object(s) tagged '{target_tag}'." if matched else f"No visible objects tagged '{target_tag}'.")
        return {'FINISHED'}


class QUICKINFILL_OT_pick_resolution_tag(Operator):
    bl_idname = "quick_infill.pick_resolution_tag"
    bl_label = "Pick Resolution"
    bl_description = "Set the voxel size and ratio preset from the active object's tag (or active face's tag in Edit Mode)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from .blender_meshlib_utils import read_object_resolution_tag, RES_TAG_ATTR, _index_to_tag

        tag = None
        if context.mode == 'EDIT_MESH':
            import bmesh
            obj = context.edit_object
            if obj is not None and obj.type == 'MESH':
                bm = bmesh.from_edit_mesh(obj.data)
                layer = bm.faces.layers.int.get(RES_TAG_ATTR)
                if layer is not None:
                    face = bm.faces.active or next((f for f in bm.faces if f.select), None)
                    if face is not None:
                        tag = _index_to_tag(face[layer])
        else:
            tag = read_object_resolution_tag(context.active_object)

        if tag is None:
            self.report({'WARNING'}, "No resolution tag found to pick.")
            return {'CANCELLED'}
        s = context.scene.quick_infill_settings
        s.shared_voxel_size, s.decimation_ratio = RESOLUTION_PRESETS[tag]
        s.active_offset_preset = tag if tag in OFFSET_PRESETS else 'NONE'
        self.report({'INFO'}, f"Set preset to '{tag}'.")
        return {'FINISHED'}


class QUICKINFILL_OT_test_cuda(Operator):
    bl_idname = "quick_infill.test_cuda"
    bl_label = "Test CUDA"
    bl_description = "Print CUDA availability to the Console"

    def execute(self, context):
        try:
            from .meshlib_utils import get_mrcudapy
            ok = get_mrcudapy().isCudaAvailable()
            print(f"[Quick Infill] CUDA available = {ok}")
            self.report({'INFO'}, f"CUDA available: {ok}")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Quick Infill Test failed: {e}")
            return {'CANCELLED'}


class QUICKINFILL_PT_sidebar(Panel):
    bl_label = "Quick Infill"
    bl_idname = "QUICKINFILL_PT_sidebar"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Quick Infill"

    def draw(self, context):
        s = context.scene.quick_infill_settings
        col = self.layout.column(align=True)

        def prop_with_suffix(layout, attr, label, suffix):
            split = layout.split(factor=0.9, align=True)
            c = split.column(align=True)
            c.use_property_split = False
            c.use_property_decorate = False
            c.prop(s, attr, text=label)
            split.label(text=suffix)

        # ---------------- Voxel Tools ----------------
        box = col.box()
        icon = 'DOWNARROW_HLT' if s.show_voxel_tools_menu else 'RIGHTARROW'
        box.row().prop(s, "show_voxel_tools_menu", text="Voxel Tools", icon=icon, emboss=False)
        if s.show_voxel_tools_menu:
            vc = box.column(align=True)
            split = vc.row(align=True).split(factor=0.4, align=True)
            toggles = split.row(align=True)
            toggles.prop(s, "replace_original", text="", icon='PASTEDOWN', toggle=True)
            toggles.prop(s, "process_islands", text="", icon='GEOMETRY_SET', toggle=True)
            toggles.prop(s, "apply_modifiers_on_export", text="", toggle=True,
                         icon='MODIFIER_ON' if s.apply_modifiers_on_export else 'MODIFIER_OFF')
            bools = split.row(align=True)
            bools.operator("quick_infill.voxel_intersect", text="", icon='SELECT_INTERSECT')
            bools.operator("quick_infill.voxel_union", text="", icon='SELECT_EXTEND')
            bools.operator("quick_infill.voxel_diff", text="", icon='SELECT_DIFFERENCE')

            vc.separator(factor=2.0)
            vc.label(text="Presets:")
            row = vc.row(align=True)
            active = get_active_offset_preset(s)
            for key in OFFSET_PRESETS:
                row.operator("quick_infill.offset_preset", text=key.title(), depress=(active == key)).preset_name = key

            vc.separator(factor=1.0)
            vc.prop(s, "shared_voxel_size", text="Voxel Size")
            row = vc.row(align=True)
            for size in (0.025, 0.05, 0.1, 0.2, 0.3, 0.4):
                row.operator("quick_infill.voxel_preset", text=str(size)).size = size

            vc.separator(factor=1.0)
            vc.label(text="Auto Decimation")
            vc.row(align=True).prop(s, "decimate_mode", expand=True)
            vc.prop(s, "decimation_ratio", text="Ratio")
            row = vc.row(align=True)
            row.prop(s, "smart_resolution", text="Smart Res", toggle=True)
            row.operator("quick_infill.select_by_resolution_tag", text="", icon='VIEWZOOM')
            row.operator("quick_infill.pick_resolution_tag", text="", icon='EYEDROPPER')

            vc.separator(factor=1.0)
            tools_panel.draw_offset_tools(vc, context)
            vc.separator(factor=1.0)
            support_tools.draw_support_tools(vc, context)

        # ---------------- Infill Tools (own resolution / grow / shrink) ----------------
        box = col.box()
        icon = 'DOWNARROW_HLT' if s.show_infill_tools_menu else 'RIGHTARROW'
        box.row().prop(s, "show_infill_tools_menu", text="Infill Tools", icon=icon, emboss=False)
        if s.show_infill_tools_menu:
            ic = box.column(align=True)
            ic.row(align=True).operator("quick_infill.heal_cavity", text="Create Cavity Infill")
            ic.separator(factor=1.0)
            ic.label(text="Presets:")
            active = get_active_building_preset(s)
            pc = ic.column(align=True)
            row = pc.row(align=True)
            row.label(text="Building:")
            row.operator("quick_infill.infill_preset", text="Fast", depress=(active == 'BUILDING_FAST')).preset_name = 'BUILDING_FAST'
            row.operator("quick_infill.infill_preset", text="Accurate", depress=(active == 'BUILDING_ACCURATE')).preset_name = 'BUILDING_ACCURATE'
            row = pc.row(align=True)
            row.label(text="Mini:")
            row.operator("quick_infill.infill_preset", text="Large Holes", depress=(active == 'MINI_LARGE_HOLES')).preset_name = 'MINI_LARGE_HOLES'
            row.operator("quick_infill.infill_preset", text="Accurate", depress=(active == 'MINI_ACCURATE')).preset_name = 'MINI_ACCURATE'

            ic.separator(factor=1.0)
            ic.label(text="Settings:")
            sc = ic.column(align=True)
            split = sc.split(factor=0.4, align=True)
            split.label(text="Voxel Mode")
            split.prop(s, "voxel_mode", text="")
            split = sc.split(factor=0.4, align=True)
            split.label(text="Method")
            split.prop(s, "method", text="")
            sc.separator(factor=1.0)
            prop_with_suffix(sc, "target_res", "Working Res", "M")
            if s.voxel_mode == 'RESOLUTION':
                prop_with_suffix(sc, "resolution", "Resolution", "mm")
            prop_with_suffix(sc, "grow", "Grow", "mm")
            sc.prop(s, "shrink_mult")
            sc.separator(factor=1.0)
            sc.prop(s, "trim_thin")


classes = (
    QUICKINFILL_OT_voxel_preset,
    QUICKINFILL_OT_offset_preset,
    QUICKINFILL_OT_infill_preset,
    QUICKINFILL_OT_preset_building_fast,
    QUICKINFILL_OT_preset_building_accurate,
    QUICKINFILL_OT_preset_mini_large_holes,
    QUICKINFILL_OT_preset_mini_accurate,
    QUICKINFILL_OT_select_by_resolution_tag,
    QUICKINFILL_OT_pick_resolution_tag,
    QUICKINFILL_OT_test_cuda,
    QUICKINFILL_PT_sidebar,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
