import bpy
from bpy.types import Panel, Operator, PropertyGroup
from bpy.props import FloatProperty, IntProperty, PointerProperty, EnumProperty, BoolProperty
from . import tools_panel
from . import support_tools
from .offset_utils import RESOLUTION_PRESETS


def get_active_building_preset(settings):
    """Return the matching preset for the current building settings."""
    if getattr(settings, 'voxel_mode', 'TARGET_VOXELS') == 'TARGET_VOXELS':
        if abs(getattr(settings, 'target_res', 1.0) - 0.5) < 1e-6 and abs(getattr(settings, 'resolution', 0.1) - 0.1) < 1e-6 and abs(getattr(settings, 'grow', 1.0) - 0.75) < 1e-6 and abs(getattr(settings, 'shrink_mult', 1.0) - 2.0) < 1e-6 and getattr(settings, 'method', 'ACCURATE') == 'NAIVE' and getattr(settings, 'trim_thin', True):
            return 'BUILDING_FAST'
        if abs(getattr(settings, 'target_res', 1.0) - 3.0) < 1e-6 and abs(getattr(settings, 'resolution', 0.1) - 0.1) < 1e-6 and abs(getattr(settings, 'grow', 1.0) - 1.0) < 1e-6 and abs(getattr(settings, 'shrink_mult', 1.0) - 1.2) < 1e-6 and getattr(settings, 'method', 'ACCURATE') == 'ACCURATE' and getattr(settings, 'trim_thin', True):
            return 'BUILDING_ACCURATE'

    if getattr(settings, 'voxel_mode', 'TARGET_VOXELS') == 'RESOLUTION':
        if abs(getattr(settings, 'resolution', 0.1) - 0.1) < 1e-6 and abs(getattr(settings, 'grow', 1.0) - 1.0) < 1e-6 and abs(getattr(settings, 'shrink_mult', 1.0) - 1.5) < 1e-6 and getattr(settings, 'method', 'ACCURATE') == 'ACCURATE' and getattr(settings, 'trim_thin', True):
            return 'MINI_LARGE_HOLES'
        if abs(getattr(settings, 'resolution', 0.1) - 0.075) < 1e-6 and abs(getattr(settings, 'grow', 1.0) - 0.35) < 1e-6 and abs(getattr(settings, 'shrink_mult', 1.0) - 1.2) < 1e-6 and getattr(settings, 'method', 'ACCURATE') == 'ACCURATE' and getattr(settings, 'trim_thin', True):
            return 'MINI_ACCURATE'

    return 'NONE'


def reset_preset(self, context):
    """Reset active preset when properties are manually changed"""
    if hasattr(self, 'active_preset'):
        self.active_preset = get_active_building_preset(self)


def get_active_offset_preset(settings):
    """Return the matching preset for the current voxel and ratio values."""
    current_voxel = settings.shared_voxel_size
    current_ratio = settings.decimation_ratio
    for key, (voxel, ratio) in RESOLUTION_PRESETS.items():
        if abs(current_voxel - voxel) < 1e-6 and abs(current_ratio - ratio) < 1e-6:
            return key
    return 'NONE'


def reset_offset_preset(self, context):
    """Keep the active offset preset synced to the current values."""
    self.active_offset_preset = get_active_offset_preset(self)


def sync_shared_voxel_size(self, context):
    """Keep the shared voxel size in sync with offset/support tool settings."""
    value = self.shared_voxel_size
    reset_offset_preset(self, context)
    if hasattr(context.scene, 'quick_infill_tools_settings'):
        context.scene.quick_infill_tools_settings.voxel_size = value
    if hasattr(context.scene, 'quick_infill_support_settings'):
        context.scene.quick_infill_support_settings.voxel_size = value


def sync_shared_decimate_mode(self, context):
    """Mirror the shared decimation mode to tool settings."""
    if hasattr(context.scene, 'quick_infill_tools_settings'):
        context.scene.quick_infill_tools_settings.decimate_mode = self.decimate_mode
    if hasattr(context.scene, 'quick_infill_support_settings'):
        context.scene.quick_infill_support_settings.decimate_mode = self.decimate_mode


def sync_shared_decimation_ratio(self, context):
    """Mirror the shared decimation ratio to tool settings."""
    reset_offset_preset(self, context)
    if hasattr(context.scene, 'quick_infill_tools_settings'):
        context.scene.quick_infill_tools_settings.decimation_ratio = self.decimation_ratio
    if hasattr(context.scene, 'quick_infill_support_settings'):
        context.scene.quick_infill_support_settings.decimation_ratio = self.decimation_ratio


def sync_shared_replace_original(self, context):
    """Mirror the shared replace-original toggle to tool settings."""
    if hasattr(context.scene, 'quick_infill_tools_settings'):
        context.scene.quick_infill_tools_settings.replace_original = self.replace_original
    if hasattr(context.scene, 'quick_infill_support_settings'):
        context.scene.quick_infill_support_settings.replace_original = self.replace_original


def sync_shared_process_islands(self, context):
    """Mirror the shared islands toggle to tool settings."""
    if hasattr(context.scene, 'quick_infill_tools_settings'):
        context.scene.quick_infill_tools_settings.process_islands = self.process_islands
    if hasattr(context.scene, 'quick_infill_support_settings'):
        context.scene.quick_infill_support_settings.process_islands = self.process_islands


def sync_shared_smart_resolution(self, context):
    """Mirror the shared Smart resolution toggle to tool settings."""
    if hasattr(context.scene, 'quick_infill_tools_settings'):
        context.scene.quick_infill_tools_settings.smart_resolution = self.smart_resolution
    if hasattr(context.scene, 'quick_infill_support_settings'):
        context.scene.quick_infill_support_settings.smart_resolution = self.smart_resolution


def sync_shared_apply_modifiers(self, context):
    """Mirror the shared apply-modifiers-on-export toggle to tool settings."""
    if hasattr(context.scene, 'quick_infill_tools_settings'):
        context.scene.quick_infill_tools_settings.apply_modifiers_on_export = self.apply_modifiers_on_export
    if hasattr(context.scene, 'quick_infill_support_settings'):
        context.scene.quick_infill_support_settings.apply_modifiers_on_export = self.apply_modifiers_on_export


class QuickInfillSettings(PropertyGroup):
    active_offset_preset: EnumProperty(
        name="Active Offset Preset",
        description="Current Preset selection for Stone / Detail / Fine / Mini",
        items=[
            ("NONE", "None", "No preset selected"),
            ("STONE", "Stone", "Stone preset selected"),
            ("DETAIL", "Detail", "Detail preset selected"),
            ("FINE", "Fine", "Fine preset selected"),
            ("MINI", "Mini", "Mini preset selected"),
        ],
        default="STONE",
    )

    shared_voxel_size: FloatProperty(
        name="Voxel Size",
        description="Shared voxel size for offset and support tools",
        default=0.075,
        min=0.025,
        max=0.4,
        precision=3,
        update=sync_shared_voxel_size,
    )

    replace_original: BoolProperty(
        name="Replace Original",
        description="Replace the original object with the result instead of creating a new object",
        default=True,
        update=sync_shared_replace_original,
    )

    process_islands: BoolProperty(
        name="Islands",
        description="Process each mesh island individually, then join the result back into one object",
        default=False,
        update=sync_shared_process_islands,
    )

    smart_resolution: BoolProperty(
        name="Smart",
        description="Use each island/object's existing resolution tag (if any) for its voxel size and decimation ratio, falling back to the current settings for untagged geometry",
        default=False,
        update=sync_shared_smart_resolution,
    )

    apply_modifiers_on_export: BoolProperty(
        name="Apply Modifiers",
        description="Apply each object's modifiers (viewport result) when exporting meshes for processing. Off = ignore modifiers",
        default=False,
        update=sync_shared_apply_modifiers,
    )

    decimate_mode: EnumProperty(
        name="Decimate Mode",
        description="Choose when automatic decimation is active",
        items=[
            ("VOXEL_RATIO", "Ratio", "Decimate based on the current ratio value"),
            ("ORIGINAL", "Original", "Decimate back to the original polycount"),
            ("OFF", "Off", "Disable auto-decimation"),
        ],
        default="VOXEL_RATIO",
        update=sync_shared_decimate_mode,
    )

    decimation_ratio: FloatProperty(
        name="Decimation Ratio",
        description="Fraction of final faces to keep during auto-decimation",
        default=0.015,
        min=0.0,
        max=1.0,
        precision=4,
        update=sync_shared_decimation_ratio,
    )

    resolution: FloatProperty(  # type: ignore
        name="Resolution",
        description="Voxel size",
        default=0.1,
        min=0.05,
        max=1.0,
        precision=3,
        update=reset_preset,
    )
    grow: FloatProperty(  
        name="Grow",
        description="+grow / -shrink distance",
        default=1.0,
        min=0.1,
        max=2.0,
        precision=3,
        update=reset_preset,
    )
    shrink_mult: FloatProperty( 
        name="Shrink Multiplier",
        description="Multiplier for shrink distance",
        default=1.2,
        min=0.1,
        max=3.0,
        step=0.1,
        subtype='FACTOR',
        update=reset_preset,
    )
    target_res: FloatProperty(  # type: ignore
        name="Working Resolution (M)",
        description="Working resolution in millions - drives voxel count and decimation limit (~1M vertices)",
        default=3.0,
        min=0.2,
        max=3.0,
        update=reset_preset,
    )
    voxel_mode: EnumProperty( 
        name="Voxel Mode",
        description="Choose how voxel size is set",
        items=[
            ("TARGET_VOXELS", "Target Voxels", "Derive voxel size from target voxel count and clamp by Resolution"),
            ("RESOLUTION", "Resolution", "Use the Resolution value directly"),
        ],
        default="TARGET_VOXELS",
        update=reset_preset,
    )
    show_voxel_tools_menu: BoolProperty(
        name="Show Voxel Tools",
        description="Expand/collapse the Voxel Tools menu",
        default=True,
    )
    show_infill_tools_menu: BoolProperty(
        name="Show Infill Tools",
        description="Expand/collapse the Infill Tools menu",
        default=True,
    )
    # Preset tracking
    active_preset: EnumProperty(
        name="Active Preset",
        description="Currently active preset",
        items=[
            ("NONE", "None", "No preset active"),
            ("BUILDING_FAST", "Building Fast", "Fast settings for buildings"),
            ("BUILDING_ACCURATE", "Building Accurate", "Accurate settings for buildings"),
            ("MINI_LARGE_HOLES", "Mini Large Holes", "Large holes settings for miniatures"),
            ("MINI_ACCURATE", "Mini Accurate", "Accurate settings for miniatures"),
        ],
        default="BUILDING_ACCURATE",
    )
    method: EnumProperty(
        name="Method",
        description="Processing method for cavity healing",
        items=[
            ("ACCURATE", "Accurate", "High-quality accurate method"),
            ("NAIVE", "Naive", "Fast naive method"),
        ],
        default="ACCURATE",
        update=reset_preset,
    )
    trim_thin: BoolProperty(
        name="Trim Thin",
        description="Remove thin elements from the result",
        default=True,
        update=reset_preset,
    )


class QUICKINFILL_OT_voxel_preset(Operator):
    bl_idname = "quick_infill.voxel_preset"
    bl_label = "Voxel Preset"
    bl_description = "Set voxel size for Offset Tools and Support Tools"
    bl_options = {'REGISTER', 'UNDO'}

    size: FloatProperty(name="Size", default=0.1)  # type: ignore

    def execute(self, context):
        context.scene.quick_infill_settings.shared_voxel_size = self.size
        return {'FINISHED'}


class QUICKINFILL_OT_select_by_resolution_tag(Operator):
    bl_idname = "quick_infill.select_by_resolution_tag"
    bl_label = "Select Same Resolution"
    bl_description = "Select currently visible objects (or mesh islands in Edit Mode) tagged with the current voxel size preset"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from .offset_utils import RESOLUTION_TAG_ORDER, nearest_resolution_tag
        from .blender_meshlib_utils import read_object_resolution_tag, RES_TAG_ATTR

        settings = context.scene.quick_infill_settings
        target_tag = nearest_resolution_tag(settings.shared_voxel_size)
        target_index = RESOLUTION_TAG_ORDER.index(target_tag)

        if context.mode == 'EDIT_MESH':
            import bmesh
            edit_objs = [obj for obj in context.objects_in_mode if obj.type == 'MESH']
            if not edit_objs:
                self.report({'ERROR'}, "No mesh in edit mode.")
                return {'CANCELLED'}
            matched = 0
            untagged_objs = []
            for obj in edit_objs:
                me = obj.data
                bm = bmesh.from_edit_mesh(me)
                layer = bm.faces.layers.int.get(RES_TAG_ATTR)
                if layer is None:
                    untagged_objs.append(obj.name)
                    continue
                for f in bm.faces:
                    is_match = f[layer] == target_index
                    f.select = is_match
                    if is_match:
                        matched += 1
                bm.select_flush_mode()
                bmesh.update_edit_mesh(me)
            if matched == 0:
                self.report({'INFO'}, f"No faces tagged '{target_tag}'.")
            elif untagged_objs:
                names_str = ", ".join(f"'{n}'" for n in untagged_objs)
                self.report({'INFO'}, f"Selected {matched} face(s) tagged '{target_tag}'. No tags on: {names_str}")
            else:
                self.report({'INFO'}, f"Selected {matched} face(s) tagged '{target_tag}'.")
            return {'FINISHED'}

        # Object mode: only touch currently visible objects; hidden objects
        # are left untouched entirely.
        matched = 0
        last_match = None
        for obj in bpy.data.objects:
            if obj.type != 'MESH' or not obj.visible_get():
                continue
            if read_object_resolution_tag(obj) == target_tag:
                obj.select_set(True)
                matched += 1
                last_match = obj
            else:
                obj.select_set(False)

        if last_match is not None:
            context.view_layer.objects.active = last_match

        if matched == 0:
            self.report({'INFO'}, f"No visible objects tagged '{target_tag}'.")
        else:
            self.report({'INFO'}, f"Selected {matched} object(s) tagged '{target_tag}'.")
        return {'FINISHED'}


class QUICKINFILL_OT_pick_resolution_tag(Operator):
    bl_idname = "quick_infill.pick_resolution_tag"
    bl_label = "Pick Resolution"
    bl_description = "Set the voxel size and ratio preset from the active object's tag (or active face's tag in Edit Mode)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from .offset_utils import RESOLUTION_TAG_ORDER
        from .blender_meshlib_utils import read_object_resolution_tag, RES_TAG_ATTR

        tag = None

        if context.mode == 'EDIT_MESH':
            import bmesh
            obj = context.edit_object
            if obj is not None and obj.type == 'MESH':
                bm = bmesh.from_edit_mesh(obj.data)
                layer = bm.faces.layers.int.get(RES_TAG_ATTR)
                if layer is not None:
                    active_face = bm.faces.active
                    if active_face is None:
                        selected = [f for f in bm.faces if f.select]
                        active_face = selected[0] if selected else None
                    if active_face is not None:
                        idx = active_face[layer]
                        if 0 <= idx < len(RESOLUTION_TAG_ORDER):
                            tag = RESOLUTION_TAG_ORDER[idx]
        else:
            tag = read_object_resolution_tag(context.active_object)

        if tag is None:
            self.report({'WARNING'}, "No resolution tag found to pick.")
            return {'CANCELLED'}

        voxel_size, ratio = RESOLUTION_PRESETS[tag]
        settings = context.scene.quick_infill_settings
        settings.shared_voxel_size = voxel_size
        settings.decimation_ratio = ratio
        settings.active_offset_preset = tag
        self.report({'INFO'}, f"Set preset to '{tag}'.")
        return {'FINISHED'}


class QUICKINFILL_OT_offset_preset(Operator):
    bl_idname = "quick_infill.offset_preset"
    bl_label = "Offset Preset"
    bl_description = "Set offset distance, voxel size, trim edge x factor and decimation ratio"
    bl_options = {'REGISTER', 'UNDO'}

    voxel_size: FloatProperty(name="Voxel Size", default=0.05)
    distance: FloatProperty(name="Distance", default=0.7)
    x: FloatProperty(name="X", default=0.25)
    ratio: FloatProperty(name="Ratio", default=0.015)

    preset_name: EnumProperty(
        name="Preset Name",
        items=[
            ("STONE", "Stone", "Stone preset"),
            ("DETAIL", "Detail", "Detail preset"),
            ("FINE", "Fine", "Fine preset"),
            ("MINI", "Mini", "Mini preset"),
        ],
        default="STONE",
    )

    def execute(self, context):
        settings = context.scene.quick_infill_settings
        tools = context.scene.quick_infill_tools_settings
        tools.voxel_size = self.voxel_size
        tools.distance = self.distance
        tools.trim_edges_x = self.x
        settings.shared_voxel_size = self.voxel_size
        settings.decimation_ratio = self.ratio
        context.scene.quick_infill_support_settings.voxel_size = self.voxel_size
        context.scene.quick_infill_support_settings.decimation_ratio = self.ratio
        settings.active_offset_preset = self.preset_name
        return {'FINISHED'}


class QUICKINFILL_OT_test_cuda(Operator):
    bl_idname = "quick_infill.test_cuda"
    bl_label = "Test CUDA"
    bl_description = "Print CUDA availability to the Console"

    def execute(self, context):
        try:
            # Import here to avoid import-time errors if environment isn't ready
            from meshlib import mrcudapy as mc
            print(f"Cuda is Available  ={mc.isCudaAvailable ()}")
            # Also show a brief status message in Blender's UI
            self.report({'INFO'}, f"CUDA available: {mc.isCudaAvailable()}")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Quick Infill Test failed: {e}")
            print(f"[Quick Infill] Test error: {e}")
            return {'CANCELLED'}


class QUICKINFILL_OT_preset_building_fast(Operator):
    bl_idname = "quick_infill.preset_building_fast"
    bl_label = "Fast"
    bl_description = "Fast settings for buildings"

    def execute(self, context):
        settings = context.scene.quick_infill_settings
        settings.voxel_mode = "TARGET_VOXELS"
        settings.target_res = 0.5
        settings.resolution = 0.1
        settings.grow = 0.75
        settings.shrink_mult = 2.0
        settings.method = "NAIVE"
        settings.trim_thin = True
        settings.active_preset = "BUILDING_FAST"
        return {'FINISHED'}


class QUICKINFILL_OT_preset_building_accurate(Operator):
    bl_idname = "quick_infill.preset_building_accurate"
    bl_label = "Accurate"
    bl_description = "Accurate settings for buildings"

    def execute(self, context):
        settings = context.scene.quick_infill_settings
        settings.voxel_mode = "TARGET_VOXELS"
        settings.target_res = 3.0
        settings.resolution = 0.1
        settings.grow = 1.0
        settings.shrink_mult = 1.2
        settings.method = "ACCURATE"
        settings.trim_thin = True
        settings.active_preset = "BUILDING_ACCURATE"
        return {'FINISHED'}


class QUICKINFILL_OT_preset_mini_large_holes(Operator):
    bl_idname = "quick_infill.preset_mini_large_holes"
    bl_label = "Large Holes"
    bl_description = "Large holes settings for miniatures"

    def execute(self, context):
        settings = context.scene.quick_infill_settings
        settings.voxel_mode = "RESOLUTION"
        settings.target_res = 1.0
        settings.resolution = 0.1
        settings.grow = 1.0
        settings.shrink_mult = 1.5
        settings.method = "ACCURATE"
        settings.trim_thin = True
        settings.active_preset = "MINI_LARGE_HOLES"
        return {'FINISHED'}


class QUICKINFILL_OT_preset_mini_accurate(Operator):
    bl_idname = "quick_infill.preset_mini_accurate"
    bl_label = "Accurate"
    bl_description = "Accurate settings for miniatures"

    def execute(self, context):
        settings = context.scene.quick_infill_settings
        settings.voxel_mode = "RESOLUTION"
        settings.target_res = 2.0
        settings.resolution = 0.075
        settings.grow = 0.35
        settings.shrink_mult = 1.2
        settings.method = "ACCURATE"
        settings.trim_thin = True
        settings.active_preset = "MINI_ACCURATE"
        return {'FINISHED'}


class QUICKINFILL_PT_sidebar(Panel):
    bl_label = "Quick Infill"
    bl_idname = "QUICKINFILL_PT_sidebar"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Quick Infill"  # This is the tab name in the side panel

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)

        settings = getattr(context.scene, 'quick_infill_settings', None)

        def prop_with_suffix(layout, data, attr, label="", suffix="mm"):
            split = layout.split(factor=0.9, align=True)
            c = split.column(align=True)
            c.use_property_split = False
            c.use_property_decorate = False
            c.prop(data, attr, text=label)
            split.label(text=suffix)

        # =================================================================
        # Voxel Tools (collapsible)
        # =================================================================
        voxel_box = col.box()
        voxel_header = voxel_box.row()
        show_voxel_tools = getattr(settings, 'show_voxel_tools_menu', True)
        icon = 'DOWNARROW_HLT' if show_voxel_tools else 'RIGHTARROW'
        voxel_header.prop(settings, "show_voxel_tools_menu", text="Voxel Tools", icon=icon, emboss=False)

        if show_voxel_tools:
            voxel_col = voxel_box.column(align=True)

            tools_row = voxel_col.row(align=True)
            tools_split = tools_row.split(factor=0.4, align=True)

            toggle_group = tools_split.row(align=True)
            toggle_group.prop(settings, "replace_original", text="", icon='PASTEDOWN', toggle=True)
            toggle_group.prop(settings, "process_islands", text="", icon='GEOMETRY_SET', toggle=True)
            mod_icon = 'MODIFIER_ON' if settings.apply_modifiers_on_export else 'MODIFIER_OFF'
            toggle_group.prop(settings, "apply_modifiers_on_export", text="", icon=mod_icon, toggle=True)

            bool_group = tools_split.row(align=True)
            bool_group.operator("quick_infill.voxel_intersect", text="", icon='SELECT_INTERSECT')
            bool_group.operator("quick_infill.voxel_union", text="", icon='SELECT_EXTEND')
            bool_group.operator("quick_infill.voxel_diff", text="", icon='SELECT_DIFFERENCE')

            voxel_col.separator(factor=2.0)
            voxel_col.prop(settings, "shared_voxel_size", text="Voxel Size")
            size_row = voxel_col.row(align=True)
            for size in (0.025, 0.05, 0.1, 0.2, 0.3, 0.4):
                size_row.operator("quick_infill.voxel_preset", text=str(size)).size = size

            voxel_col.separator(factor=1.0)
            voxel_col.label(text="Auto Decimation")
            decimate_row = voxel_col.row(align=True)
            decimate_row.prop(settings, "decimate_mode", expand=True)
            voxel_col.prop(settings, "decimation_ratio", text="Ratio")

            voxel_col.separator(factor=1.0)
            voxel_col.label(text="Presets:")
            preset_row = voxel_col.row(align=True)
            active_offset_preset = get_active_offset_preset(settings)
            stone = preset_row.operator("quick_infill.offset_preset", text="Stone", depress=(active_offset_preset == 'STONE'))
            stone.preset_name = 'STONE'
            stone.voxel_size = 0.075
            stone.distance = 0.7
            stone.x = 0.25
            stone.ratio = 0.0175

            detail = preset_row.operator("quick_infill.offset_preset", text="Detail", depress=(active_offset_preset == 'DETAIL'))
            detail.preset_name = 'DETAIL'
            detail.voxel_size = 0.05
            detail.distance = 0.3
            detail.x = 0.4
            detail.ratio = 0.03

            fine = preset_row.operator("quick_infill.offset_preset", text="Fine", depress=(active_offset_preset == 'FINE'))
            fine.preset_name = 'FINE'
            fine.voxel_size = 0.03
            fine.distance = 0.2
            fine.x = 0.25
            fine.ratio = 0.02

            mini = preset_row.operator("quick_infill.offset_preset", text="Mini", depress=(active_offset_preset == 'MINI'))
            mini.preset_name = 'MINI'
            mini.voxel_size = 0.025
            mini.distance = 0.2
            mini.x = 0.25
            mini.ratio = 0.04

            tag_tools_row = voxel_col.row(align=True)
            tag_tools_row.prop(settings, "smart_resolution", text="Smart Res", toggle=True)
            tag_tools_row.operator("quick_infill.select_by_resolution_tag", text="", icon='VIEWZOOM')
            tag_tools_row.operator("quick_infill.pick_resolution_tag", text="", icon='EYEDROPPER')

            # --- Offset (collapsible) ---
            voxel_col.separator(factor=1.0)
            tools_panel.draw_offset_tools(voxel_col, context)

            # --- Support (collapsible) ---
            voxel_col.separator(factor=1.0)
            support_tools.draw_support_tools(voxel_col, context)

        # =================================================================
        # Infill Tools (collapsible)
        # =================================================================
        infill_box = col.box()
        infill_header = infill_box.row()
        show_infill_tools = getattr(settings, 'show_infill_tools_menu', True)
        icon = 'DOWNARROW_HLT' if show_infill_tools else 'RIGHTARROW'
        infill_header.prop(settings, "show_infill_tools_menu", text="Infill Tools", icon=icon, emboss=False)

        if show_infill_tools:
            infill_col = infill_box.column(align=True)

            # --- Create Cavity Infill button ---
            ifRow = infill_col.row(align=True)
            ifRow.operator("quick_infill.heal_cavity", text="Create Cavity Infill")

            infill_col.separator(factor=1.0)

            # --- Presets (static header) ---
            infill_col.label(text="Presets:")

            presets_col = infill_col.column(align=True)
            active_building_preset = get_active_building_preset(settings)

            building_row = presets_col.row(align=True)
            building_row.label(text="Building:")
            building_row.operator("quick_infill.preset_building_fast", text="Fast", depress=(active_building_preset == 'BUILDING_FAST'))
            building_row.operator("quick_infill.preset_building_accurate", text="Accurate", depress=(active_building_preset == 'BUILDING_ACCURATE'))

            mini_row = presets_col.row(align=True)
            mini_row.label(text="Mini:")
            mini_row.operator("quick_infill.preset_mini_large_holes", text="Large Holes", depress=(active_building_preset == 'MINI_LARGE_HOLES'))
            mini_row.operator("quick_infill.preset_mini_accurate", text="Accurate", depress=(active_building_preset == 'MINI_ACCURATE'))

            # --- Settings (static header) ---
            infill_col.separator(factor=1.0)
            infill_col.label(text="Settings:")

            settings_col = infill_col.column(align=True)
            split = settings_col.split(factor=0.4, align=True)
            split.label(text="Voxel Mode")
            split.prop(settings, "voxel_mode", text="")

            method_split = settings_col.split(factor=0.4, align=True)
            method_split.label(text="Method")
            method_split.prop(settings, "method", text="")

            settings_col.separator(factor=1.0)
            prop_with_suffix(settings_col, settings, "target_res", "Working Res", "M")

            mode = getattr(settings, 'voxel_mode', 'TARGET_VOXELS')
            if mode == 'RESOLUTION':
                prop_with_suffix(settings_col, settings, "resolution", "Resolution", "mm")

            prop_with_suffix(settings_col, settings, "grow", "Grow", "mm")
            settings_col.prop(settings, "shrink_mult")
            settings_col.separator(factor=1.0)
            settings_col.prop(settings, "trim_thin")


classes = (
    QuickInfillSettings,
    QUICKINFILL_OT_voxel_preset,
    QUICKINFILL_OT_select_by_resolution_tag,
    QUICKINFILL_OT_pick_resolution_tag,
    QUICKINFILL_OT_offset_preset,
    QUICKINFILL_OT_test_cuda,
    QUICKINFILL_OT_preset_building_fast,
    QUICKINFILL_OT_preset_building_accurate,
    QUICKINFILL_OT_preset_mini_large_holes,
    QUICKINFILL_OT_preset_mini_accurate,
    QUICKINFILL_PT_sidebar,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.quick_infill_settings = PointerProperty(type=QuickInfillSettings)
    tools_panel.register()
    support_tools.register()


def unregister():
    support_tools.unregister()
    tools_panel.unregister()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.quick_infill_settings
