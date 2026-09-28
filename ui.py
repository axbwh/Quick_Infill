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
    show_settings: BoolProperty(
        name="Show Settings",
        description="Expand/collapse settings panel",
        default=True,
    )
    show_presets: BoolProperty(
        name="Show Presets",
        description="Expand/collapse presets panel",
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
        
        # Scene Settings
        settings = getattr(context.scene, 'quick_infill_settings', None)
        
        # Create Cavity Infill button at top
        ifRow = col.row(align=True)
        ifRow.operator("quick_infill.heal_cavity", text="Create Cavity Infill")
        
        col.separator(factor=1.0)
        
        # Presets section (collapsible)
        presets_box = col.box()
        presets_header = presets_box.row()
        
        # Collapsible header with arrow icon
        show_presets = getattr(settings, 'show_presets', True)
        icon = 'DOWNARROW_HLT' if show_presets else 'RIGHTARROW'
        presets_header.prop(settings, "show_presets", text="Presets", icon=icon, emboss=False)
        
        # Show presets when expanded
        if show_presets:
            presets_col = presets_box.column(align=True)
            active_preset = get_active_building_preset(settings)

            # Building row
            building_row = presets_col.row(align=True)
            building_row.label(text="Building:")
            fast_op = building_row.operator("quick_infill.preset_building_fast", text="Fast", depress=(active_preset == 'BUILDING_FAST'))
            accurate_op = building_row.operator("quick_infill.preset_building_accurate", text="Accurate", depress=(active_preset == 'BUILDING_ACCURATE'))

            # Mini row
            mini_row = presets_col.row(align=True)
            mini_row.label(text="Mini:")
            holes_op = mini_row.operator("quick_infill.preset_mini_large_holes", text="Large Holes", depress=(active_preset == 'MINI_LARGE_HOLES'))
            mini_acc_op = mini_row.operator("quick_infill.preset_mini_accurate", text="Accurate", depress=(active_preset == 'MINI_ACCURATE'))
        def prop_with_suffix(layout, data, attr, label="", suffix="mm"):
            split = layout.split(factor=0.9, align=True)
            col = split.column(align=True)
            col.use_property_split = False
            col.use_property_decorate = False
            col.prop(data, attr, text=label)
            split.label(text=suffix)
        
        # Scene Settings
        settings = getattr(context.scene, 'quick_infill_settings', None)
        def _prop(name: str, text: str | None = None):
            try:
                if text is None:
                    col.prop(settings, name)
                else:
                    col.prop(settings, name, text=text)
            except Exception:
                col.label(text=f"{name} (unavailable)")

        # Collapsible Settings section
        box = col.box()
        header = box.row()
        
        # Collapsible header with arrow icon
        show_settings = getattr(settings, 'show_settings', True)
        icon = 'DOWNARROW_HLT' if show_settings else 'RIGHTARROW'
        header.prop(settings, "show_settings", text="Settings", icon=icon, emboss=False)
        
        # Show settings when expanded
        if show_settings:
            # Voxel mode selection with custom label/dropdown proportions
            settings_col = box.column(align=True)
            split = settings_col.split(factor=0.4, align=True)
            split.label(text="Voxel Mode")
            split.prop(settings, "voxel_mode", text="")
            
            # Method selection
            method_split = settings_col.split(factor=0.4, align=True)
            method_split.label(text="Method")
            method_split.prop(settings, "method", text="")
            
            # Target Resolution (shown in both modes)
            prop_with_suffix(settings_col, settings, "target_res", "Working Res", "M")
            
            # Show mode-specific settings
            mode = getattr(settings, 'voxel_mode', 'TARGET_VOXELS')
            
            if mode == 'RESOLUTION':
                prop_with_suffix(settings_col, settings, "resolution", "Resolution", "mm")

            prop_with_suffix(settings_col, settings, "grow", "Grow", "mm")
            settings_col.prop(settings, "shrink_mult")
            settings_col.prop(settings, "trim_thin")
        
        # Offset Tools section
        tools_panel.draw_offset_tools(col, context)
        
        # Support Tools section
        support_tools.draw_support_tools(col, context)

        col.label(text="Tool Options")
        voxel_row = col.row(align=True)
        voxel_row.operator("quick_infill.voxel_intersect", text="Voxel Intersect", icon='MOD_BOOLEAN')

        toggle_row = col.row(align=True)
        toggle_row.prop(context.scene.quick_infill_settings, "replace_original", text="Replace", toggle=True)
        toggle_row.prop(context.scene.quick_infill_settings, "process_islands", text="Islands", toggle=True)

        col.separator(factor=0.5)
        col.prop(context.scene.quick_infill_settings, "smart_resolution", text="Smart Res", toggle=True)
        col.prop(context.scene.quick_infill_settings, "shared_voxel_size", text="Voxel Size")
        row = col.row(align=True)
        for size in (0.025, 0.05, 0.1, 0.2, 0.3, 0.4):
            row.operator("quick_infill.voxel_preset", text=str(size)).size = size

        col.separator(factor=0.5)
        auto_decimation_box = col.box()
        auto_decimation_box.label(text="Auto Decimation")
        decimate_row = auto_decimation_box.row(align=True)
        decimate_row.prop(context.scene.quick_infill_settings, "decimate_mode", expand=True)
        auto_decimation_box.prop(context.scene.quick_infill_settings, "decimation_ratio", text="Ratio")

        col.label(text="Presets")
        preset_row = col.row(align=True)
        active_preset = get_active_offset_preset(context.scene.quick_infill_settings)
        stone = preset_row.operator("quick_infill.offset_preset", text="Stone", depress=(active_preset == 'STONE'))
        stone.preset_name = 'STONE'
        stone.voxel_size = 0.075
        stone.distance = 0.7
        stone.x = 0.25
        stone.ratio = 0.0175

        detail = preset_row.operator("quick_infill.offset_preset", text="Detail", depress=(active_preset == 'DETAIL'))
        detail.preset_name = 'DETAIL'
        detail.voxel_size = 0.05
        detail.distance = 0.3
        detail.x = 0.4
        detail.ratio = 0.03

        fine = preset_row.operator("quick_infill.offset_preset", text="Fine", depress=(active_preset == 'FINE'))
        fine.preset_name = 'FINE'
        fine.voxel_size = 0.03
        fine.distance = 0.2
        fine.x = 0.25
        fine.ratio = 0.02

        mini = preset_row.operator("quick_infill.offset_preset", text="Mini", depress=(active_preset == 'MINI'))
        mini.preset_name = 'MINI'
        mini.voxel_size = 0.025
        mini.distance = 0.2
        mini.x = 0.25
        mini.ratio = 0.04


classes = (
    QuickInfillSettings,
    QUICKINFILL_OT_voxel_preset,
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
