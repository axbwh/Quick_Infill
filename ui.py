from sqlite3 import Row
import bpy
from bpy.types import Panel, Operator, PropertyGroup
from bpy.props import FloatProperty, IntProperty, PointerProperty, EnumProperty, BoolProperty


def reset_preset(self, context):
    """Reset active preset when properties are manually changed"""
    if hasattr(self, 'active_preset'):
        self.active_preset = 'NONE'


class QuickInfillSettings(PropertyGroup):
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
        default=2.0,
        min=0.1,
        max=2.0,
        precision=3,
        update=reset_preset,
    )
    shrink_mult: FloatProperty( 
        name="Shrink Multiplier",
        description="Multiplier for shrink distance",
        default=1.0,
        min=0.1,
        max=3.0,
        step=0.1,
        subtype='FACTOR',
        update=reset_preset,
    )
    target_res: FloatProperty(  # type: ignore
        name="Working Resolution (M)",
        description="Working resolution in millions - drives voxel count and decimation limit (~1M vertices)",
        default=1.0,
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
        default="NONE",
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
        settings.target_res = 1.0
        settings.resolution = 0.1
        settings.grow = 2.0
        settings.shrink_mult = 1.0
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
        
        # Presets section
        presets_box = col.box()
        presets_col = presets_box.column(align=True)
        presets_col.label(text="Presets")
        
        # Building row
        building_row = presets_col.row(align=True)
        building_row.label(text="Building:")
        
        # Fast button - highlight if active
        active_preset = getattr(settings, 'active_preset', 'NONE')
        fast_op = building_row.operator("quick_infill.preset_building_fast", text="Fast", depress=(active_preset == 'BUILDING_FAST'))
        
        # Accurate button - highlight if active  
        accurate_op = building_row.operator("quick_infill.preset_building_accurate", text="Accurate", depress=(active_preset == 'BUILDING_ACCURATE'))
        
        # Mini row
        mini_row = presets_col.row(align=True)
        mini_row.label(text="Mini:")
        
        # Large Holes button - highlight if active
        holes_op = mini_row.operator("quick_infill.preset_mini_large_holes", text="Large Holes", depress=(active_preset == 'MINI_LARGE_HOLES'))
        
        # Mini Accurate button - highlight if active
        mini_acc_op = mini_row.operator("quick_infill.preset_mini_accurate", text="Accurate", depress=(active_preset == 'MINI_ACCURATE'))
        
        col.separator(factor=1.0)
        # Heal Cavity Operator
        ifRow = col.row(align=True)
        ifRow.operator("quick_infill.heal_cavity", text="Create Cavity Infill")

        col.separator(factor=1.0)
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
        
        # Test CUDA button
        col.operator(QUICKINFILL_OT_test_cuda.bl_idname, text="Test CUDA")


classes = (
    QuickInfillSettings,
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


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.quick_infill_settings
