import bpy
from bpy.types import PropertyGroup
from bpy.props import FloatProperty

# Import operators from separate module
from . import tools_operators


class QuickInfillToolsSettings(PropertyGroup):
    distance: FloatProperty(
        name="Distance",
        description="Offset distance for grow/shrink operations",
        default=0.3,
        min=0.0,
        max=4.0,
        precision=3,
    )
    
    resolution: FloatProperty(
        name="Resolution",
        description="Voxel resolution for remesh operations",
        default=0.2,
        min=0.05,
        max=0.4,
        precision=3,
    )
    
    auto_decimate: bpy.props.BoolProperty(
        name="Auto Decimate",
        description="Automatically decimate after operations",
        default=True,
    )
    
    replace_original: bpy.props.BoolProperty(
        name="Replace Original",
        description="Replace the original object with the result instead of creating a new object",
        default=False,
    )
    
    show_tools: bpy.props.BoolProperty(
        name="Show Tools",
        description="Expand/collapse offset tools panel",
        default=False,
    )


def draw_offset_tools(layout, context):
    """Draw the Offset Tools UI into the given layout.
    
    Args:
        layout: The parent Blender UI layout to draw into
        context: The Blender context
    """
    settings = context.scene.quick_infill_tools_settings
    col = layout.column(align=True)
    
    # Collapsible box similar to Settings panel
    box = col.box()
    header = box.row()
    
    # Collapsible header with arrow icon
    show_tools = getattr(settings, 'show_tools', False)
    icon = 'DOWNARROW_HLT' if show_tools else 'RIGHTARROW'
    header.prop(settings, "show_tools", text="Offset Tools", icon=icon, emboss=False)
    
    # Show tools when expanded
    if show_tools:
        tools_col = box.column(align=True)
        
        def prop_with_suffix(parent_layout, data, attr, label="", suffix="mm"):
            split = parent_layout.split(factor=0.9, align=True)
            c = split.column(align=True)
            c.use_property_split = False
            c.use_property_decorate = False
            c.prop(data, attr, text=label)
            split.label(text=suffix)
        
        # Auto Decimate and Replace Original checkboxes on same row
        row = tools_col.row(align=True)
        row.prop(settings, "auto_decimate", text="Auto Decimate")
        row.prop(settings, "replace_original", text="Replace Original")
        
        tools_col.separator()
        
        # Distance slider
        prop_with_suffix(tools_col, settings, "distance", "Distance", "mm")
        
        # Resolution slider
        prop_with_suffix(tools_col, settings, "resolution", "Resolution", "mm")
        
        tools_col.separator()
        
        # Buttons row
        row = tools_col.row(align=True)
        row.operator("quick_infill.grow", text="Grow", icon='PROP_CON')
        row.operator("quick_infill.shrink", text="Shrink", icon='PROP_OFF')
        
        tools_col.separator()
        
        # Remesh and Trim Thin buttons
        row = tools_col.row(align=True)
        row.operator("quick_infill.remesh", text="Remesh", icon='ALIASED')
        row.operator("quick_infill.trim_thin", text="Trim Thin", icon='MOD_WARP')


classes = (
    QuickInfillToolsSettings,
)


def register():
    # Register operators first
    tools_operators.register()
    
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.quick_infill_tools_settings = bpy.props.PointerProperty(
        type=QuickInfillToolsSettings
    )


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    del bpy.types.Scene.quick_infill_tools_settings
    
    # Unregister operators last
    tools_operators.unregister()
