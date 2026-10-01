import bpy
from bpy.types import PropertyGroup
from bpy.props import FloatProperty

# Import operators from separate module
from . import tools_operators


class QuickInfillToolsSettings(PropertyGroup):
    distance: FloatProperty(
        name="Distance",
        description="Offset distance for grow/shrink operations",
        default=0.7,
        min=0.0,
        max=4.0,
        precision=3,
    )
    
    voxel_size: FloatProperty(
        name="Voxel Size",
        description="Internal compatibility value for the shared voxel size",
        default=0.05,
        min=0.025,
        max=0.4,
        precision=3,
        options={'HIDDEN'},
    )

    trim_edges_x: FloatProperty(
        name="Trim Edges X",
        description="Extra factor for Trim Edges final grow step (1 + x)",
        default=0.25,
        min=0.0,
        max=0.5,
        precision=3,
        subtype='FACTOR',
    )

    trim_edges_density: FloatProperty(
        name="Trim Edges Density",
        description="Target face density before Trim Edges boolean (faces per area unit)",
        default=1.0,
        min=0.5,
        max=4.0,
        precision=4,
    )
    
    decimate_mode: bpy.props.EnumProperty(
        name="Decimate Mode",
        description="Choose when auto-decimation is active",
        items=[
            ("OFF", "Off", "Disable auto-decimation"),
            ("ORIGINAL", "Original", "Decimate back to the original polycount"),
            ("VOXEL_RATIO", "Ratio", "Decimate based on the current ratio value"),
        ],
        default="VOXEL_RATIO",
        options={'HIDDEN'},
    )

    decimation_ratio: bpy.props.FloatProperty(
        name="Decimation Ratio",
        description="Fraction of final faces to keep during auto-decimation",
        default=0.015,
        min=0.0,
        max=1.0,
        precision=4,
        options={'HIDDEN'},
    )
    
    replace_original: bpy.props.BoolProperty(
        name="Replace Original",
        description="Replace the original object with the result instead of creating a new object",
        default=True,
        options={'HIDDEN'},
    )

    process_islands: bpy.props.BoolProperty(
        name="Islands",
        description="Process each mesh island individually, then join the result back into one object",
        default=False,
        options={'HIDDEN'},
    )

    smart_resolution: bpy.props.BoolProperty(
        name="Smart",
        description="Use each island/object's existing resolution tag (if any) for its voxel size and decimation ratio",
        default=False,
        options={'HIDDEN'},
    )

    apply_modifiers_on_export: bpy.props.BoolProperty(
        name="Apply Modifiers",
        description="Apply each object's modifiers (viewport result) when exporting meshes for processing. Off = ignore modifiers",
        default=False,
        options={'HIDDEN'},
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
        
        tools_col.separator()
        
        # Distance slider
        prop_with_suffix(tools_col, settings, "distance", "Distance", "mm")
        
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

        tools_col.separator()

        # Trim Edges controls (last row)
        row = tools_col.row(align=True)
        row.prop(settings, "trim_edges_x", text="X")
        row.operator("quick_infill.trim_edges", text="Trim Edges", icon='MOD_BOOLEAN')

        # Density used for decimation before Trim Edges boolean
        row = tools_col.row(align=True)
        row.prop(settings, "trim_edges_density", text="Trim Density")


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
