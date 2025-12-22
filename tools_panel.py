import bpy
from bpy.types import Panel, PropertyGroup
from bpy.props import FloatProperty

# Import operators from separate module
from . import tools_operators


class QuickInfillToolsSettings(PropertyGroup):
    distance: FloatProperty(
        name="Distance",
        description="Offset distance for grow/shrink operations",
        default=0.3,
        min=0.0,
        max=0.3,
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
        default=False,
    )
    
    replace_original: bpy.props.BoolProperty(
        name="Replace Original",
        description="Replace the original object with the result instead of creating a new object",
        default=False,
    )


class QUICKINFILL_PT_tools(Panel):
    bl_label = "Advanced Tools"
    bl_idname = "QUICKINFILL_PT_tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Quick Infill"
    bl_parent_id = "QUICKINFILL_PT_sidebar"  # Makes this a subpanel
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.quick_infill_tools_settings
        col = layout.column(align=True)
        
        def prop_with_suffix(layout, data, attr, label="", suffix="mm"):
            split = layout.split(factor=0.9, align=True)
            col = split.column(align=True)
            col.use_property_split = False
            col.use_property_decorate = False
            col.prop(data, attr, text=label)
            split.label(text=suffix)
        
        # Auto Decimate checkbox at the top
        col.prop(settings, "auto_decimate", text="Auto Decimate")
        col.prop(settings, "replace_original", text="Replace Original")
        
        col.separator()
        
        # Distance slider
        prop_with_suffix(col, settings, "distance", "Distance", "mm")
        
        # Resolution slider
        prop_with_suffix(col, settings, "resolution", "Resolution", "mm")
        
        col.separator()
        
        # Buttons row
        row = col.row(align=True)
        row.operator("quick_infill.grow", text="Grow", icon='PROP_CON')
        row.operator("quick_infill.shrink", text="Shrink", icon='PROP_OFF')
        
        col.separator()
        
        # Remesh and Trim Thin buttons
        row = col.row(align=True)
        row.operator("quick_infill.remesh", text="Remesh", icon='ALIASED')
        row.operator("quick_infill.trim_thin", text="Trim Thin", icon='X')


classes = (
    QuickInfillToolsSettings,
    QUICKINFILL_PT_tools,
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
