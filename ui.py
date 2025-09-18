from sqlite3 import Row
import bpy
from bpy.types import Panel, Operator, PropertyGroup
from bpy.props import FloatProperty, IntProperty, PointerProperty, EnumProperty


class QuickInfillSettings(PropertyGroup):
    resolution: FloatProperty(  # type: ignore
        name="Resolution",
        description="Voxel size",
        default=0.1,
        min=0.05,
        max=1.0,
        precision=3,
    )
    grow: FloatProperty(  
        name="Grow",
        description="+grow / -shrink distance",
        default=2.0,
        min=0.1,
        max=3.0,
        precision=3,
    )
    shrink_mult: FloatProperty( 
        name="Shrink Multiplier",
        description="Multiplier for shrink distance",
        default=1.0,
        min=0.1,
        max=2.0,
        step=0.1,
        subtype='FACTOR'
    )
    target_voxels: FloatProperty(  # type: ignore
        name="Voxels (M)",
        description="Target voxel count in millions",
        default=2.0,
        min=0.2,
        max=2.0,
    )
    voxel_mode: EnumProperty( 
        name="Voxel Mode",
        description="Choose how voxel size is set",
        items=[
            ("RESOLUTION", "Resolution", "Use the Resolution value directly"),
            ("TARGET_VOXELS", "Target Voxels", "Derive voxel size from target voxel count and clamp by Resolution"),
        ],
        default="TARGET_VOXELS",
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


class QUICKINFILL_PT_sidebar(Panel):
    bl_label = "Quick Infill"
    bl_idname = "QUICKINFILL_PT_sidebar"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Quick Infill"  # This is the tab name in the side panel

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
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

        _prop("voxel_mode", text="Voxel Size Mode")
        
        # Nested controls under voxel mode dropdown
        mode = getattr(settings, 'voxel_mode', 'TARGET_VOXELS')
        
        # Create a nested box for the mode-specific controls
        box = col.box()
        if mode == 'RESOLUTION':
            prop_with_suffix(box, settings, "resolution", "Resolution", "mm")
        elif mode == 'TARGET_VOXELS':
            box.prop(settings, "target_voxels", text="Voxels (M)")

        prop_with_suffix(box, settings, "grow", "Grow", "mm")
        _prop("shrink_mult")
        
        # Test CUDA button
        col.operator(QUICKINFILL_OT_test_cuda.bl_idname, text="Test CUDA")


classes = (
    QuickInfillSettings,
    QUICKINFILL_OT_test_cuda,
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
