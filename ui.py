import bpy
from bpy.types import Panel, Operator, PropertyGroup
from bpy.props import FloatProperty, IntProperty, PointerProperty, EnumProperty


class QuickInfillSettings(PropertyGroup):
    resolution: FloatProperty(  # type: ignore
        name="Resolution",
        description="Voxel size",
        default=0.1,
        min=0.01,
        max=1.0
    )
    grow: FloatProperty(  
        name="Grow",
        description="+grow / -shrink distance",
        default=2.0,
        min=0.1,
        max=10.0
    )
    shrink_mult: FloatProperty( 
        name="Shrink Multiplier",
        description="Multiplier for shrink distance",
        default=1.5,
        min=0.1,
        max=5.0
    )
    target_voxels: FloatProperty(  # type: ignore
        name="Target Voxels (M)",
        description="Target voxel count in millions",
        default=2.0,
        min=0.2,
        max=2.0,
        step=1,
        precision=2
    )
    voxel_mode: EnumProperty( 
        name="Voxel Mode",
        description="Choose how voxel size is set",
        items=[
            ("RESOLUTION", "Use Resolution", "Use the Resolution value directly"),
            ("TARGET_VOXELS", "Use Target Voxels", "Derive voxel size from target voxel count and clamp by Resolution"),
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
        
        # Scene Settings
        settings = getattr(context.scene, 'quick_infill_settings', None)
        if not settings:
            col.label(text="Quick Infill settings not available.")
            col.label(text="Reload the add-on to initialize settings.")
        else:
            def _prop(name: str, text: str | None = None):
                try:
                    if text is None:
                        col.prop(settings, name)
                    else:
                        col.prop(settings, name, text=text)
                except Exception:
                    col.label(text=f"{name} (unavailable)")

            _prop("resolution")
            _prop("voxel_mode", text="Voxel Size Mode")
            mode = getattr(settings, 'voxel_mode', 'TARGET_VOXELS')
            if mode == 'TARGET_VOXELS':
                _prop("target_voxels", text="Target Voxels (M)")
            _prop("grow")
            _prop("shrink_mult")

        # Heal Cavity Operator
        col.operator("quick_infill.heal_cavity", text="Heal Cavity")
        
        # Separator
        col.separator()
        
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
