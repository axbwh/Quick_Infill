import bpy
from bpy.types import Panel, Operator


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
        col.operator(QUICKINFILL_OT_test_cuda.bl_idname, text="Test")


classes = (
    QUICKINFILL_OT_test_cuda,
    QUICKINFILL_PT_sidebar,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
