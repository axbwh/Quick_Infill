"""Offset Tools section of the Quick Infill sidebar."""


def draw_offset_tools(layout, context):
    s = context.scene.quick_infill_settings
    box = layout.column(align=True).box()
    icon = 'DOWNARROW_HLT' if s.show_tools else 'RIGHTARROW'
    box.row().prop(s, "show_tools", text="Offset Tools", icon=icon, emboss=False)
    if not s.show_tools:
        return

    col = box.column(align=True)
    col.separator()
    split = col.split(factor=0.9, align=True)
    c = split.column(align=True)
    c.use_property_split = False
    c.use_property_decorate = False
    c.prop(s, "distance", text="Distance")
    split.label(text="mm")

    col.separator()
    row = col.row(align=True)
    row.operator("quick_infill.grow", text="Grow", icon='PROP_CON')
    row.operator("quick_infill.shrink", text="Shrink", icon='PROP_OFF')
    col.separator()
    row = col.row(align=True)
    row.operator("quick_infill.remesh", text="Remesh", icon='ALIASED')
    row.operator("quick_infill.trim_thin", text="Trim Thin", icon='MOD_WARP')
    col.separator()
    row = col.row(align=True)
    row.prop(s, "trim_edges_x", text="X")
    row.operator("quick_infill.trim_edges", text="Trim Edges", icon='MOD_BOOLEAN')
    col.row(align=True).prop(s, "trim_edges_density", text="Trim Density")
