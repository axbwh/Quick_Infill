"""
All Quick Infill settings live in ONE property group, scene.quick_infill_settings.

Before this, the Offset, Support and main panels each had their own copy of
voxel size, decimation, Replace, Islands, Smart and Apply Modifiers, kept in
step by update callbacks, and their defaults disagreed (the panel showed Stone
while the tools ran at 0.05). One group removes the copies and the callbacks.

The Cavity Infill tool deliberately keeps its OWN resolution, grow and shrink
multiplier (resolution / grow / shrink_mult / target_res / voxel_mode); those
are not shared with the voxel tools.
"""

import bpy
from bpy.types import PropertyGroup
from bpy.props import FloatProperty, BoolProperty, EnumProperty, PointerProperty, StringProperty

from .offset_utils import OFFSET_PRESETS, INFILL_PRESETS, ANGLE_PRESETS


# ---------------------------------------------------------------------------
# Preset matching (computed when drawn, so the highlighted button always
# reflects the real values)
# ---------------------------------------------------------------------------

def _close(a, b, eps=1e-5):
    return abs(float(a) - float(b)) < eps


def get_active_offset_preset(s):
    for key, p in OFFSET_PRESETS.items():
        if _close(s.shared_voxel_size, p["voxel"]) and _close(s.decimation_ratio, p["ratio"]):
            return key
    return 'NONE'


def get_active_building_preset(s):
    for key, p in INFILL_PRESETS.items():
        if s.voxel_mode != p["voxel_mode"] or s.method != p["method"] or bool(s.trim_thin) != p["trim_thin"]:
            continue
        if not (_close(s.resolution, p["resolution"]) and _close(s.grow, p["grow"]) and _close(s.shrink_mult, p["shrink_mult"])):
            continue
        # target_res only matters (and is only shown) in Target Voxels mode
        if p["voxel_mode"] == 'TARGET_VOXELS' and not _close(s.target_res, p["target_res"]):
            continue
        return key
    return 'NONE'


def get_active_angle_preset(s):
    for a in ANGLE_PRESETS:
        if _close(s.undercut_angle, a):
            return str(int(a))
    return 'NONE'


class QuickInfillSettings(PropertyGroup):
    # ----- shared by every voxel tool (Offset, Support, booleans) -----
    shared_voxel_size: FloatProperty(
        name="Voxel Size", description="Voxel size used by the offset, support and boolean tools",
        default=OFFSET_PRESETS['STONE']["voxel"], min=0.025, max=0.4, precision=3,
    )
    decimation_ratio: FloatProperty(
        name="Decimation Ratio", description="Fraction of final faces to keep during auto-decimation",
        default=OFFSET_PRESETS['STONE']["ratio"], min=0.001, max=1.0, precision=4,
    )
    decimate_mode: EnumProperty(
        name="Decimate Mode", description="Choose when automatic decimation is active",
        items=[
            ("VOXEL_RATIO", "Ratio", "Decimate based on the current ratio value"),
            ("ORIGINAL", "Original", "Decimate back to the original polycount"),
            ("OFF", "Off", "Disable auto-decimation"),
        ],
        default="VOXEL_RATIO",
    )
    replace_original: BoolProperty(
        name="Replace Original",
        description="Replace the original object with the result instead of creating a new object",
        default=True,
    )
    process_islands: BoolProperty(
        name="Islands",
        description="Process each mesh island individually, then join the result back into one object",
        default=False,
    )
    smart_resolution: BoolProperty(
        name="Smart",
        description="Use each island/object's existing resolution tag (if any) for its voxel size and decimation ratio, "
                    "falling back to the current settings for untagged geometry",
        default=False,
    )
    apply_modifiers_on_export: BoolProperty(
        name="Apply Modifiers",
        description="Use each object's modifier result (viewport) as the input. Off = ignore modifiers",
        default=False,
    )
    ask_above_seconds: FloatProperty(
        name="Ask Above",
        description="Before starting a job estimated to take longer than this, ask first (0 = never ask). "
                    "Running jobs can always be stopped with Esc",
        default=30.0, min=0.0, soft_max=600.0, step=500, precision=0,
    )
    # Kept for files saved by older versions; the highlighted preset is computed when drawn.
    active_offset_preset: EnumProperty(
        name="Active Offset Preset",
        items=[("NONE", "None", ""), ("STONE", "Stone", ""), ("DETAIL", "Detail", ""), ("FINE", "Fine", ""), ("MINI", "Mini", "")],
        default="STONE", options={'HIDDEN'},
    )

    # ----- Offset tools -----
    distance: FloatProperty(
        name="Distance", description="Offset distance for grow/shrink operations",
        default=OFFSET_PRESETS['STONE']["distance"], min=0.0, max=4.0, precision=3,
    )
    trim_edges_x: FloatProperty(
        name="Trim Edges X", description="Extra factor for Trim Edges final grow step (1 + x)",
        default=OFFSET_PRESETS['STONE']["x"], min=0.0, max=0.5, precision=3, subtype='FACTOR',
    )
    trim_edges_density: FloatProperty(
        name="Trim Edges Density", description="Target face density before Trim Edges boolean (faces per area unit)",
        default=1.0, min=0.5, max=4.0, precision=4,
    )

    # ----- Support tools -----
    undercut_angle: FloatProperty(
        name="Angle",
        description="Undercut angle in degrees. 0° = vertical (Z up), +/-90° = horizontal (selected axis). "
                    "Negative angles use voxel union when combining directions",
        default=60.0, min=-90.0, max=90.0, precision=1,
    )
    dir_x_pos: BoolProperty(name="+X", description="Include +X direction", default=True)
    dir_x_neg: BoolProperty(name="-X", description="Include -X direction", default=True)
    dir_y_pos: BoolProperty(name="+Y", description="Include +Y direction", default=True)
    dir_y_neg: BoolProperty(name="-Y", description="Include -Y direction", default=True)
    dir_xpos_ypos: BoolProperty(name="+X+Y", description="Include +X+Y diagonal direction", default=True)
    dir_xpos_yneg: BoolProperty(name="+X-Y", description="Include +X-Y diagonal direction", default=True)
    dir_xneg_ypos: BoolProperty(name="-X+Y", description="Include -X+Y diagonal direction", default=True)
    dir_xneg_yneg: BoolProperty(name="-X-Y", description="Include -X-Y diagonal direction", default=True)
    dir_z: BoolProperty(name="Z", description="Include straight-on direction (no tilt, like 0° angle)", default=False)
    auto_shrink: BoolProperty(
        name="Auto Shrink", description="Shrink the result to compensate for material added by undercut fixing",
        default=False,
    )
    shrink_amount: FloatProperty(
        name="Shrink Amount", description="Amount to shrink the mesh after undercut fixing",
        default=0.1, min=0.01, max=1.0, precision=2,
    )
    shrink_angle_threshold: FloatProperty(
        name="Shrink Angle", description="Only shrink faces within this angle (degrees) of pointing straight up",
        default=70.0, min=5.0, max=90.0, precision=1,
    )

    # ----- Remove Scoop (support tools) -----
    scoop_method: EnumProperty(
        name="Method",
        items=(('FILL', "Fill", "Delete the faces no 35-degree side view can see (the scoop) and fill the hole smoothly, "
                                "like repairing a 3D-scan defect (no flat cut)"),
               ('VIEW', "View", "Cut flat where the scoop ends, seen looking down at each side of the stone at the View Angle"),
               ('CONTACT', "Contact", "Cut where a flat cut covers the Contact share of the stone's footprint")),
        default='FILL',
    )
    scoop_angle: FloatProperty(
        name="View Angle", description="Angle the four side views look at the stone from, like the support angle: "
                                        "90 = straight at the side, 0 = from the top. The scoop hides under the face",
        default=60.0, min=20.0, max=80.0, precision=0, subtype='NONE',
    )
    scoop_share: FloatProperty(
        name="Bottom Line", description="Cut at this share of the bottom points the four views see (50% = median, higher cuts more)",
        default=50.0, min=0.0, max=100.0, precision=0, subtype='PERCENTAGE',
    )
    scoop_contact: FloatProperty(
        name="Contact", description="Cut the scoop where the flat cut covers this share of the stone's footprint "
                                    "(same contact-area rule as the stone packer's flat cuts)",
        default=90.0, min=50.0, max=100.0, precision=0, subtype='PERCENTAGE',
    )
    scoop_offset: FloatProperty(
        name="Scoop Offset", description="Move the Remove Scoop cut up (+) or down (-) from the found height, in mm",
        default=0.0, min=-5.0, max=5.0, precision=2,
    )
    scoop_damage: BoolProperty(
        name="Damage Cut", description="Run Stone Damage on the new flat face only (needs the Stone_Damage node group in the file)",
        default=True,
    )
    scoop_trim: BoolProperty(
        name="Trim Edges", description="Run Trim Edges after the cut (and damage)", default=True,
    )

    # ----- Cavity Infill (its own resolution / grow / shrink, NOT shared) -----
    resolution: FloatProperty(name="Resolution", description="Infill voxel size", default=0.1, min=0.05, max=1.0, precision=3)
    grow: FloatProperty(name="Grow", description="+grow / -shrink distance", default=1.0, min=0.1, max=2.0, precision=3)
    shrink_mult: FloatProperty(
        name="Shrink Multiplier", description="Multiplier for shrink distance",
        default=1.2, min=0.1, max=3.0, step=0.1, subtype='FACTOR',
    )
    target_res: FloatProperty(
        name="Working Resolution (M)",
        description="Working resolution in millions - drives voxel count and decimation limit",
        default=3.0, min=0.2, max=3.0,
    )
    voxel_mode: EnumProperty(
        name="Voxel Mode", description="Choose how voxel size is set",
        items=[
            ("TARGET_VOXELS", "Target Voxels", "Derive voxel size from target voxel count and clamp by Resolution"),
            ("RESOLUTION", "Resolution", "Use the Resolution value directly"),
        ],
        default="TARGET_VOXELS",
    )
    method: EnumProperty(
        name="Method", description="Processing method for cavity healing",
        items=[("ACCURATE", "Accurate", "High-quality accurate method"), ("NAIVE", "Naive", "Fast naive method")],
        default="ACCURATE",
    )
    trim_thin: BoolProperty(name="Trim Thin", description="Remove thin elements from the result", default=True)
    # Kept for files saved by older versions; the highlighted preset is computed when drawn.
    active_preset: EnumProperty(
        name="Active Preset",
        items=[("NONE", "None", ""), ("BUILDING_FAST", "Building Fast", ""), ("BUILDING_ACCURATE", "Building Accurate", ""),
               ("MINI_LARGE_HOLES", "Mini Large Holes", ""), ("MINI_ACCURATE", "Mini Accurate", "")],
        default="BUILDING_ACCURATE", options={'HIDDEN'},
    )

    # ----- Print Check / Seal Cavities / Make Manifold -----
    check_voxel: FloatProperty(
        name="Check Voxel", description="Voxel size (and layer height) Print Check and Seal Cavities look at the model with, in mm",
        default=0.2, min=0.05, max=1.0, precision=3,
    )
    check_drain_gap: FloatProperty(
        name="Min Drain Gap",
        description="Openings narrower than this (mm) count as closed, because resin does not drain through them. "
                    "0 = only cavities with no opening at all, which is what a slicer reports",
        default=0.0, min=0.0, max=2.0, precision=2,
    )
    check_plate: EnumProperty(
        name="Build Plate",
        items=[("Z0", "Plate at Z = 0", "The model is cut at world Z = 0 and printed on that face"),
               ("LOWEST", "Lowest Point", "The plate touches the lowest point of the selection")],
        default="Z0",
    )
    manifold_cut_plate: BoolProperty(
        name="Cut at Plate", description="Make Manifold also cuts the result flat at the build plate (world Z = 0)",
        default=True,
    )
    manifold_blank: PointerProperty(
        name="Intersect With", type=bpy.types.Object,
        description="Optional blank: Make Manifold intersects the united result with this object (read with its "
                    "modifiers), after the repair, so the blank's flat faces stay flat",
        poll=lambda self, ob: ob.type == 'MESH',
    )
    check_result_cavities: StringProperty(default="", options={'SKIP_SAVE'})
    check_result_cups: StringProperty(default="", options={'SKIP_SAVE'})
    check_result_islands: StringProperty(default="", options={'SKIP_SAVE'})
    check_result_contact: StringProperty(default="", options={'SKIP_SAVE'})

    # ----- UI state -----
    show_voxel_tools_menu: BoolProperty(name="Show Voxel Tools", default=True)
    show_infill_tools_menu: BoolProperty(name="Show Infill Tools", default=True)
    show_tools: BoolProperty(name="Show Offset Tools", default=False)
    show_support_tools: BoolProperty(name="Show Support Tools", default=False)
    show_print_check: BoolProperty(name="Show Print Check", default=True)


def get_settings(context=None):
    return (context or bpy.context).scene.quick_infill_settings


def register():
    bpy.utils.register_class(QuickInfillSettings)
    bpy.types.Scene.quick_infill_settings = PointerProperty(type=QuickInfillSettings)


def unregister():
    del bpy.types.Scene.quick_infill_settings
    bpy.utils.unregister_class(QuickInfillSettings)
