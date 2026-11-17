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

import math

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


# ----- Smart scoop: the slope comes from what prints, not from an angle -----
# A printed layer may stick out past the one below by a "lip". Fix Undercuts builds its scoop as
# slopes, and on a slope every layer steps out by LAYER * tan(angle): so a lip is an angle.
# Stringing is that lip, in mm per layer (Alex 2026-11-17).
SMART_LAYER = 0.2       # mm: the print layer the lips are counted for (a 0.4 mm nozzle's usual layer)
SMART_LIP_MIN = 0.2     # mm: all green in Ashlarize's Undercuts overlay, a 45 degree scoop
SMART_LIP_60 = SMART_LAYER * math.tan(math.radians(60.0))   # 0.35 mm: the 60 degree scoop (the default)
SMART_LIP_ROUGH = 0.45  # mm: full cyan in the overlay. One line of a 0.4 mm nozzle; past it the outer line is in the air
SMART_LIP_MAX = 0.6     # mm: the most ("increase the max stringing to around 0.6")
# the buttons under the slider: (mm per layer, what it is)
STRINGING_PRESETS = (
    (SMART_LIP_MIN, "All green in the Undercuts overlay: prints clean. A 45 degree scoop, the deepest"),
    (SMART_LIP_60, "A 60 degree scoop: the underside is a little rough"),
    (SMART_LIP_ROUGH, "Full cyan in the Undercuts overlay: the most that still holds. A 66 degree scoop"),
    (SMART_LIP_MAX, "The outer line droops and strings show. A 72 degree scoop, the shallowest"),
)


def smart_lip(stringing):
    """mm a layer of the scoop may stick out past the one below: Stringing, kept inside its range."""
    return min(SMART_LIP_MAX, max(SMART_LIP_MIN, float(stringing)))


def smart_angle(stringing):
    """The Fix Undercuts angle (degrees from straight down) that gives that lip on every layer."""
    return math.degrees(math.atan(smart_lip(stringing) / SMART_LAYER))


def stringing_for_angle(angle):
    """The Stringing (mm per layer) whose scoop has this angle."""
    return smart_lip(SMART_LAYER * math.tan(math.radians(float(angle))))


def stringing_status(stringing):
    """One line that says what a Stringing value prints like."""
    lip = smart_lip(stringing)
    if lip <= SMART_LIP_MIN + 0.005:
        words = "prints clean"
    elif lip <= SMART_LIP_ROUGH + 0.005:
        words = "underside a little rough"
    else:
        words = "outer line droops, strings show"
    return f"{smart_angle(stringing):.0f}° scoop: {words}"


def get_active_stringing_preset(s):
    """The preset (mm per layer) the Stringing slider stands on, or None."""
    return next((lip for lip, _ in STRINGING_PRESETS if abs(float(s.undercut_stringing) - lip) < 0.005), None)


def get_active_angle_preset(s):
    for a in ANGLE_PRESETS:
        if _close(s.undercut_angle, a):
            return str(int(a))
    return 'NONE'


class QuickInfillSettings(PropertyGroup):
    # ----- shared by every voxel tool (Offset, Support, booleans) -----
    shared_voxel_size: FloatProperty(
        name="Voxel Size", description="Size of the grid a mesh is rebuilt on.\nSmaller is finer and slower. "
                                        "Used by the offset, support and boolean tools",
        default=OFFSET_PRESETS['STONE']["voxel"], min=0.025, max=0.4, precision=3,
    )
    decimation_ratio: FloatProperty(
        name="Decimation Ratio", description="How many of the result's faces are kept.\n"
                                              "0.02 keeps one face in fifty. Used when Decimate is set to Ratio",
        default=OFFSET_PRESETS['STONE']["ratio"], min=0.001, max=1.0, precision=4,
    )
    decimate_mode: EnumProperty(
        name="Decimate Mode", description="How a result is reduced to fewer faces",
        items=[
            ("VOXEL_RATIO", "Ratio", "Reduce the result to the share of faces set by Ratio"),
            ("ORIGINAL", "Original", "Reduce the result to the face count the mesh had before"),
            ("OFF", "Off", "Do not reduce the result.\nIt keeps every face the voxel grid makes"),
        ],
        default="VOXEL_RATIO",
    )
    replace_original: BoolProperty(
        name="Replace Original",
        description="Replace the mesh with the result.\nOff: the original is kept and the result is a new object",
        default=True,
    )
    process_islands: BoolProperty(
        name="Islands",
        description="Work on each loose part of a mesh on its own.\nThe parts are joined back into one object "
                    "afterwards, so parts that touch do not melt together",
        default=False,
    )
    smart_resolution: BoolProperty(
        name="Smart",
        description="Work each mesh at its own resolution tag.\nA tagged object (or island) uses its tag's voxel "
                    "size and ratio. The values set here are only used for meshes without a tag",
        default=True,
    )
    apply_modifiers_on_export: BoolProperty(
        name="Apply Modifiers",
        description="Work on each object as its modifiers leave it.\nOff: the modifiers are ignored",
        default=False,
    )
    ask_above_seconds: FloatProperty(
        name="Ask Above",
        description="Ask before starting a job expected to take longer than this many seconds.\n"
                    "0: never ask. A running job can always be stopped with Esc",
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
        name="Distance", description="How far Grow and Shrink move the surface, in mm.\n"
                                      "Also the base distance of Trim Edges and the slit size Create Outline closes",
        default=OFFSET_PRESETS['STONE']["distance"], min=0.0, max=4.0, precision=3,
    )
    trim_edges_x: FloatProperty(
        name="Trim Edges X", description="How much Trim Edges gives back after rounding the edges off.\n"
                                          "Higher keeps more of the stone (the last grow step is 1 + X times Distance)",
        default=OFFSET_PRESETS['STONE']["x"], min=0.0, max=0.5, precision=3, subtype='FACTOR',
    )
    trim_edges_method: EnumProperty(
        name="Trim Edges Method",
        description="How Trim Edges takes off the thin, sharp edges",
        items=[
            ("ROUND", "Round", "Round the thin edges off, as the old Trim Edges did, about 5 times faster.\n"
                               "Same grow, shrink and regrow (Distance, X, Trim Density); the result is cut "
                               "back to the stone exactly instead of through voxels"),
            ("CHIP", "Flat Chip", "Knock a flat chip off each sharp edge, like a chip off a stone.\n"
                                  "Fast: works on the mesh as it is, no voxels"),
        ],
        default="ROUND",
    )
    trim_edges_thin: FloatProperty(
        name="Thin", description="How big the flat chips are, in Blender units (cm).\n"
                                 "About the width of the chip across a square edge; sharper edges get "
                                 "bigger chips, blunt ones small or none. 0.03 matches the old Trim Edges",
        default=0.04, min=0.005, max=0.2, soft_min=0.02, soft_max=0.12, precision=3, step=0.1,
    )
    trim_edges_only_cut: BoolProperty(
        name="Only Cut Edges", description="Chip only the edges where a cut face meets the stone.\n"
                                           "Cut faces are the ones Ashlarize marked as cut (sp_cut); on a "
                                           "mesh without marks, flat faces left by a boolean are found by shape",
        default=False,
    )
    trim_edges_density: FloatProperty(
        name="Trim Edges Density", description="How finely the trimmed surface is meshed.\n"
                                                "Faces per unit of area before it is cut back to the stone",
        default=1.0, min=0.5, max=4.0, precision=4,
    )

    # ----- Support tools -----
    undercut_angle: FloatProperty(
        name="Angle",
        description="How far the lit directions are tilted from straight down.\n"
                    "0 = straight down, 90 = level. Remove Scoop looks at the stone from this angle too. "
                    "Negative angles join the directions with a voxel union",
        default=60.0, min=-90.0, max=90.0, precision=1,
    )
    undercut_smart: BoolProperty(
        name="Smart",
        description="Work the scoop out from what prints, not from an Angle.\n"
                    "No layer of the scoop sticks out further past the one below than Stringing allows. "
                    "With several objects selected, each scoop stops at the objects around it and joins them, "
                    "so it rests on the object below",
        default=True,
    )
    undercut_floor: EnumProperty(
        name="Floor",
        description="A floor a Smart scoop goes down to and rests on, like the top of an object below, when it is "
                    "within 0.8 units (8 mm) under the object",
        items=[('GROUND', "Ground", "Z = 0 (the print plate), or the lowest point of the selected objects if they go below it"),
               ('HEIGHT', "Height", "A height you set, such as the bottom of a blank"),
               ('NONE', "None", "No floor: a scoop only rests on other objects")],
        default='GROUND',
    )
    undercut_floor_z: FloatProperty(
        name="Floor Height",
        description="Height of the floor (Floor: Height), such as the bottom of a blank",
        default=0.0, subtype='DISTANCE', unit='LENGTH',
    )
    undercut_stringing: FloatProperty(
        name="Stringing",
        description="How far each printed layer of the scoop may stick out past the one below, in mm "
                    "(for 0.2 mm layers).\n"
                    "0.2 is all green in the Undercuts overlay and gives the deepest scoop. "
                    "Higher gives a shallower scoop that keeps more of the stone but prints rougher: "
                    "0.45 is full cyan, 0.6 is where the outer line droops",
        default=SMART_LIP_60, min=SMART_LIP_MIN, max=SMART_LIP_MAX, precision=2, step=1,
    )
    dir_x_pos: BoolProperty(name="+X", description="Fix undercuts seen from +X", default=True)
    dir_x_neg: BoolProperty(name="-X", description="Fix undercuts seen from -X", default=True)
    dir_y_pos: BoolProperty(name="+Y", description="Fix undercuts seen from +Y", default=True)
    dir_y_neg: BoolProperty(name="-Y", description="Fix undercuts seen from -Y", default=True)
    dir_xpos_ypos: BoolProperty(name="+X+Y", description="Fix undercuts seen from +X+Y", default=True)
    dir_xpos_yneg: BoolProperty(name="+X-Y", description="Fix undercuts seen from +X-Y", default=True)
    dir_xneg_ypos: BoolProperty(name="-X+Y", description="Fix undercuts seen from -X+Y", default=True)
    dir_xneg_yneg: BoolProperty(name="-X-Y", description="Fix undercuts seen from -X-Y", default=True)
    dir_z: BoolProperty(name="Z", description="Fix undercuts seen from straight above, with no tilt", default=False)
    auto_shrink: BoolProperty(
        name="Auto Shrink", description="Shrink the result after fixing undercuts.\nTakes back the material the fix "
                                         "added, on faces within 70 degrees of straight up, so stones still fit",
        default=False,
    )
    shrink_amount: FloatProperty(
        name="Shrink Amount", description="How far Auto Shrink pulls the surface in, in mm",
        default=0.1, min=0.01, max=1.0, precision=2,
    )
    shrink_angle_threshold: FloatProperty(
        name="Shrink Angle", description="Only shrink faces within this angle (degrees) of pointing straight up",
        default=70.0, min=5.0, max=90.0, precision=1,
    )

    # ----- Remove Scoop (support tools) -----
    scoop_method: EnumProperty(
        name="Method", description="How the scoop is taken off",
        items=(('FILL', "Fill", "Fill the scoop in.\nThe faces no side view can see (the scoop) are deleted and the "
                                "hole is filled smoothly, like repairing a 3D-scan defect. No flat cut"),
               ('VIEW', "View", "Cut the scoop off flat, where it ends.\nFound by looking down at each side of the "
                                "stone at the Angle"),
               ('CONTACT', "Contact", "Cut the scoop off flat, where the cut is big enough.\nWhere a flat cut covers "
                                      "the Contact share of the stone's footprint")),
        default='FILL',
    )
    scoop_angle: FloatProperty(
        name="View Angle", description="Angle the four side views look at the stone from, like the support angle: "
                                        "90 = straight at the side, 0 = from the top. The scoop hides under the face",
        default=60.0, min=20.0, max=80.0, precision=0, subtype='NONE',
    )
    scoop_share: FloatProperty(
        name="Bottom Line", description="How much of the scoop is cut off.\nThe cut is at this share of the bottom points "
                                         "the four views see: 50% is the middle, higher cuts more",
        default=50.0, min=0.0, max=100.0, precision=0, subtype='PERCENTAGE',
    )
    scoop_contact: FloatProperty(
        name="Contact", description="How big the flat cut must be.\nThe scoop is cut where the flat cut covers this "
                                    "share of the stone's footprint",
        default=90.0, min=50.0, max=100.0, precision=0, subtype='PERCENTAGE',
    )
    scoop_offset: FloatProperty(
        name="Scoop Offset", description="Move the cut up (+) or down (-) from where it was found, in mm",
        default=0.0, min=-5.0, max=5.0, precision=2,
    )
    scoop_damage: BoolProperty(
        name="Damage Cut", description="Damage the new flat face so it matches the stone.\n"
                                        "Needs the Stone_Damage node group in the file",
        default=True,
    )
    scoop_trim: BoolProperty(
        name="Trim Edges", description="Round the edges off after the cut.\nRuns Trim Edges (Round, whatever the Trim Edges method is set to) on the stone, after the damage",
        default=True,
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
    show_remove_scoop: BoolProperty(name="Remove Scoop",
                                    description="Show or fold away Remove Scoop and its settings.\n"
                                                "Take the scoop Fix Undercuts made back off a stone",
                                    default=False)
    show_print_check: BoolProperty(name="Show Print Check", default=True)


def get_settings(context=None):
    return (context or bpy.context).scene.quick_infill_settings


# Trim Edges' "Round (old)" was taken out (Alex 2026-11-18). A file saved with it stores the
# method as its number (Round 0, Flat Chip 1, Round (old) 2): set those to Round when it opens.
OLD_ROUND = 2


@bpy.app.handlers.persistent
def _old_round_to_round(_dummy=None):
    for scene in bpy.data.scenes:
        s = getattr(scene, "quick_infill_settings", None)
        if s is not None and s.get("trim_edges_method") == OLD_ROUND:
            s["trim_edges_method"] = 0


def register():
    bpy.utils.register_class(QuickInfillSettings)
    bpy.types.Scene.quick_infill_settings = PointerProperty(type=QuickInfillSettings)
    bpy.app.handlers.load_post.append(_old_round_to_round)


def unregister():
    if _old_round_to_round in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_old_round_to_round)
    del bpy.types.Scene.quick_infill_settings
    bpy.utils.unregister_class(QuickInfillSettings)
