import bpy
import mathutils
from .offset_utils import RESOLUTION_PRESETS, RESOLUTION_TAG_ORDER, nearest_resolution_tag

# Scale factor used for the Blender <-> meshlib STL round-trip (export uses this
# as global_scale, import uses its reciprocal). Kept as one constant so island
# centroid math (world space) and the STL export call always agree.
STL_EXPORT_SCALE = 10.0

# Name of the per-face integer attribute + object custom property used to tag
# mesh geometry with the Stone/Detail/Fine/Mini resolution it was processed at.
RES_TAG_ATTR = "qi_res_tag"

# Name of the visual color attribute mirroring the tag, for viewport shading.
RES_COLOR_ATTR = "qi_res_color"
RES_TAG_COLORS = {
    'STONE': (0.55, 0.35, 0.2, 1.0),
    'DETAIL': (0.2, 0.55, 0.9, 1.0),
    'FINE': (0.25, 0.8, 0.3, 1.0),
    'MINI': (0.95, 0.75, 0.1, 1.0),
    'COARSE': (0.8, 0.15, 0.15, 1.0),
}
RES_UNTAGGED_COLOR = (0.5, 0.5, 0.5, 1.0)


def _tag_to_index(tag_name):
    """Map a resolution tag name to its stable storage index, or -1 if unknown."""
    try:
        return RESOLUTION_TAG_ORDER.index(tag_name)
    except (ValueError, TypeError):
        return -1


def _index_to_tag(index):
    """Map a stored resolution tag index back to its name, or None if invalid."""
    if index is None or index < 0 or index >= len(RESOLUTION_TAG_ORDER):
        return None
    return RESOLUTION_TAG_ORDER[index]


def _dominant_tag_label(face_tag_indices):
    """Return a single tag name if all faces share one tag, "MIXED" if they
    differ, or None if none of the faces carry a valid tag."""
    valid = [t for t in face_tag_indices if t is not None and t >= 0]
    if not valid:
        return None
    unique = set(valid)
    if len(unique) == 1:
        return _index_to_tag(next(iter(unique)))
    return "MIXED"


def _dominant_face_tag(mesh):
    """Return the majority resolution tag from a mesh's per-face qi_res_tag
    attribute, or None if the mesh has no such attribute.

    This is the ground truth for a mesh's current geometry: Blender's native
    join/separate operators correctly split/merge per-face attribute data
    along with the polygons, but only copy the object-level custom property
    from a single source object, which goes stale the moment islands with
    different tags are combined or split outside our own pipeline.
    """
    if mesh is None:
        return None
    attr = mesh.attributes.get(RES_TAG_ATTR)
    if attr is None or len(attr.data) == 0:
        return None
    counts = {}
    for item in attr.data:
        counts[item.value] = counts.get(item.value, 0) + 1
    best_index = max(counts, key=counts.get)
    return _index_to_tag(best_index)


def read_object_resolution_tag(obj):
    """Read the whole-object resolution tag for Smart resolution lookups.

    Prefers the mesh's own per-face attribute (accurate even after a native
    Blender join/separate reshuffles geometry), falls back to the object-level
    custom property, then to an approximation from remesh_voxel_size.
    """
    if obj is None:
        return None
    if obj.type == 'MESH':
        face_tag = _dominant_face_tag(obj.data)
        if face_tag:
            return face_tag
    tag = obj.get(RES_TAG_ATTR)
    if tag in RESOLUTION_TAG_ORDER:
        return tag
    if obj.type == 'MESH':
        return _approximate_tag_from_remesh_voxel_size(obj.data)
    return None


def _approximate_tag_from_remesh_voxel_size(mesh):
    """Guess a resolution tag from mesh.remesh_voxel_size when no explicit
    qi_res_tag is present. Only trusted within 0.1 of a known preset's voxel
    size; otherwise returns None so the caller falls back to current settings.

    mesh.remesh_voxel_size is in Blender's own local units, while the presets
    and the rest of the pipeline operate in meshlib/STL space (scaled up by
    STL_EXPORT_SCALE), so the read value must be converted before comparing.
    """
    if mesh is None:
        return None
    blender_voxel_size = getattr(mesh, "remesh_voxel_size", 0.0)
    if not blender_voxel_size:
        return None
    voxel_size = blender_voxel_size * STL_EXPORT_SCALE
    tag = nearest_resolution_tag(voxel_size)
    preset_voxel_size = RESOLUTION_PRESETS[tag][0]
    if abs(voxel_size - preset_voxel_size) <= 0.1:
        return tag
    return None


def write_mesh_resolution_tags(mesh, face_tag_indices, remesh_voxel_size=None):
    """Write the per-face qi_res_tag attribute, a matching color attribute for
    viewport visualization, and remesh_voxel_size onto a mesh datablock, so
    Blender's own voxel remesh in Sculpt Mode stays consistent with the
    resolution Quick Infill last processed it at.

    remesh_voxel_size is given in meshlib/STL space (scaled by STL_EXPORT_SCALE);
    it is converted down to Blender's own local units before being stored.
    """
    if mesh is None or len(mesh.polygons) == 0:
        return
    polycount = len(mesh.polygons)
    attr = mesh.attributes.get(RES_TAG_ATTR)
    if attr is None:
        attr = mesh.attributes.new(name=RES_TAG_ATTR, type='INT', domain='FACE')
    if len(face_tag_indices) == polycount:
        for i, item in enumerate(attr.data):
            item.value = face_tag_indices[i]
        _write_resolution_colors(mesh, face_tag_indices)
    if remesh_voxel_size is not None:
        mesh.remesh_voxel_size = float(remesh_voxel_size) / STL_EXPORT_SCALE


def _write_resolution_colors(mesh, face_tag_indices):
    """Mirror the per-face resolution tag onto a CORNER color attribute so it
    can be visualized directly in the viewport (Color Attribute shading)."""
    color_attr = mesh.color_attributes.get(RES_COLOR_ATTR)
    if color_attr is None:
        color_attr = mesh.color_attributes.new(name=RES_COLOR_ATTR, type='BYTE_COLOR', domain='CORNER')
    for poly in mesh.polygons:
        tag = _index_to_tag(face_tag_indices[poly.index])
        color = RES_TAG_COLORS.get(tag, RES_UNTAGGED_COLOR)
        for loop_index in poly.loop_indices:
            color_attr.data[loop_index].color = color
    try:
        mesh.color_attributes.active_color = color_attr
    except (AttributeError, TypeError):
        pass


def set_object_resolution_label(obj, label):
    """Set (or clear) the object-level resolution tag custom property."""
    if obj is None:
        return
    if label:
        obj[RES_TAG_ATTR] = label
    elif RES_TAG_ATTR in obj:
        del obj[RES_TAG_ATTR]


def compute_blender_island_tags(blender_obj):
    """Read-only: find each loose part's world-space centroid and dominant
    resolution tag directly from the Blender mesh via bmesh connectivity, with
    no object creation. Returns a list of (centroid_world, tag_name_or_None).
    """
    import bmesh
    mesh = blender_obj.data
    if mesh is None or len(mesh.polygons) == 0:
        return []

    attr = mesh.attributes.get(RES_TAG_ATTR)
    face_tags = [item.value for item in attr.data] if attr is not None else None
    fallback_tag = _approximate_tag_from_remesh_voxel_size(mesh)

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()

    visited = set()
    islands = []
    for seed in bm.faces:
        if seed.index in visited:
            continue
        stack = [seed]
        visited.add(seed.index)
        island_faces = []
        while stack:
            f = stack.pop()
            island_faces.append(f)
            for e in f.edges:
                for lf in e.link_faces:
                    if lf.index not in visited:
                        visited.add(lf.index)
                        stack.append(lf)

        centroid_local = mathutils.Vector((0.0, 0.0, 0.0))
        for f in island_faces:
            centroid_local += f.calc_center_median()
        centroid_local /= len(island_faces)
        centroid_world = blender_obj.matrix_world @ centroid_local

        tag = None
        if face_tags is not None:
            counts = {}
            for f in island_faces:
                v = face_tags[f.index]
                counts[v] = counts.get(v, 0) + 1
            best_index = max(counts, key=counts.get)
            tag = _index_to_tag(best_index)
        if tag is None:
            tag = fallback_tag

        islands.append((centroid_world, tag))

    bm.free()
    return islands


def _nearest_island_tag(centroid_world, island_tags):
    """Match a meshlib island centroid (world space) to the closest known
    Blender-side island and return its resolution tag, or None if unmatched."""
    if not island_tags:
        return None
    _, tag = min(island_tags, key=lambda pair: (pair[0] - centroid_world).length_squared)
    return tag


def blender_to_meshlib(blender_obj):
    """
    Convert a Blender mesh object to a meshlib Mesh using MeshBuilder.
    """
    from .meshlib_utils import get_meshlib
    mm, _ = get_meshlib()
    
    if blender_obj.type != 'MESH':
        raise ValueError("Selected object is not a mesh.")

    mesh_data = blender_obj.data

    # Extract vertices
    vertices = []
    for v in mesh_data.vertices:
        vertices.append([v.co.x, v.co.y, v.co.z])

    # Extract faces (triangulate ngons via fan)
    faces = []
    for poly in mesh_data.polygons:
        idx = list(poly.vertices)
        if len(idx) == 3:
            faces.append(idx)
        elif len(idx) == 4:
            faces.append([idx[0], idx[1], idx[2]])
            faces.append([idx[0], idx[2], idx[3]])
        else:
            for i in range(2, len(idx)):
                faces.append([idx[0], idx[i-1], idx[i]])

    # Build meshlib mesh
    try:
        tris = [mm.MeshBuilder.Triangle(f[0], f[1], f[2]) for f in faces]
        ml_mesh = mm.MeshBuilder.fromTriangles(vertices, tris)
    except (AttributeError, TypeError):
        try:
            ml_mesh = mm.MeshBuilder.fromPointTriples(vertices, faces)
        except Exception as e:
            raise RuntimeError("MeshBuilder creation failed") from e

    return ml_mesh


def blender_to_meshlib_via_stl(blender_obj, tmp_dir=None, apply_modifiers=True):
    """
    Export the given Blender object to a temporary STL and load it via meshlib.
    This avoids constructing huge Python-side vectors for very dense meshes.
    """
    import os
    import tempfile
    import bpy
    from .meshlib_utils import get_meshlib
    mm, _ = get_meshlib()
    # Do not rely on enabling addons; use available operators

    # Prepare temp filepath
    tmp_dir = tmp_dir or tempfile.gettempdir()
    os.makedirs(tmp_dir, exist_ok=True)
    fd, stl_path = tempfile.mkstemp(prefix="quick_infill_", suffix=".stl", dir=tmp_dir)
    os.close(fd)

    # Preserve selection state
    view_layer = bpy.context.view_layer
    prev_active = view_layer.objects.active
    prev_selection = [obj for obj in bpy.context.selected_objects]

    try:
        # Select only the target object
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        blender_obj.select_set(True)
        view_layer.objects.active = blender_obj

        # Export selection to STL using Blender native operator
        res = bpy.ops.wm.stl_export(
            'EXEC_DEFAULT',
            filepath=stl_path,
            export_selected_objects=True,
            use_batch=False,
            global_scale=STL_EXPORT_SCALE,
            apply_modifiers=apply_modifiers,
            evaluation_mode='DAG_EVAL_VIEWPORT',
        )
        if res != {'FINISHED'}:
            raise RuntimeError("Could not export STL with Blender 4.5 native operator")

        # Load via meshlib
        loaded = mm.loadMesh(stl_path)
        mesh = loaded.mesh if hasattr(loaded, 'mesh') else loaded
        return mesh
    finally:
        # Restore selection
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        for obj in prev_selection:
            obj.select_set(True)
        view_layer.objects.active = prev_active
        # Clean up temp file
        try:
            os.remove(stl_path)
        except Exception:
            pass


def meshlib_to_blender_via_stl(meshlib_mesh, name: str = "Converted Mesh", import_scale: float = 0.1):
    """
    Save a meshlib mesh to a temporary STL and import it back into Blender at a given scale.
    Returns the imported Blender object.
    """
    import os
    import tempfile
    import bpy
    from .meshlib_utils import get_meshlib
    mm, _ = get_meshlib()
    # Do not rely on enabling addons; use available operators

    tmp_dir = tempfile.gettempdir()
    os.makedirs(tmp_dir, exist_ok=True)
    fd, stl_path = tempfile.mkstemp(prefix="quick_infill_out_", suffix=".stl", dir=tmp_dir)
    os.close(fd)

    # Save meshlib mesh to STL
    saved = False
    try:
        mm.saveMesh(meshlib_mesh, stl_path)
        saved = True
    except Exception:
        try:
            mm.saveMeshAs(meshlib_mesh, stl_path)
            saved = True
        except Exception:
            saved = False
    if not saved:
        raise RuntimeError("Could not save meshlib mesh to STL; consider direct meshlib_to_blender fallback")

    # Track current objects to detect newly imported one
    prev_objs = set(bpy.data.objects)

    # Import STL
    imported = False
    # Prefer wm.stl_import if available
    if hasattr(bpy.ops.wm, "stl_import"):
        try:
            res = bpy.ops.wm.stl_import('EXEC_DEFAULT', filepath=stl_path, global_scale=float(import_scale))
            imported = (res == {'FINISHED'})
        except Exception:
            imported = False
    # Fallback to legacy importer
    if not imported and hasattr(bpy.ops, "import_mesh") and hasattr(bpy.ops.import_mesh, "stl"):
        try:
            res = bpy.ops.import_mesh.stl('EXEC_DEFAULT', filepath=stl_path, global_scale=float(import_scale))
            imported = (res == {'FINISHED'})
        except Exception:
            imported = False
    if not imported:
        raise RuntimeError("STL import failed")

    new_objs = [obj for obj in bpy.data.objects if obj not in prev_objs]
    if not new_objs:
        # As a fallback, try selected objects
        new_objs = list(bpy.context.selected_objects)
    if not new_objs:
        raise RuntimeError("Imported STL but could not find the new object")

    obj = new_objs[0]
    obj.name = name

    # Clean up temp STL
    try:
        os.remove(stl_path)
    except Exception:
        pass

    return obj


def select_results(result_objs):
    """Select the given result objects and make the first one active.
    
    Call after multi-object operator loops to keep selection consistent.
    """
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    for obj in result_objs:
        if obj and obj.name in bpy.data.objects:
            obj.select_set(True)
    if result_objs:
        bpy.context.view_layer.objects.active = result_objs[0]


def meshlib_to_blender(meshlib_mesh, name="Converted Mesh"):
    """
    Convert a meshlib Mesh to a Blender mesh object.
    
    Args:
        meshlib_mesh (mm.Mesh): The meshlib Mesh to convert.
        name (str): Name for the new Blender object. 
    
    Returns:
        bpy.types.Object: The new Blender mesh object.
    """
    from .meshlib_utils import get_meshlib
    mm, _ = get_meshlib()
    
    # Extract vertices
    vertices = []
    for i in range(meshlib_mesh.points.size()):
        vert_id = mm.VertId(i)
        point = meshlib_mesh.points[vert_id]
        vertices.append((point.x, point.y, point.z))
    
    # Extract faces
    faces = []
    for i in range(meshlib_mesh.topology.numValidFaces()):
        face = meshlib_mesh.topology.getFace(i)
        if face.valid():
            face_indices = [face[0], face[1], face[2]]
            faces.append(face_indices)
    
    # Create new mesh in Blender
    new_mesh = bpy.data.meshes.new(name)
    new_mesh.from_pydata(vertices, [], faces)
    new_mesh.update()
    
    # Create new object
    new_obj = bpy.data.objects.new(name, new_mesh)
    
    # Link to current scene
    bpy.context.scene.collection.objects.link(new_obj)
    
    return new_obj


def _process_mesh_operation_single(blender_obj, operation_fn, output_suffix, auto_decimate=False, import_scale=0.1, replace_original=False, resolution=None):
    """Actual operation pipeline without island splitting."""
    from .offset_utils import decimate_mesh, should_auto_decimate_faces

    obj_name = blender_obj.name

    settings = getattr(bpy.context.scene, "quick_infill_tools_settings", None)
    mode = getattr(settings, "decimate_mode", "VOXEL_RATIO") if settings is not None else "VOXEL_RATIO"
    ratio = getattr(settings, "decimation_ratio", None) if settings is not None else None
    smart = bool(getattr(settings, "smart_resolution", False)) if settings is not None else False
    apply_modifiers = bool(getattr(settings, "apply_modifiers_on_export", False)) if settings is not None else False

    effective_resolution = resolution
    if smart:
        existing_tag = read_object_resolution_tag(blender_obj)
        if existing_tag:
            effective_resolution, ratio = RESOLUTION_PRESETS[existing_tag]
            mode = "VOXEL_RATIO"

    # Convert to meshlib
    src_mesh = blender_to_meshlib_via_stl(blender_obj, apply_modifiers=apply_modifiers)
    initial_face_count = src_mesh.topology.numValidFaces()
    initial_vertex_count = src_mesh.topology.numValidVerts()

    # Apply the operation
    out_mesh = operation_fn(src_mesh, effective_resolution)

    # Auto decimate if enabled - only when significant face growth occurred
    final_face_count = out_mesh.topology.numValidFaces()
    if auto_decimate:
        do_decimate, target_faces = should_auto_decimate_faces(
            initial_face_count,
            final_face_count,
            voxel_size=effective_resolution,
            mode=mode,
            ratio=ratio,
        )
        if do_decimate:
            out_mesh = decimate_mesh(out_mesh, target_face_count=target_faces, resolution=effective_resolution)

    final_vertex_count = out_mesh.topology.numValidVerts()

    # Convert back to Blender
    result_obj = meshlib_to_blender_via_stl(out_mesh, obj_name + output_suffix, import_scale=import_scale)

    tag_index = _tag_to_index(nearest_resolution_tag(effective_resolution)) if effective_resolution is not None else -1
    face_tags = [tag_index] * len(result_obj.data.polygons)
    write_mesh_resolution_tags(result_obj.data, face_tags, remesh_voxel_size=effective_resolution)

    # If replace_original is enabled, swap mesh data and delete the temp object
    if replace_original:
        result_obj = replace_mesh_keep_transforms(blender_obj, result_obj)
    set_object_resolution_label(result_obj, _dominant_tag_label(face_tags))

    return result_obj, initial_vertex_count, final_vertex_count


def process_object_with_island_split(blender_obj, operation_fn, output_suffix, auto_decimate=False, import_scale=0.1, replace_original=False, resolution=None):
    """Process a mesh one connected-component (island) at a time, entirely inside
    meshlib, then merge the results and import back into Blender exactly once.

    This avoids Blender-side object duplication/split/join (bpy.ops.mesh.separate,
    bpy.ops.object.join, bpy.data.objects.remove in a loop), which is slow and
    prone to invalidating other Python Object references when objects are removed.
    """
    from .meshlib_utils import get_meshlib
    from .offset_utils import decimate_mesh, should_auto_decimate_faces
    mm, _ = get_meshlib()

    settings = getattr(bpy.context.scene, "quick_infill_settings", None)
    if settings is None:
        settings = getattr(bpy.context.scene, "quick_infill_tools_settings", None)
    process_islands = bool(getattr(settings, "process_islands", False)) if settings is not None else False

    if not process_islands:
        return _process_mesh_operation_single(
            blender_obj,
            operation_fn,
            output_suffix,
            auto_decimate=auto_decimate,
            import_scale=import_scale,
            replace_original=replace_original,
            resolution=resolution,
        )

    obj_name = blender_obj.name
    smart = bool(getattr(settings, "smart_resolution", False)) if settings is not None else False
    apply_modifiers = bool(getattr(settings, "apply_modifiers_on_export", False)) if settings is not None else False
    blender_island_tags = compute_blender_island_tags(blender_obj) if smart else []

    src_mesh = blender_to_meshlib_via_stl(blender_obj, apply_modifiers=apply_modifiers)

    component_bitsets = mm.MeshComponents.getAllComponents(mm.MeshPart(src_mesh))
    if len(component_bitsets) <= 1:
        return _process_mesh_operation_single(
            blender_obj,
            operation_fn,
            output_suffix,
            auto_decimate=auto_decimate,
            import_scale=import_scale,
            replace_original=replace_original,
            resolution=resolution,
        )

    mode = getattr(settings, "decimate_mode", "VOXEL_RATIO") if settings is not None else "VOXEL_RATIO"
    ratio = getattr(settings, "decimation_ratio", None) if settings is not None else None

    total_initial = 0
    total_final = 0
    merged_mesh = mm.Mesh()
    face_tags = []
    voxel_sizes_used = []

    for bitset in component_bitsets:
        island_mesh = mm.Mesh()
        island_mesh.addMeshPart(mm.MeshPart(src_mesh, bitset))

        initial_faces = island_mesh.topology.numValidFaces()

        island_resolution, island_ratio, island_mode = resolution, ratio, mode
        if smart and blender_island_tags:
            bbox = src_mesh.computeBoundingBox(bitset)
            centroid = (bbox.min + bbox.max) * 0.5
            centroid_world = mathutils.Vector((centroid.x, centroid.y, centroid.z)) / STL_EXPORT_SCALE
            matched_tag = _nearest_island_tag(centroid_world, blender_island_tags)
            if matched_tag:
                island_resolution, island_ratio = RESOLUTION_PRESETS[matched_tag]
                island_mode = "VOXEL_RATIO"

        try:
            out_mesh = operation_fn(island_mesh, island_resolution)
        except Exception:
            # This island collapsed under the operation (e.g. Trim Edges/Trim
            # Thin on a thin sliver); drop just this island instead of
            # aborting the whole multi-island merge.
            continue

        total_initial += island_mesh.topology.numValidVerts()

        if auto_decimate:
            final_faces = out_mesh.topology.numValidFaces()
            do_decimate, target_faces = should_auto_decimate_faces(
                initial_faces,
                final_faces,
                voxel_size=island_resolution,
                mode=island_mode,
                ratio=island_ratio,
            )
            if do_decimate:
                out_mesh = decimate_mesh(out_mesh, target_face_count=target_faces, resolution=island_resolution)

        n_faces = out_mesh.topology.numValidFaces()
        tag_index = _tag_to_index(nearest_resolution_tag(island_resolution)) if island_resolution is not None else -1
        face_tags.extend([tag_index] * n_faces)
        if island_resolution is not None:
            voxel_sizes_used.append(island_resolution)

        total_final += out_mesh.topology.numValidVerts()
        merged_mesh.addMesh(out_mesh)

    if merged_mesh.topology.numValidFaces() == 0:
        raise RuntimeError(f"All mesh islands collapsed for '{blender_obj.name}'")

    result_obj = meshlib_to_blender_via_stl(merged_mesh, obj_name + output_suffix, import_scale=import_scale)

    remesh_voxel_size = min(voxel_sizes_used) if voxel_sizes_used else resolution
    write_mesh_resolution_tags(result_obj.data, face_tags, remesh_voxel_size=remesh_voxel_size)

    if replace_original:
        result_obj = replace_mesh_keep_transforms(blender_obj, result_obj)
    set_object_resolution_label(result_obj, _dominant_tag_label(face_tags))

    return result_obj, total_initial, total_final


def process_mesh_operation(blender_obj, operation_fn, output_suffix, auto_decimate=False, import_scale=0.1, replace_original=False, resolution=None):
    """
    Generic wrapper for mesh operations: import -> process -> optional decimate -> export.
    """
    settings = getattr(bpy.context.scene, "quick_infill_settings", None)
    if settings is None:
        settings = getattr(bpy.context.scene, "quick_infill_tools_settings", None)
    process_islands = bool(getattr(settings, "process_islands", False)) if settings is not None else False

    if process_islands:
        return process_object_with_island_split(
            blender_obj,
            operation_fn,
            output_suffix,
            auto_decimate=auto_decimate,
            import_scale=import_scale,
            replace_original=replace_original,
            resolution=resolution,
        )

    return _process_mesh_operation_single(
        blender_obj,
        operation_fn,
        output_suffix,
        auto_decimate=auto_decimate,
        import_scale=import_scale,
        replace_original=replace_original,
        resolution=resolution,
    )


def replace_mesh_keep_transforms(original_obj, new_obj):
    """
    Replace the mesh data of original_obj with new_obj's mesh data,
    keeping original_obj's transforms (location, rotation, scale) intact.
    
    The new mesh is transformed so it appears in the correct position
    when using the original object's transforms.
    
    Args:
        original_obj: The original Blender object to update
        new_obj: The new object with the mesh data to use
    
    Returns:
        The original object (now with new mesh data)
    """
    import bpy
    import bmesh
    
    # Get the transformation matrices
    # new_obj is at origin with some scale (e.g., 0.1 from STL import)
    # original_obj has its own transforms
    # We need to transform new_obj's mesh so it appears correct with original_obj's transforms
    
    # The mesh vertices in new_obj are in new_obj's local space
    # When we assign them to original_obj, they'll be interpreted in original_obj's local space
    # So we need: new_world_position = original_world_position
    # new_obj.matrix_world @ new_local = original_obj.matrix_world @ original_local
    # original_local = original_obj.matrix_world.inverted() @ new_obj.matrix_world @ new_local
    
    transform_matrix = original_obj.matrix_world.inverted() @ new_obj.matrix_world
    
    # Apply the transformation to the new mesh vertices
    new_mesh = new_obj.data
    bm = bmesh.new()
    bm.from_mesh(new_mesh)
    bmesh.ops.transform(bm, matrix=transform_matrix, verts=bm.verts)
    bm.to_mesh(new_mesh)
    bm.free()
    new_mesh.update()
    
    # Store reference to old mesh data for cleanup
    old_mesh = original_obj.data
    
    # Assign the new mesh data to the original object
    original_obj.data = new_mesh
    
    # Give the mesh a proper name
    original_obj.data.name = original_obj.name

    # Remove the temporary imported object (but not its mesh data, which is now in use)
    bpy.data.objects.remove(new_obj, do_unlink=True)

    # Reselect the original object and make it active
    original_obj.select_set(True)
    bpy.context.view_layer.objects.active = original_obj

    # Sync the depsgraph so it transitions from old_mesh to new_mesh
    # BEFORE freeing old_mesh.  Without this the depsgraph still holds a
    # reference to old_mesh internally; removing the mesh while that reference
    # is live leaves a dangling pointer that Blender dereferences on the next
    # file save, causing a freeze.
    bpy.context.view_layer.update()

    # Now safe to free the old mesh – the depsgraph no longer references it.
    if old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)

    return original_obj


def batch_process_mesh_operation(blender_objs, operation_fn, output_suffix, auto_decimate=False, import_scale=0.1, replace_original=False, resolution=None):
    """
    Optimized batch wrapper for mesh operations on multiple objects.
    Processes each object individually but batches Blender I/O for efficiency.
    
    Args:
        blender_objs: List of source Blender mesh objects
        operation_fn: Function that takes meshlib mesh and returns processed meshlib mesh
        output_suffix: Suffix for output object names (e.g., "_Grown")
        auto_decimate: If True, decimate output to match initial vertex count per object
        import_scale: Scale factor for STL import (default 0.1)
        replace_original: If True, replace original objects' mesh data instead of creating new objects
    
    Returns:
        list of tuples: [(output_blender_obj, initial_vertex_count, final_vertex_count), ...]
    """
    import os
    import tempfile
    from .meshlib_utils import get_meshlib
    from .offset_utils import decimate_mesh
    mm, _ = get_meshlib()

    settings = getattr(bpy.context.scene, "quick_infill_settings", None)
    if settings is None:
        settings = getattr(bpy.context.scene, "quick_infill_tools_settings", None)
    process_islands = bool(getattr(settings, "process_islands", False)) if settings is not None else False
    
    if not blender_objs:
        return [], []

    if process_islands:
        results = []
        collapsed_objs = []
        # Restore the pre-operation selection afterwards so behavior matches
        # the non-island batch path below (selection state is not meant to
        # reflect the newly created results in either mode).
        view_layer = bpy.context.view_layer
        prev_active_name = view_layer.objects.active.name if view_layer.objects.active else None
        prev_selection_names = [obj.name for obj in bpy.context.selected_objects]
        try:
            for obj in blender_objs:
                try:
                    result_obj, initial_verts, final_verts = process_object_with_island_split(
                        obj,
                        operation_fn,
                        output_suffix,
                        auto_decimate=auto_decimate,
                        import_scale=import_scale,
                        replace_original=replace_original,
                        resolution=resolution,
                    )
                    results.append((result_obj, initial_verts, final_verts))
                except Exception as exc:
                    collapsed_objs.append((obj, exc))
        finally:
            for obj in bpy.context.selected_objects:
                obj.select_set(False)
            for name in prev_selection_names:
                if name in bpy.data.objects:
                    bpy.data.objects[name].select_set(True)
            if prev_active_name and prev_active_name in bpy.data.objects:
                view_layer.objects.active = bpy.data.objects[prev_active_name]
        return results, collapsed_objs
    
    results = []
    collapsed_objs = []
    tmp_dir = tempfile.gettempdir()

    # Precompute the effective (voxel_size, ratio, mode) per object up front:
    # under Smart resolution, an object with an existing tag uses that tag's
    # preset values instead of the current UI settings.
    smart = bool(getattr(settings, "smart_resolution", False)) if settings is not None else False
    default_mode = getattr(settings, "decimate_mode", "VOXEL_RATIO") if settings is not None else "VOXEL_RATIO"
    default_ratio = getattr(settings, "decimation_ratio", None) if settings is not None else None

    effective_resolutions = []
    effective_ratios = []
    effective_modes = []
    for obj in blender_objs:
        eff_res, eff_ratio, eff_mode = resolution, default_ratio, default_mode
        if smart:
            existing_tag = read_object_resolution_tag(obj)
            if existing_tag:
                eff_res, eff_ratio = RESOLUTION_PRESETS[existing_tag]
                eff_mode = "VOXEL_RATIO"
        effective_resolutions.append(eff_res)
        effective_ratios.append(eff_ratio)
        effective_modes.append(eff_mode)
    
    # Save current selection state once (store names to avoid stale StructRNA references)
    view_layer = bpy.context.view_layer
    prev_active_name = view_layer.objects.active.name if view_layer.objects.active else None
    prev_selection_names = [obj.name for obj in bpy.context.selected_objects]
    
    try:
        # Phase 1: Export all objects to STL files (batch export setup)
        stl_paths = []
        for obj in blender_objs:
            fd, stl_path = tempfile.mkstemp(prefix=f"qi_batch_{obj.name}_", suffix=".stl", dir=tmp_dir)
            os.close(fd)
            stl_paths.append(stl_path)
        
        # Export each object (Blender requires individual selection for STL export)
        meshlib_meshes = []
        initial_face_counts = []
        initial_vert_counts = []
        
        apply_modifiers = bool(getattr(settings, "apply_modifiers_on_export", False)) if settings is not None else False

        for i, obj in enumerate(blender_objs):
            # Select only this object
            for o in bpy.context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            view_layer.objects.active = obj
            
            # Export to STL
            bpy.ops.wm.stl_export(
                'EXEC_DEFAULT',
                filepath=stl_paths[i],
                export_selected_objects=True,
                use_batch=False,
                global_scale=STL_EXPORT_SCALE,
                apply_modifiers=apply_modifiers,
                evaluation_mode='DAG_EVAL_VIEWPORT',
            )
            
            # Load into meshlib
            loaded = mm.loadMesh(stl_paths[i])
            mesh = loaded.mesh if hasattr(loaded, 'mesh') else loaded
            meshlib_meshes.append(mesh)
            initial_face_counts.append(mesh.topology.numValidFaces())
            initial_vert_counts.append(mesh.topology.numValidVerts())
            
            # Clean up input STL immediately
            try:
                os.remove(stl_paths[i])
            except Exception:
                pass
        
        # Phase 2: Process meshes in parallel (pure meshlib, no Blender API).
        # ThreadPoolExecutor lets multiple C++ meshlib operations run concurrently
        # because MeshLib releases the GIL. Worker count is capped to avoid
        # saturating the GPU if CUDA offsets are in use.
        from .offset_utils import should_auto_decimate_faces
        from concurrent.futures import ThreadPoolExecutor, as_completed

        n_workers = min(len(meshlib_meshes), 4)

        def _process_one(args):
            i, mesh = args
            eff_res = effective_resolutions[i]
            out_mesh = operation_fn(mesh, eff_res)
            if auto_decimate:
                final_faces = out_mesh.topology.numValidFaces()
                do_decimate, target_faces = should_auto_decimate_faces(
                    initial_face_counts[i],
                    final_faces,
                    voxel_size=eff_res,
                    mode=effective_modes[i],
                    ratio=effective_ratios[i],
                )
                if do_decimate:
                    out_mesh = decimate_mesh(out_mesh, target_face_count=target_faces, resolution=eff_res)
            return i, out_mesh

        success_map = {}
        error_map = {}
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_process_one, (i, mesh)): i for i, mesh in enumerate(meshlib_meshes)}
            for future in as_completed(futures):
                i = futures[future]
                try:
                    idx, out_mesh = future.result()
                    success_map[idx] = out_mesh
                except Exception as exc:
                    error_map[i] = exc

        # Rebuild survivors in original order so subsequent phases stay aligned
        processed_meshes = []
        surviving_indices = []
        for i in range(len(meshlib_meshes)):
            if i in success_map:
                processed_meshes.append(success_map[i])
                surviving_indices.append(i)
            else:
                collapsed_objs.append((blender_objs[i], error_map.get(i, RuntimeError("unknown"))))

        # Phase 3: Save surviving meshes to STL in parallel (pure file I/O)
        output_stl_paths = []
        for j in range(len(processed_meshes)):
            src_idx = surviving_indices[j]
            fd, stl_path = tempfile.mkstemp(prefix=f"qi_out_{blender_objs[src_idx].name}_", suffix=".stl", dir=tmp_dir)
            os.close(fd)
            output_stl_paths.append(stl_path)

        def _save_one(args):
            mesh, stl_path = args
            try:
                mm.saveMesh(mesh, stl_path)
            except Exception:
                mm.saveMeshAs(mesh, stl_path)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            list(executor.map(_save_one, zip(processed_meshes, output_stl_paths)))

        # Phase 4: Import surviving results back to Blender
        result_objs = []
        for j, stl_path in enumerate(output_stl_paths):
            src_idx = surviving_indices[j]
            prev_objs = set(bpy.data.objects)

            # Import STL
            if hasattr(bpy.ops.wm, "stl_import"):
                bpy.ops.wm.stl_import('EXEC_DEFAULT', filepath=stl_path, global_scale=float(import_scale))
            else:
                bpy.ops.import_mesh.stl('EXEC_DEFAULT', filepath=stl_path, global_scale=float(import_scale))

            # Find the newly imported object
            new_objs = [obj for obj in bpy.data.objects if obj not in prev_objs]
            if not new_objs:
                new_objs = list(bpy.context.selected_objects)

            if new_objs:
                result_obj = new_objs[0]
                result_obj.name = blender_objs[src_idx].name + output_suffix
                result_objs.append(result_obj)

            # Clean up output STL
            try:
                os.remove(stl_path)
            except Exception:
                pass

        # Phase 5: Handle replace_original and build final results (survivors only)
        for j, result_obj in enumerate(result_objs):
            src_idx = surviving_indices[j]
            original_obj = blender_objs[src_idx]
            final_verts = processed_meshes[j].topology.numValidVerts()

            eff_res = effective_resolutions[src_idx]
            tag_index = _tag_to_index(nearest_resolution_tag(eff_res)) if eff_res is not None else -1
            face_tags = [tag_index] * len(result_obj.data.polygons)
            write_mesh_resolution_tags(result_obj.data, face_tags, remesh_voxel_size=eff_res)

            if replace_original:
                result_obj = replace_mesh_keep_transforms(original_obj, result_obj)
            set_object_resolution_label(result_obj, _dominant_tag_label(face_tags))

            results.append((result_obj, initial_vert_counts[src_idx], final_verts))
    
    finally:
        # Restore selection state (use names to avoid stale StructRNA references)
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        for name in prev_selection_names:
            if name in bpy.data.objects:
                bpy.data.objects[name].select_set(True)
        if prev_active_name and prev_active_name in bpy.data.objects:
            view_layer.objects.active = bpy.data.objects[prev_active_name]
    
    return results, collapsed_objs