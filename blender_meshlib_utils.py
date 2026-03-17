import bpy


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


def blender_to_meshlib_via_stl(blender_obj, tmp_dir=None):
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

        # Export selection to STL using Blender 4.5 native operator
        res = bpy.ops.wm.stl_export(
            'EXEC_DEFAULT',
            filepath=stl_path,
            export_selected_objects=True,
            use_batch=False,
            global_scale=10.0,
            apply_modifiers=True,
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


def process_mesh_operation(blender_obj, operation_fn, output_suffix, auto_decimate=False, import_scale=0.1, replace_original=False, resolution=None):
    """
    Generic wrapper for mesh operations: import -> process -> optional decimate -> export.
    
    Args:
        blender_obj: Source Blender mesh object
        operation_fn: Function that takes meshlib mesh and returns processed meshlib mesh
        output_suffix: Suffix for the output object name (e.g., "_Grown")
        auto_decimate: If True, decimate output to match initial vertex count
        import_scale: Scale factor for STL import (default 0.1)
        replace_original: If True, replace the original object's mesh data instead of creating new object
    
    Returns:
        tuple: (output_blender_obj, initial_vertex_count, final_vertex_count)
    """
    from .offset_utils import decimate_mesh, should_auto_decimate_faces
    
    obj_name = blender_obj.name
    
    # Convert to meshlib
    src_mesh = blender_to_meshlib_via_stl(blender_obj)
    initial_face_count = src_mesh.topology.numValidFaces()
    initial_vertex_count = src_mesh.topology.numValidVerts()
    
    # Apply the operation
    out_mesh = operation_fn(src_mesh)
    
    # Auto decimate if enabled - only when significant face growth occurred
    final_face_count = out_mesh.topology.numValidFaces()
    if auto_decimate:
        do_decimate, target_faces = should_auto_decimate_faces(initial_face_count, final_face_count)
        if do_decimate:
            out_mesh = decimate_mesh(out_mesh, target_face_count=target_faces, resolution=resolution)
    
    final_vertex_count = out_mesh.topology.numValidVerts()
    
    # Convert back to Blender
    result_obj = meshlib_to_blender_via_stl(out_mesh, obj_name + output_suffix, import_scale=import_scale)
    
    # If replace_original is enabled, swap mesh data and delete the temp object
    if replace_original:
        result_obj = replace_mesh_keep_transforms(blender_obj, result_obj)
    
    return result_obj, initial_vertex_count, final_vertex_count


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
    
    # Clean up old mesh data if it has no users
    if old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)
    
    # Reselect the original object and make it active
    original_obj.select_set(True)
    bpy.context.view_layer.objects.active = original_obj
    
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
    
    if not blender_objs:
        return []
    
    results = []
    tmp_dir = tempfile.gettempdir()
    
    # Save current selection state once
    view_layer = bpy.context.view_layer
    prev_active = view_layer.objects.active
    prev_selection = [obj for obj in bpy.context.selected_objects]
    
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
                global_scale=10.0,
                apply_modifiers=True,
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
        
        # Phase 2: Process all meshes through the operation (pure meshlib, no Blender calls)
        from .offset_utils import should_auto_decimate_faces
        processed_meshes = []
        for i, mesh in enumerate(meshlib_meshes):
            out_mesh = operation_fn(mesh)
            
            # Auto decimate if enabled - only when significant face growth occurred
            if auto_decimate:
                final_faces = out_mesh.topology.numValidFaces()
                do_decimate, target_faces = should_auto_decimate_faces(initial_face_counts[i], final_faces)
                if do_decimate:
                    out_mesh = decimate_mesh(out_mesh, target_face_count=target_faces, resolution=resolution)
            
            processed_meshes.append(out_mesh)
        
        # Phase 3: Export all processed meshes to STL and import back to Blender
        output_stl_paths = []
        for i, mesh in enumerate(processed_meshes):
            fd, stl_path = tempfile.mkstemp(prefix=f"qi_out_{blender_objs[i].name}_", suffix=".stl", dir=tmp_dir)
            os.close(fd)
            output_stl_paths.append(stl_path)
            
            # Save meshlib mesh to STL
            try:
                mm.saveMesh(mesh, stl_path)
            except Exception:
                mm.saveMeshAs(mesh, stl_path)
        
        # Phase 4: Import all results back to Blender
        result_objs = []
        for i, stl_path in enumerate(output_stl_paths):
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
                result_obj.name = blender_objs[i].name + output_suffix
                result_objs.append(result_obj)
            
            # Clean up output STL
            try:
                os.remove(stl_path)
            except Exception:
                pass
        
        # Phase 5: Handle replace_original and build final results
        for i, result_obj in enumerate(result_objs):
            original_obj = blender_objs[i]
            final_verts = processed_meshes[i].topology.numValidVerts()
            
            if replace_original:
                result_obj = replace_mesh_keep_transforms(original_obj, result_obj)
            
            results.append((result_obj, initial_vert_counts[i], final_verts))
    
    finally:
        # Restore selection state
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        for obj in prev_selection:
            if obj and obj.name in bpy.data.objects:
                obj.select_set(True)
        if prev_active and prev_active.name in bpy.data.objects:
            view_layer.objects.active = prev_active
    
    return results