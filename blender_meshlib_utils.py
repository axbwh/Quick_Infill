import bpy
from meshlib import mrmeshpy as mm


def blender_to_meshlib(blender_obj):
    """
    Convert a Blender mesh object to a meshlib Mesh using MeshBuilder.
    """
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


def meshlib_to_blender_via_stl(meshlib_mesh: mm.Mesh, name: str = "Converted Mesh", import_scale: float = 0.1):
    """
    Save a meshlib mesh to a temporary STL and import it back into Blender at a given scale.
    Returns the imported Blender object.
    """
    import os
    import tempfile
    import bpy
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