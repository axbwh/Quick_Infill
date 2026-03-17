"""
Reusable mesh offset utilities for Quick Infill.
"""

from typing import Optional
# Auto-decimate: decimate back to initial if mesh grew at all
# This prevents both progressive detail loss AND progressive growth


def should_auto_decimate_faces(initial_faces: int, final_faces: int) -> tuple:
    """
    Determine if auto-decimation should be applied and calculate target face count.
    
    Simple logic: if the mesh grew, decimate back to initial count.
    This prevents progressive changes in either direction.
    
    Args:
        initial_faces: Face count before the operation
        final_faces: Face count after the operation
    
    Returns:
        tuple: (should_decimate: bool, target_face_count: int)
    """
    if final_faces > initial_faces:
        # Mesh grew - decimate back to initial
        return True, initial_faces
    
    # Mesh stayed same or shrunk - no decimation needed
    return False, final_faces

def cuda_offset(mesh, resolution: float, distance: float):
	"""
	CUDA-based general offset on a mesh.
	- resolution: voxel size used for offset grid
	- distance: positive grows, negative shrinks
	"""
	from .meshlib_utils import get_meshlib
	mm, mc = get_meshlib()
	
	p = mm.GeneralOffsetParameters()
	p.voxelSize = float(resolution)
	p.signDetectionMode = mm.SignDetectionMode.HoleWindingRule
	p.fwn = mc.FastWindingNumber(mesh)
	return mm.generalOffsetMesh(mp=mesh, offset=float(distance), params=p)


def weighted_dist_shell(
	mesh_to_offset,
	reference_mesh,
	voxel_size: float,
	shrink_mult: float,
	max_vertices: Optional[int] = None,
	target_resolution: Optional[float] = None,
):
	"""
	Creates a variable-width shell based on distance to reference_mesh.
	Thicker where farther from reference, used to reduce occlusion when booleaned.
	
	Args:
		shrink_mult: Multiplier for shell thickness based on distance
		max_vertices: Maximum vertex count before decimation (default: 100M for safety)
	
	Automatically decimates mesh if too dense to prevent "vector too long" errors.
	"""
	from .meshlib_utils import get_meshlib
	mm, _ = get_meshlib()
	
	# Check if mesh is too dense for vector operations
	# Use provided limit or fallback to conservative default
	MAX_VERTICES = max_vertices if max_vertices is not None else 1_000_000
	
	working_mesh = mesh_to_offset
	working_ref = reference_mesh
	
	if mesh_to_offset.points.size() > MAX_VERTICES:
		# Calculate reduction ratio to get under the limit
		target_vertices = MAX_VERTICES // 2  # Use half the limit for safety
		reduction_ratio = target_vertices / mesh_to_offset.points.size()
		working_mesh = decimate_mesh(mesh_to_offset, reduction_ratio=reduction_ratio)
		print(f"[Quick Infill] Decimated mesh from {mesh_to_offset.points.size()} to {working_mesh.points.size()} vertices")
		
    	
	if reference_mesh.points.size() > MAX_VERTICES:
		# Also decimate reference mesh if needed
		target_vertices = MAX_VERTICES // 2
		reduction_ratio = target_vertices / reference_mesh.points.size()
		working_ref = decimate_mesh(reference_mesh, reduction_ratio=reduction_ratio)
		print(f"[Quick Infill] Decimated reference mesh from {reference_mesh.points.size()} to {working_ref.points.size()} vertices")
	
	sd = mm.findSignedDistances(working_ref, working_mesh)
	W = [abs(d) * float(shrink_mult) for d in sd]

	n = working_mesh.points.size()
	scalars = mm.VertScalars(n)
	for i in range(n):
		scalars.vec[i] = W[i]

	# Calculate adaptive voxel size based on target resolution to prevent memory explosion
	max_weight = max(scalars.vec) if scalars.vec else 1.0
	
	# Use target resolution to determine safe voxel size for weighted shell operation
	if target_resolution is not None and max_vertices is not None:
		# Use meshlib's built-in voxel size suggestion with target resolution
		target_voxel_count = min(int(max_vertices), int(target_resolution))


		suggested_voxel_size = mm.suggestVoxelSize(mm.MeshPart(working_mesh), target_voxel_count)
		adaptive_voxel_size = max(float(voxel_size), suggested_voxel_size)
		
		print(f"[Quick Infill] WeightedShell voxel adaptation: {voxel_size:.4f} → {adaptive_voxel_size:.4f} (target_res: {target_resolution}, weight: {max_weight:.2f})")
	else:
		# Fallback to original voxel size if target resolution not available
		adaptive_voxel_size = float(voxel_size)

	params = mm.WeightedShell.ParametersMetric()
	params.voxelSize = float(adaptive_voxel_size)
	params.offset = float(adaptive_voxel_size)
	params.dist.maxWeight = max_weight

	return mm.WeightedShell.meshShell(working_mesh, scalars, params)


def compute_voxel_size(mesh, target_voxels: int, min_resolution: float) -> float:
	"""
	Suggest voxel size from mesh and clamp by the minimum resolution provided.
	"""
	from .meshlib_utils import get_meshlib
	mm, _ = get_meshlib()
	
	suggested = mm.suggestVoxelSize(mm.MeshPart(mesh), float(target_voxels))
	return max(float(suggested), float(min_resolution))


def decimate_mesh(mesh, target_face_count: Optional[int] = None, reduction_ratio: Optional[float] = None, max_error: Optional[float] = None, resolution: Optional[float] = None):
	"""
	Decimate mesh to reduce face count using mrmeshpy decimation.
	
	Args:
		mesh: Input mesh to decimate
		target_face_count: Target number of faces
		reduction_ratio: Ratio of faces to keep (0.0-1.0, e.g., 0.5 = keep 50% of faces)
		max_error: Maximum geometric deviation allowed (in mesh units). 
		resolution: Voxel/remesh resolution - if provided, maxError defaults to 0.5 * resolution
	
	Returns:
		Decimated mesh (modified in-place)
	"""
	from .meshlib_utils import get_meshlib
	mm, _ = get_meshlib()
	
	current_faces = mesh.topology.numValidFaces()
	
	if target_face_count is None and reduction_ratio is None:
		reduction_ratio = 0.5  # Default to 50% reduction
	
	settings = mm.DecimateSettings()
	
	# Calculate target face count
	if target_face_count is not None:
		target = int(target_face_count)
	elif reduction_ratio is not None:
		target = max(1, int(current_faces * float(reduction_ratio)))
	else:
		target = current_faces
	
	# MeshLib decimation uses maxDeletedFaces (limit on faces to delete)
	# Calculate how many faces need to be deleted to reach target
	faces_to_delete = max(0, current_faces - target)
	settings.maxDeletedFaces = faces_to_delete
	
	# Set maxError to preserve detail - limits geometric deviation
	# Note: MeshLib mesh is scaled 10x from Blender, so resolution must be scaled too
	if max_error is not None:
		settings.maxError = float(max_error)
	elif resolution is not None:
		# Resolution is in Blender units, scale to match MeshLib (10x)
		# Allow error equal to voxel size since voxel ops already quantize to this
		settings.maxError = float(resolution) * 10.0
	else:
		# Fallback: estimate from bounding box (0.5% of diagonal)
		bbox = mesh.computeBoundingBox()
		diagonal = (bbox.max - bbox.min).length()
		settings.maxError = diagonal * 0.005
	
	# Use parallel processing for better performance
	settings.packMesh = True
	
	# Apply decimation
	result = mm.decimateMesh(mesh, settings)
	
	final_faces = mesh.topology.numValidFaces()
	print(f"[Quick Infill] Decimation: {current_faces} → {final_faces} faces (maxError: {settings.maxError:.4f}, {result.vertsDeleted} verts removed)")
	
	return mesh

