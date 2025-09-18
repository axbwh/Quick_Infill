"""
Reusable mesh offset utilities for Quick Infill.
"""

from typing import Optional

from meshlib import mrmeshpy as mm
from meshlib import mrcudapy as mc


def cuda_offset(mesh: mm.Mesh, resolution: float, distance: float) -> mm.Mesh:
	"""
	CUDA-based general offset on a mesh.
	- resolution: voxel size used for offset grid
	- distance: positive grows, negative shrinks
	"""
	p = mm.GeneralOffsetParameters()
	p.voxelSize = float(resolution)
	p.signDetectionMode = mm.SignDetectionMode.HoleWindingRule
	p.fwn = mc.FastWindingNumber(mesh)
	return mm.generalOffsetMesh(mp=mesh, offset=float(distance), params=p)


def weighted_dist_shell(
	mesh_to_offset: mm.Mesh,
	reference_mesh: mm.Mesh,
	voxel_size: float,
	shrink_mult: float,
	max_vertices: Optional[int] = None,
	target_resolution: Optional[float] = None,
) -> mm.Mesh:
	"""
	Creates a variable-width shell based on distance to reference_mesh.
	Thicker where farther from reference, used to reduce occlusion when booleaned.
	
	Args:
		max_vertices: Maximum vertex count before decimation (default: 100M for safety)
	
	Automatically decimates mesh if too dense to prevent "vector too long" errors.
	"""
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


def compute_voxel_size(mesh: mm.Mesh, target_voxels: int, min_resolution: float) -> float:
	"""
	Suggest voxel size from mesh and clamp by the minimum resolution provided.
	"""
	suggested = mm.suggestVoxelSize(mm.MeshPart(mesh), float(target_voxels))
	return max(float(suggested), float(min_resolution))


def decimate_mesh(mesh: mm.Mesh, target_face_count: Optional[int] = None, reduction_ratio: Optional[float] = None, target_vertex_count: Optional[int] = None) -> mm.Mesh:
	"""
	Decimate mesh to reduce face count using mrmeshpy decimation.
	
	Args:
		mesh: Input mesh to decimate
		target_face_count: Target number of faces (if specified, takes priority)
		reduction_ratio: Ratio of faces to keep (0.0-1.0, e.g., 0.5 = keep 50% of faces)
		target_vertex_count: Target number of vertices (approximated via face reduction)
	
	Returns:
		Decimated mesh
	"""
	if target_face_count is None and reduction_ratio is None and target_vertex_count is None:
		reduction_ratio = 0.5  # Default to 50% reduction
	
	settings = mm.DecimateSettings()
	
	if target_face_count is not None:
		# Use target face count - calculate how many faces to delete
		current_faces = mesh.topology.numValidFaces()
		faces_to_delete = max(0, current_faces - int(target_face_count))
		settings.maxDeletedFaces = faces_to_delete
	elif target_vertex_count is not None:
		# Use target vertex count - estimate face reduction needed
		# Rough approximation: vertex/face ratio is usually around 0.5-2.0
		current_vertices = mesh.topology.numValidVerts()
		if current_vertices > target_vertex_count:
			vertex_ratio = float(target_vertex_count) / float(current_vertices)
			# Use slightly more aggressive face reduction to ensure vertex target is met
			face_ratio = vertex_ratio * 0.8  # Reduce faces more aggressively
			current_faces = mesh.topology.numValidFaces()
			faces_to_keep = max(1, int(current_faces * face_ratio))
			faces_to_delete = max(0, current_faces - faces_to_keep)
			settings.maxDeletedFaces = faces_to_delete
		else:
			# Already below target, no decimation needed
			settings.maxDeletedFaces = 0
	elif reduction_ratio is not None:
		# Use reduction ratio - calculate faces to delete
		current_faces = mesh.topology.numValidFaces()
		faces_to_keep = max(1, int(current_faces * float(reduction_ratio)))
		faces_to_delete = max(0, current_faces - faces_to_keep)
		settings.maxDeletedFaces = faces_to_delete
	
	# Apply decimation
	result = mm.decimateMesh(mesh, settings)
	
	# DecimateResult contains info about the operation, mesh is modified in-place
	print(f"[Quick Infill] Decimation: deleted {result.facesDeleted} faces, {result.vertsDeleted} vertices")
	
	# Return the modified mesh
	return mesh

