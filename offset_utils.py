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
) -> mm.Mesh:
	"""
	Creates a variable-width shell based on distance to reference_mesh.
	Thicker where farther from reference, used to reduce occlusion when booleaned.
	"""
	sd = mm.findSignedDistances(reference_mesh, mesh_to_offset)
	W = [abs(d) * float(shrink_mult) for d in sd]

	n = mesh_to_offset.points.size()
	scalars = mm.VertScalars(n)
	for i in range(n):
		scalars.vec[i] = W[i]

	params = mm.WeightedShell.ParametersMetric()
	params.voxelSize = float(voxel_size)
	params.offset = float(voxel_size)
	params.dist.maxWeight = max(scalars.vec)

	return mm.WeightedShell.meshShell(mesh_to_offset, scalars, params)


def compute_voxel_size(mesh: mm.Mesh, target_voxels: int, min_resolution: float) -> float:
	"""
	Suggest voxel size from mesh and clamp by the minimum resolution provided.
	"""
	suggested = mm.suggestVoxelSize(mm.MeshPart(mesh), float(target_voxels))
	return max(float(suggested), float(min_resolution))

