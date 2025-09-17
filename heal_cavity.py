# CUDA offset of selected mesh → adds "<original>Infill"
# Saves the initial vertex count as a variable at the start, and logs it.

from meshlib import mrmeshpy as mm
from meshlib import mrviewerpy as mv
from meshlib import mrcudapy as mc

# ---- Options ----
RESOLUTION = 0.1   # voxel size
GROW    = 2.0  # +grow / -shrink
SHRINK_MULT = 1.5
SMALLEST_SIZE=RESOLUTION*2
TARGET_VOXELS = 2000000
MAX_THICKNESS_FACTOR = 0.25      # at most 25% of bbox diagonal

print(f"Cuda available Memory ={mc.getCudaAvailableMemory()}")
print(f"Cuda is Available  ={mc.isCudaAvailable ()}")

def cuda_offset(mesh, resolution, distance):
    p = mm.GeneralOffsetParameters()
    p.voxelSize = float(resolution)
    p.signDetectionMode = mm.SignDetectionMode.HoleWindingRule
    p.fwn = mc.FastWindingNumber(mesh)
    return mm.generalOffsetMesh(mp=mesh, offset=float(distance), params=p)

def weighted_dist_shell(mesh_to_offset: mm.Mesh, reference_mesh: mm.Mesh, voxel_size=RESOLUTION, shrink_mult=SHRINK_MULT) -> mm.Mesh:
    """
    Calculates distance between Original Mesh and infill Mesh
    Creates a variable width shell, with parts further from the mesh thicker
    When booleaned with the infill mesh, this results in shrinking the parts that are occluding detail
    Since  problematic cavities are expected to be further from the infill surfaces than 2*unsigned distance they are not affected
    """
    # Signed distances (per vertex of A relative to B)

    sd = mm.findSignedDistances(reference_mesh, mesh_to_offset)  # iterable VertScalars
    # Outside-only distances W_i = max(d_i, 0)
    W = [abs(d)*shrink_mult for d in sd]

    n = mesh_to_offset.points.size()
    scalars = mm.VertScalars(n)
    for i in range(n):
        scalars.vec[i] =  W[i]

    params = mm.WeightedShell.ParametersMetric()
    params.voxelSize = voxel_size
    params.offset = voxel_size
    params.dist.maxWeight = max(scalars.vec)  # must be the maximum provided weight (= maxW)

    return mm.WeightedShell.meshShell(mesh_to_offset, scalars, params)


# -- Start: get selection and save initial vertex count
src_mesh = mv.getSelectedMeshes()[0]
INITIAL_VERTEX_COUNT = src_mesh.topology.numValidVerts()   # <— saved variable
print(f"Initial vertex count: {INITIAL_VERTEX_COUNT}")

# object name for output
obj_name = mv.getSelectedObjects()[0].name()

vox = max(mm.suggestVoxelSize(src_mesh, TARGET_VOXELS), RESOLUTION)
print(f"Voxel Size: {vox}")
# offset + add to scene
grow_mesh = cuda_offset(src_mesh, vox, GROW)
shrink_mesh = cuda_offset(grow_mesh, vox, -GROW)
shell_mesh = weighted_dist_shell(shrink_mesh, src_mesh,vox, SHRINK_MULT)
# trim_mesh = mm.voxelBooleanSubtract(shrink_mesh, shell_mesh, RESOLUTION)
trim_mesh = mm.boolean(shrink_mesh, shell_mesh, mm.BooleanOperation.DifferenceAB).mesh

ss_mesh_shrink = cuda_offset(trim_mesh, vox, -vox)
ss_mesh_grow = cuda_offset(ss_mesh_shrink, vox, vox)


out_mesh = ss_mesh_grow
# out_mesh =  trim_mesh
mv.addMeshToScene(out_mesh, obj_name + "Infill")
