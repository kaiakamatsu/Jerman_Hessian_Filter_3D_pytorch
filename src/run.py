import numpy as np
from scipy import ndimage as ndi
from .hessian_filter_gpu import vesselness3D

def run_hessian_vesselness_multiscale_python_gpu(volume_segmented_zyx, voxel_dimensions_xyz, lungmask_zyx, sigmas=[0.9, 1.6, 2.3, 3.0], tau=1.0, brightondark=True, lung_surface_thickness=4):
    """
    Jerman Enhancement Filter 
    T. Jerman, F. Pernus, B. Likar, Z. Spiclin, "Enhancement of Vascular Structures in 3D and 2D Angiographic Images", IEEE Transactions on Medical Imaging, 35(9), p. 2107-2118 (2016), doi={10.1109/TMI.2016.2550102} 

    I implemented this python version that can support GPU acceleration using pytorch. 
    """

    spacing_zyx = np.array([voxel_dimensions_xyz[2], voxel_dimensions_xyz[1], voxel_dimensions_xyz[0]], dtype=np.float32)

    vesselness = vesselness3D(
        volume_segmented_zyx,
        sigmas=sigmas,
        spacing=spacing_zyx,
        tau=float(tau),
        brightondark=bool(brightondark),
        device="cuda",
        verbose=False,
    )

    vesselness_cpu = vesselness.detach().float().cpu().numpy()

    surf = extract_lung_surface(lungmask_zyx, thickness=lung_surface_thickness)
    vesselness_cpu[surf == 1] = 0.0

    # normalize to 0-255
    if vesselness_cpu.min() != vesselness_cpu.max():
        vesselness_cpu = (vesselness_cpu - vesselness_cpu.min()) / (vesselness_cpu.max() - vesselness_cpu.min())
        vesselness_ready = (vesselness_cpu * 255).astype(np.uint8)
    else: # should not happen, but if this failure case occurs, return the original volume
        vesselness_ready = vesselness_cpu

    return vesselness_ready # vesselness between 0 and 255

def extract_lung_surface(binary_mask, thickness=4):
    """
    Returns a mask containing only the surface voxels of the lung.
    
    Parameters:
    -----------
    binary_mask : np.ndarray
        3D binary array (1 for lung, 0 for background).
    thickness : int
        How many voxels deep the surface should be. 
        Higher values create a thicker "shell."
    """
    # Ensure it's a boolean array
    mask = binary_mask.astype(bool)
    
    # Create a structure element (3D cross)
    # This defines how the erosion "looks" at neighbors
    struct = ndi.generate_binary_structure(3, 1)
    
    # Erode the mask: this shrinks the 1s by 'thickness' voxels
    eroded_mask = ndi.binary_erosion(mask, structure=struct, iterations=thickness)
    
    # The surface is the original mask MINUS the shrunk version
    # (Exclusive OR also works: mask ^ eroded_mask)
    surface_mask = mask & ~eroded_mask
    
    return surface_mask.astype(np.uint8)