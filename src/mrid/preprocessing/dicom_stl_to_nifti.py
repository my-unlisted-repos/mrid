import os
import warnings

import numpy as np
import SimpleITK as sitk
import trimesh


def dicom_stl_to_nifti(dicom_dir: str, stl_path: str, output_ct_path: str, output_seg_path: str, verbose:bool=False):
    """
    Converts a DICOM series and an STL segmentation file into two spatially
    aligned NIfTI files (CT image and segmentation mask).

    Args:
        dicom_dir (str): Path to the directory containing DICOM slices.
        stl_path (str): Path to the STL segmentation file. The STL coordinates
                        MUST be in the same coordinate system as the DICOM
                        Patient Coordinate System (usually LPS).
        output_ct_path (str): Path where the CT NIfTI file will be saved.
        output_seg_path (str): Path where the segmentation NIfTI file will be saved.

    Raises:
        FileNotFoundError: If the DICOM directory or STL file does not exist.
        RuntimeError: If DICOM loading or processing fails.
        Exception: For other potential errors during processing.
    """
    def verbose_print(*args,**kwargs):
        if verbose: print(*args,**kwargs)

    verbose_print(f"Processing DICOM directory: {dicom_dir}")
    verbose_print(f"Processing STL file: {stl_path}")

    # --- 1. Input Validation ---
    if not os.path.isdir(dicom_dir):
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")
    if not os.path.isfile(stl_path):
        raise FileNotFoundError(f"STL file not found: {stl_path}")

    # --- 2. Load DICOM Series ---
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    if not dicom_names:
        raise RuntimeError(f"No DICOM series found in directory: {dicom_dir}")

    reader.SetFileNames(dicom_names)
    ct_image = reader.Execute()
    verbose_print("DICOM series loaded successfully.")

    # --- 3. Extract CT Metadata ---
    ct_origin = np.array(ct_image.GetOrigin())
    ct_spacing = np.array(ct_image.GetSpacing())
    # ct_direction = np.array(ct_image.GetDirection()).reshape(3, 3)
    ct_size = np.array(ct_image.GetSize()) # Order: x, y, z
    ct_shape_xyz = ct_size                    # For clarity
    ct_shape_zyx = ct_size[::-1]              # NumPy shape: z, y, x
    verbose_print(f"CT Metadata - Size: {ct_size}, Spacing: {ct_spacing}, Origin: {ct_origin}")


    # --- 4. Load STL Mesh ---
    # Ensure STL is loaded using units consistent with DICOM (usually mm)
    mesh = trimesh.load_mesh(stl_path)
    # Optional: Check if mesh is watertight, which is ideal for .contains()
    if not mesh.is_watertight:
        warnings.warn(f"Warning: STL mesh '{stl_path}' is not watertight. Voxelization using 'contains' might be inaccurate.")
        # Optionally try to fix it, though this can be slow or alter geometry:
        # mesh.fill_holes()
        # if not mesh.is_watertight:
        #     print("Warning: Failed to make mesh watertight after filling holes.")

    verbose_print(f"STL mesh loaded: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")
    # --- Assume STL coordinates are in DICOM's Patient Coordinate System (LPS) ---
    # If not, a transformation matrix would need to be applied to mesh.vertices here
    # e.g., mesh.apply_transform(transformation_matrix)


    # --- 5. Generate Voxel Center Coordinates for the CT Grid ---
    # Create coordinate arrays for each dimension
    x_coords = ct_origin[0] + np.arange(ct_shape_xyz[0]) * ct_spacing[0]
    y_coords = ct_origin[1] + np.arange(ct_shape_xyz[1]) * ct_spacing[1]
    z_coords = ct_origin[2] + np.arange(ct_shape_xyz[2]) * ct_spacing[2]

    # Use meshgrid to create a grid of coordinates
    # Note the 'ij' indexing to match the z, y, x array structure
    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')

    # Stack coordinates into a (N, 3) array where N = Z*Y*X
    voxel_centers_xyz = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
    verbose_print(f"Generated {voxel_centers_xyz.shape[0]} voxel center coordinates.")

    # --- 6. Voxelize STL using mesh.contains ---
    # This checks which voxel center points fall inside the mesh volume
    verbose_print("Checking which voxel centers are inside the STL mesh (this may take time)...")

    voxel_mask_flat = mesh.contains(voxel_centers_xyz)
    verbose_print("Voxel containment check complete.")


    # --- 7. Create Segmentation Array ---
    # Reshape the flat boolean mask back into the 3D CT shape (z, y, x)
    segmentation_array = voxel_mask_flat.reshape(ct_shape_zyx).astype(np.uint8) # Use uint8 for masks
    verbose_print(f"Created segmentation array with shape {segmentation_array.shape} and {segmentation_array.sum()} foreground voxels.")

    # --- 8. Create Segmentation SimpleITK Image ---
    segmentation_image = sitk.GetImageFromArray(segmentation_array)

    # --- 9. Copy Metadata from CT to Segmentation ---
    # This is crucial for ensuring they are in the same space!
    segmentation_image.SetOrigin(ct_image.GetOrigin())
    segmentation_image.SetSpacing(ct_image.GetSpacing())
    segmentation_image.SetDirection(ct_image.GetDirection())
    verbose_print("Copied spatial metadata from CT to segmentation image.")

    # --- 10. Save NIfTI Files ---
    verbose_print(f"Saving CT NIfTI to: {output_ct_path}")
    sitk.WriteImage(ct_image, output_ct_path)

    verbose_print(f"Saving Segmentation NIfTI to: {output_seg_path}")
    sitk.WriteImage(segmentation_image, output_seg_path)
    verbose_print("NIfTI files saved successfully.")


# dicom_stl_to_nifti("027", "an_027.stl", "027 v2.nii.gz", "027 seg.nii.gz")