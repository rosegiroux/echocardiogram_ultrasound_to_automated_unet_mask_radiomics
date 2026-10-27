#python39

from pathlib import Path
import os
import cv2
import nibabel as nib
import numpy as np
import sys


def save_nifti(path_to_export_nifti,
    image,
    mask,
    img_nii_name="original_image",
    mask_nii_name="first_mask_unet",
    # nii.affine =None,
    text = None): #connected_mask, last: connected_mask_nii_name="connected_mask"
    """
    Save image and masks as NIfTI files.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    mask : np.ndarray
        Original mask.
    connected_mask : np.ndarray
        Connected mask.
    """

    os.makedirs(path_to_export_nifti, exist_ok=True)

    header = nib.Nifti1Header()
    if text:
        ext = nib.nifti1.Nifti1Extension(
        code="comment",
        content=str(text).encode("utf-8")
        )

        header.extensions.append(ext)

    nib.save(nib.Nifti1Image(image.astype(np.float32), None, header),
        os.path.join(path_to_export_nifti, img_nii_name))

    nib.save(nib.Nifti1Image(mask.astype(np.uint8), None, header),
        os.path.join(path_to_export_nifti, mask_nii_name))

    # nib.save(nib.Nifti1Image(connected_mask.astype(np.uint8), None, header),
    #     os.path.join(path_to_export_nifti, connected_mask_nii_name))

