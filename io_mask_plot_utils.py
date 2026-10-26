#python 39
# import segmentation_models as sm
import numpy as np
from PIL import Image
import glob
import os

import matplotlib.pyplot as plt
from pathlib import Path
import csv
import sys
import pydicom
import segmentation_models as sm
# from tensorflow.keras.models import load_model
from typing import List, Tuple

def dicom_to_array(dicom_file_path: str, frame = 0) -> Tuple[List[str], List[np.ndarray]]:
    '''
    imports a dicom prints the timestamp, resizes to 256x256, and takes the frame-th to export to image 
    arguments: the path for the dicom file, frame that will be converted to array defaults to first frame
    returns: images is the dicom file path, an array of the image that is 256x256x3 rgb uint8, the dicom acquisition time float unix utc
    '''
    images = []
    timestamp = []
    dicom_data = pydicom.dcmread(dicom_file_path)
    print(dicom_data.timestamp) #float weird number format
    
    print(dicom_data.pixel_array.shape)
    img = dicom_data.pixel_array[frame].astype(np.uint8) #the values are 0 - 255, so already 8 bit, 

    print(img.min(), img.max())
    plt.imshow(img)

    plt.axis('off')

    plt.show(block = False)

    r = dicom_data.SequenceOfUltrasoundREgions[0]
    size_data = {"rows": dicom_data.Rows,
                "cols": dicom_data.Columns,
                "frames": int(dicom_data.get("NumberOfFrames",1)),
                "spacing_mm": (abs(r.PhysicalDeltaY) * 10 * dicom_data.Rows,  #cm->mm
                                abs(r.PhysicalDeltaX) * 10 * dicom_data.Columns),
                }

    # array_rgb_frame = img.convert('RGB')# the pixel array is 112 slices/frames, 708x1016 image heightxwidth, 
    #3 rgb color image, 
    img_resized = np.array(Image.fromarray(img).resize((256, 256)))
    # pil_img = Image.fromarray(img).convert("RGB")

    timestamp.append(dicom_data.timestamp)
    images.append(dicom_file_path)
    return images, img_resized, timestamp, size_data


def create_mask_with_model(the_rgb_array, the_model_path= rf"C:\Users\rlong5\Desktop\us_unet\best_model.keras"):

    '''
    returns: 256x256 binary uint8 array of mask '''
    # Load model

    model = load_model(
        the_model_path,
        custom_objects={
            'dice_loss': sm.losses.DiceLoss(),
            'iou_score': sm.metrics.IOUScore(threshold=0.5),
            'f1-score': sm.metrics.FScore(threshold=0.5)
        }
    )
    print(model.input_shape)

    preprocess_input = sm.get_preprocessing('seresnet34')

    '''create new directory for each image directory'''
    x = preprocess_input(the_rgb_array)
    x = np.expand_dims(x, axis = 0) #what does this do

    #make prediction
    prediction = model.predict(x, verbose = 0) 
    #make mask binary
    pred_mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)
    
    #save mask
    # mask_save_path = os.path.join(output_root, "masks", rel_path)
    # print('save mask path in masks is %s' % mask_save_path)
    # os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)

    # Image.fromarray(pred_mask * 255).save(mask_save_path) #what does this do
    return pred_mask
    


def create_2_plot(image: np.ndarray, mask: np.ndarray):
    plt.close('all')

    # Convert binary uint8 masks (0/1) to 0/255
    if (
        mask.dtype == np.uint8
        and mask.size > 0
        and np.all(np.isin(mask, [0, 1]))
    ):
        mask = mask * 255

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    image_pil = Image.fromarray(image)
    mask_pil = Image.fromarray(mask)

    ax[0].imshow(image_pil)
    ax[0].set_title("Original") #could try to add image filename to title
    ax[0].axis("off")

    ax[1].imshow(mask_pil, cmap="gray")
    ax[1].set_title("Predicted Mask")
    ax[1].axis("off")
    plt.tight_layout()

    return fig

def create_mask_image_overlay_plot(image: np.ndarray, mask: np.ndarray, transparency = 0.3):#transparency is 0 (transcprent) to 1 = most opaque
    '''
    this fades array 2 by amount alpha 0 is transparent, 1 is most opaque
    arguments: image arrays of euqal size (here, 256 x 256 2d), alpha opaqueness 0 - 1 most, set = 0.30
    returns: figure handle'''
    plt.close('all')

    # Convert binary uint8 masks (0/1) to 0/255
    # if (
    #     mask.dtype == np.uint8
    #     and mask.size > 0
    #     and np.all(np.isin(mask, [0, 1]))
    # ):
    #     mask = mask * 255

    #use the image 
    image_with_mask = image.copy()
    # if there is a mask 
    mask_region = mask > 0
    image_with_mask[mask_region] = (
        (1 - transparency) * image_with_mask[mask_region]
        + transparency * mask[mask_region]).astype(np.uint8)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    image_pil = Image.fromarray(image_with_mask)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_with_mask)
    ax.axis("off")
    plt.show()

    ax[0].imshow(image_pil)
    ax[0].set_title("Image with Mask") #could try to add image filename to title
    ax[0].axis("off")

    plt.tight_layout()

    return fig

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

