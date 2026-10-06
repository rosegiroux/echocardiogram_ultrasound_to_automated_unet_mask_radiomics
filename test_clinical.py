#python 39
# import segmentation_models as sm
import numpy as np
from PIL import Image
import glob
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import sys
import pydicom

from tensorflow.keras.models import load_model
from typing import List, Tuple

#todo: add nifti save as mask and image together. 

#loading the trained unet and testing on some of clinical images



def img_to_arr(image_dir: str) -> Tuple[List[np.ndarray], List[str]]:
    '''load all images in a directory and convert to rgb np array
    args: path to directory containing png images 
    return: list of rgb np arrays
    filenames: file paths of each png
    resizes image to 256 x 256'''
    images = []
    filepaths = []
    for img_path in sorted(glob.glob(os.path.join(image_dir, "**", "*.png"),recursive=True)): #collect relative path
        img = np.array(Image.open(img_path).convert('RGB'))
        img_resized = np.array(Image.fromarray(img).resize((256, 256)))
        rel_path = os.path.relpath(img_path, image_dir)

        images.append(img_resized)
        filepaths.append(img_path)
        rel_paths.append(rel_path)
    return images, filepaths, rel_paths

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



    # array_rgb_frame = img.convert('RGB')# the pixel array is 112 slices/frames, 708x1016 image heightxwidth, 
    #3 rgb color image, 
    img_resized = np.array(Image.fromarray(img).resize((256, 256)))
    # pil_img = Image.fromarray(img).convert("RGB")

    timestamp.append(dicom_data.timestamp)
    images.append(dicom_file_path)
    return images, img_resized, timestamp

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

if __name__ == "__main__":
    # Load model

    model = load_model(
        rf"C:\Users\rlong5\Desktop\us_unet\best_model.keras",
        custom_objects={
            'dice_loss': sm.losses.DiceLoss(),
            'iou_score': sm.metrics.IOUScore(threshold=0.5),
            'f1-score': sm.metrics.FScore(threshold=0.5)
        }
    )
    print(model.input_shape)

    images = []
    masks = []
    filenames = []
    samples = []
    rel_paths = []


    image_dir = rf"C:\Users\rlong5\SAMUS\test_images"
    #"C:\Users\rlong5\SAMUS\test_images\Echo_amyloid_HCM_pngs\5278115"#"C:\Users\rlong5\SAMUS\test_images\Echo_amyloid_HCM_pngs"


    output_root = Path(rf"C:\Users\rlong5\Desktop\results_20260715_full")
    output_root.mkdir(exist_ok=True)

    images, filenames, rel_paths = img_to_arr(image_dir)

    X = np.stack(images)
    print(X.shape)
    print(X.dtype)
    # training set: (2349, 256, 256, 3)
    # here: (9, 573, 707, 3)
    # uint8
    #uint8

    preprocess_input = sm.get_preprocessing('seresnet34')

    for img, rel_path in zip(X, rel_paths):
    #the seresnet34 encoder downsaples several times 707->353->177...45 then the decoder upsamples 45-> 90->180..720

        '''create new directory for each image directory'''
        x = preprocess_input(img)
        x = np.expand_dims(x, axis = 0) #what does this do

        #make prediction
        prediction = model.predict(x, verbose = 0) 
        #make mask binary
        pred_mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)
        
        #save mask
        mask_save_path = os.path.join(output_root, "masks", rel_path)
        print('save mask path in masks is %s' % mask_save_path)
        os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)

        Image.fromarray(pred_mask * 255).save(mask_save_path) #what does this do

        #create plot 
        figure = create_2_plot(img, pred_mask) #both are ndarray

        #save it to a directory called plot in output root
        og_mask_path = os.path.join(output_root, "plots", rel_path)

        og_mask_path = os.path.splitext(og_mask_path)[0] + ".png"

        os.makedirs(os.path.dirname(og_mask_path), exist_ok=True)

        figure.savefig(og_mask_path, bbox_inches="tight")
        plt.close(figure)

        print("mask", mask_save_path)
        print("original + mask:", og_mask_path)

        # Write record
        #
        samples.append({
            "image_path": rel_path,
            "mask_path": os.path.relpath(mask_save_path, output_root),
            "plot_path": os.path.relpath(og_mask_path, output_root)
        })
        del x
        del prediction
        del pred_mask
        del figure



