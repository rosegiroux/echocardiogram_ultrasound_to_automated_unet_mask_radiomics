#! C:/Users/rlong5/AppData/Local/anaconda3/envs/python39/python

from test_clinical import dicom_to_array 
from test_clinical import create_mask_with_model
from test_clinical import create_2_plot
from test_clinical import create_mask_image_overlay_plot
from connect_masks_with_arc import list_output_masks
from connect_masks_with_arc import save_nifti
import os
import cv2
import numpy as np

import pandas #pandas   2.3.1

# 1) For each line
# Go to directory
# Get dicom-> array
# Mask
# Output maks

data_directory = rf'C:\Users\rlong5\Desktop\Echo\1_Raw Data\20260911_files\0897724'

#there is two dicoms 
#column 6 is the path to the file
# 1/2 are 2ch, 1/2 are 4ch
# C:\Users\rlong5\Desktop\Echo\1_Raw Data\20260911_files\0897724
#choose the one with 4ch
#^start w 4ch

#for each line, go to directory
file_list_from_excel = pandas.read_excel(rf"C:\Users\rlong5\Desktop\Echo\1_Raw Data\OneDrive_File_Inventory_with_Diagnoses3.xlsm")
print(file_list_from_excel.columns)
    
for path in file_list_from_excel.iloc[12:13, 5]:
    fig_output = os.path.dirname(path)
    #make fiile figure handle from 
    # the base of fig_out_path
    figout_filename = os.path.join(fig_output, os.path.splitext(os.path.basename(path))[0])
    if '2ch' in path: #skip if 2ch in file name
        continue
    print(path)
    result, img_array, time = dicom_to_array(path) #imag_array is 256x256, 8 bit
    pred_mask = create_mask_with_model(img_array)

#create image and mask overlay plot
    fig_image_mask_overlay = create_mask_image_overlay_plot(img_array, pred_mask, transparency = 0.3)#transparency is 0 (transcprent) to 1 = most opaque
    fig_image_mask_overlay.savefig(figout_filename, dip = 300, bbox_inches = "tight")

    
    #create image and mask plot seperate panels
    fig = create_2_plot(img_array, pred_mask) #d

    #rename to make figure handle of image, mask
    fig_output = os.path.dirname(path)
    #rename to make figure handle of image, mask, image+mask
    fig_output2 = os.path.join(fig_output, "image_mask.png")
    fig.savefig(fig_output2, dpi=300, bbox_inches="tight")#path contains dicom os.
    labels, objects = list_output_masks(pred_mask)
    for obj in objects:
        fig.axes[1].annotate(
            str(obj["id"]),
            xy=obj["center"],
            color="red",
            fontsize = 12
        )


    #all of test_dicom
    #makes two .nii files

    fig.canvas.draw()
    fig.show()
    input("Press Enter to continue...")


    print(len(objects))
    print(labels, objects)

    #todo: user input select object number.
    target_id = int(input("Enter object ID: "))
    

    obj = next(o for o in objects if o["id"] == target_id)

    x = obj["bbox"]["x"]
    y = obj["bbox"]["y"]
    w = obj["bbox"]["w"]
    h = obj["bbox"]["h"]

    # Create all-False array
    mask_id3_bbox = np.zeros_like(pred_mask, dtype=bool)

    # Copy values only within the bounding box
    mask_id3_bbox[y:y+h, x:x+w] = pred_mask[y:y+h, x:x+w]

    save_nifti(os.path.dirname(path), #rf"C:\Users\rlong5\Desktop\Echo\1_Raw Data\20260911_files\0897724", #folder =
        img_array,
        mask_id3_bbox,
        img_nii_name="original_image",
        mask_nii_name="selected_mask_unet",
        text = time
    )
#then radiomics
#go trough same directory

