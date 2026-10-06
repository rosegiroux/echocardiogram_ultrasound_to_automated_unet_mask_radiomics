import os
import pandas as pd 
import numpy as mp
import matplotlib.pyplot as mlt 
import nibabel as nib
import SimpleITK as sitk

# data_dir = rf'C:\Users\rlong5\Desktop\Echo\1_Raw Data\CAMUS_public\CAMUS_public\database_nifti\patient0001'
def list_nii_parameter(data_dir):
    '''
    summarizes size of .nii
    arguments: full path to nii file
    returns: prints size'''

    image = nib.load(data_dir)

    # image.orthoview()   

    data = image.get_fdata()
    mlt.imshow(data[:,:,1])
    mlt.show()
    hdr = image.header
    print('heres the voxel size')
    print(hdr.get_zooms())
    print(hdr.get_xyzt_units())
    print('the nifti image shape is ')#, image.GetSize())
    print(image.shape)
# 

    image = sitk.ReadImage(data_dir)


# path_to_image = rf"C:\Users\rlong5\Desktop\Echo\1_Raw Data\20260911_files\0965833\original_image.nii"


# list_nii_parameter(path_to_image)