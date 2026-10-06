
This program takes an echocardiogram, takes first slice, creates mask on custom unet trained on three avaailable datasets [1]

1) For each line
Go to directory
Get dicom-> array
Mask
Output maks
2) Change env to radiomics
Send directory to radiomics 
Output radiomics to new spreadsheet. Append. 

to calculate radiomics: send an image and mask np array (256x256) to run_radiomics_on_echo_slice




view_nii.py:
This is ouput from extract nii data
Image size: (256, 256, 3)
Mask size: (256, 256)


[1]trained on 470 images: 
Echocardiography-HMCQU

The model is here:
best_model.keras
