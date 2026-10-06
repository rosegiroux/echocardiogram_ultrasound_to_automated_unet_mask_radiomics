
This program takes an echocardiogram, takes first slice, creates mask on custom unet trained on three avaailable datasets [1]

1) For each line
Go to directory
Get dicom-> array
Mask
Output maks
2) Change env to radiomics
Send directory to radiomics 
Output radiomics to new spreadsheet. Append. 

<img width="855" height="474" alt="image" src="https://github.com/user-attachments/assets/e77ff25d-999e-4757-abb7-5b913a5d8a5d" />
<img width="1024" height="620" alt="image" src="https://github.com/user-attachments/assets/97b538a3-41fa-4a7a-8d5e-3cdb6ef6b6ce" />

to calculate radiomics: send an image and mask np array (256x256) to run_radiomics_on_echo_slice

 


<img width="777" height="195" alt="image" src="https://github.com/user-attachments/assets/bfbe1c0a-6417-48d8-add2-100e5d7526ce" />




view_nii.py:
This is ouput from extract nii data
Image size: (256, 256, 3)
Mask size: (256, 256)


[1]trained on 470 images: 
Echocardiography-HMCQU

The model is here:
best_model.keras hugging face: echocardiogram_ultrasound_to_automated_unet_mask_radiomics

