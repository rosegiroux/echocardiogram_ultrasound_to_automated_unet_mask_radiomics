from run_radiomics_on_echo_slice import extract_nii_data
from run_radiomics_on_echo_slice import extract_radiomic_features


import os

import numpy as np

import pandas #pandas   2.3.1

from datetime import datetime, timezone
from view_nii import list_nii_parameter


'''this file uses pyradiomics to calculate radiomics of sultrasound slice.
use radiomix2 enviornment 
'''

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
results = []

# path_to_image = rf"C:\Users\rlong5\Desktop\Echo\1_Raw Data\20260911_files\0965833\original_image.nii"


# list_nii_parameter(path_to_image)

for path in file_list_from_excel.iloc[12:20, 5]:
   
    if '2ch' in path: #skip if 2ch in file name
        continue
    print(path) #path is to the dicom file
    print("^^starting this radiomics")
    data_directory = os.path.dirname(path)    #need the directory

    patient_data = {}
    patient_data['echo'] = extract_nii_data(data_directory)

#debugging
    image = patient_data['echo']['image']
    mask = patient_data['echo']['mask']
    
    print("Image size:", image.GetSize())
    print("Mask size:", mask.GetSize())

    print("Image spacing:", image.GetSpacing())
    print("Mask spacing:", mask.GetSpacing())

    print("Image origin:", image.GetOrigin())
    print("Mask origin:", mask.GetOrigin())

    print("Image direction:", image.GetDirection())
    print("Mask direction:", mask.GetDirection())




    dt = datetime.fromtimestamp((patient_data['echo']['image_timestamp']), tz=timezone.utc)
    patient_feature_data = extract_radiomic_features(image[:,:,0], mask)
    #convert patient feature data list to pandas dataframe
    df = pandas.DataFrame([patient_feature_data])


    df["dicom_datetime"] = dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    df['dicom_file_path'] = path
    print(path)
    print(df.head())
    results.append(df)
    del patient_feature_data, path, dt

final_df = pandas.concat(results, ignore_index=True)
output_csv = rf'C:\Users\rlong5\Desktop\Echo\1_Raw Data\20260911_files\20260928_radiomics_results4.csv'
#C:\Users\rlong5\Desktop\Echo\20260505_checking_abnormal_echos\1_Raw_data\krupien_34\krupien_34.csv'
final_df.to_csv(output_csv, index = False)
print(f"{len(results)+1} files Radiomics features saved to {output_csv}")
