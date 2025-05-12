import os
import matplotlib.pyplot as plt
from pydicom import dcmread
import numpy as np
import glob

def save_dicom_images(input_dir, output_dir, cmap="nipy_spectral"):
    os.makedirs(output_dir, exist_ok=True)

    dicom_files = glob.glob(os.path.join(input_dir, "*.dcm"))

    for i, dicom_path in enumerate(dicom_files, 1):
        try:
            ds = dcmread(dicom_path)

            output_path = os.path.join(output_dir, f"image_{i}.png")

            plt.imsave(output_path, ds.pixel_array, cmap=cmap)
            print(f"Image {i} saved to {output_path}")

        except Exception as e:
            print(f"Error processing {dicom_path}: {e}")
    
    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
    plt.title("Last Processed Image")
    plt.show()
    
# !pip install pydicom 
# input_directory = "Data/TCGA-AO-A0J8/12-20-2003-NA-NA-54853/1.000000-Loc-83949"
# output_directory = "outputs/img-output"
# save_dicom_images(input_directory, output_directory)
