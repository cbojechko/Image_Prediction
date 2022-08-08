import os

from PlotScrollNumpyArrays import plot_scroll_Image
import numpy as np
import SimpleITK as sitk

base_path = r'R:\Bojechko\PatientData2'
MRNs = os.listdir(base_path)
MRN = MRNs[6]

path = os.path.join(base_path, MRN, 'Niftiis', 'CBCT_20200128.mha')

image_handle = sitk.ReadImage(path)
image_array = sitk.GetArrayFromImage(image_handle)
Y, X = np.ogrid[:image_array.shape[1], :image_array.shape[2]]
dist_from_center = np.sqrt((X - 256) ** 2 + (Y - 256) ** 2)
for z in range(image_array.shape[0]):
    image = image_array[z, ...]
    binary_image = image > -1000
    total_max = np.sum(dist_from_center < 256 * binary_image)  # This is the absolute max
    upper_limit = 256
    lower_limit = 0
    current_guess_radii = (upper_limit - lower_limit) // 2 + lower_limit
    previous_guess_radii = upper_limit
    while previous_guess_radii != current_guess_radii:
        current_sum = np.sum(dist_from_center < current_guess_radii * binary_image)
        previous_guess_radii = current_guess_radii
        if current_sum < total_max:
            lower_limit = current_guess_radii
            current_guess_radii = lower_limit + (upper_limit - lower_limit) // 2
        else:
            upper_limit = current_guess_radii
            current_guess_radii = upper_limit - (upper_limit - lower_limit) // 2

    min_row = 1
xxx = 1