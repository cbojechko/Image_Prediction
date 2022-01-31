import os
import numpy as np
import SimpleITK as sitk
import pandas
import pandas as pd
from PreProcessingTools.Image_Processors_Module.src.Processors.MakeTFRecordProcessors import plot_scroll_Image
from tqdm import tqdm


def evaluate_data():
    out_path = os.path.join('.', "Data.xlsx")
    if os.path.exists(out_path):
        return None
    data_dictionary = {'Image': [], 'EPID': [], 'Transmission': [], 'File': [], 'Patient #': []}
    data_path = r'\\ad.ucsd.edu\ahs\radon\research\Bojechko\nifti'
    files = os.listdir(data_path)
    pbar = tqdm(total=len(files), desc='Loading through data')
    for file in files:
        data_handle = sitk.ReadImage(os.path.join(data_path, file))
        data_array = sitk.GetArrayFromImage(data_handle)
        data_max = np.max(data_array,axis=(0, 1))
        data_dictionary['Image'].append(data_max[..., -1])
        data_dictionary['EPID'].append(data_max[..., 1])
        data_dictionary['Transmission'].append(data_max[..., 0])
        data_dictionary['File'].append(file)
        data_dictionary['Patient #'].append(file.split('_')[0])
        pbar.update()
    df = pandas.DataFrame(data_dictionary)
    df.to_excel(out_path, index=0)
    return None


if __name__ == '__main__':
    pass

