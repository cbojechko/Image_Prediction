import os
import numpy as np
import SimpleITK as sitk
import pandas
import pandas as pd
from PreProcessingTools.Image_Processors_Module.src.Processors.MakeTFRecordProcessors import plot_scroll_Image
from tqdm import tqdm
from glob import glob


def evaluate_data():
    out_path = os.path.join('.', "Data.xlsx")
    if os.path.exists(out_path):
        return None
    data_dictionary = {'File': [], 'Patient #': [], 'Description': [], 'Value': []}
    base_path = r'\\ad.ucsd.edu\ahs\radon\research\Bojechko'
    for patient_data in ['PatientData2']:
        base_patient_path = os.path.join(base_path, patient_data)
        MRN_list = os.listdir(base_patient_path)
        pbar = tqdm(total=len(MRN_list), desc='Loading through data')
        for patient_MRN in MRN_list:
            print(patient_MRN)
            patient_path = os.path.join(base_patient_path, patient_MRN, "Niftiis")
            files = glob(os.path.join(patient_path, "*.mha"))
            for file in files:
                if file.find('Fluence') != -1:
                    extension = 'Fluence'
                elif file.find('PDOS') != -1:
                    extension = 'PDOS'
                elif file.find('Half') != -1:
                    extension = 'Half'
                elif file.find('DRR') != -1:
                    extension = 'DRR'
                else:
                    continue
                data_handle = sitk.ReadImage(file)
                data_array = sitk.GetArrayFromImage(data_handle)
                data_dictionary['Patient #'].append(patient_MRN)
                data_dictionary['File'].append(os.path.split(file)[-1])
                data_dictionary['Description'].append(extension)
                data_dictionary['Value'].append(np.max(data_array))
            pbar.update()
    df = pandas.DataFrame(data_dictionary)
    df.to_excel(out_path, index=0)
    return None


if __name__ == '__main__':
    evaluate_data()
    pass

