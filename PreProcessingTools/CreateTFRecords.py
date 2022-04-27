import os
import numpy as np
import PreProcessingTools.Image_Processors_Module.src.Processors.MakeTFRecordProcessors as Processors
import PreProcessingTools.Image_Processors_Module.src.Processors.TFRecordWriter as RecordWriter
from glob import glob
import pandas as pd


def return_dictionary_list(base_path, out_path, rewrite):
    """
    :param path:
    :return:
    """
    """
    We'll start by finding all of the PDOS files, this ensures that we have a PDOS
    """
    output_list = []
    excel_path = os.path.join('.', "Patient_Keys.xlsx")
    excel_path = r'R:\Bojechko\patientlist_032222.xlsx'
    # print("We are not adding patients in the excel file! This is only loading from an available excel file, we aware!")
    patient_id_column = 'MRN'
    if not os.path.exists(excel_path):
        data_dictionary = {patient_id_column: [], 'Index': []}
        df = pd.DataFrame(data_dictionary)
        df.to_excel(excel_path, index=0)
    else:
        df = pd.read_excel(excel_path, engine='openpyxl', sheet_name='folds')
    rewrite_excel = False
    for patient_data in ['phantom', 'PatientData2']:
        base_patient_path = os.path.join(base_path, patient_data)
        MRN_list = os.listdir(base_patient_path)
        for patient_MRN in MRN_list:
            previous_run = df.loc[df[patient_id_column].astype('str') == patient_MRN]
            if previous_run.shape[0] == 0:
                previous_run = df.loc[df[patient_id_column].astype('str') == patient_MRN[1:]]
            if previous_run.shape[0] == 0:
                try:
                    previous_run = df.loc[df[patient_id_column].astype('int') == int(patient_MRN)]
                except:
                    print('Issue with {}'.format(patient_MRN))
            if previous_run.shape[0] == 0:
                print("Issue with {}".format(patient_MRN))
                continue
                rewrite_excel = True
                i = 0
                while i in df['Index'].values:
                    i += 1
                df = df.append(pd.DataFrame({patient_id_column: [patient_MRN], 'Index': [i]}))
            else:
                i = int(previous_run['Index'].values[0])
            print(patient_MRN)
            path = os.path.join(base_patient_path, patient_MRN, 'Niftiis')
            pdos_files = glob(os.path.join(path, 'PDOS_G*'))
            for pdos_file in pdos_files:
                angle = pdos_file.split('PDOS_')[1].split('_')[0]
                """
                Next, find the fluence files with the same angle used
                """
                fluence_files = glob(os.path.join(path, "Fluence_{}_*".format(angle)))
                for fluence_file in fluence_files:
                    date = fluence_file.split('_')[-1].split('.')[0]
                    addition = "{}_{}".format(angle, date)
                    half_proj_file = os.path.join(path, "HalfProj_{}.mha".format(addition))
                    full_drr_file = os.path.join(path, "DRR_{}.mha".format(addition))
                    examples_exist = [os.path.exists(os.path.join(out_path, "{}_{}_{}_{}.tfrecord".format(i,
                                                                                                          angle, date,
                                                                                                          _)))
                                      for _ in range(5)]
                    if max(examples_exist) and not rewrite:  # If we have the record, move on
                        continue
                    if os.path.exists(full_drr_file) and os.path.exists(half_proj_file):
                        patient_dict = {'pdos_path': pdos_file, 'fluence_path': fluence_file,
                                        'half_drr_path': half_proj_file, 'full_drr_path': full_drr_file,
                                        'out_file_name': "{}_{}_{}.tfrecord".format(i, angle, date)}
                        output_list.append(patient_dict)
    if rewrite_excel:
        df.to_excel(excel_path, index=0)
    return output_list


def make_train_records(base_path, rewrite=False):
    out_path = os.path.join(base_path, 'TFRecords', 'Train')
    train_list = return_dictionary_list(base_path, out_path, rewrite)
    if not train_list:
        print("No new files found for record making")
        return None
    record_writer = RecordWriter.RecordWriter(out_path=out_path,
                                              file_name_key='out_file_name', rewrite=rewrite)
    keys = ('pdos_handle', 'fluence_handle', 'half_drr_handle', 'drr_handle')
    array_keys = tuple(i.replace('_handle', '_array') for i in keys)
    """
    Load all of the files into SITK handles
    Then resample the PDOS to be 1x1x1mm
    Resample everything else to fit around the PDOS
    """
    train_processors = [
        Processors.LoadNifti(nifti_path_keys=('pdos_path', 'fluence_path', 'half_drr_path', 'full_drr_path'),
                             out_keys=keys),
        Processors.ResampleSITKHandles(desired_output_spacing=(2.0, 2.0, 1.0), resample_keys=('pdos_handle',),
                                       resample_interpolators=['Linear',]),
        Processors.ResampleSITKHandlesToAnotherHandle(resample_keys=keys,
                                                      reference_handle_keys=['pdos_handle' for _ in range(len(keys))],
                                                      resample_interpolators=['Linear' for _ in range(len(keys))]),
        Processors.SimpleITKImageToArray(nifti_keys=keys, out_keys=array_keys,
                                         dtypes=['float32' for _ in range(len(keys))]),
        Processors.PadImages(bounding_box_expansion=(0, 0, 0), power_val_z=1, power_val_x=2**8, power_val_y=2**8,
                             image_keys=array_keys, mode='linear_ramp', min_val=None),
        Processors.DeleteKeys(keys_to_delete=keys),
        # Processors.AddByValues(image_keys=('image',), values=(0,)),
        Processors.ChangeArrayByArgInArray(reference_keys=('pdos_array', 'pdos_array'),  # Scale fluence based on pdos
                                           value_args=(np.max, np.max), target_keys=('fluence_array', 'pdos_array'),
                                           change_args=(np.divide, np.divide)),
        Processors.MultiplyByValues(image_keys=('drr_array', 'half_drr_array', 'pdos_array', 'fluence_array'),
                                    values=(255/300, 255/300, 255, 255)),  # 0-255

        # Processors.Threshold_Images(image_keys=('image_array',), lower_bound=-3, upper_bound=3),
        # Processors.AddByValues(image_keys=('image_array',), values=(3,)),
        # Processors.DivideByValues(image_keys=('image_array',), values=(6,)),
    ]

    RecordWriter.parallel_record_writer(dictionary_list=train_list, thread_count=1, recordwriter=record_writer,
                                        image_processors=train_processors, debug=True, verbose=False)
    return None


def create_tf_records(base_path, rewrite=False):
    make_train_records(base_path, rewrite)
    return None


if __name__ == '__main__':
    pass
