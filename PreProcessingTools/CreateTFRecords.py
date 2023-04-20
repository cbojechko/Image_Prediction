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
    excel_path = r'R:\Bojechko\patientlist_081722.xlsx'
    # print("We are not adding patients in the excel file! This is only loading from an available excel file, we aware!")
    patient_id_column = 'MRN'
    if not os.path.exists(excel_path):
        data_dictionary = {patient_id_column: [], 'Index': []}
        df = pd.DataFrame(data_dictionary)
        df.to_excel(excel_path, index=0)
    else:
        df = pd.read_excel(excel_path, engine='openpyxl', sheet_name='Sheet1')
    rewrite_excel = False
    for patient_data in ['phantom']:  #'PatientData2',
        base_patient_path = os.path.join(base_path, patient_data)
        MRN_list = os.listdir(base_patient_path)
        for patient_MRN in MRN_list:
            previous_run = df.loc[df[patient_id_column].astype('str') == patient_MRN]
            if previous_run.shape[0] == 0:
                previous_run = df.loc[df[patient_id_column].astype('str') == patient_MRN[1:]]
            if previous_run.shape[0] == 0:
                previous_run = df.loc[df[patient_id_column].astype('str') == patient_MRN[1:]+'.0']
            if previous_run.shape[0] == 0:
                previous_run = df.loc[df[patient_id_column].astype('str') == patient_MRN+'.0']
            if previous_run.shape[0] == 0:
                try:
                    previous_run = df.loc[df[patient_id_column].astype('int') == int(patient_MRN)]
                except:
                    print('Issue with {}'.format(patient_MRN))
            if previous_run.shape[0] == 0:
                print("Issue with {}".format(patient_MRN))
                rewrite_excel = True
                i = max(df['Index'].values)+1
                # while i in df['Index'].values:
                #     i += 1
                df = df.append(pd.DataFrame({patient_id_column: [patient_MRN], 'Index': [i]}))
            else:
                i = int(previous_run['Index'].values[0])
            if i < 9982:
                continue
            print(patient_MRN)
            path = os.path.join(base_patient_path, patient_MRN, 'Niftiis')
            pdos_files = glob(os.path.join(path, 'PDOS_G*'))
            for pdos_file in pdos_files:
                angle = "".join([f"{_}_" for _ in pdos_file.split('PDOS_')[1].split('_')[:-1]])
                """
                Next, find the fluence files with the same angle used
                """
                fluence_files = glob(os.path.join(path, f"Fluence_{angle}*"))
                for fluence_file in fluence_files:
                    file_info = fluence_file.split('_')
                    addition = "".join([f"{_}_" for _ in file_info[1:]]).split('.mha')[0]  # Gantry_field name_date
                    iso_proj_file = os.path.join(path, f"Proj_0cm_to_iso_{addition}.mha")
                    deep_proj_file = os.path.join(path, f"Proj_5cm_to_iso_{addition}.mha")
                    shallow_proj_file = os.path.join(path, f"Proj_-5cm_to_iso_{addition}.mha")
                    deep_proj_to_panel_file = os.path.join(path, f"Proj_5cm_from_iso_to_panel_{addition}.mha")
                    iso_proj_to_panel_file = os.path.join(path, f"Proj_0cm_from_iso_to_panel_{addition}.mha")
                    shallow_proj_to_panel_file = os.path.join(path, f"Proj_-5cm_from_iso_to_panel_{addition}.mha")
                    full_drr_file = os.path.join(path, f"DRR_{addition}.mha")
                    examples_exist = [os.path.exists(os.path.join(out_path, f"{i}_{addition}_{_}.tfrecord"))
                                      for _ in range(5)]
                    if max(examples_exist) and not rewrite:  # If we have the record, move on
                        continue
                    if os.path.exists(full_drr_file) and os.path.exists(iso_proj_file):
                        patient_dict = {'pdos_path': pdos_file, 'fluence_path': fluence_file,
                                        '5cm_shallow_path': shallow_proj_file,
                                        '5cm_deep_path': deep_proj_file,
                                        'iso_drr_path': iso_proj_file, 'full_drr_path': full_drr_file,
                                        'deep_to_panel_path': deep_proj_to_panel_file,
                                        'iso_to_panel_path': iso_proj_to_panel_file,
                                        'shallow_to_panel_path': shallow_proj_to_panel_file,
                                        'out_file_name': f"{i}_{addition}.tfrecord"}
                        output_list.append(patient_dict)
    if rewrite_excel:
        df.to_excel(excel_path, index=0)
    return output_list


def make_train_records(base_path, rewrite=False):
    out_path = os.path.join(base_path, 'TFRecords', 'TrainNoNormalizationMultipleProj')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    train_list = return_dictionary_list(base_path, out_path, rewrite)
    if not train_list:
        print("No new files found for record making")
        return None
    record_writer = RecordWriter.RecordWriter(out_path=out_path,
                                              file_name_key='out_file_name', rewrite=rewrite)
    keys = ('pdos_handle', 'fluence_handle', 'drr_handle', '5cm_deep_handle', 'iso_handle', '5cm_shallow_handle',
            'deep_to_panel_handle', 'iso_to_panel_handle', 'shallow_to_panel_handle')
    array_keys = tuple(i.replace('_handle', '_array') for i in keys)
    """
    Load all of the files into SITK handles
    Then resample the PDOS to be 1x1x1mm
    Resample everything else to fit around the PDOS
    """
    spacing = 1.68
    train_processors = [
        Processors.LoadNifti(nifti_path_keys=('pdos_path', 'fluence_path', 'full_drr_path', '5cm_deep_path',
                                              'iso_drr_path', '5cm_shallow_path', 'deep_to_panel_path',
                                              'iso_to_panel_path', 'shallow_to_panel_path'),
                             out_keys=keys),
        Processors.ResampleSITKHandles(desired_output_spacing=(spacing, spacing, 1.0), resample_keys=('fluence_handle',),
                                       resample_interpolators=('Linear',)),
        Processors.SetSITKOrigin(keys=keys, desired_output_origin=(None, None, -1540)),
        Processors.ResampleSITKHandlesToAnotherHandle(resample_keys=keys,
                                                      reference_handle_keys=['fluence_handle' for _ in range(len(keys))],
                                                      resample_interpolators=['Linear' for _ in range(len(keys))]),
        Processors.SimpleITKImageToArray(nifti_keys=keys, out_keys=array_keys,
                                         dtypes=['float32' for _ in range(len(keys))]),
        Processors.PadImages(bounding_box_expansion=(0, 0, 0), power_val_z=1, power_val_x=2**8, power_val_y=2**8,
                             image_keys=array_keys, mode='linear_ramp', min_val=None),
        Processors.DeleteKeys(keys_to_delete=keys),
        # Processors.AddByValues(image_keys=('image',), values=(0,)),
        # Processors.ChangeArrayByArgInArray(reference_keys=('pdos_array', 'pdos_array'),  # Scale fluence based on pdos
        #                                    value_args=(np.max, np.max), target_keys=('fluence_array', 'pdos_array'),
        #                                    change_args=(np.divide, np.divide)),
        # Processors.MultiplyByValues(image_keys=('drr_array', 'half_drr_array', 'pdos_array', 'fluence_array'),
        #                             values=(255/300, 255/300, 255, 255)),  # 0-255
    ]

    RecordWriter.parallel_record_writer(dictionary_list=train_list, thread_count=1, recordwriter=record_writer,
                                        image_processors=train_processors, debug=True, verbose=False)
    return None


def create_tf_records(base_path, rewrite=False):
    make_train_records(base_path, rewrite)
    return None


if __name__ == '__main__':
    pass
