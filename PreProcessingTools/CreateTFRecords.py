import os
import PreProcessingTools.Image_Processors_Module.src.Processors.MakeTFRecordProcessors as Processors
import PreProcessingTools.Image_Processors_Module.src.Processors.TFRecordWriter as RecordWriter
from glob import glob
import pandas


def return_dictionary_list(base_path):
    """
    :param path:
    :return:
    """
    """
    We'll start by finding all of the PDOS files, this ensures that we have a PDOS
    """
    output_list = []
    i = -1
    out_path = os.path.join('.', "Patient_Keys.xlsx")
    data_dictionary = {'Patient #': [], 'Index': []}
    for patient_data in ['PatientData2']:
        base_patient_path = os.path.join(base_path, patient_data)
        MRN_list = os.listdir(base_patient_path)
        for patient_MRN in MRN_list:
            print(patient_MRN)
            i += 1
            data_dictionary['Patient #'].append(patient_MRN)
            data_dictionary['Index'].append(i)
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
                    addition = "{}_{}.mha".format(angle, date)
                    half_proj_file = os.path.join(path, "HalfProj_{}".format(addition))
                    full_drr_file = os.path.join(path, "DRR_{}".format(addition))
                    if os.path.exists(full_drr_file) and os.path.exists(half_proj_file):
                        patient_dict = {'pdos_path': pdos_file, 'fluence_path': fluence_file,
                                        'half_drr_path': half_proj_file, 'full_drr_path': full_drr_file,
                                        'out_file_name': "{}_G{}_{}.tfrecord".format(i, angle, date)}
                        output_list.append(patient_dict)
    df = pandas.DataFrame(data_dictionary)
    df.to_excel(out_path, index=0)
    return output_list


def make_train_records(base_path):
    train_list = return_dictionary_list(base_path)
    record_writer = RecordWriter.RecordWriter(out_path=os.path.join(base_path, 'TFRecords', 'Train'),
                                              file_name_key='out_file_name', rewrite=False)
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
        Processors.PadImages(bounding_box_expansion=(0, 0, 0), power_val_z=1, power_val_x=2**5, power_val_y=2**5,
                             image_keys=array_keys),
        Processors.DeleteKeys(keys_to_delete=keys),
        # Processors.AddByValues(image_keys=('image',), values=(0,)),
        # Processors.DivideByValues(image_keys=('image',), values=(1,)),
        # Processors.Threshold_Images(image_keys=('image_array',), lower_bound=-3, upper_bound=3),
        # Processors.AddByValues(image_keys=('image_array',), values=(3,)),
        # Processors.DivideByValues(image_keys=('image_array',), values=(6,)),
    ]

    RecordWriter.parallel_record_writer(dictionary_list=train_list, thread_count=8, recordwriter=record_writer,
                                        image_processors=train_processors, debug=False)
    return None


def create_tf_records(base_path):
    make_train_records(base_path)
    return None


if __name__ == '__main__':
    pass
