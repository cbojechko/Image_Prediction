import os
import PreProcessingTools.Image_Processors_Module.src.Processors.MakeTFRecordProcessors as Processors
import PreProcessingTools.Image_Processors_Module.src.Processors.TFRecordWriter as RecordWriter
from glob import glob


def return_dictionary_list(base_path):
    """
    :param path:
    :return:
    """
    """
    We'll start by finding all of the PDOS files, this ensures that we have a PDOS
    """
    output_list = []
    for patient_data in ['PatientData2']:
        base_patient_path = os.path.join(base_path, patient_data)
        MRN_list = os.listdir(base_patient_path)
        for patient_MRN in MRN_list:
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
                    addition = "{}_{}.mha".format(angle, date)
                    half_proj_file = os.path.join(path, "HalfProj_{}".format(addition))
                    full_drr_file = os.path.join(path, "DRR_{}".format(addition))
                    patient_dict = {'pdos_path': pdos_file, 'fluence_path': fluence_file,
                                    'half_drr_path': half_proj_file, 'full_drr_file': full_drr_file}
                    output_list.append(patient_dict)
            return output_list
    return output_list


def make_train_records(base_path):
    train_list = return_dictionary_list(base_path)
    record_writer = RecordWriter.RecordWriter(out_path=os.path.join(base_path, 'TFRecords', 'Train'),
                                              file_name_key='out_file_name', rewrite=True)
    train_processors = [
        Processors.LoadNifti(nifti_path_keys=('data_path',), out_keys=('data_handle',)),
        Processors.ResampleSITKHandles(desired_output_spacing=(1.0, 1.0, 1.0), resample_keys=('data_handle',),
                                       resample_interpolators=('Linear',)), # They are already 1x1x1, but keep this for now
        Processors.SimpleITKImageToArray(nifti_keys=('data_handle',),
                                         out_keys=('data_array',)),
        Processors.SplitArray(array_keys=('data_array','data_array', 'data_array'),
                              out_keys=('image', 'epid', 'transmission'), axis_index=(3, 1, 0)),
        Processors.DeleteKeys(keys_to_delete=('data_handle',)),
        Processors.ExpandDimensions(image_keys=('image', 'epid', 'transmission'), axis=-1),
        Processors.AddByValues(image_keys=('image',), values=(0,)),
        Processors.DivideByValues(image_keys=('image',), values=(1,)),
        # Processors.Threshold_Images(image_keys=('image_array',), lower_bound=-3, upper_bound=3),
        # Processors.AddByValues(image_keys=('image_array',), values=(3,)),
        # Processors.DivideByValues(image_keys=('image_array',), values=(6,)),
    ]

    RecordWriter.parallel_record_writer(dictionary_list=train_list, thread_count=1, recordwriter=record_writer,
                                        image_processors=train_processors, debug=True)
    return None


def create_tf_records(base_path):
    make_train_records(base_path)
    return None


if __name__ == '__main__':
    pass
