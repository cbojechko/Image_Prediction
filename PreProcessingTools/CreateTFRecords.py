import os
import PreProcessingTools.Image_Processors_Module.src.Processors.MakeTFRecordProcessors as Processors
import PreProcessingTools.Image_Processors_Module.src.Processors.TFRecordWriter as RecordWriter


def return_dictionary_list(path):
    files = [i for i in os.listdir(path) if i.endswith('.nii.gz')]
    output_list = []
    for file in files:
        breakdown = file.split('.nii')[0]
        patient_dict = {'data_path': os.path.join(path, file),
                        'out_file_name': '{}.tfrecord'.format(breakdown)}
        output_list.append(patient_dict)
    return output_list


def make_train_records(base_path):
    train_path = os.path.join(base_path, 'nifti')
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
    train_list = return_dictionary_list(train_path)
    RecordWriter.parallel_record_writer(dictionary_list=train_list, thread_count=8, recordwriter=record_writer,
                                        image_processors=train_processors, debug=False)
    return None


def create_tf_records(base_path):
    make_train_records(base_path)
    return None


if __name__ == '__main__':
    pass
