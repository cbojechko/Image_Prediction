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


def create_tf_records(data_path):
    output_list = return_dictionary_list(data_path)
    xxx = 1
    return None


if __name__ == '__main__':
    pass
