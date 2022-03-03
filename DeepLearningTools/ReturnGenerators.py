import numpy as np
import os
from DeepLearningTools.Data_Generators.TFRecord_to_Dataset_Generator import DataGeneratorClass
import DeepLearningTools.Data_Generators.Image_Processors_Module.src.Processors.TFDataSetProcessors as Processors
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image
import tensorflow as tf
from PIL import Image


def get_mean_std(train_generator):
    iter_generator = iter(train_generator.data_set)
    print(len(train_generator))
    for i in range(len(train_generator)):
        print(i)
        x, y = next(iter_generator)
        print(tf.reduce_max(x[0][..., 0]))
        print(tf.reduce_max(x[0][..., 1]))
        print(tf.reduce_max(x[0][..., 2]))
        print(tf.reduce_max(y[0][..., 0]))
        #temp_values = x[0][y[0] == 1]
        #if values is None:
        #    values = temp_values
        #else:
        #    values = np.concatenate([values, temp_values])
    return None


def create_files_for_streamline(records_path):
    out_path_numpy = os.path.join(records_path, 'NumpyFiles')
    out_path_jpeg = os.path.join(records_path, 'Jpegs')
    train_path = os.path.join(records_path, 'Train')
    train_generator = DataGeneratorClass(record_paths=[train_path])
    all_keys = ('pdos_array', 'drr_array', 'half_drr_array', 'fluence_array')
    processors = [
        Processors.Squeeze(image_keys=all_keys),
        Processors.ExpandDimension(axis=-1, image_keys=('pdos_array', 'drr_array', 'half_drr_array', 'fluence_array')),
        Processors.Resize_with_crop_pad(keys=all_keys, image_rows=[256 for _ in range(len(all_keys))],
                                        image_cols=[256 for _ in range(len(all_keys))],
                                        is_mask=[False for _ in range(len(all_keys))]),
        Processors.CombineKeys(axis=-1, image_keys=('pdos_array', 'drr_array', 'half_drr_array', 'fluence_array'),
                               output_key='combined'),
        Processors.ReturnOutputs(input_keys=('combined',), output_keys=('out_file_name',)),
        {'batch': 1}, {'repeat'}
    ]
    train_generator.compile_data_set(image_processors=processors, debug=False)
    iterator = iter(train_generator.data_set)
    for i in range(len(train_generator)):
        x, y = next(iterator)
        numpy_array = x[0].numpy()
        file_info = str(y[0][0]).split('b')[-1][1:].split('.tf')[0]
        np.save(os.path.join(out_path_numpy, "{}.npy".format(file_info)), numpy_array)
        out_array = np.zeros((256, 256 * 4))
        for i in range(4):
            out_array[..., 256 * i:256 * (i + 1)] = numpy_array[..., i]
        image = Image.fromarray(out_array.astype('uint64'))
        image.save(os.path.join(out_path_jpeg, "{}.jpeg".format(file_info)))
    return None


def return_train_generator(records_path):
    train_path = os.path.join(records_path, 'Train')
    train_generator = DataGeneratorClass(record_paths=[train_path])
    all_keys = ('pdos_array', 'drr_array', 'half_drr_array', 'fluence_array')
    processors = [
        Processors.Squeeze(image_keys=all_keys),
        Processors.ExpandDimension(axis=-1, image_keys=('pdos_array', 'drr_array', 'half_drr_array', 'fluence_array')),
        Processors.Resize_with_crop_pad(keys=all_keys, image_rows=[256 for _ in range(len(all_keys))],
                                        image_cols=[256 for _ in range(len(all_keys))],
                                        is_mask=[False for _ in range(len(all_keys))]),
        Processors.CombineKeys(axis=-1, image_keys=('pdos_array', 'drr_array', 'half_drr_array'),
                               output_key='combined'),
        Processors.ReturnOutputs(input_keys=('combined',), output_keys=('fluence_array',)),
        {'shuffle': len(train_generator)//3},
        {'batch': 1}, {'repeat'}
    ]
    train_generator.compile_data_set(image_processors=processors, debug=False)
    return train_generator


def return_validation_generator(records_path):
    validation_path = os.path.join(records_path, 'Validation')
    validation_generator = DataGeneratorClass(record_paths=[validation_path])
    processors = [
        Processors.ExpandDimension(axis=-1, image_keys=('image_array', 'annotation_array')),
        # Processors.RandomCrop(keys_to_crop=('image_array', 'annotation_array'), crop_dimensions=((32, 32, 32, 1),
        #                                                                                          (32, 32, 32, 1))),
        Processors.ReturnOutputs(input_keys=('image_array',), output_keys=('annotation_array',)),
        {'batch': 1}, {'repeat'}
    ]
    validation_generator.compile_data_set(image_processors=processors, debug=True)
    return validation_generator


def return_generators():
    records_path = r'\\ad.ucsd.edu\ahs\radon\research\Bojechko\TFRecords'
    if not os.path.exists(records_path):
        records_path = os.path.abspath(os.path.join('..', 'Data'))
    print(records_path)
    create_files_for_streamline(records_path)
    train_generator = return_train_generator(records_path=records_path)
    # validation_generator = return_validation_generator(records_path=records_path)
    xxx = 1
    return train_generator


if __name__ == '__main__':
    train_generator = return_generators()
    #get_mean_std(train_generator)
    pass
