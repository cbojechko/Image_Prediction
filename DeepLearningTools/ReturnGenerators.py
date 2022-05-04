import numpy as np
import os
from Data_Generators.TFRecord_to_Dataset_Generator import DataGeneratorClass
from Data_Generators.Image_Processors_Module.src.Processors.TFDataSets import ConstantProcessors as CProcessors,\
    RelativeProcessors as RProcessors
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
    out_path_jpeg = os.path.join(records_path, 'JpegsNoNormalization')
    for fold in [0]:
        train_path = os.path.join(records_path, 'Train', 'fold{}'.format(fold))
        if fold == 0:
            train_path = os.path.join(records_path, 'TrainNoNormalization')
        train_generator = DataGeneratorClass(record_paths=[train_path])
        all_keys = ('pdos_array', 'drr_array', 'half_drr_array', 'fluence_array')
        processors = [
            CProcessors.Squeeze(image_keys=all_keys),
            CProcessors.ExpandDimension(axis=-1, image_keys=('pdos_array', 'drr_array', 'half_drr_array', 'fluence_array')),
            CProcessors.CombineKeys(axis=-1, image_keys=('pdos_array', 'drr_array', 'half_drr_array', 'fluence_array'),
                                    output_key='combined'),
            CProcessors.ReturnOutputs(input_keys=('combined',), output_keys=('out_file_name',)),
            {'batch': 1}, {'repeat'}
        ]
        train_generator.compile_data_set(image_processors=processors, debug=False)
        iterator = iter(train_generator.data_set)
        for i in range(len(train_generator)):
            x, y = next(iterator)
            numpy_array = x[0].numpy()
            file_info = str(y[0][0]).split('b')[-1][1:].split('.tf')[0]
            print(file_info)
            if file_info.split('_')[0] == '10':
                xxx = 1
            numpy_array[..., 0] /= np.max(numpy_array[..., 0])
            numpy_array[..., 1] /= np.max(numpy_array[..., 1])
            numpy_array[..., 2] /= np.max(numpy_array[..., 2])
            numpy_array[..., 3] /= np.max(numpy_array[..., 3])
            numpy_array *= 255
            np.save(os.path.join(out_path_numpy, "{}.npy".format(file_info)), numpy_array)
            max_val = np.max(numpy_array[...,-1])

            if max_val < 20:
                print("{} max is {}".format(file_info, max_val))
            out_array = np.zeros((256, 256 * 4))
            for i in range(4):
                out_array[..., 256 * i:256 * (i + 1)] = numpy_array[..., i]
            image = Image.fromarray(out_array.astype('uint8'))
            image.save(os.path.join(out_path_jpeg, "{}.jpeg".format(file_info)))
    return None


def build_cache(generator):
    iterator = iter(generator.data_set)
    for i in range(len(generator)*2):
        x, y = next(iterator)
    return None


def return_generators(records_path):
    train_path = os.path.join(records_path, 'Train', 'fold1')
    validation_path = os.path.join(records_path, 'Train', 'fold2')
    train_generator = DataGeneratorClass(record_paths=[train_path])
    validation_generator = DataGeneratorClass(record_paths=[validation_path])
    all_keys = ('pdos_array', 'drr_array', 'half_drr_array', 'fluence_array')
    base_processors = [
        CProcessors.Squeeze(image_keys=all_keys),
        CProcessors.ExpandDimension(axis=-1, image_keys=('pdos_array', 'drr_array', 'half_drr_array', 'fluence_array')),
        CProcessors.Resize_with_crop_pad(keys=all_keys, image_rows=[256 for _ in range(len(all_keys))],
                                        image_cols=[256 for _ in range(len(all_keys))],
                                        is_mask=[False for _ in range(len(all_keys))]),
        CProcessors.MultiplyImagesByConstant(all_keys, values=(2/255, 2/255, 2/255, 1/255)), # Scale to be 0-2, and 0-1
        CProcessors.Add_Constant(all_keys, values=(-1, -1, -1, 0)), # Slide -1-1, and 0-1
        CProcessors.CombineKeys(axis=-1, image_keys=('pdos_array', 'drr_array', 'half_drr_array'),
                               output_key='combined'),
        CProcessors.ReturnOutputs(input_keys=('combined',), output_keys=('fluence_array',))
        ]
    train_processors = [
        # {'cache': train_path},
        {'shuffle': len(train_generator) // 3},
        {'batch': 1}, {'repeat'}
    ]
    validation_processors = [
        # {'cache': validation_path},
        {'batch': 1}, {'repeat'}
    ]
    train_generator.compile_data_set(image_processors=base_processors + train_processors, debug=False)
    validation_generator.compile_data_set(image_processors=base_processors + validation_processors, debug=False)
    if not os.path.exists(os.path.join(train_path, 'cache.tfrecord.index')):
        build_cache(train_generator)
    if not os.path.exists(os.path.join(validation_path, 'cache.tfrecord.index')):
        build_cache(validation_generator)
    return train_generator, validation_generator


def load_data_from_generator(generator):
    data = {'input' : [], 'rtimg' : []}
    iterator = iter(generator.data_set)
    for _ in range(len(generator)):
        x, y = next(iterator)
        data['input'].append(x[0][0])
        data['rtimg'].append(y[0][0])
    return data


def return_datasets(data_generators):
    all_datasets = {}
    for i in data_generators.keys():
        generator = data_generators[i]
        all_datasets[i] = tf.data.Dataset.from_tensor_slices((load_data_from_generator(generator)))
    return all_datasets


def return_generator(records_paths):
    generator = DataGeneratorClass(record_paths=records_paths)
    all_keys = ('pdos_array', 'drr_array', 'half_drr_array', 'fluence_array')
    base_processors = [
        CProcessors.Squeeze(image_keys=all_keys),
        CProcessors.ExpandDimension(axis=-1, image_keys=('pdos_array', 'drr_array', 'half_drr_array', 'fluence_array')),
        # RProcessors.NormalizeBasedOnOther(guiding_keys=('pdos_array', 'pdos_array'),
        #                                   changing_keys=('fluence_array', 'pdos_array'),
        #                                   reference_method=('reduce_max', 'reduce_max'),
        #                                   changing_methods=('divide', 'divide')),
        CProcessors.MultiplyImagesByConstant(keys=('pdos_array', 'fluence_array',
                                                   'drr_array', 'half_drr_array'), values=(1/3.448, 1/2.226,
                                                                                           1/325, 1/175)),
        CProcessors.CombineKeys(axis=-1, image_keys=('pdos_array', 'drr_array', 'half_drr_array'),
                                output_key='combined'),
        CProcessors.ReturnOutputs(input_keys=('combined',), output_keys=('fluence_array',)),
        {'shuffle': len(generator) // 3},
        {'batch': 1}, {'repeat'}
        ]
    generator.compile_data_set(image_processors=base_processors, debug=False)
    values_pdos = []
    values_fluence = []
    values_drr = []
    values_drr_half = []
    iterator = iter(generator.data_set)
    for _ in range(len(generator)):
        x, y = next(iterator)
        values_pdos.append(np.max(x[0][..., 0].numpy()))
        values_drr.append(np.max(x[0][..., 1].numpy()))
        values_drr_half.append(np.max(x[0][..., 2].numpy()))
        values_fluence.append(np.max(y[0].numpy()))
    xxx = 1
    return generator


def return_fold_datasets(data_generators, excluded_fold=5, batch_size=1):
    all_datasets = return_datasets(data_generators)
    train_dataset = None
    for i in data_generators.keys():
        print(i)
        dataset = all_datasets[i]
        print(dataset)
        if i == excluded_fold:
            valid_dataset = dataset
            valid_dataset = valid_dataset.shuffle(len(dataset))
            valid_dataset = valid_dataset.batch(batch_size)
        elif train_dataset is None:
            train_dataset = dataset
        else:
            train_dataset = train_dataset.concatenate(dataset)
    train_dataset.shuffle(len(train_dataset))
    train_dataset = train_dataset.batch(batch_size)
    return train_dataset, valid_dataset


def main():
    records_path = r'\\ad.ucsd.edu\ahs\radon\research\Bojechko\TFRecords'
    # create_files_for_streamline(records_path)
    # return None
    data_generators = {}
    xxx = return_generator([r'\\ad.ucsd.edu\ahs\radon\research\Bojechko\TFRecords\TrainNoNormalization\fold{}'.format(i) for i in range(1,6)])
    return None
    for i in range(1, 6):
        data_generators[i] = return_generator([r'\\ad.ucsd.edu\ahs\radon\research\Bojechko\TFRecords\Train\fold{}'.format(i)])
    train_dataset, valid_dataset = return_fold_datasets(data_generators, excluded_fold=5, batch_size=1)
    if not os.path.exists(records_path):
        records_path = os.path.abspath(os.path.join('..', 'Data'))
    print(records_path)

    train_generator, validation_generator = return_generators(records_path=records_path)
    iterator = iter(train_generator.data_set)
    for _ in range(1):
        for i in range(len(train_generator)):
            x, y = next(iterator)
            print(i)
    # return train_generator


if __name__ == '__main__':
    train_generator = main()
    pass
