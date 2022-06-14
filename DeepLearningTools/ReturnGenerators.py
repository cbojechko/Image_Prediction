import numpy as np
import os
from DeepLearningTools.Data_Generators.TFRecord_to_Dataset_Generator import DataGeneratorClass
from DeepLearningTools.Data_Generators.Image_Processors_Module.src.Processors.TFDataSets import ConstantProcessors as CProcessors,\
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


def return_generator(records_path, cache, batch=1):
    global_norm = True
    add_5cm_keys = True
    generator = DataGeneratorClass(record_paths=records_path, delete_old_cache=True)
    all_keys = ('pdos_array', 'fluence_array','drr_array', '5cm_deep_array', 'iso_array', '5cm_shallow_array')
    drr_keys = ('drr_array', '5cm_deep_array', 'iso_array', '5cm_shallow_array', )
    normalize_keys = ('iso_array',)
    input_keys = ('pdos_array', 'drr_array', 'iso_array')
    if add_5cm_keys:
        input_keys = ('pdos_array', 'drr_array', '5cm_deep_array', 'iso_array', '5cm_shallow_array')
        normalize_keys = ('5cm_deep_array', 'iso_array', '5cm_shallow_array')
    print(f"Inputs are {input_keys}")
    base_processors = [
                     CProcessors.Squeeze(image_keys=all_keys),
                     CProcessors.ExpandDimension(axis=-1, image_keys=all_keys),
                     CProcessors.MultiplyImagesByConstant(keys=drr_keys,
                                                          values=(1/90, 1/90, 1/90, 1/90)),
                     ]
    if global_norm:
        base_processors += [
                            CProcessors.MultiplyImagesByConstant(keys=('pdos_array',
                                                                    'fluence_array'),
                                                                 values=(1, 1)) #(1/2, 1/(.3876*2))
                            ]
    else:
        base_processors += [
                            RProcessors.NormalizeBasedOnOther(guiding_keys=('pdos_array', 'pdos_array'),
                                                              changing_keys=('fluence_array', 'pdos_array'),
                                                              reference_method=('reduce_max', 'reduce_max'),
                                                              changing_methods=('divide', 'divide'))
                            ]
    base_processors += [
                      CProcessors.CombineKeys(axis=-1,
                                              image_keys=input_keys,
                                              output_key='test'),
                      CProcessors.ReturnOutputs(input_keys=('test',),
                                                output_keys=('fluence_array',))
    ]
    base_processors += [
                      #{'cache': train_cache},
                      {'batch': batch}, {'repeat'}
                      ]
    generator.compile_data_set(image_processors=base_processors, debug=False)
    return generator


def create_files_for_streamline(records_path):
    out_path_numpy = os.path.join(records_path, 'NumpyFiles')
    out_path_jpeg = os.path.join(records_path, 'JpegsNoNormalizationMultipleProj')
    for fold in [0]:
        train_path = os.path.join(records_path, 'Train', 'fold{}'.format(fold))
        record_paths = [train_path]
        if fold == 0:
            train_path = os.path.join(records_path, 'TrainNoNormalizationMultipleProj')
            record_paths = [os.path.join(train_path, 'phantom_valid'), os.path.join(train_path, 'phantom_train')]
        # record_paths = [train_path]
        train_generator = DataGeneratorClass(record_paths=record_paths)
        all_keys = ('pdos_array', 'fluence_array', 'drr_array', '5cm_deep_array', 'iso_array', '5cm_shallow_array',
                    'deep_to_panel_array', 'iso_to_panel_array', 'shallow_to_panel_array')
        # ['pdos_array', 'drr_array', 'iso_array', '-5cm_array', '5cm_array']
        processors = [
            CProcessors.Squeeze(image_keys=all_keys),
            CProcessors.MultiplyImagesByConstant(keys=('drr_array', 'iso_array', '5cm_deep_array', '5cm_shallow_array'),
                                                 values=(1/1, 1/1, 1/1, 1/1)),#(1 / 325, 1 / 325, 1/325, 1/325)
            CProcessors.MultiplyImagesByConstant(keys=('pdos_array', 'fluence_array'),
                                                 values=(1, 1)),#(1 / 3.448, 1 / 2.226)
            CProcessors.ExpandDimension(axis=-1, image_keys=all_keys),
            CProcessors.CombineKeys(axis=-1, image_keys=all_keys, output_key='combined'),
            CProcessors.ReturnOutputs(input_keys=('combined',), output_keys=('out_file_name',)),
            {'batch': 1}, {'repeat'}
        ]
        train_generator.compile_data_set(image_processors=processors, debug=True)
        iterator = iter(train_generator.data_set)
        ratios = []
        # for i in range(len(train_generator)):
        #     x, y = next(iterator)
        #     ratio = x[0][0, 125, 125, 2] / x[0][0, 125, 125, 3]
        #     ratios.append(ratio)
        #     if ratio < 1:
        #         xxx = 1
        for i in range(len(train_generator)):
            x, y = next(iterator)
            #output, resized_pdos = generator.predict(x)
            numpy_array = x[0].numpy()
            file_info = str(y[0][0]).split('b')[-1][1:].split('.tf')[0]
            print(file_info)
            if file_info.split('_')[0] == '12':
                xxx = 1
            for i in range(numpy_array.shape[-1]):
                numpy_array[..., i] /= np.max(numpy_array[..., i])
            numpy_array *= 255
            np.save(os.path.join(out_path_numpy, "{}.npy".format(file_info)), numpy_array)
            max_val = np.max(numpy_array[...,-1])

            if max_val < 20:
                print("{} max is {}".format(file_info, max_val))
            out_array = np.zeros((256, 256 * numpy_array.shape[-1]))
            for i in range(numpy_array.shape[-1]):
                out_array[..., 256 * i:256 * (i + 1)] = numpy_array[..., i]
            image = Image.fromarray(out_array.astype('uint8'))
            image.save(os.path.join(out_path_jpeg, "{}.jpeg".format(file_info)))
    return None


def convlayerBMA(x, filters, size, apply_batchnorm=True):
    x = tf.keras.layers.Conv2D(filters, size, strides=1, padding='same', use_bias=True)(x)
    if apply_batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    return x


def resize_tensor(x, wanted_distance=1000, acquired_distance=1540):
    output_size = int(x.shape[1]//2*wanted_distance/acquired_distance*2)
    current_size = x.shape[1]
    x = tf.image.resize(x, [output_size, output_size])
    if wanted_distance < acquired_distance:
        x = tf.image.pad_to_bounding_box(x, (current_size - output_size) // 2,
                                         (current_size - output_size) // 2, current_size, current_size)
    else:
        x = x[:, (output_size-current_size)//2:-(output_size-current_size)//2,
            (output_size-current_size)//2:-(output_size-current_size)//2]
    return x


def GeneratorBMA2(top_layers=2, size=4, filters_start=64):
    """
    default values creates the original generator, filters double from start
    to a max after the number of 'double layers'
    Size is the kernel size
    Layers is the number of layers
    """
    """
    Back to basic physics
    """
    inputs = x = tf.keras.layers.Input(shape=[256, 256, 6])
    PDOS = tf.expand_dims(inputs[..., 0], axis=-1, name='PDOS')
    Fluence = tf.expand_dims(inputs[..., 1], axis=-1, name='PDOS')
    fulldrr = tf.expand_dims(inputs[..., 2], axis=-1)
    drr_deep = tf.expand_dims(inputs[..., 3], axis=-1)
    iso_drr = tf.expand_dims(inputs[..., 4], axis=-1)
    drr_shallow = tf.expand_dims(inputs[..., 5], axis=-1)

    drr_deep_to_iso = resize_tensor(drr_deep, wanted_distance=1000, acquired_distance=1050)
    drr_dif = drr_deep_to_iso - iso_drr
    drr_dif = tf.keras.layers.ReLU()(drr_dif)
    exp_shallow = tf.math.exp(-convlayerBMA(drr_shallow, 1, size, apply_batchnorm=False)) * PDOS * (1540**2/950**2)
    exp_iso = tf.math.exp(-convlayerBMA(iso_drr - drr_shallow, 1, size, apply_batchnorm=False)) * PDOS * (950**2/1000**2)
    exp_deeper = tf.math.exp(-convlayerBMA(drr_deep - iso_drr, 1, size, apply_batchnorm=False)) * PDOS * (1000**2 / 1050**2)
    exp_full = tf.math.exp(-convlayerBMA(fulldrr - drr_deep, 1, size, apply_batchnorm=False)) * PDOS * (1050**2 / 1540**2)

    x = tf.keras.layers.Concatenate()([exp_shallow, exp_iso, exp_deeper, exp_full])
    x = convlayerBMA(x, filters_start, size, apply_batchnorm=True)
    for _ in range(top_layers):
        x = convlayerBMA(x, filters_start, size, apply_batchnorm=True)
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=1, activation=None, padding='same', use_bias=True)(x)
    return tf.keras.Model(inputs=inputs, outputs=(x,drr_dif))


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
    create_files_for_streamline(records_path)
    return None
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
