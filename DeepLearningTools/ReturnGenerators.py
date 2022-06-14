import numpy as np
import os
from DeepLearningTools.Data_Generators.TFRecord_to_Dataset_Generator import DataGeneratorClass
from DeepLearningTools.Data_Generators.Image_Processors_Module.src.Processors.TFDataSets import ConstantProcessors as CProcessors,\
    RelativeProcessors as RProcessors
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image
import tensorflow as tf


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


def return_generator(records_path, batch=1, proj_to_panel=True, add_5cm_keys=True):
    generator = DataGeneratorClass(record_paths=records_path, delete_old_cache=True)
    all_keys = ('pdos_array', 'fluence_array','drr_array', 'deep_to_panel_array', 'iso_to_panel_array', 'shallow_to_panel_array')
    drr_keys = ('drr_array', 'deep_to_panel_array', 'iso_to_panel_array', 'shallow_to_panel_array', )
    input_keys = ('pdos_array', 'drr_array', 'iso_to_panel_array')
    if add_5cm_keys:
        if proj_to_panel:
            input_keys = ('pdos_array', 'drr_array', 'deep_to_panel_array', 'iso_to_panel_array',
                          'shallow_to_panel_array')
        else:
            input_keys = ('pdos_array', 'drr_array', '5cm_deep_array', 'iso_array', 'shallow_array')
    print(f"Inputs are {input_keys}")
    base_processors = [
        CProcessors.Squeeze(image_keys=all_keys),
        CProcessors.ExpandDimension(axis=-1, image_keys=all_keys),
        CProcessors.MultiplyImagesByConstant(keys=drr_keys, values=(1/90, 1/90, 1/90, 1/90)),
        CProcessors.CombineKeys(axis=-1,
                                image_keys=input_keys,
                                output_key='test'),
        CProcessors.ReturnOutputs(input_keys=('test',),
                                  output_keys=('fluence_array',)),
        {'batch': batch}, {'repeat'}
    ]
    generator.compile_data_set(image_processors=base_processors, debug=False)
    return generator


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


def return_fold_datasets(data_generators, batch_size=1):
    all_datasets = return_datasets(data_generators)

    train_dataset = all_datasets['train']
    train_dataset = train_dataset.shuffle(len(train_dataset))
    train_dataset = train_dataset.batch(batch_size)

    valid_dataset = all_datasets['validation']
    # valid_dataset = valid_dataset.shuffle(len(valid_dataset))
    valid_dataset = valid_dataset.batch(1)
    return train_dataset, valid_dataset


def main():
    pass


if __name__ == '__main__':
    pass
