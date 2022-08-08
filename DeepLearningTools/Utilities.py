import numpy as np
import os
from DeepLearningTools.Data_Generators.TFRecord_to_Dataset_Generator import DataGeneratorClass
from DeepLearningTools.Data_Generators.Image_Processors_Module.src.Processors.TFDataSets import ConstantProcessors as CProcessors,\
    RelativeProcessors as RProcessors
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image
from PIL import Image


def create_files_for_streamline(records_path):
    out_path_numpy = os.path.join(records_path, 'NumpyFiles')
    out_path_jpeg = os.path.join(records_path, 'JpegsNoNormalizationMultipleProj')
    for fold in [0]:
        train_path = os.path.join(records_path, 'Train', 'fold{}'.format(fold))
        record_paths = [train_path]
        if fold == 0:
            train_path = os.path.join(records_path, 'TrainNoNormalizationMultipleProj')
            record_paths = [
                os.path.join(train_path, 'phantom_valid'), os.path.join(train_path, 'phantom_train'),
                os.path.join(train_path, 'fold1'),
                os.path.join(train_path, 'fold2'),
                os.path.join(train_path, 'fold3'),
                os.path.join(train_path, 'fold4'),
                os.path.join(train_path, 'fold5')
            ]
        # record_paths = [train_path]
        train_generator = DataGeneratorClass(record_paths=record_paths)
        all_keys = ('pdos_array', 'fluence_array', 'drr_array', '5cm_deep_array', 'iso_array', '5cm_shallow_array',
                    'deep_to_panel_array', 'iso_to_panel_array', 'shallow_to_panel_array')
        # ['pdos_array', 'drr_array', 'iso_array', '-5cm_array', '5cm_array']
        processors = [
            CProcessors.Squeeze(image_keys=all_keys),
            CProcessors.MultiplyImagesByConstant(keys=('drr_array', 'iso_array', '5cm_deep_array', '5cm_shallow_array'),
                                                 values=(255/300, 255/300, 255/300, 255/300)),#(1 / 325, 1 / 325, 1/325, 1/325)
            RProcessors.NormalizeBasedOnOther(guiding_keys=('pdos_array', 'pdos_array'),
                                              changing_keys=('fluence_array', 'pdos_array'),
                                              reference_method=('reduce_max', 'reduce_max'),
                                              changing_methods=('divide', 'divide')),
            CProcessors.MultiplyImagesByConstant(keys=('pdos_array', 'fluence_array'),
                                                 values=(255, 255)),  # (1 / 3.448, 1 / 2.226)
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
            # for i in range(numpy_array.shape[-1]):
            #     numpy_array[..., i] /= np.max(numpy_array[..., i])
            # numpy_array *= 255
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


def main():
    records_path = r'\\ad.ucsd.edu\ahs\radon\research\Bojechko\TFRecords'
    create_files_for_streamline(records_path)
    return None


if __name__ == '__main__':
    main()
    pass
