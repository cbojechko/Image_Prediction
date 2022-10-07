import numpy as np
import tensorflow as tf
import tensorboard as tb
import os
from DeepLearningTools.Finding_Optimization_Parameters import History_Plotter_TF2 as hp


def main():
    base_path = r'\\ad.ucsd.edu\ahs\radon\research\Bojechko\Model_Outputs'
    excel_sheet = os.path.join(base_path, 'Model_Parameters.xlsx')
    #tb.data.experimental.ExperimentFromDev('5')
    all_dictionaries = {}
    hp.build_from_backend(os.path.join(base_path, 'Model_1', 'validation'), all_dictionaries=all_dictionaries)
    hp.iterate_paths_add_to_dictionary(path=base_path, all_dictionaries=all_dictionaries,
                                       fraction_start=.25, weight_smoothing=.6,
                                       metric_name_and_criteria={'epoch_mean_absolute_error': np.min})
    xxx = 1


if __name__ == '__main__':
    main()
