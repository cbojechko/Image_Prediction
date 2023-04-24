import SimpleITK as sitk
import os
import numpy as np
from glob import glob
from PlotScrollNumpyArrays import plot_scroll_Image


def get_binary_image(annotation_handle, lowerThreshold, upperThreshold):
    thresholded_image = sitk.BinaryThreshold(annotation_handle, lowerThreshold=lowerThreshold,
                                             upperThreshold=upperThreshold)
    return thresholded_image


def pad_cbct(meta_handle: sitk.Image, cbct_handle: sitk.Image, ct_handle: sitk.Image,
             erode_filter: sitk.BinaryErodeImageFilter, couch_start: int):
    """
    :param cbct_handle:
    :param ct_handle:
    :param expansion: expansion to explore, in cm
    :return:
    """
    ct_array = sitk.GetArrayFromImage(ct_handle)
    cbct_array = sitk.GetArrayFromImage(cbct_handle)
    cbct_s = cbct_array.shape
    spacing = cbct_handle.GetSpacing()
    couch_stop = couch_start + int(50 * spacing[1])
    ct_array[:, couch_stop:, :] = -1000
    cbct_array[:, couch_stop:, :] = -1000
    ct_array[:, couch_start:couch_stop, :] = cbct_array[cbct_s[0]//2, couch_start:couch_stop, cbct_s[-1]//2][None, ..., None]
    cbct_array[:, couch_start:couch_stop, :] = cbct_array[:, couch_start:couch_stop, cbct_s[-1]//2][..., None]
    binary_meta = get_binary_image(meta_handle, lowerThreshold=1, upperThreshold=2)
    eroded_meta = erode_filter.Execute(binary_meta)
    eroded_meta_array = sitk.GetArrayFromImage(eroded_meta)
    cbct_array[eroded_meta_array != 1] = ct_array[eroded_meta_array != 1]
    padded_cbct_handle = array_to_sitk(cbct_array, cbct_handle)
    return padded_cbct_handle


def array_to_sitk(array: np.ndarray, reference_handle: sitk.Image):
    out_handle = sitk.GetImageFromArray(array)
    out_handle.SetSpacing(reference_handle.GetSpacing())
    out_handle.SetOrigin(reference_handle.GetOrigin())
    out_handle.SetDirection(reference_handle.GetDirection())
    return out_handle


def main_update_CBCT(path=r'R:\Bojechko\phantom'):
    phantom_directories = os.listdir(path)
    for phantom_dir in phantom_directories:
        print(phantom_dir)
        full_path = os.path.join(path, phantom_dir)
        update_CBCT(os.path.join(full_path, 'Niftiis'))


if __name__ == '__main__':
    pass
