import SimpleITK as sitk
import os
import numpy as np
from glob import glob
from PlotScrollNumpyArrays import plot_scroll_Image


def array_to_sitk(array: np.ndarray, reference_handle: sitk.Image):
    out_handle = sitk.GetImageFromArray(array)
    out_handle.SetSpacing(reference_handle.GetSpacing())
    out_handle.SetOrigin(reference_handle.GetOrigin())
    out_handle.SetDirection(reference_handle.GetDirection())
    return out_handle


def update_CBCT(nifti_path, rewrite=False):
    status_file = os.path.join(nifti_path, "Padded_from_air.txt")
    if os.path.exists(status_file) and not rewrite:
        return None
    padded_cbcts = glob(os.path.join(nifti_path, "Padded_CBCT_*"))
    for padded_cbct in padded_cbcts:
        padded_handle = sitk.ReadImage(padded_cbct)
        spacing = padded_handle.GetSpacing()
        padded_np = sitk.GetArrayFromImage(padded_handle)
        flattened_padded = np.max(padded_np, axis=(1, 2))
        air = np.where(flattened_padded == -1000)[0]
        not_air = np.where(flattened_padded > -1000)[0]
        if len(air) > 0:
            if len(not_air) > 0:
                start = not_air[0] + int(20/spacing[-1])
                stop = not_air[-1] - int(20/spacing[-1])
                padded_np[:start] = padded_np[start]
                padded_np[stop:] = padded_np[stop]
                new_padded_handle = array_to_sitk(padded_np, reference_handle=padded_handle)
                sitk.WriteImage(new_padded_handle, padded_cbct)
                print('Rewriting {}'.format(nifti_path))
        fid = open(status_file, 'w+')
        fid.close()


def main_update_CBCT(path=r'R:\Bojechko\phantom'):
    phantom_directories = os.listdir(path)
    for phantom_dir in phantom_directories:
        print(phantom_dir)
        full_path = os.path.join(path, phantom_dir)
        update_CBCT(os.path.join(full_path, 'Niftiis'))


if __name__ == '__main__':
    pass
