import SimpleITK as sitk
import numpy as np
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image
from PreProcessingTools.Tools import *
import os
from glob import glob


def replace_CT_couch(primary_CT_handle: sitk.Image, cbct_handle: sitk.Image, table_vert: int) -> sitk.Image:
    spacing = cbct_handle.GetSpacing()
    ct_array = sitk.GetArrayFromImage(primary_CT_handle)
    cbct_array = sitk.GetArrayFromImage(cbct_handle)
    cbct_s = cbct_array.shape
    couch_stop = table_vert + int(50 * spacing[1])
    ct_array[:, couch_stop:, :] = -1000
    cbct_array[:, couch_stop:, :] = -1000
    ct_array[:, table_vert:couch_stop, :] = cbct_array[cbct_s[0] // 2, table_vert:couch_stop, cbct_s[-1] // 2][
        None, ..., None]
    fixed_ct_handle = array_to_sitk(ct_array, primary_CT_handle)
    return fixed_ct_handle


def remove_rods(primary_CT_handle: sitk.Image) -> sitk.Image:
    spacing = primary_CT_handle.GetSpacing()
    stats = sitk.LabelShapeStatisticsImageFilter()
    center = primary_CT_handle.TransformPhysicalPointToIndex((0, 0, 0))
    center_handle = primary_CT_handle[:, :, center[-1]]
    connected_handle = get_connected_image(center_handle, lowerThreshold=-900, upperThreshold=5000)
    stats.Execute(connected_handle)
    connected_array = sitk.GetArrayFromImage(connected_handle)
    for label in stats.GetLabels():
        centroid = stats.GetCentroid(label)
    x = 1


def running(patient_path, rewrite: True):
    patient_path = os.path.join(patient_path, "Niftiis")
    status_file = os.path.join(patient_path, "Finished_Padded_CBCT.txt")
    if os.path.exists(status_file) and not rewrite:
        return None
    CT_handle = sitk.ReadImage(os.path.join(patient_path, "Primary_CT_Updated.mha"))
    remove_rods(CT_handle)
    return None


def running_replace_couch(patient_path, rewrite: True):
    patient_path = os.path.join(patient_path, "Niftiis")
    status_file = os.path.join(patient_path, "Finished_Padded_CBCT.txt")
    if os.path.exists(status_file) and not rewrite:
        return None
    CT_handle = sitk.ReadImage(os.path.join(patient_path, "Primary_CT.mha"))
    out_file = os.path.join(patient_path, "Primary_CT_Updated.mha")
    CBCT_Files = glob(os.path.join(patient_path, 'Registered_CBCT*.mha'))
    for CBCT_File in CBCT_Files:
        table_file = CBCT_File.replace("Registered_", "TableHeight_").replace(".mha", ".txt")
        registered_handle = sitk.ReadImage(CBCT_File)
        fid = open(table_file)
        table_vert = int(fid.readline().split(', ')[1])
        fid.close()
        fixed_ct_handle = replace_CT_couch(CT_handle, registered_handle, table_vert)
        sitk.WriteImage(fixed_ct_handle, out_file)
        fid = open(status_file, 'w+')
        fid.close()
        return None


def main():
    patient_path = r'R:\Bojechko\phantom\1000'
    running(patient_path, True)


if __name__ == "__main__":
    main()
