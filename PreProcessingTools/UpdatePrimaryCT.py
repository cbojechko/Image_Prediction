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
    """
    General flow is this
    Find the iso-center of the CT and take that 2D slice
    Identify the rails by making a binary image
    Each rail should be almost an exact circle
    Make a binary mask of these circles. Expand them by 3 centimeters
    Mask the primary CT handle
    """
    stats = sitk.LabelShapeStatisticsImageFilter()
    center = primary_CT_handle.TransformPhysicalPointToIndex((0, 0, 0))
    center_handle = primary_CT_handle[:, :, center[-1]]
    connected_handle = get_connected_image(center_handle, lowerThreshold=-900, upperThreshold=5000)
    stats.Execute(connected_handle)
    connected_array = sitk.GetArrayFromImage(connected_handle)
    mask_rails = np.zeros(connected_array.shape)
    for label in stats.GetLabels():
        radius = stats.GetEquivalentSphericalRadius(label)
        perimeter = stats.GetPerimeter(label)
        equivalent_radius = perimeter/(2*3.141529654)
        percent_difference = abs((radius - equivalent_radius) / radius * 100)
        """
        Check if it is 'pretty much' a circle
        """
        if percent_difference < 5:
            mask_rails[connected_array == label] = 1
    """
    Now, dilate the rails mask by 3 cm
    """
    spacing = primary_CT_handle.GetSpacing()
    dilate_filter = sitk.BinaryDilateImageFilter()
    dilate_filter.SetKernelType(sitk.sitkBall)
    dilate_filter.SetKernelRadius((int(15 / spacing[0]), int(15 / spacing[1])))  # x, y
    mask_rails_handle = sitk.GetImageFromArray(mask_rails.astype('int'))
    dilated_rails_handle = dilate_filter.Execute(mask_rails_handle)
    dilated_rails_array = sitk.GetArrayFromImage(dilated_rails_handle)
    """
    Duplicate this mask in the sup-inf direction
    Replace the values with air at -1000
    """
    dilated_rails_array = np.concatenate([dilated_rails_array[None, ...] for _ in range(primary_CT_handle.GetSize()[-1])])
    primary_CT_array = sitk.GetArrayFromImage(primary_CT_handle)
    primary_CT_array[dilated_rails_array == 1] = -1000
    out_handle = array_to_sitk(primary_CT_array, primary_CT_handle)
    return out_handle


def replace_ionchamber(primary_CT_handle: sitk.Image) -> sitk.Image:
    stats = sitk.LabelShapeStatisticsImageFilter()
    connected_handle = get_connected_image(primary_CT_handle, lowerThreshold=2500, upperThreshold=9999)
    stats.Execute(connected_handle)
    """
    We will end up getting slightly more volumes, because we are getting the BBs
    """
    spacing = primary_CT_handle.GetSpacing()
    dilate_filter = sitk.BinaryDilateImageFilter()
    dilate_filter.SetKernelType(sitk.sitkBall)
    dilate_filter.SetKernelRadius((int(12 / spacing[0]), int(12 / spacing[1]), int(20 / spacing[2])))  # x, y, z
    for label in stats.GetLabels():
        if stats.GetEquivalentSphericalRadius(label) < 3:  # Should be at least 5 cm in equivalent radius from volume...
            continue
        binary_chamber = connected_handle == label
        dilated_chamber = dilate_filter.Execute(binary_chamber)
        dilated_chamber_array = sitk.GetArrayFromImage(dilated_chamber)
        rolled_dilated_chamber_array = np.roll(dilated_chamber_array, -int(31/spacing[1]), axis=1)
        primary_CT_array = sitk.GetArrayFromImage(primary_CT_handle)
        primary_CT_array[dilated_chamber_array == 1] = primary_CT_array[rolled_dilated_chamber_array == 1]
        primary_CT_handle = array_to_sitk(primary_CT_array, primary_CT_handle)
        return primary_CT_handle
    print("No chamber found here")
    return primary_CT_handle


def update_HU(primary_CT_handle: sitk.Image) -> sitk.Image:
    scale = 1.0125
    bias = -25.27
    primary_CT_handle = primary_CT_handle * scale + bias
    return primary_CT_handle


def update_primary_CT(patient_path, rewrite: True):
    patient_path = os.path.join(patient_path, "Niftiis")
    status_file = os.path.join(patient_path, "Finished_UpdatingPrimary.txt")
    if os.path.exists(status_file) and not rewrite:
        return None
    primary_path = os.path.join(patient_path, "Primary_CT.mha")
    if not os.path.exists(primary_path):
        return None
    CT_handle = sitk.ReadImage(primary_path)
    print("Replacing couch")
    CT_handle = replace_couch(patient_path, CT_handle)
    print("Removing rods")
    CT_handle = remove_rods(CT_handle)
    print("Removing ion chamber")
    CT_handle = replace_ionchamber(CT_handle)
    print("Shifting HU")
    CT_handle = update_HU(CT_handle)
    sitk.WriteImage(CT_handle, os.path.join(patient_path, "Primary_CT_Updated.mha"))
    fid = open(status_file, 'w+')
    fid.close()
    return None


def replace_couch(patient_path, CT_handle: sitk.Image) -> sitk.Image:
    CBCT_Files = glob(os.path.join(patient_path, 'Registered_CBCT*.mha'))
    for CBCT_File in CBCT_Files:
        table_file = CBCT_File.replace("Registered_", "TableHeight_").replace(".mha", ".txt")
        registered_handle = sitk.ReadImage(CBCT_File)
        fid = open(table_file)
        table_vert = int(fid.readline().split(', ')[1])
        fid.close()
        fixed_ct_handle = replace_CT_couch(CT_handle, registered_handle, table_vert)
        return fixed_ct_handle


def main():
    patient_path = r'R:\Bojechko\phantom'
    for p in os.listdir(patient_path):
        print(f"Running for patient {p}")
        update_primary_CT(os.path.join(patient_path, p), False)


if __name__ == "__main__":
    pass
