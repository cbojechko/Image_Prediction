import numpy as np
import SimpleITK as sitk


def array_to_sitk(array: np.ndarray, reference_handle: sitk.Image):
    out_handle = sitk.GetImageFromArray(array)
    out_handle.SetSpacing(reference_handle.GetSpacing())
    out_handle.SetOrigin(reference_handle.GetOrigin())
    out_handle.SetDirection(reference_handle.GetDirection())
    return out_handle


def get_binary_image(annotation_handle, lowerThreshold, upperThreshold):
    thresholded_image = sitk.BinaryThreshold(annotation_handle, lowerThreshold=lowerThreshold,
                                             upperThreshold=upperThreshold)
    return thresholded_image


def get_connected_image(annotation_handle, lowerThreshold=-900, upperThreshold=5000) -> sitk.Image:
    Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
    Connected_Component_Filter.FullyConnectedOff()
    RelabelComponent = sitk.RelabelComponentImageFilter()
    RelabelComponent.SortByObjectSizeOn()
    thresholded_image = get_binary_image(annotation_handle, lowerThreshold, upperThreshold)
    connected_image = Connected_Component_Filter.Execute(thresholded_image)
    connected_image = RelabelComponent.Execute(connected_image)
    return connected_image


if __name__ == '__main__':
    pass
