from matplotlib import pyplot as plt
from glob import glob
from PreProcessingTools.itk_sitk_converter import *
import SimpleITK as sitk
import os
# from itk import RTK as rtk
from PreProcessingTools.RegisteringImages.src.RegisterImages.WithDicomReg import registerDicom
import itk
import numpy as np
from DicomRTTool.ReaderWriter import DicomReaderWriter, plot_scroll_Image, pydicom


def rotate_and_translate_image(itk_image, translations=(0, 0, 0), rotations=(0, 0, 0)):
    """
    :param itk_image:
    :param translations: translations in mm
    :param rotations: in degrees
    :return:
    """
    origin = itk_image.GetOrigin()
    imSize = itk_image.GetLargestPossibleRegion().GetSize()
    imRes = itk_image.GetSpacing()
    imDirection = itk_image.GetDirection()
    isocenter = [0, 0, 0]
    for i in range(3):
        isocenter[i] = origin[i] + imRes[i] * imSize[i] / 2

    transform = itk.Euler3DTransform[itk.D].New()
    transform.SetComputeZYX(True)
    transform.SetCenter(isocenter)
    transform.SetTranslation(translations)
    transform.SetRotation(np.deg2rad(rotations[0]), np.deg2rad(rotations[1]), np.deg2rad(rotations[2]))

    interpolator = itk.LinearInterpolateImageFunction[itk.Image[itk.F, 3], itk.D].New()

    filter = itk.ResampleImageFilter[itk.Image[itk.F, 3], itk.Image[itk.F, 3]].New()
    filter.SetInterpolator(interpolator)
    filter.SetDefaultPixelValue(-1000)
    filter.SetOutputOrigin(origin)
    filter.SetOutputSpacing(imRes)
    filter.SetOutputDirection(imDirection)
    filter.SetSize(imSize)
    filter.SetTransform(transform)

    filter.SetInput(itk_image)
    filter.Update()
    output = filter.GetOutput()
    return output


def create_drr(sitk_handle, sid=1000, spd=1540, gantry_angle=0, out_path=os.path.join('.', 'Output.mha')):
    """
    :param sitk_handle: handle from SimpleITK, usually from DICOMRTTool
    :param sid: source to iso-center distance, in mm
    :param spd: source to panel distance, in mm
    :param gantry_angle: angle of gantry
    :param out_path: name of out file
    :return:
    """
    image = ConvertSimpleItkImageToItkImage(sitk_handle, itk.F)
    rotations = [0, 0, gantry_angle]  # x, y, z

    translations = [0, 0, 0]  # x, y, z

    rprojection = 0.  # Projection angle in degrees
    threshold = -1000
    imRes = image.GetSpacing()

    im_sx = imRes[0]
    im_sy = imRes[1]

    imSize = image.GetLargestPossibleRegion().GetSize()
    dx = imSize[0]
    dy = imSize[1]

    Dimension = 3

    InputImageType = itk.Image[itk.F, Dimension]

    origin = image.GetOrigin()

    spacing = image.GetSpacing()

    final_filter = itk.ResampleImageFilter[InputImageType, InputImageType].New()
    final_filter.SetDefaultPixelValue(0)
    transformed_image = rotate_and_translate_image(image, translations=translations, rotations=rotations)
    final_filter.SetInput(transformed_image)

    transform = itk.Euler3DTransform[itk.D].New()
    transform.SetComputeZYX(True)

    transform.SetTranslation((0, 0, 0)) # Do not change these!
    transform.SetRotation(0, 0, 0) # Do not change these!
    isocenter = [0, 0, 0]
    for i in range(3):
        isocenter[i] = origin[i] + imRes[i] * imSize[i] / 2
    transform.SetCenter(isocenter)

    interpolator = itk.SiddonJacobsRayCastInterpolateImageFunction[InputImageType, itk.D].New()
    interpolator.SetProjectionAngle(np.deg2rad(rprojection))
    interpolator.SetFocalPointToIsocenterDistance(sid)
    interpolator.SetThreshold(threshold)
    interpolator.SetTransform(transform)
    interpolator.Initialize()
    final_filter.SetInterpolator(interpolator)

    final_filter.SetSize([dx, dy, 1])
    final_filter.SetOutputSpacing(spacing)

    o2Dx = (dx - 1) / 2
    o2Dy = (dy - 1) / 2

    origin[0] += im_sx * o2Dx
    origin[1] += im_sy * o2Dy
    origin[2] = -spd

    final_filter.SetOutputOrigin(origin)
    final_filter.SetOutputDirection(image.GetDirection())
    final_filter.Update()

    flipFilter = itk.FlipImageFilter[InputImageType].New()
    flipFilter.SetInput(final_filter.GetOutput())
    flipFilter.SetFlipAxes((False, True, False))

    writer = itk.ImageFileWriter[InputImageType].New()
    writer.SetFileName(out_path)
    writer.SetInput(flipFilter.GetOutput())
    writer.Update()
    return None


def array_to_sitk(array: np.ndarray, reference_handle: sitk.Image):
    out_handle = sitk.GetImageFromArray(array)
    out_handle.SetSpacing(reference_handle.GetSpacing())
    out_handle.SetOrigin(reference_handle.GetOrigin())
    out_handle.SetDirection(reference_handle.GetDirection())
    return out_handle


def fix_DRR(cbct_drr: sitk.Image, ct_drr: sitk.Image):
    slice_thickness = cbct_drr.GetSpacing()[-1]
    cbct_array = sitk.GetArrayFromImage(cbct_drr)
    ct_array = sitk.GetArrayFromImage(ct_drr)
    y = np.sum(cbct_array.astype('bool')[0], axis=-1) # Flatten the array into a line, looking for the spot to move
    values = np.where(y > 0)[0]
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    dy = np.diff(y, 1)
    dx = np.ones(y.shape[0] - 1)
    yfirst = np.abs(dy / dx)
    yfirst[yfirst < 1] = 0
    yfirst_convolved = np.convolve(yfirst, kernel, mode='same')
    start_vals = np.where((yfirst_convolved[:-1] > .75) & (yfirst_convolved[1:] < .75))[0]
    start = int(start_vals[0] + 20 / slice_thickness)
    stop_vals = np.where((yfirst_convolved[:-1] < .75) & (yfirst_convolved[1:] > .75))[0]
    stop = int(stop_vals[-1] - 20 / slice_thickness)
    cbct_array[:, :start, :] = ct_array[:, :start, :]
    cbct_array[:, stop:, :] = ct_array[:, stop:, :]
    # plt.plot(axis_sum) # <-- can see the hump
    cbct_handle = array_to_sitk(cbct_array, reference_handle=ct_drr)
    return cbct_handle


def expandDRR(patient_path):
    if not os.path.exists(os.path.join(patient_path, 'Primary_CT_DRR.mha')):
        print("Could not find Primary_CT_DRR!")
        return None
    CBCT_DRR_Files = glob(os.path.join(patient_path, 'CBCT*DRR.mha'))
    CTDRRhandle = sitk.ReadImage(os.path.join(patient_path, 'Primary_CT_DRR.mha'))
    for cbct_drr_file in CBCT_DRR_Files:
        print("Writing padded CBCT for {}".format(cbct_drr_file))
        file_name = os.path.split(cbct_drr_file)[-1]
        CBCTDRRhandle = sitk.ReadImage(cbct_drr_file)
        cbct_handle = fix_DRR(cbct_drr=CBCTDRRhandle, ct_drr=CTDRRhandle)
        sitk.WriteImage(cbct_handle, os.path.join(patient_path, file_name.replace(".mha", "_Padded.mha")))
    return None


def get_binary_image(annotation_handle, lowerThreshold, upperThreshold):
    thresholded_image = sitk.BinaryThreshold(annotation_handle, lowerThreshold=lowerThreshold,
                                             upperThreshold=upperThreshold)
    return thresholded_image


def get_connected_image(annotation_handle, lowerThreshold=-900, upperThreshold=5000):
    Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
    Connected_Component_Filter.FullyConnectedOff()
    RelabelComponent = sitk.RelabelComponentImageFilter()
    RelabelComponent.SortByObjectSizeOn()
    thresholded_image = get_binary_image(annotation_handle, lowerThreshold, upperThreshold)
    connected_image = Connected_Component_Filter.Execute(thresholded_image)
    connected_image = RelabelComponent.Execute(connected_image)
    return connected_image


def get_outside_body_contour(annotation_handle, lowerThreshold, upperThreshold):
    connected_images = get_connected_image(annotation_handle, lowerThreshold, upperThreshold)
    outside_body = get_binary_image(connected_images, lowerThreshold=1, upperThreshold=1)
    for i in range(outside_body.GetSize()[-1]):
        connected_image = get_connected_image(outside_body[:, :, i], lowerThreshold=1, upperThreshold=1)
        binary_image = get_binary_image(connected_image, lowerThreshold=1, upperThreshold=1)
        outside_body[:, :, i] = binary_image
    return outside_body


def createDRRs(patient_path):
    dilate_filter = sitk.BinaryDilateImageFilter()
    dilate_filter.SetKernelType(sitk.sitkBall)
    dilate_filter.SetKernelRadius((0, 0, 5))
    Dicom_reader = DicomReaderWriter(description='Examples', verbose=True)
    Dicom_reader.down_folder(os.path.join(patient_path, 'CT'))
    # for index in Dicom_reader.indexes_with_contours:
    #   Dicom_reader.set_index(index)
    #   Dicom_reader.get_images()
    #   date = Dicom_reader.reader.GetMetaData(0, "0008|0022")
    #   print(index)
    #   print(date)
    # Loading 3D CT image
    CT_SIUID = None
    for index in Dicom_reader.indexes_with_contours:
        if Dicom_reader.series_instances_dictionary[index]['Description'] is not None:
            Dicom_reader.set_index(index)  # Primary CT
            Dicom_reader.get_images()
            CT_handle = Dicom_reader.dicom_handle
            ct_array = Dicom_reader.ArrayDicom
            sitk.WriteImage(CT_handle, os.path.join(patient_path, "Primary_CT.mha"))
            CT_SIUID = Dicom_reader.series_instances_dictionary[6]['SeriesInstanceUID']

    reg_path = os.path.join(patient_path, 'REG')
    for file in os.listdir(reg_path):
        ds = pydicom.read_file(os.path.join(reg_path, file))
        for ref in ds.ReferencedSeriesSequence:
            from_uid = ref.SeriesInstanceUID
            if from_uid == CT_SIUID:
                continue
            for index in Dicom_reader.indexes_with_contours:
                if Dicom_reader.series_instances_dictionary[index]['SeriesInstanceUID'] == from_uid:
                    Dicom_reader.set_index(index)  # Primary CT
                    Dicom_reader.get_images()
                    cbct_handle = Dicom_reader.dicom_handle
                    sitk.WriteImage(cbct_handle, os.path.join('.', "CBCT_{}.mha".format(index)))
                    registered_handle = registerDicom(fixed_image=CT_handle,  moving_image=cbct_handle,
                                                      moving_series_instance_uid=from_uid,
                                                      dicom_registration=ds, min_value=-1000, method=sitk.sitkLinear)
                    outside_body = get_outside_body_contour(registered_handle, lowerThreshold=-2000, upperThreshold=-300)
                    dilated_handle = dilate_filter.Execute(outside_body)
                    cbct_array = sitk.GetArrayFromImage(registered_handle)
                    cbct_array[sitk.GetArrayFromImage(dilated_handle) == 1] = ct_array[sitk.GetArrayFromImage(dilated_handle) == 1]
                    registered_handle = array_to_sitk(cbct_array, registered_handle)
                    sitk.WriteImage(registered_handle, os.path.join('.', "Registered_CBCT_{}.mha".format(index)))
                    create_drr(registered_handle, gantry_angle=0, sid=1000, spd=1540,
                               out_path=os.path.join('.', 'CBCT_{}_DRR.mha'.format(index)))
    create_drr(CT_handle, gantry_angle=0, sid=1000, spd=1540, out_path=os.path.join('.', 'Primary_CT_DRR.mha'))


def main():
    patient_path = r'C:\Users\b5anderson\Desktop\Modular_Projects\Image_Prediction\Data\Patient'
    if True:
        createDRRs(patient_path=patient_path)
    if False:
        expandDRR(patient_path='.')
    return None


if __name__ == '__main__':
    main()
