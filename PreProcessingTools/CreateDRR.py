import sys
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


def main():
    patient_path = r'C:\Users\b5anderson\Desktop\Modular_Projects\Image_Prediction\Data\Patient'
    Dicom_reader = DicomReaderWriter(description='Examples', verbose=True)
    Dicom_reader.down_folder(os.path.join(patient_path, 'CT'))
    # for index in Dicom_reader.indexes_with_contours:
    #   Dicom_reader.set_index(index)
    #   Dicom_reader.get_images()
    #   date = Dicom_reader.reader.GetMetaData(0, "0008|0022")
    #   print(index)
    #   print(date)
    # Loading 3D CT image
    Dicom_reader.set_index(6)  # Primary CT
    Dicom_reader.get_images()
    CT_handle = Dicom_reader.dicom_handle
    sitk.WriteImage(CT_handle, os.path.join('.', "Primary_CT.mha"))
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
                    sitk.WriteImage(registered_handle, os.path.join('.', "Registered_CBCT_{}.mha".format(index)))
                    create_drr(registered_handle, gantry_angle=45, sid=1000, spd=1540,
                               out_path=os.path.join('.', 'CBCT_{}_DRR.mha'.format(index)))



    create_drr(CT_handle, gantry_angle=45, sid=1000, spd=1540, out_path=os.path.join('.', 'Primary_CT_DRR.mha'))

    return None


if __name__ == '__main__':
    main()
