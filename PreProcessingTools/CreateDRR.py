import sys

import SimpleITK as sitk
import os
# from itk import RTK as rtk
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


def main():
    if os.path.exists(os.path.join('.', 'stack.mha')):
        image_handle = sitk.ReadImage('stack.mha')

    data_path = r'C:\Users\b5anderson\Desktop\Modular_Projects\Image_Prediction\Data\Patient1\CT'
    if not os.path.exists(os.path.join('.', "Image.mha")):
        Dicom_reader = DicomReaderWriter(description='Examples', verbose=True)
        Dicom_reader.down_folder(data_path)
        Dicom_reader.set_index(0)  # 5
        Dicom_reader.get_images()
        dicom_handle = Dicom_reader.dicom_handle
        # for index in Dicom_reader.indexes_with_contours:
        #   Dicom_reader.set_index(index)
        #   Dicom_reader.get_images()
        #   date = Dicom_reader.reader.GetMetaData(0, "0008|0022")
        #   print(index)
        #   print(date)
        # Loading 3D CT image
        sitk.WriteImage(dicom_handle, os.path.join('.', "Image.mha"))
    else:
        dicom_handle = sitk.ReadImage(os.path.join('.', "Image.mha"))
    # dicom_handle.SetDirection((0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    # sitk.WriteImage(dicom_handle, "Test.nii.gz")
    image = itk.imread(os.path.join('.', "Image.mha"), pixel_type=itk.F)
    rotations = [0, 0, 45]  # x, y, z

    translations = [0, 0, 0]  # x, y, z

    rprojection = 0.  # Projection angle in degrees
    sid = 1000  # source to isocenter
    spd = 1000  # source to panel distance
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
    transformed_image = rotate_and_translate_image(image, rotations)
    final_filter.SetInput(transformed_image)

    transform = itk.Euler3DTransform[itk.D].New()
    transform.SetComputeZYX(True)

    transform.SetTranslation((0, 0, 0))
    transform.SetRotation(0, 0, 0)
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
    # origin = [0, 0, 0]
    origin[0] += im_sx * o2Dx
    origin[1] += im_sy * o2Dy
    origin[2] = -spd

    final_filter.SetOutputOrigin(origin)
    # filter.SetOutputDirection(image.GetDirection())
    final_filter.Update()

    flipFilter = itk.FlipImageFilter[InputImageType].New()
    flipFilter.SetInput(final_filter.GetOutput())
    flipFilter.SetFlipAxes((False, True, False))

    writer = itk.ImageFileWriter[InputImageType].New()
    writer.SetFileName(os.path.join('.', 'Output.mha'))
    writer.SetInput(flipFilter.GetOutput())
    writer.Update()


if __name__ == '__main__':
    main()
