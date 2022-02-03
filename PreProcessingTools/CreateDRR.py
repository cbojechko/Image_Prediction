import sys

import SimpleITK as sitk
import os
#from itk import RTK as rtk
import itk
import numpy as np
from DicomRTTool.ReaderWriter import DicomReaderWriter, plot_scroll_Image, pydicom

if os.path.exists(os.path.join('.', 'stack.mha')):
  image_handle = sitk.ReadImage('stack.mha')
  array = sitk.GetArrayFromImage(image_handle)


data_path = r'C:\Users\b5anderson\Desktop\Modular_Projects\Image_Prediction\Data\Patient1\CT'
if not os.path.exists(os.path.join('.', "Image.mha")):
  Dicom_reader = DicomReaderWriter(description='Examples', verbose=True)
  Dicom_reader.down_folder(data_path)
  Dicom_reader.set_index(0) # 5
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
rotations = [0, 0, 0] # x, y, z

translations = [0, 0, 0] # x, y, z

pixel_iso_center = [0, 0, 0] # x, y, z
rprojection = 0. # Projection angle in degrees
sid = 1000 # source to isocenter
imRes = dicom_handle.GetSpacing()

im_sx = imRes[0]
im_sy = imRes[1]

imSize = dicom_handle.GetSize()
dx = imSize[0]
dy = imSize[1]

o2Dx = 0.
o2Dy = 0.

threshold = 0

Dimension = 3

InputImageType = itk.Image[itk.F, Dimension]
OutputImageType = itk.Image[itk.F, Dimension]


image = itk.imread(os.path.join('.', "Image.mha"), pixel_type=itk.F)
origin = image.GetOrigin()

spacing = image.GetSpacing()
region = image.GetBufferedRegion()

filter = itk.ResampleImageFilter[InputImageType, InputImageType].New()
filter.SetDefaultPixelValue(0)
filter.SetInput(image)

transform = itk.Euler3DTransform[itk.D].New()
transform.SetComputeZYX(True)

dtr = np.arctan(1)*4/180

transform.SetTranslation(translations)
transform.SetRotation(rotations[0] * dtr, rotations[1] * dtr, rotations[2] * dtr)

isocenter = [0, 0, 0]
for i in range(3):
  isocenter[i] = origin[i] + imRes[i] * imSize[i]/2
transform.SetCenter(isocenter)

interpolator = itk.SiddonJacobsRayCastInterpolateImageFunction[InputImageType, itk.D].New()
interpolator.SetProjectionAngle(rprojection*dtr)
interpolator.SetFocalPointToIsocenterDistance(sid)
interpolator.SetThreshold(threshold)
interpolator.SetTransform(transform)
interpolator.Initialize()
filter.SetInterpolator(interpolator)

filter.SetSize([dx, dy, 1])
filter.SetOutputSpacing(spacing)

o2Dx = (dx-1)/2
o2Dy = (dy-1)/2
#origin = [0, 0, 0]
origin[0] += im_sx * o2Dx
origin[1] += im_sy * o2Dy
origin[2] = -sid

filter.SetOutputOrigin(origin)
filter.SetOutputDirection(image.GetDirection())
filter.Update()

output = filter.GetOutput()

writer = itk.ImageFileWriter[InputImageType].New()
writer.SetFileName(os.path.join('.','Output.mha'))
writer.SetInput(output)
writer.Update()