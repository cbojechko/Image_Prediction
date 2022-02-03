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
InputImageType = itk.Image[itk.F, 3]

CT = itk.imread(os.path.join('.', "Image.mha"), pixel_type=itk.F)
region = CT.GetBufferedRegion()

filter = itk.ResampleImageFilter[InputImageType, InputImageType].New()
filter.SetDefaultPixelValue(0)
filter.SetInput(CT)

transform = itk.Euler3DTransform[itk.D].New()
transform.SetComputeZYX(True)

dtr = np.arctan(1)*4/180
transform.SetTranslation((0, 0, 0))
transform.SetRotation(dtr*90, 0, 0)

imSize = region.GetSize()
imOrigin = dicom_handle.GetOrigin()

isocenter = [0, 0, 0]
imRes = dicom_handle.GetSpacing()
isocenter[0] = imOrigin[0] + imRes[0] * (imSize[0]) / 2.0
isocenter[1] = imOrigin[1] + imRes[1] * (imSize[1]) / 2.0
isocenter[2] = imOrigin[2] + imRes[2] * (imSize[2]) / 2.0

transform.SetCenter(imOrigin)
imRes = dicom_handle.GetSpacing()
imSize = dicom_handle.GetSize()

o2Dx = (512-1)/2
o2Dy = (512-1)/2
origin = [0, 0, 0]

origin[0] = -512 * o2Dx
origin[1] = -512 * o2Dy
origin[2] = -1000
filter.SetOutputOrigin(imOrigin)

interpolator = itk.SiddonJacobsRayCastInterpolateImageFunction[InputImageType, itk.D].New()
interpolator.SetProjectionAngle(dtr*0)
interpolator.SetFocalPointToIsocenterDistance(1000)
interpolator.SetThreshold(0.)
interpolator.SetTransform(transform)
interpolator.Initialize()

filter.SetInterpolator(interpolator)

dicom_size = dicom_handle.GetSize()
sizeOutput = [ dicom_size[0], dicom_size[1], 1]
spacing = dicom_handle.GetSpacing()

filter.SetSize(sizeOutput)
filter.SetOutputSpacing(spacing)
filter.Update()
output = filter.GetOutput()

writer = itk.ImageFileWriter[InputImageType].New()
writer.SetFileName(os.path.join('.','Output.mha'))
writer.SetInput(output)
writer.Update()
# Defines the image type
Dimension_CT = 3
PixelType = itk.F
ImageType = itk.Image[PixelType, Dimension_CT]
# Create a stack of empty projection images
print("Got here")
ConstantImageSourceType = rtk.ConstantImageSource[ImageType]
constantImageSource = ConstantImageSourceType.New()
print("And here")
# Define origin, sizeOutput and spacing (still need to change these)

numberOfProjections = 8 # 360



constantImageSource.SetOrigin( origin )
constantImageSource.SetSpacing( spacing )
constantImageSource.SetSize( sizeOutput )
constantImageSource.SetConstant(0.)

# Defines the RTK geometry object
geometry = rtk.ThreeDCircularProjectionGeometry.New()
firstAngle = 0.
angularArc = 360.
# firstangle = 180.
sid = 600 # source to isocenter distance
sdd = 1200 # source to detector distance

sdd = 1540  # source to imager distance
sid = 1000  # source to isocenter
for x in range(numberOfProjections):
  angle = firstAngle + x * angularArc / numberOfProjections
  geometry.AddProjection(sid, sdd, angle, 0., 0., 90., 0., 0., 0.)
for x in range(numberOfProjections):
  angle = firstAngle + x * angularArc / numberOfProjections
  geometry.AddProjection(sid, sdd, 0., 0., 0., 90., angle, 0., 0.)
# geometry.AddProjection(sid, sdd, 0, 0., 0., 90., 0., 0., 0.)
# sid, ssd, gantryAngle, projOffsetX, projOffsetY, outOfPlaneAngle, inPlaneAngle, sourceOffsetX, sourceOffsetY
# Writing the geometry to disk
xmlWriter = rtk.ThreeDCircularProjectionGeometryXMLFileWriter.New()
xmlWriter.SetFilename('geo.xml')
xmlWriter.SetObject(geometry)
xmlWriter.WriteFile()

REIType = rtk.JosephForwardProjectionImageFilter[ImageType, ImageType]
rei = REIType.New()
rei.SetGeometry(geometry)
rei.SetInput(0, constantImageSource.GetOutput())
rei.SetInput(1, CT)
rei.Update()
itk.imwrite(rei.GetOutput(), 'stack1.mha')