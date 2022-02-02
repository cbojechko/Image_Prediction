import sys

import SimpleITK as sitk
import os
from itk import RTK as rtk
import itk
from DicomRTTool.ReaderWriter import DicomReaderWriter, plot_scroll_Image, pydicom

if os.path.exists(os.path.join('.', 'stack.mha')):
  image_handle = sitk.ReadImage('stack.mha')
  array = sitk.GetArrayFromImage(image_handle)


data_path = r'C:\Users\b5anderson\Desktop\Modular_Projects\Image_Prediction\Data\Patient\CT'
if not os.path.exists(os.path.join(data_path, "Image.mha")):
  Dicom_reader = DicomReaderWriter(description='Examples', verbose=True)
  Dicom_reader.down_folder(data_path)
  Dicom_reader.set_index(5)
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
CT = itk.imread(os.path.join('.', "Image.mha"), pixel_type=itk.F)

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
origin = dicom_handle.GetOrigin()
numberOfProjections = 180 # 360
dicom_size = dicom_handle.GetSize()
sizeOutput = [ dicom_size[0], dicom_size[1], numberOfProjections]
spacing = dicom_handle.GetSpacing()

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
for x in range(0,numberOfProjections):
  angle = firstAngle + x * angularArc / numberOfProjections
  geometry.AddProjection(sid, sdd, 0., 0., 0., 90., angle)
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
itk.imwrite(rei.GetOutput(), 'stack.mha')