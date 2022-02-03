import itk
import SimpleITK as sitk
import os
#from itk import RTK as rtk
import itk
import numpy as np
from DicomRTTool.ReaderWriter import DicomReaderWriter, plot_scroll_Image, pydicom

def write_itk_file(itk_file, path):
  writer = itk.ImageFileWriter[itk.Image[itk.F, 3]].New()
  writer.SetFileName(path)
  writer.SetInput(itk_file)
  writer.Update()

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

rotations = [0, 0, 45]
image = itk.imread(os.path.join('.', "Image.mha"), pixel_type=itk.F)
origin = dicom_handle.GetOrigin()
imSize = dicom_handle.GetSize()
imRes = dicom_handle.GetSpacing()
isocenter = [0, 0, 0]
for i in range(3):
  isocenter[i] = origin[i] + imRes[i] * imSize[i]/2

transform = itk.Euler3DTransform[itk.D].New()
transform.SetComputeZYX(True)
transform.SetCenter(isocenter)
transform.SetTranslation([0, 0, 0])
transform.SetRotation(np.deg2rad(rotations[0]), np.deg2rad(rotations[1]), np.deg2rad(rotations[2]))

interpolator = itk.LinearInterpolateImageFunction[itk.Image[itk.F, 3], itk.D].New()

filter = itk.ResampleImageFilter[itk.Image[itk.F, 3], itk.Image[itk.F, 3]].New()
filter.SetInterpolator(interpolator)
filter.SetDefaultPixelValue(-1000)
filter.SetOutputOrigin(image.GetOrigin())
filter.SetOutputSpacing(image.GetSpacing())
filter.SetOutputDirection(image.GetDirection())
filter.SetSize(image.GetLargestPossibleRegion().GetSize())

filter.SetTransform(transform)

filter.SetInput(image)
filter.Update()
output = filter.GetOutput()
write_itk_file(output, os.path.join('.', 'Rotated.mha'))





euler_transform = sitk.Euler3DTransform()

euler_transform.SetCenter(isocenter)
euler_transform.SetTranslation((0, 0, 0))
euler_transform.SetRotation(np.deg2rad(rotations[0]), np.deg2rad(rotations[1]), np.deg2rad(rotations[2]))
resampled = sitk.Resample(dicom_handle, dicom_handle, euler_transform, sitk.sitkLinear, -1000)
sitk.WriteImage(resampled, os.path.join('.', "Image_new.mha"))
