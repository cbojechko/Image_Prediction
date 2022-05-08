import pydicom
import os

path = r'K:'
updated = os.listdir(os.path.join(path, 'BMA_Export'))[0]
path = r'K:\BMA_Export'
files = os.listdir(path)
for file in files:
  print(file)
  ds = pydicom.read_file(os.path.join(path, file))
  ds.Id = "Test"
  pydicom.write_file(os.path.join(r'Z:\DICOM\BMA_Export', file), ds)

for key in ds.keys():
  print(ds[key])
path = r'K:\ANON398831'
file = os.listdir(path)[0]
ds2 = pydicom.read_file(os.path.join(path, file))
xxx = 1