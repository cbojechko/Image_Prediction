# importing neccessary libraries 
# file mangagment 
#%%
import os 
import zipfile
from six.moves import urllib

# array manipulation and plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# medical image manipulation 
import SimpleITK as sitk
from DicomRTTool.ReaderWriter import DicomReaderWriter

# path to CT image 
patpath = os.path.join('P:\Image_Prediction','11657988')
myCTpath = os.path.join('P:\Image_Prediction','11657988','frac2cbct')

print('path ' + myCTpath)
#%%
Dicom_reader = DicomReaderWriter(description='Examples',verbose=True)
print('Estimated 30 seconds, depending on number of cores present in your computer')
Dicom_reader.walk_through_folders(myCTpath) # need to define in order to use all_roi method

Dicom_reader.set_index(0)  # This index has all the structures, corresponds to pre-RT T1-w image for patient 011
Dicom_reader.get_images()

image = Dicom_reader.ArrayDicom 
#mask = Dicom_reader.mask

dicom_sitk_handle = Dicom_reader.dicom_handle
#fileout = myCTpath + "test.nii"
numpyout = myCTpath + "test.npy"
#print('fileout ' + fileout)
#sitk.WriteImage(dicom_sitk_handle,fileout)

imarr = sitk.GetArrayFromImage(dicom_sitk_handle)
#print("Size " + imarr.shape)
origin = dicom_sitk_handle.GetOrigin() 
xorigin = origin[0] 
yorigin = origin[1] 
zorigin = origin[2] 

print("Orgin " + str(origin))
spacing = dicom_sitk_handle.GetSpacing() 
xspace = spacing[0]
yspace = spacing[1]
zspace = spacing[2]
print("Spacing " + str(spacing))
size = dicom_sitk_handle.GetSize()
xsize = size[0]
ysize = size[1]
zsize = size[2]
print("Size " + str(size))

cbctlist = []
for i in range(0,xsize):
    for j in range(0,ysize):
         for k in range(0,zsize):
             cbctlist.append(np.int16(dicom_sitk_handle.GetPixel(i,j,k)))  
             cbctlist.append(np.float32(xorigin + i*xspace))
             cbctlist.append(np.float32(yorigin + j*yspace))
             cbctlist.append(np.float32(zorigin + k*zspace))


print(len(cbctlist))

cbctvector = np.array(cbctlist)
npfileout = "cbctvector"
arrout = os.path.join(patpath, npfileout)
print("Saving CBCT vector "+ str(arrout))
np.savez_compressed(arrout,cbctvector)


#dicom_sitk_handle.GetPixel(0,0,0)  
#plt.imshow(imarr[0], cmap='gray', interpolation='none')

#plt.show()

print(imarr.dtype)
#np.save(numpyout,imarr)

#x = np.linspace(0,20,100)
#plt.plot(x,np.sin(x))
#plt.show()
#plt.imshow(image[0],cmap='gray', interpolation='none')
#plt.show()
#n_slices_skip = 1
#display_slices(image,skip = n_slices_skip)
# %%
