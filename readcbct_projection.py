# importing neccessary libraries 
# file mangagment 

from cProfile import Profile
import os 
import zipfile
#from six.moves import urllib

# array manipulation and plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# medical image manipulation 
import SimpleITK as sitk
from DicomRTTool.ReaderWriter import DicomReaderWriter

import rays
import time

import cProfile, pstats, io

def profile(fnc):
    
    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


# path to CT image 
patpath = os.path.join('P:\Image_Prediction','11657988')
myCTpath = os.path.join('P:\Image_Prediction','11657988','frac2cbct')

print('path ' + myCTpath)

Dicom_reader = DicomReaderWriter(description='Examples',verbose=True)
print('Estimated 30 seconds, depending on number of cores present in your computer')
Dicom_reader.walk_through_folders(myCTpath) # need to define in order to use all_roi method

Dicom_reader.set_index(0)  # This index has all the structures, corresponds to pre-RT T1-w image for patient 011
Dicom_reader.get_images()

image = Dicom_reader.ArrayDicom 

dicom_sitk_handle = Dicom_reader.dicom_handle

numpyout = myCTpath + "test.npy"
#print('fileout ' + fileout)
#sitk.WriteImage(dicom_sitk_handle,fileout)
#%%
voxData = sitk.GetArrayFromImage(dicom_sitk_handle)
#print("Size " + imarr.shape)
origin = dicom_sitk_handle.GetOrigin() 
xorigin = origin[0] 
yorigin = origin[1] 
zorigin = origin[2] 

print("Orgin " + str(origin))
voxDim = dicom_sitk_handle.GetSpacing() 
xspace = voxDim[0]
yspace = voxDim[1]
zspace = voxDim[2]
print("voxDim " + str(voxDim))
voxSize = dicom_sitk_handle.GetSize()
xsize = voxSize[0]
ysize = voxSize[1]
zsize = voxSize[2]
print("VoxSize " + str(voxSize))

voxDim = np.asarray(voxDim)
voxSize = np.asarray(voxSize)
origin = np.asarray(origin)

# print(len(cbctlist))
SID = 1540 #source to imager distance
SAD = 1000 # source to isocenter
print("Ray Tracing ")


# Size of the Panel 
nx = 320
nz = 160

#rayvec = np.zeros((1280,1280))
rayvec = np.zeros((nz,nx))
zstep = 2.688
xstep = 1.344

epidEdgeX = -nx/2*xstep
epidEdgeZ = -nz/2*zstep


#define gantry angle this will rotate the source and the EPID positions 
gantryang =45
rotsource = rays.source_rotate(gantryang,origin)

start = time.time()
print(" Starting Loop ", str(start))

pr = cProfile.Profile()
pr.enable()
# # Testing 2D scan
for i in range(1,nx):
    for j in range(1,nz):
        #print(" Ray Index i " +str(i) + " postion on EPID " + str(epidEdgeX+i*xstep) + " index j " + str(j) + " Position on EPID " + str(epidEdgeZ+j*zstep) )
        #PointOnEPID = np.array([(epidEdge+i*0.336)+origin[0],(SID-SAD)+origin[1],(epidEdge+j*0.336)+origin[2]]) 
        PointOnEPID = np.array([(epidEdgeX+i*xstep),SID-SAD,(epidEdgeZ+j*zstep)]) 
        ray= rays.EPID_rotate(gantryang,origin,PointOnEPID)-rotsource
        #print("It's RAY! " + str(ray))
        rayvec[nz-j,nx-i] = rays.new_trace(image,origin,rotsource,ray,voxDim,voxSize)


pr.disable()
s = io.StringIO()
#sortby = SortKey.CUMULATIVE
#ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
pr.print_stats()
print(s.getvalue()) 

# end = time.time()
# print(" Ending Loop ", str(end) + " Time for loop " + str(end-start))


plt.imshow(rayvec)
plt.show()


# # Testing Scan in one direction 
# for i in range(0,1280):
#     print(" Ray Index" +str(i) + " postion on EPID " + str(epidEdge+i*0.336) )
#     PointOnEPID = np.array([(epidEdge+640*0.336)+origin[0],(SID-SAD)+origin[1],(epidEdge+i*0.336)+origin[2]])  # select midway point for X 
#     #PointOnEPID = np.array([(epidEdge+i*0.336)+origin[0],(SID-SAD)+origin[1],(epidEdge+640*0.336)+origin[2]])  # select midway point for Z 
#     raysum = rays.ray_trace(dicom_sitk_handle,PointOnEPID)
#     rayvec[i] = rays.ray_trace(dicom_sitk_handle,PointOnEPID)
#     print("Ray Sum " + str(raysum))


# plt.plot(rayvec)
# plt.show()



# epidEdge = -1280/2*0.336
# PointOnEPID1 = np.array([(epidEdge+640*0.336)+origin[0],(SID-SAD),(epidEdge+600*0.336)+origin[2]])  # select midway point for X 
"""
PointOnEPID1 = np.array([(epidEdgeX+160*xstep),SID-SAD,(epidEdgeZ+80*zstep)]) 
print("Point on EPID" +str(PointOnEPID1))
ray1= rays.EPID_rotate(gantryang,origin,PointOnEPID1)-rotsource
test1 = rays.new_trace(image,origin,rotsource,ray1,voxDim,voxSize)
print("Ray Sum 1   " + str(test1))
"""

