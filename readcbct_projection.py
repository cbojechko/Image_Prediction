# Depreciated
# importing neccessary libraries 
# file mangagment 

#%%
from cProfile import Profile
import os 
import zipfile
import glob
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

# path to CT image
fid = open(os.path.join('.', 'MRN.txt'))
for _ in range(5):
    MRN = fid.readline()
MRN = MRN.strip('\n')
fid.close()
patpath = os.path.join('P:\Image_Prediction', 'Testing', MRN)
myCTpath = os.path.join('P:\Image_Prediction', 'Testing', MRN, 'CBCT')

print('path ' + myCTpath)

#Search for a numpy file 
npfile = glob.glob(str(myCTpath) + '\*.npz')
if(len(npfile) == 0):
    print("No numpy file for CBCT found , make numpy file")

    Dicom_reader = DicomReaderWriter(description='Examples',verbose=True)
    print('Read CBCT Dicom Files ......')
    Dicom_reader.walk_through_folders(myCTpath) # need to define in order to use all_roi method

    Dicom_reader.set_index(0)  # This index has all the structures, corresponds to pre-RT T1-w image for patient 011
    Dicom_reader.get_images()

    dicom_sitk_handle = Dicom_reader.dicom_handle

    #voxData = sitk.GetArrayFromImage(dicom_sitk_handle)

    origin = dicom_sitk_handle.GetOrigin() 
    print("Orgin " + str(origin))

    voxDim = dicom_sitk_handle.GetSpacing() 
    print("voxDim " + str(voxDim))

    voxSize = dicom_sitk_handle.GetSize()
    print("VoxSize " + str(voxSize))

    voxDim = np.asarray(voxDim)
    voxSize = np.asarray(voxSize)
    origin = np.asarray(origin)

    image = Dicom_reader.ArrayDicom 

    cbctnp = np.array(image)
    npfileout = "cbct"
    arrout = os.path.join(myCTpath, npfileout)
    print("Saving CBCT vector "+ str(arrout))
    np.savez_compressed(arrout,cbct=cbctnp,origin=origin,voxDim=voxDim,voxSize=voxSize)
else:
    print("Numpy file found")
    npfin = np.load(npfile[0])
    image = npfin['cbct']
    origin = npfin['origin']
    voxDim = npfin['voxDim']
    voxSize = npfin['voxSize']
#%%

SID = 1540 #source to imager distance
SAD = 1000 # source to isocenter
print("Ray Tracing ")


# Size of the Panel 
nx = 125
nz = 125

#rayvec = np.zeros((1280,1280))
rayvec = np.zeros((nz,nx))
zstep = 3.44
xstep = 3.44

epidEdgeX = -nx/2*xstep
epidEdgeZ = -nz/2*zstep


#define gantry angle this will rotate the source and the EPID positions 
gantryang =0
rotsource = rays.source_rotate(gantryang,origin)


# # Testing 2D scan
for i in range(1,nx):
    for j in range(1,nz):
        #print(" Ray Index i " +str(i) + " postion on EPID " + str(epidEdgeX+i*xstep) + " index j " + str(j) + " Position on EPID " + str(epidEdgeZ+j*zstep) )
        #PointOnEPID = np.array([(epidEdge+i*0.336)+origin[0],(SID-SAD)+origin[1],(epidEdge+j*0.336)+origin[2]]) 
        PointOnEPID = np.array([(epidEdgeX+i*xstep),SID-SAD,(epidEdgeZ+j*zstep)]) 
        ray= rays.EPID_rotate(gantryang,origin,PointOnEPID)-rotsource
        #print("It's RAY! " + str(ray))
        rayvec[nz-j,nx-i] = rays.new_trace(image,origin,rotsource,ray,voxDim,voxSize)


#plt.imshow(rayvec)
#plt.show()

#Save projection 
projfileout = "cbctprojection" + str(gantryang)
arrout = os.path.join(patpath, projfileout)
print("Saving Projection "+ str(arrout))
np.savez_compressed(arrout,projfileout)


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


