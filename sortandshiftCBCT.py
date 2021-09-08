#%%
from cProfile import Profile
import os 
import zipfile
import glob
from PIL.Image import AFFINE
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
import pydicom

#path to folder with Dicom files 
CTpath = 'P:\\Image_Prediction\\Testing\\18987107\\CBCT'
#CTpath = 'P:\\Image_Prediction\\Testing\\30318785\\CBCT'


imageReader = sitk.ImageSeriesReader()
#dicom_names = imageReader.GetGDCMSeriesFileNames(CTpath)
series_ids = imageReader.GetGDCMSeriesIDs(CTpath)
print("Found " + str(len(series_ids)) + " Different Series IDs")
if( len(series_ids) == 0):
    print("No Dicom Files in Directory")
    #return
series_dates=np.zeros(len(series_ids))
series_times=np.zeros(len(series_ids))
#Loop over the series and find the dates and time and assign an index.
for i in range(0,len(series_ids)):
    print("Series ID " + str(series_ids[i]) )
    seriesf = imageReader.GetGDCMSeriesFileNames(CTpath, series_ids[i])
    ds = pydicom.read_file(seriesf[0])
    aqdata = ds.AcquisitionDate
    cbctUID = ds.SeriesInstanceUID
    aqtime = ds.AcquisitionTime
    series_dates[i] = int(aqdata)
    series_times[i] = float(aqtime)
    print(" Date " + str(aqdata) + " Time " + str(aqtime) + " Index " + str(i))
    #aqtime = ds.AcquisitionTime
# Find if there are any cone beams collected on the same day
# Find the indicies of the series dates. 
#%%
REfiles  =  glob.glob(CTpath + '\RE*.dcm')

#%%
#find out if there are scans given on the same day. 
double_dates = []
ddidx = []
for j in range(0,len(series_dates)):
    for k in range(j+1,len(series_dates)):
        if(series_dates[j] == series_dates[k]):
            double_dates.append(series_dates[j])
            ddidx.append(j)
            ddidx.append(k)
            break
        else:
            continue
        break
for i in range(0,len(double_dates)):
    print("Found 2 CTs taken on " + str(double_dates[i]))

badidx = np.zeros(len(double_dates),dtype=int)
# This code will not work if there are 3 CT's collected on a given day 
for m in range(0,len(ddidx),2):
    if(series_times[ddidx[m]] > series_times[ddidx[m+1]]):
        badidx[m//2] = int(ddidx[m])
    else:
        badidx[m//2] = int(ddidx[m+1])

print("Need to remove indicies" + str(badidx))
#Delete files of duplicate CT scans
"""
for n in badidx:
    seriesf = imageReader.GetGDCMSeriesFileNames(CTpath, series_ids[int(n)])
    tt = list(seriesf)
    for i in range(0,len(tt)):
        os.remove(tt[i]) # Remove deletion for now 
"""
series_idx = np.arange(len(series_ids))
series_idx = np.delete(series_idx,badidx)
print("Loop and save indicies" + str(series_idx))

translate = sitk.AffineTransform(3)

# Save remaing CT Scans as numpy arrays 
#%%
for i in series_idx:
    print("Index to save " + str(i))
    seriesf = imageReader.GetGDCMSeriesFileNames(CTpath, series_ids[i])
    ds = pydicom.read_file(seriesf[0])
    aqdata = ds.AcquisitionDate
    #Open the RP dicom file to get the isocenter 
    RPfiles = glob.glob(CTpath + '\RP*.dcm')
    if(len(RPfiles) != 1):
        print("Single RP file not found do not make npz file")
        break
    dp = pydicom.read_file(RPfiles[0])
    planiso = dp.BeamSequence[0].ControlPointSequence[0].IsocenterPosition

    #planiso = ds.BeamSequence[0].ControlPointSequence[0].IsocenterPosition
    print("CBCT date " + aqdata)
    #Loop over registration files and find matching date. 
    for REfile in REfiles:
        dr = pydicom.read_file(REfile)
        regdate = dr.ContentDate
        regUID = dr.ReferencedSeriesSequence[0].SeriesInstanceUID
        print("Reg ID " + str(regUID) + " Series idx " + str(series_ids[i]))
        if(series_ids[i] == regUID):
            print(" Registration UID match " + str(regUID))
    
            transform = dr.RegistrationSequence[0].MatrixRegistrationSequence[0].MatrixSequence[0].FrameOfReferenceTransformationMatrix
            transx = transform[3]
            transy = transform[7]
            transz = transform[11]

            shiftlat =  (transx-planiso[0])
            shiftvrt =  (transy-planiso[1])
            shiftlong = (transz-planiso[2])
            print("iso 0 " + str(planiso[0]) + " iso 1 " + str(planiso[1]) + " iso 2 " + str(planiso[2]) )
            print("transform3 " + str(transform[3]) + " transform7 " + str(transform[7]) + " transform11 " + str(transform[11]) )
    
            print("shifts lat " +str(shiftlat) + " vrt " + str(shiftvrt) + " long " + str(shiftlong))
            imageReader.SetFileNames(seriesf)
            img = imageReader.Execute()
            img = img +1000
            #first val is along axis3 of cbct
            #second val is along axis2 of cbct
            #third val is along axis1 of cbct
            #torigin = img.GetOrigin() 
            #shiftlong = shiftlong -(-torigin[2]-93.432414)
            print("Corrected long " + str(shiftlong))
            translate.SetTranslation((-shiftlat,-shiftvrt,-shiftlong))
            newimg = sitk.Resample(img,translate)
            newimg = newimg - 1000
            imarr = sitk.GetArrayFromImage(newimg)
            origin = newimg.GetOrigin() 
            print("Orgin " + str(origin))

            voxDim = img.GetSpacing() 
            print("voxDim " + str(voxDim))
            voxSize = img.GetSize()
            print("VoxSize " + str(voxSize))

            voxDim = np.asarray(voxDim)
            voxSize = np.asarray(voxSize)
            origin = np.asarray(origin)
            cbctnp = np.array(imarr)
            npfileout = "cbct" + str(int(aqdata))
            arrout = os.path.join(CTpath, npfileout)
            print("Saving CBCT vector "+ str(arrout))
            np.savez_compressed(arrout,cbct=cbctnp,origin=origin,voxDim=voxDim,voxSize=voxSize)

"""
print("Clean Up DICOM Files")
for i in range(0,len(series_ids)):
    seriesf = imageReader.GetGDCMSeriesFileNames(CTpath, series_ids[int(i)])
    tt = list(seriesf)
    for k in range(0,len(tt)):
        os.remove(tt[k]) 
return 
"""




# Main loop 
# Basepath = 'P:\Image_Prediction\PatientList'
# MRNs = os.listdir(Basepath)

# for i in range(0,len(MRNs)):
#     CBCTpath = os.path.join(Basepath,MRNs[i],'CBCT')
#     print(CBCTpath)
#     SortShiftCBCT(CBCTpath)
# %%
