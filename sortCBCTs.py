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
import pydicom

# path to CT image 
patpath = os.path.join('P:\Image_Prediction','04455192')
myCTpath = os.path.join('P:\Image_Prediction','04455192','cbct_test')

print('path ' + myCTpath)


imageReader = sitk.ImageSeriesReader()
#dicom_names = imageReader.GetGDCMSeriesFileNames(myCTpath)
series_ids = imageReader.GetGDCMSeriesIDs(myCTpath)
print("Found " + str(len(series_ids)) + " Two Different Series IDs")
series_dates=np.zeros(len(series_ids))
series_times=np.zeros(len(series_ids))
for i in range(0,len(series_ids)):
    print("Series ID " + str(series_ids[i]))
    seriesf = imageReader.GetGDCMSeriesFileNames(myCTpath, series_ids[i])
    ds = pydicom.read_file(seriesf[0])
    aqdata = ds.AcquisitionDate
    aqtime = ds.AcquisitionTime
    series_dates[i] = int(aqdata)
    series_times[i] = float(aqtime)
    print(" Date " + str(aqdata) + " Time " + str(aqtime))
    #aqtime = ds.AcquisitionTime
# Find if there are any cone beams collected on the same day
# Find the indicies of the series dates. 
#%%
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

badidx = np.zeros(len(double_dates))
# This code will not work if there are 3 CT's collected on a given day 
for m in range(0,len(ddidx),2):
    if(series_times[ddidx[m]] > series_times[ddidx[m+1]]):
        badidx[m//2] = ddidx[m]
    else:
        badidx[m//2] = ddidx[m+1]

print("Need to remove indicies" + str(badidx))
#Delete files of duplicate CT scans
for n in badidx:
    seriesf = imageReader.GetGDCMSeriesFileNames(myCTpath, series_ids[int(n)])
    tt = list(seriesf)
    for i in range(0,len(tt)):
        os.remove(tt[i])

# Save remaing CT Scans as numpy arrays 



#imageReader.SetFileNames(seriesf)

#img = imageReader.Execute()

#imarr = sitk.GetArrayFromImage(img)

#mdata = imageReader.GetMetaDataKeys(1)

# Dicom_reader.set_index(1)  # This index has all the structures, corresponds to pre-RT T1-w image for patient 011
# Dicom_reader.get_images()

# dicom_sitk_handle = Dicom_reader.dicom_handle

# dicom_sitk_handle.GetOrigin()


# %%
