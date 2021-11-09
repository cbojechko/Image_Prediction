
#%%#Script to loop over all DICOM RT files and make numpy arrays
import os 
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import SimpleITK as sitk
import glob
import re
from scipy import interpolate

def SortRTIMAGE(Basepath,Ndownsample):

    
    
    #Search for a dicom files
    RIpath = os.path.join(Basepath,'RTIMAGE')
    RPpath = os.path.join(Basepath,'RTPLAN')
    RIfiles = glob.glob(str(RIpath) + '\*.dcm')
    RPfile =  glob.glob(str(RPpath) + '\*.dcm')
    
    """
    npfile = glob.glob(str(RIpath) + '\*.npz')
    if(npfile):
         print("Numpy files exists skip making new ones")
         return
    """
    print()
    dr = pydicom.read_file(RPfile[0])
    rbs = dr.FractionGroupSequence[0].ReferencedBeamSequence
    
    #print(RIfiles)

    for file in RIfiles:
        PredictedPDOS = False
        ds = pydicom.read_file(file)
        colang = ds.BeamLimitingDeviceAngle
        print("file " + str(file))
        try:
            ds.FractionNumber
            fxnum = ds.FractionNumber
        except AttributeError:
            print("No fx number")
            try:
                ds.ImageComments
                if(re.search('Predicted Portal Dose Image',ds.ImageComments)):
                    print("Predicted Portal Dose")
                    fxnum = 0
                    PredictedPDOS = True
            except AttributeError:
                print("Not a Prediction PDOS")
                continue
        
        aqdata = ds.AcquisitionDate
        print("Fraction number " + str(fxnum) + " Date " + str(aqdata))
        print("file" + str(file))
        gantryang = np.rint(ds.GantryAngle)
        if(gantryang == 360.0):
            gantryang =0
        print("Gantry Angle " + str(int(np.rint(gantryang))))
        image = ds.pixel_array
 
        print("Image shape " + str(image.shape) )
        #print("Position" + str(ds.RTImagePosition))
        # if(image.shape[0] < 1280):
        #     print("Image too small ?")
        #     continue

        rescale = ds.RescaleSlope
        imagescale = np.multiply(image,rescale)
        if(PredictedPDOS):
            #Geting the scaling factor for predicted images is tricky
            #Get the beam number in PDOS file, loop over the beam numbers in RT file and get number of MU, rescale by MU
            pdosbeamn = ds.ReferencedBeamNumber
            for i in range(0,len(rbs)):

                #print(rbs[i]. ReferencedBeamNumber)
                try:
                    rbs[i].BeamMeterset
                    #print(rbs[i].BeamMeterset)
                except AttributeError:
                    continue
                if(pdosbeamn == rbs[i]. ReferencedBeamNumber):
                    print("found a match in RT PLAN")
                    imagescale = np.multiply(imagescale,rbs[i].BeamMeterset)

            #Is the create PDOS padded? 
            #imagescale = imagescale[4:imagescale.shape[1]-3,3:imagescale.shape[1]-4]
            #print("Remove pad shape " + str(imagescale.shape) + " " + str(imagescale) + " " + str(imagescale[429,429]) )
            # First up sample image then crop and downsample.  
            x = np.array(range(imagescale.shape[1]))
            y = np.array(range(imagescale.shape[0]))
            #xx, yy = np.meshgrid(x, y)
            f = interpolate.interp2d(x, y, imagescale, kind='cubic')
            #xnew = np.linspace(0, imagescale.shape[1], 3*imagescale.shape[1])
            #ynew = np.linspace(0, imagescale.shape[0], 3*imagescale.shape[0])
            #xnew = np.linspace(0, imagescale.shape[1], 1280)
            #ynew = np.linspace(0, imagescale.shape[0], 1280)
            xnew = np.linspace(0, imagescale.shape[1], 1294)
            ynew = np.linspace(0, imagescale.shape[0], 1294)
            pdosnew = f(xnew, ynew)
            crop = (pdosnew.shape[1]-1280)//2
            #Testing the cropping seems to be a shift between pdos and rt image 
            #pdoscrop = pdosnew[crop+1:pdosnew.shape[1]-crop,crop+1:pdosnew.shape[1]-crop]
            shift = -5
            pdoscrop = pdosnew[crop+shift:pdosnew.shape[1]-(crop-shift),crop+shift:pdosnew.shape[1]-(crop-shift)]
            downsample = pdoscrop.reshape(1280//Ndownsample,Ndownsample,1280//Ndownsample,Ndownsample).mean(-1).mean(1)
            #downsample = pdosnew.reshape(1280//Ndownsample,Ndownsample,1280//Ndownsample,Ndownsample).mean(-1).mean(1)
        else:
            downsample = imagescale.reshape(1280//Ndownsample,Ndownsample,1280//Ndownsample,Ndownsample).mean(-1).mean(1)
        #downsample = imagescale.reshape(1280//Ndownsample,Ndownsample,1280//Ndownsample,Ndownsample).mean(-1).mean(1)
        #ndims = downsample.shape
        rtimg = np.float32(downsample)
        
        
        if(fxnum ==0):
            print("Found PDOS Image")
            # try:
            #     ds[0x5000,0x2500].value
            # except KeyError:
            #     print("No Label")
            #     break
            # label = ds[0x5000,0x2500].value
            npfileout = "PDOS_G" + str(int(np.rint(gantryang))) + "_" + str(int(aqdata))
            arrout = os.path.join(RIpath, npfileout)
            print("Saving PDOS Image "+ str(arrout))
            np.savez_compressed(arrout,rtimg)
        else:
            print("Found In-vivo Image")
            npfileout = "RI" + str(fxnum) + "_G" + str(int(np.rint(gantryang))) + "_" + str(int(aqdata))
            arrout = os.path.join(RIpath, npfileout)
            print("Saving RT Image "+ str(arrout))
            np.savez_compressed(arrout,rtimg)

       
    #Clean up dicom files 
    """
    print("Clean up DICOM files")
    for files in RIfiles:
        os.remove(files)  
    """
    return

"""
# Main loop 
Basepath = 'P:\Image_Prediction\Marginal'
MRNs = os.listdir(Basepath)
#Factor with which to downsample EPID images are 1280x1280 
Ndownsample = 5
#print(MRNs)
for i in range(0,len(MRNs)):
    RTIpath = os.path.join(Basepath,MRNs[i])
    #RTIpath = os.path.join(Basepath,'RTIMAGE')
    print(RTIpath)
    SortRTIMAGE(RTIpath,Ndownsample)
"""

Basepath = 'P:\Image_Prediction\PatientData\\31148288'
MRNs = os.listdir(Basepath)
#Factor with which to downsample EPID images are 1280x1280 
Ndownsample = 5

#RTIpath = os.path.join(Basepath,MRNs[i],'RTIMAGE')
#RTIpath = os.path.join(Basepath,'RTIMAGE')
#print(RTIpath)
SortRTIMAGE(Basepath,Ndownsample)

