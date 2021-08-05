#Script to loop over all DICOM RT files and make numpy arrays
import os 
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import SimpleITK as sitk
import glob



def SortRTIMAGE(RIpath,Ndownsample):

    #Search for a dicom files
    RIfiles = glob.glob(str(RIpath) + '\*.dcm')

    for file in RIfiles:
        ds = pydicom.read_file(file)
        colang = ds.BeamLimitingDeviceAngle
        print("file " + str(file))
        try:
            ds.FractionNumber
        except AttributeError:
            print("No fx number")
            break
        

      
       
        fxnum = ds.FractionNumber
        
        aqdata = ds.AcquisitionDate
        print("Fraction number " + str(fxnum) + " Date " + str(aqdata))
        gantryang = ds.GantryAngle
        gantryang = np.rint(gantryang)
        if(gantryang == 360.0):
            gantryang =0
        print("Gantry Angle " + str(int(np.rint(gantryang))))
        image = ds.pixel_array
        print("Image shape " + str(image.shape) )
        if(image.shape[0] < 1280):
            print("Image too small ?")
            continue
        rescale = ds.RescaleSlope
        print(" Re-Scaling  factor "  +str(rescale))
        window_max = ds.WindowCenter+ds.WindowWidth/2
        window_min = ds.WindowCenter-ds.WindowWidth/2
        print("Window max "  + str(window_max) + " Window min " + str(window_min) )
        #multiply by the rescaling 
        imagescale = np.multiply(image,rescale)
        downsample = imagescale.reshape(1280//Ndownsample,Ndownsample,1280//Ndownsample,Ndownsample).mean(-1).mean(1)
        ndims = downsample.shape
        rtimg = np.float32(downsample)
        
        
        if(fxnum ==0):
            print("Found PDOS Image")
            try:
                ds[0x5000,0x2500].value
            except KeyError:
                print("No Label")
                break
            label = ds[0x5000,0x2500].value
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


# Main loop 
Basepath = 'P:\Image_Prediction\PatientList\\31171124'
MRNs = os.listdir(Basepath)
#Factor with which to downsample EPID images are 1280x1280 
Ndownsample = 5
print(MRNs)
for i in range(0,len(MRNs)):
    #RTIpath = os.path.join(Basepath,MRNs[i],'RTIMAGE')
    RTIpath = os.path.join(Basepath,'RTIMAGE')
    print(RTIpath)
    SortRTIMAGE(RTIpath,Ndownsample)
# %%
