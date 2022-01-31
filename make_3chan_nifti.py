#%%
import os 
import numpy as np
import glob
import re
from scipy import ndimage
import nibabel as nib 


def PadNP(in_array):

    ucropidx = 0
    lcropidx = 255
    #make a profile along the sup/inf direction
    #sarr = np.sum(in_array, axis=1)
    sarr = in_array[0:256,128] 
    #print("sarr " + str(sarr.shape))
    #np.sum(in_array, axis=1)
    #Find the slope looking for max and min slope
    slope = np.zeros(255)
    for k in range(0,254):
        slope[k] = sarr[k+1]-sarr[k]
    
    #print("slope " + str(slope.shape))
    maxidx = np.argmax(slope[0:128])
    minidx = np.argmin(slope[128:256])
    minidx = 128+minidx
    #print("Max idx " + str(maxidx) + "  Min idx " + str(minidx))
    #move inward from max slope to where slope gets small use those indexs to crop image 
    #Only search half of image. 
    for i in range(0,128):
        if(slope[maxidx+i] < 0.5):
            lcropidx = maxidx+i
            break

    for j in range(1,128):
        if(slope[minidx-j] > -0.5):
            ucropidx = minidx-j
            break

    #print(" l idx " + str(lcropidx) + " u idx " + str(ucropidx) )
    crop_array = in_array[lcropidx:ucropidx,0:256]

    pad_array = np.pad(crop_array,((lcropidx,256-ucropidx),(0,0)),'edge')

    return pad_array


def MakeNIFTI(Patpath,pcntr):

    cntr = 1
    maxRI = 2.26
    maxPDOS = 3.46
    maxLen = 40.0
    Basepath = os.path.dirname(os.path.dirname(Patpath))
    #print(Basepath)
    RIpath = os.path.join(Patpath,'RTIMAGE')
    myCTpath = os.path.join(Patpath,'CT')
    
    RIfiles = glob.glob(str(RIpath) + '\RI*.npz')

    for file in RIfiles:
        tt = re.search('_G\d+_',file)
        gang = tt[0][2:len(tt[0])-1]
      
        rr = re.search('_\d+\.',file)
        date = rr[0][1:len(rr[0])-1]
        ff = re.search('RI\d+_',file)
        fxnum = ff[0][2:len(ff[0])-1]
       
        print("Gantry Angle " + str(gang) + " Date " + str(date) + " Fraction " + str(fxnum))
        npRT = np.load(file)
        RTarr = npRT['arr_0']
        
    
        #Load PDOS with same angle 
        PDOSfile = glob.glob(str(RIpath) + '\PDOS_G'+ str(gang) + '_' + '*.npz')
        #print(PDOSfile)
        npPDOS = np.load(PDOSfile[0])
        PDOSarr = npPDOS['arr_0']

        # Load CBCT projection with same date/fraction and angle   
        cbctprojfile = str(myCTpath) + '\cbctprojection' + str(int(fxnum)) + '_G' + str(gang) +'_' + str(int(date)) +'.npz'
        #print(cbctprojfile)
        if(os.path.exists(cbctprojfile) == False):
            print( " Projection File does not exsit " + cbctprojfile  )
            continue
        
        #print(cbctprojfile)
        npproj = np.load(cbctprojfile)
        halfprojfile = str(myCTpath) + '\halfprojection' + str(int(fxnum)) + '_G' + str(gang) +'_' + str(int(date)) +'.npz'
        #print(cbctprojfile)
        if(os.path.exists(halfprojfile) == False):
            print( " ISO Projection File does not exsit " + halfprojfile  )
            continue

        nphalf = np.load(halfprojfile)
        PROJarr = npproj['arr_0']
        HALFarr = nphalf['arr_0']

        PROJarr = PadNP(PROJarr)
        HALFarr = PadNP(HALFarr)

        PDOSgs = np.zeros((PDOSarr.shape[0],PDOSarr.shape[1]),np.uint32)
        RTgs = np.zeros((PDOSarr.shape[0],PDOSarr.shape[1]),np.uint32)
        CBCTgs = np.zeros((PROJarr.shape[0],PROJarr.shape[1]),np.uint32)
        HALFgs = np.zeros((HALFarr.shape[0],HALFarr.shape[1]),np.uint32)

        for kk in range(0,PDOSarr.shape[0]):
            for ll in range(PDOSarr.shape[1]):
                PDOSgs[kk,ll] = PDOSarr[kk,ll]*511.0/maxPDOS
                RTgs[kk,ll] = RTarr[kk,ll]*511.0/maxRI
        

         
        for pp in range(0,PROJarr.shape[0]):
            for qq in range(PROJarr.shape[1]):
                if(PROJarr[pp,qq] > 40.0):
                    CBCTgs[pp,qq] = 511.0
                else: 
                    CBCTgs[pp,qq] = PROJarr[pp,qq]*511.0/maxLen
       
        for rr in range(0,HALFarr.shape[0]):
            for ss in range(HALFarr.shape[1]):
                if(HALFarr[rr,ss] > 40.0):
                    HALFgs[rr,ss] = 511.0
                else: 
                    HALFgs[rr,ss] = HALFarr[rr,ss]*511.0/maxLen
        
        
        datastack = np.dstack((CBCTgs,HALFgs,PDOSgs,RTgs))
        # Something weird happens with passing to ITK and reading in tensor need to rotate one more time 
        datastack = ndimage.rotate(datastack, 90, axes=(0,2), reshape=True,order=1)
        
        print(datastack.shape)
        img = nib.Nifti1Image(datastack,affine=np.eye(4))
        filein = str(pcntr) + '_' + str(fxnum) + '_' + str(gang)
        print(filein)
        fileinp = os.path.join(Basepath,'dstack',filein)

        img.to_filename(fileinp + ".nii.gz")

    print("counter " + str(cntr))
    


######################################################
# Main loop loop over all patients
Basepath = 'P:\Image_Prediction\PatientData'
MRNs = os.listdir(Basepath)
pcntr = 1
for i in range(0,len(MRNs)):
    Patpath = os.path.join(Basepath,MRNs[i])
    print(Patpath)
    MakeNIFTI(Patpath,pcntr)
    pcntr = pcntr+1

"""
######################################################
# Generate for single patient
Patpath = 'P:\Image_Prediction\PatientData\\04536843'
#MRNs = os.listdir(Basepath)
cntr = 1
#for i in range(0,len(MRNs)):
#Patpath = os.path.join(Basepath,MRNs[i])
print(Patpath)
cntr = MakeNIFTI(Patpath,cntr)
"""
