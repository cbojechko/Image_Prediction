#%%
import os 
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import SimpleITK as sitk
import glob
import re
import rays
import PIL
from pathlib import Path

def MakeJPEG(Patpath,cntr):

    maxRI = 2.3
    maxPDOS = 3.5
    maxFlu = 2.0
    maxLen = 40
    Basepath = os.path.dirname(os.path.dirname(Patpath))
    print(Basepath)
    RIpath = os.path.join(Patpath,'RTIMAGE')
    myCTpath = os.path.join(Patpath,'CBCT')
    Flupath = os.path.join(Patpath,'Fluence')


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
        # Load CBCT projection with same date/fraction and angle   
        cbctprojfile = str(myCTpath) + '\cbctprojection' + str(int(fxnum)) + '_G' + str(gang) +'_' + str(int(date)) +'.npz'
        print(cbctprojfile)
        if(os.path.exists(cbctprojfile) == False):
            continue
        print(cbctprojfile)
        npproj = np.load(cbctprojfile)
        #Load Fluence File with same angle 
        flufile = str(Flupath) + '\Fluence'+ str(gang) + '.npz'
        print(flufile)
        npflu = np.load(flufile)
        #Load PDOS with same angle 
        PDOSfile = glob.glob(str(RIpath) + '\PDOS_G'+ str(gang) + '_' + '*.npz')
        print(PDOSfile)
        npPDOS = np.load(PDOSfile[0])
        FLUarr = npflu['arr_0']
        PDOSarr = npPDOS['arr_0']
        PROJarr = npproj['arr_0']
        PDOSgs = np.zeros((PDOSarr.shape[0],PDOSarr.shape[1]),np.uint8)
        FLUgs  = np.zeros((FLUarr.shape[0],FLUarr.shape[1]),np.uint8)
        CBCTgs = np.zeros((PROJarr.shape[0],PROJarr.shape[1]),np.uint8)
        RTgs = np.zeros((PDOSarr.shape[0],PDOSarr.shape[1]),np.uint8)
        for kk in range(0,PDOSarr.shape[0]):
            for ll in range(PDOSarr.shape[1]):
                PDOSgs[kk,ll] = int(np.rint(PDOSarr[kk,ll]*255/maxPDOS))
                RTgs[kk,ll] = int(np.rint(RTarr[kk,ll]*255/maxRI))
        
        for mm in range(0,FLUarr.shape[0]):
            for nn in range(FLUarr.shape[1]):
                FLUgs[mm,nn] = int(np.rint(FLUarr[mm,nn]*255/maxFlu))
        
        for pp in range(0,PROJarr.shape[0]):
            for qq in range(PROJarr.shape[1]):
                if(PROJarr[pp,qq] > 40):
                    CBCTgs[pp,qq] = 255
                else: 
                    CBCTgs[pp,qq] = int(np.rint(PROJarr[pp,qq]*255/maxLen))

        #Join together fluence PDOS and CBCT projection 
        #join = np.dstack((FLUgs,PDOSgs,CBCTgs))
        join = np.hstack((CBCTgs,FLUgs,PDOSgs,RTgs))
        filein = str(cntr+1)
        fileinp = os.path.join(Basepath,'testing_jpeg',filein)
        print("Saving  "+ fileinp)
        #PDOSim = PIL.Image.fromarray(PDOSgs)
        Inputim = PIL.Image.fromarray(join)
        Inputim.save(fileinp + ".jpeg")
        
        #fileoutp = os.path.join(BASEpath,'train\output',filein)
        #Outputim = PIL.Image.fromarray(RTgs)
        #Outputim.save(fileoutp + ".jpeg") 
        #np.savez_compressed(fileinp,join)
        
        #print("Saving output "+ str(fileoutp))
        #np.savez_compressed(fileoutp,RTarr)
        cntr=cntr+1
    return cntr

"""
######################################################
# Main loop loop over all patients
Basepath = 'P:\Image_Prediction\PatientList'
MRNs = os.listdir(Basepath)
cntr = 0
for i in range(0,len(MRNs)):
    Patpath = os.path.join(Basepath,MRNs[i])
    print(Patpath)
    cntr = MakeJPEG(Patpath,cntr)
"""

######################################################
# Generate for single patient
Patpath = 'P:\Image_Prediction\PatientList\\22933592'
#MRNs = os.listdir(Basepath)
cntr = 228
#for i in range(0,len(MRNs)):
#Patpath = os.path.join(Basepath,MRNs[i])
print(Patpath)
cntr = MakeJPEG(Patpath,cntr)





