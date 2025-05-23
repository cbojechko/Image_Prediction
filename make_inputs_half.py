#%%
import os 
import numpy as np
import glob
import re
import PIL
from pathlib import Path
from PIL import Image
##############


#################
def MakeJPEG(Patpath,pcntr):

    cntr = 1
    maxRI = 2.26
    maxPDOS = 3.46
    maxFlu = 1.86
    maxLen = 40.0
    Basepath = os.path.dirname(os.path.dirname(Patpath))
    #print(Basepath)
    RIpath = os.path.join(Patpath,'RTIMAGE')
    myCTpath = os.path.join(Patpath,'CT')
    Flupath = os.path.join(Patpath,'Fluence')


    RIfiles = glob.glob(str(RIpath) + '\RI*.npz')

    for file in RIfiles:
        tt = re.search('_G\d+_',file)
        gang = tt[0][2:len(tt[0])-1]
      
        rr = re.search('_\d+\.',file)
        date = rr[0][1:len(rr[0])-1]
        ff = re.search('RI\d+_',file)
        fxnum = ff[0][2:len(ff[0])-1]
       
        
        #print("Gantry Angle " + str(gang) + " Date " + str(date) + " Fraction " + str(fxnum))
        npRT = np.load(file)
        RTarr = npRT['arr_0']
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
        
        #print(cbctprojfile)
        nphalf = np.load(halfprojfile)
        #Load Fluence File with same angle 
        flufile = str(Flupath) + '\Fluence'+ str(gang) + '.npz'
        #print(flufile)
        npflu = np.load(flufile)
        #Load PDOS with same angle 
        PDOSfile = glob.glob(str(RIpath) + '\PDOS_G'+ str(gang) + '_' + '*.npz')
        #print(PDOSfile)
        npPDOS = np.load(PDOSfile[0])
        FLUarr = npflu['arr_0']
        PDOSarr = npPDOS['arr_0']
        PROJarr = npproj['arr_0']
        HALFarr = nphalf['arr_0']
        PDOSgs = np.zeros((PDOSarr.shape[0],PDOSarr.shape[1]),np.uint8)
        FLUgs  = np.zeros((FLUarr.shape[0],FLUarr.shape[1]),np.uint8)
        CBCTgs = np.zeros((PROJarr.shape[0],PROJarr.shape[1]),np.uint8)
        HALFgs = np.zeros((HALFarr.shape[0],HALFarr.shape[1]),np.uint8)
        RTgs = np.zeros((PDOSarr.shape[0],PDOSarr.shape[1]),np.uint8)

        for kk in range(0,PDOSarr.shape[0]):
            for ll in range(PDOSarr.shape[1]):
                PDOSgs[kk,ll] = int(np.rint(PDOSarr[kk,ll]*255/maxPDOS))
                RTgs[kk,ll] = int(np.rint(RTarr[kk,ll]*255/maxRI))
        
        # for mm in range(0,FLUarr.shape[0]):
        #     for nn in range(FLUarr.shape[1]):
        #         FLUgs[mm,nn] = int(np.rint(FLUarr[mm,nn]*255/maxFlu))
        
        for pp in range(0,PROJarr.shape[0]):
            for qq in range(PROJarr.shape[1]):
                if(PROJarr[pp,qq] > 40.0):
                    CBCTgs[pp,qq] = 255
                else: 
                    CBCTgs[pp,qq] = int(np.rint(PROJarr[pp,qq]*255/maxLen))
       
        for rr in range(0,HALFarr.shape[0]):
            for ss in range(HALFarr.shape[1]):
                if(HALFarr[rr,ss] > 40):
                    HALFgs[rr,ss] = 255
                else: 
                    HALFgs[rr,ss] = int(np.rint(HALFarr[rr,ss]*255/maxLen))

        #Join together PDOS and CBCT projection and half CBCT projection (projection to iso)
        #join = np.dstack((FLUgs,PDOSgs,CBCTgs))
        join = np.hstack((CBCTgs,HALFgs,PDOSgs,RTgs))
        filein = str(pcntr) + '_' + str(fxnum) + '_' + str(gang)
        print(filein)
        fileinp = os.path.join(Basepath,'half_jpeg',filein)
        #print("Saving  "+ fileinp)
        #PDOSim = PIL.Image.fromarray(PDOSgs)
        Inputim = PIL.Image.fromarray(join)
        Inputim.save(fileinp + ".jpeg",quality=100,subsampling=0)
        
        #fileoutp = os.path.join(BASEpath,'train\output',filein)
        #Outputim = PIL.Image.fromarray(RTgs)
        #Outputim.save(fileoutp + ".jpeg") 
        #np.savez_compressed(fileinp,join)
        
        #print("Saving output "+ str(fileoutp))
        #np.savez_compressed(fileoutp,RTarr)
        cntr=cntr+1
    print("counter " + str(cntr))
    


######################################################
# Main loop loop over all patients
Basepath = 'P:\Image_Prediction\PatientData'
MRNs = os.listdir(Basepath)
pcntr = 1
for i in range(0,len(MRNs)):
    Patpath = os.path.join(Basepath,MRNs[i])
    print(Patpath)
    MakeJPEG(Patpath,pcntr)
    pcntr = pcntr+1
"""
######################################################
# Generate for single patient
# fid = open(os.path.join('.', 'MRN.txt'))
# fid.readline()
# fid.readline()
# MRN = fid.readline().strip('\n')
# fid.close()
Patpath = 'P:\Image_Prediction\Marginal\\'
#MRNs = os.listdir(Basepath)
cntr = 50
#for i in range(0,len(MRNs)):
#Patpath = os.path.join(Basepath,MRNs[i])
print(Patpath)
cntr = MakeJPEG(Patpath,cntr)
"""

