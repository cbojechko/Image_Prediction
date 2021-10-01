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


def Histdata(Basepath):
    
    #Basepath = 'P:\Image_Prediction\PatientList\\13281530'
    RIpath = os.path.join(Basepath,'RTIMAGE')
    myCTpath = os.path.join(Basepath,'CT')
    Flupath = os.path.join(Basepath,'Fluence')


    RIfiles = glob.glob(str(RIpath) + '\RI*.npz')
    CBCTfiles = glob.glob(str(myCTpath) + '\cbct*.npz')
    PDOSfiles = glob.glob(str(RIpath) + '\PDOS*.npz')
    Flufiles = glob.glob(str(Flupath) + '\Flu*.npz')

    angles = np.zeros(len(RIfiles))
    dates = np.zeros(len(RIfiles))
    fxs =  np.zeros(len(RIfiles))
    #fileout.write("Patient " + str(Basepath) + '\n')
    nmatchdate = 0
    nmatch_pdos_ri_ang =0
    nmatch_flu_ri_ang =0
    RTarrmax =[]
    CBCTmax =[]
    Flumax = []
    print("Number of images " + str(len(RIfiles)))
    for file in RIfiles:
        tt = re.search('_G\d+_',file)
        gang = tt[0][2:len(tt[0])-1]
        riangle = gang
        rr = re.search('_\d+\.',file)
        ridate = rr[0][1:len(rr[0])-1]
        ff = re.search('RI\d+_',file)
        fxnum = ff[0][2:len(ff[0])-1]
        npRT = np.load(file)
        RTarr = npRT['arr_0']
        PDOSfile = glob.glob(str(RIpath) + '\PDOS_G'+ str(gang) + '*.npz')
        #print(PDOSfile)
        npPDOS = np.load(PDOSfile[0])
        #FLUarr = npflu['arr_0']
        PDOSarr = npPDOS['arr_0']
        #print("RT arr max " + str(RTarr.max(1).max(0)))
        RTarrmax = np.append(RTarrmax,[RTarr.max(1).max(0)])
        #RTarrmax = np.append(RTarrmax,[PDOSarr.max(1).max(0)])

        flufile = str(Flupath) + '\Fluence'+ str(gang) + '.npz'
        #print(flufile)
        npflu = np.load(flufile)
        FLUarr = npflu['arr_0']
        Flumax = np.append(Flumax,FLUarr.max(1).max(0))
        
        cbctfile = str(myCTpath) + '\cbctprojection' + str(int(fxnum)) + '_G' + str(int(riangle)) + '_' + str(int(ridate)) + '.npz'
        if(os.path.exists(cbctfile) == False):
            continue
        else:
            #print(cbctfile)
            npcbct = np.load(cbctfile)
            CBCTarr = npcbct['arr_0']
            CBCTmax = np.append(CBCTmax,CBCTarr.max(1).max(0))
            if(CBCTarr.max(1).max(0) > 40):
                print("Larger than 40 cm thickness")
        #print("RT arr max " +str(RTarrmax))



       
    return RTarrmax,Flumax,CBCTmax
    #print('path ' + myCTpath)



# Main loop 
Basepath = 'P:\Image_Prediction\PatientData'
MRNs = os.listdir(Basepath)
#Basepath = 'P:\Image_Prediction\PatientList\\13281530'
#QApatdata(Basepath)
fileoutpath = str(Basepath) + '\\fileout.txt'
fileout = open(fileoutpath, 'w')

RTmaxtot =[]
Flumaxtot =[]
CBCTmaxtot =[]
for i in range(0,len(MRNs)):
    Patpath = os.path.join(Basepath,MRNs[i])
    print(Patpath)
    RTmax,Flumax,CBCTmax = Histdata(Patpath)
    #print("RTmax")
    RTmaxtot = np.append(RTmaxtot,RTmax)
    Flumaxtot = np.append(Flumaxtot,Flumax)
    CBCTmaxtot = np.append(CBCTmaxtot,CBCTmax)
    # print(Flumaxtot)

fig, axs = plt.subplots(3)
axs[0].hist(RTmaxtot, 50, density=True, facecolor='g', alpha=0.75)
axs[1].hist(Flumaxtot, 50, density=True, facecolor='r', alpha=0.75)
axs[2].hist(CBCTmaxtot, 50, density=True, facecolor='b', alpha=0.75)

print('RT max ' + str(np.max(RTmaxtot))) 
print('Flu max ' + str(np.max(Flumaxtot)))
print('CBCT max ' + str(np.max(CBCTmaxtot)))
plt.show()

