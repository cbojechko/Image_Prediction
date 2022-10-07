#%%
import os 
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import SimpleITK as sitk
from glob import glob
import re
import PIL

from itertools import groupby
from collections import defaultdict

def getprofiles(inputjpeg,crossval=128,supinf=True):

    inputpdos = np.array(inputjpeg.crop((0,0,256,256)))
    inputcbct = np.array(inputjpeg.crop((256,0,512,256)))
    inputhalf = np.array(inputjpeg.crop((512,0,768,256)))
    inputrt = np.array(inputjpeg.crop((768,0,1024,256)))


    if(supinf):
        procbct = inputcbct[1:256,crossval]
        prohalf = inputhalf[1:256,crossval]
        propdos = inputpdos[1:256,crossval]
        prort = inputrt[1:256,crossval]
    else:
        procbct = inputcbct[crossval,1:256]
        prohalf = inputhalf[crossval,1:256]
        propdos = inputpdos[crossval,1:256]
        prort = inputrt[crossval,1:256]

    return procbct,prohalf,propdos,prort


def getCAX(inputjpeg):

    cbct,half,pdos,rt = getimages(inputjpeg)
    cbctcax = np.mean(cbct[126:130,126:130])
    halfcax = np.mean(half[126:130,126:130])
    pdoscax =np.mean(pdos[126:130,126:130])
    rtcax = np.mean(rt[126:130,126:130])

    return cbctcax, halfcax,pdoscax,rtcax



def getimages(inputjpeg):
    inputpdos = np.array(inputjpeg.crop((0, 0, 256, 256)))
    inputcbct = np.array(inputjpeg.crop((256, 0, 512, 256)))
    inputhalf = np.array(inputjpeg.crop((512, 0, 768, 256)))
    inputrt = np.array(inputjpeg.crop((768, 0, 1024, 256)))


    return inputcbct, inputhalf, inputpdos, inputrt


def make_summary_plot(path,idx_and_angles):
    # Make the CAX the middle of the image

    outpath = os.path.join(os.path.dirname(path),'summary_plots')
    crossval = 128
    idx = idx_and_angles[0]
    print("Patient idx " +str(idx))
    nplots = len(idx_and_angles)-1
    print("Making " + str(nplots) + " Plots ")

    for i in range(1,len(idx_and_angles)):
        filen = str(idx) + '_G' + str(idx_and_angles[i]) + '_*.jpeg'
        filesiout = str(idx) + '_G' + str(idx_and_angles[i]) +'_supinf.jpeg'
        filelatout = str(idx) + '_G' + str(idx_and_angles[i]) + '_lat.jpeg'
        siout = os.path.join(outpath,filesiout )
        latout = os.path.join(outpath, filelatout)
        jfiles = glob(os.path.join(path,filen))
        #print(jfiles)
        figsi,axsi = plt.subplots(2)
        figlat, axlat = plt.subplots(2)
        for imgfile in jfiles:
            inputjpeg = PIL.Image.open(imgfile)
            sicbct, sihalf, sipdos, sirt = getprofiles(inputjpeg, crossval)
            axsi[0].plot(sicbct)
            axsi[1].plot(sirt)
            latcbct, lathalf, latpdos, latrt = getprofiles(inputjpeg, crossval,False)
            axlat[0].plot(latcbct)
            axlat[1].plot(latrt)
        figsi.savefig(siout)
        figsi.clf()
        figlat.savefig(latout)
        figlat.clf()


    #plt.figure()
    #plt.plot(cbctvals,ratiovals,'bo')
    #plt.show()
    #return cbctvals,ratiovals
    return None

def main():
    path = 'R:\TFRecords\JpegsNoNormalization'
    jpeg_files = glob(os.path.join(path, '*.jpeg'))

    allthings =[]
    for file in jpeg_files:
        tt = file.split('\\')
        rr = tt[3].split('_')
        patidx = rr[0]
        gang = rr[1].split('G')[1]
        ss = rr[2].split('.')
        date = ss[0]
        tupdata = (patidx,gang)
        allthings.append(tupdata)

    idxandgang = list(set([i for i in allthings]))
    idxandgang = sorted(idxandgang, key=lambda x: x[0])

    mapp = defaultdict(list)
    for key, val in idxandgang:
        mapp[key].append(val)
    res = [(key, *val) for key, val in mapp.items()]

    for line in res:
        make_summary_plot(path,line)


    return None

if __name__ == '__main__':
    main()


#plt.figure(2)
#plt.plot(hFlucax,hPDOScax,'bo')
#plt.plot(hsum,hratio,'bo')
#plt.plot(hsumuni,hratiouni,'ro')
#plt.plot(hCBCTcax,hratio2,'bo')
#plt.plot(hratio2uni,hratiouni,'ro')
#plt.plot(hCBCTcax,hratio,'bo')
#plt.plot(hCBCTuni,hratiouni,'ro')
#plt.show()




# %%
