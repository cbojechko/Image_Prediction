#%%
import glob

# array manipulation and plotting
import numpy as np
from numpy.core.numeric import False_
from numpy.lib.twodim_base import _trilu_dispatcher
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt

#import pymedphys
import os 
import re
import PIL
from glob import glob
import pydicom

#from scipy import interpolate
#from scipy import ndimage
from collections import defaultdict
import xlsxwriter

def get_HGHD(rtimage,gthres=0.05,dthres=0.5):
    gradient = get_Grad(rtimage)

    nx, ny = np.shape(rtimage)
    mask = np.zeros((nx, ny))
    rtmax = rtimage.max(0).max(-1)
    dthres = dthres*rtmax

    for i in range(0, nx):
        for j in range(0, ny):
            if(gradient[i,j] < gthres and rtimage[i,j] > dthres):
             mask[i,j] = 1.0

    return mask

def get_Grad(rtimage):


    nx, ny = np.shape(rtimage)
    gradient = np.zeros((nx,ny))
    pixs = 1.68

    for i in range(1,nx-1,2):
        for j in range(1,ny-1,2):
            gradient[i,j] = np.sqrt((((rtimage[i,j]-rtimage[i-1,j])/pixs)**2) + (((rtimage[i,j]-rtimage[i+1,j])/pixs)**2) + (((rtimage[i,j]-rtimage[i,j-1])/pixs)**2) + (((rtimage[i,j]-rtimage[i,j+1])/pixs)**2))

    return gradient

def primaryatten(inputjpeg):
    cbct, half, pdos, rt = getimages(inputjpeg)

    predict = 1.0 * np.exp(-4.0 * cbct)  + 0.15 * cbct -0.10*cbct*cbct
    #predict = 1.02 * pdos * np.exp(-4.0 * cbct)*np.exp(-0.5*half/cbct) +0.35*pdos*cbct
    #predict = 1.02 * pdos * np.exp(-4.0 * cbct)
    #predict = 1.02 * pdos * np.exp(-4.0 * cbct)+0.15*pdos*cbct - 0.05*np.square(0.4-cbct)
    #predict = 1.02*pdos*np.exp(-4.0*cbct)*np.exp(-0.05*half/cbct)+0.07*pdos*cbct
    #predict = 1.02*pdos*np.exp(-4.7*cbct)+0.4*pdos*cbct
    #predict = 1.02 * pdos * np.exp(-4.0 * cbct) + 0.10 * pdos * cbct + pdos*0.2*(cbct-half)
    #predict = 1.02 * pdos * np.exp(-5.0 * half) *np.exp(-3.0 * (cbct-half)) + 0.10 * pdos * cbct + pdos * 0.2 * (cbct - half)
    #predict = 1.02 * pdos * np.exp(-2.0 * half) *np.exp(-5.0 * (cbct-half))  + 0.15 * pdos * cbct -0.10*pdos*cbct*cbct
    return predict

def getpredprofiles(inputjpeg,crossval,supinf = False):


    predimg = primaryatten(inputjpeg)

    if (supinf):
        propred = predimg[1:256, crossval]
    else:
        propred = predimg[crossval, 1:256]

    return propred

def getimages(inputjpeg):
    inputpdos = np.array(inputjpeg.crop((0,0,256,256)))
    inputrt = np.array(inputjpeg.crop((256,0,512,256)))
    inputcbct = np.array(inputjpeg.crop((512,0,768,256)))
    inputhalf = np.array(inputjpeg.crop((768,0,1024,256)))

    #return inputcbct,inputhalf,inputpdos,inputrt
    return inputcbct/255.0,inputhalf/255.0,inputpdos/255.0,inputrt/255.0

def getprofiles(inputjpeg,crossval,supinf=False):

    inputpdos = np.array(inputjpeg.crop((0,0,256,256)))
    inputrt = np.array(inputjpeg.crop((256,0,512,256)))
    inputcbct = np.array(inputjpeg.crop((512,0,768,256)))
    inputhalf = np.array(inputjpeg.crop((768,0,1024,256)))

    #inputpdos = inputpdos/np.max(np.max(inputpdos))

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

    #return procbct , prohalf , propdos , prort
    return procbct/255.0,prohalf/255.0, propdos/255.0, prort/255.0

def getprofilesnew(inputjpeg,crossval,supinf=False):

    inputpdos = np.array(inputjpeg.crop((0,0,256,256)))
    inputrt = np.array(inputjpeg.crop((256,0,512,256)))
    inputcbct = np.array(inputjpeg.crop((512,0,768,256)))
    inputhalf = np.array(inputjpeg.crop((768,0,1024,256)))

    #inputrt = inputrt / np.max(np.max(inputpdos))
    #inputpdos = inputpdos / np.max(np.max(inputpdos))


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

    #return procbct , prohalf , propdos , prort
    return procbct/255.0,prohalf/255.0, propdos/255.0, prort/255.0

def calcgamma(rt,prediction):

    gamma_options = {
        'dose_percent_threshold': 3,  # Try with 2%/2mm
        'distance_mm_threshold': 3,
        'lower_percent_dose_cutoff': 10,
        'interp_fraction': 20,  # Should be 10 or more, see the paper referenced above
        'max_gamma': 2,
        'random_subset': None,  # Can be used to get quick pass rates
        'local_gamma': False,  # Change to false for global gamma
        'ram_available': 2 ** 29  # 1/2 GB
    }

    xepidmin = -256
    xepidmax = 256
    yepidmin = -256
    yepidmax = 256
    grid = 2
    xepid = np.arange(xepidmin, xepidmax, grid)
    yepid = np.arange(yepidmin, yepidmax, grid)
    coords = (yepid, xepid)

    gamma_test = pymedphys.gamma(coords, rt, coords, prediction, **gamma_options)
    valid_gamma = gamma_test[~np.isnan(gamma_test)]
    pass_ratio = np.sum(valid_gamma <= 1) / len(valid_gamma)
    return pass_ratio

def getratio(inputjpeg):

    cbct,half,pdos,rt = getimages(inputjpeg)
    rtmax = rt.max(0).max(-1)
    #print(rtmax)
    cbctval = []
    ratio = []

    mask = get_HGHD(rt)

    #rt*mask


    for i in range(0,256):
        for j in range(0,256):
            if(mask[i,j] == 1.0):
                cbctval = np.append(cbctval,cbct[i,j])
                ratio = np.append(ratio,rt[i,j]/pdos[i,j])
                if(rt[i,j]/pdos[i,j]> 1.0 and cbct[i,j] > 0.6):
                        print("OUTLIER ")

    return cbctval, ratio

def make_plots(path,idx_and_angles,dfin):
    # Make the CAX the middle of the image
    #print(idx_and_angles)
    #outpath = os.path.join(os.path.dirname(path),'QAplots')
    outpath = os.path.join(os.path.dirname(path),'QAother')
    #crossval = 128
    idx = idx_and_angles[0]
    #print("Patient idx " +str(idx))
    nplots = len(idx_and_angles)-1
    #print("Making " + str(nplots) + " Plots ")

    delx = []
    dely = []
    for i in range(1,len(idx_and_angles)):
        filen = str(idx) + '_G' + str(idx_and_angles[i]) + '*.jpeg'
        #print(filen)
        #filesiout = str(idx) + '_G' + str(idx_and_angles[i]) + '_' + str(date) + '_rt_cbct.jpeg'
        #print(filesiout)
        #filelatout = str(idx) + '_G' + str(idx_and_angles[i]) + '_lat.jpeg'

        #print("test")
        #print(idx)
        #print("Angle " +  str(idx_and_angles[i]))

        #print(imgout)
        #latout = os.path.join(outpath, filelatout)
        jfiles = glob(os.path.join(path,filen))
        jfiles = np.sort(jfiles)

        leg = []

        for jdx in range(0,len(jfiles)):
            imgfile = jfiles[jdx]
            #print(imgfile)
            #image1 = PIL.Image.open(jfiles[0])
            image2 = PIL.Image.open(imgfile)
            tt = imgfile.split('/')
            rr = tt[7].split('_')
            patidx = rr[0]
            gang = rr[1].split('G')[1]
            print("Ang " + str(gang))
            ss = rr[2].split('.')
            date = ss[0]
            leg = np.append(leg,date)
            #print(image1)
            delCT,delRT = getratio(image2)
            #Calc variance
            toterr =0
            n = len(delCT)

            filesiout = str(idx) + '_G' + str(idx_and_angles[i]) + '_' + str(date) + '_rt_cbct.jpeg'
            imgout = os.path.join(outpath, filesiout)
            for j in range(0,n):
                err = np.square(delRT[j]- (1.0 * np.exp(-4.0 * delCT[j]) + 0.15 * delCT[j] - 0.1 * delCT[j] * delCT[j]))
                toterr = toterr + err

            print("total error " + str(toterr) + " Normalized " + str(toterr/n))
            dfin.append( {'idx': idx, 'angle': gang, 'err': toterr/n})


            delx = np.append(delx,delRT)
            dely = np.append(dely, delCT)

            #fig = plt.figure()
            plt.plot(delCT,delRT,'o')
            plt.ylabel("RT")
            plt.xlabel("CBCT")
            xp = np.arange(0, 1.0, 0.01)
            y = 1.0 * np.exp(-4.0 * xp) + 0.15 * xp - 0.1 * xp * xp
            plt.plot(xp, y)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 2.0])
            #corrr = np.corrcoef(delx, dely)
            #print("Correlation " + str(corrr[0,1]))
            plt.legend(leg)
            plt.savefig(imgout)
            plt.clf()

    return dely, delx, dfin

def sortJpegs(path,dfin):
    jpeg_files = glob(os.path.join(path, '*.jpeg'))

    allthings = []
    for file in jpeg_files:
        print(file)
        tt = file.split('/')
        rr = tt[7].split('_')
        patidx = rr[0]
        gang = rr[1].split('G')[1]
        ss = rr[2].split('.')
        date = ss[0]
        tupdata = (patidx, gang)
        allthings.append(tupdata)

    idxandgang = list(set([i for i in allthings]))
    idxandgang = sorted(idxandgang, key=lambda x: x[0])

    mapp = defaultdict(list)
    for key, val in idxandgang:
        mapp[key].append(val)
    res = [(key, *val) for key, val in mapp.items()]

    fullCT = []
    fullRT = []
    for line in res:
        #print(line[0])

        delCT, delRT, dfout = make_plots(path,line,dfin)
        fullCT = np.append(delCT, fullCT)
        fullRT = np.append(delRT, fullRT)


    plt.plot(fullCT, fullRT, 'bo')

    xp = np.arange(0, 1.0, 0.01)
    y = 1.0 * np.exp(-4.0 * xp) + 0.15 * xp - 0.1 * xp * xp
    plt.plot(xp, y)
    plt.show()
    return dfout

def main():
    #path = 'R:\TFRecords\Jpegs'
    #path = 'R:\TFRecords\JpegsNoNormalization'
    bpath = "/Users/caseybojechko/Documents/Image_Prediction/jpeg"
    #path = 'R:\TFRecords\JpegsNoNormalizationMultipleProj
    phantom = True
    dfin = []
    if phantom:
        dfin = []
        #folder = "testpat"
        folder = "otherfrac"
        path = os.path.join(bpath, folder)
        dfout = sortJpegs(path,dfin)
        #df = pd.DataFrame(dfout)
        #df.sort_values(by=['idx'])
        #df.to_excel("/Users/caseybojechko/Documents/Image_Prediction/testphan.xlsx")
    else:
        dfall =[]
        for fold in range(1,6):
            dfin = []
            folder = "fold" + str(fold)
            path = os.path.join(bpath,folder)
            dfout = sortJpegs(path,dfin)
            dfall = dfall + dfout

        df = pd.DataFrame(dfall)
        df.idx = df.idx.astype(int)
        df = df.sort_values(by=['idx'])
        df.to_excel("/Users/caseybojechko/Documents/Image_Prediction/testpat.xlsx")

    return None

if __name__ == '__main__':
    main()


