
#%%
import glob

# array manipulation and plotting
import numpy as np
from numpy.core.numeric import False_
from numpy.lib.twodim_base import _trilu_dispatcher
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy import misc

#import pymedphys
import os 
import re
import PIL
import pydicom

from scipy import interpolate
from scipy import ndimage
from scipy.constants import e,h,hbar,alpha,c,m_e

def get_HGHD(rtimage,gthres=0.05,dthres=0.2):
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
    predict = 1.0 * pdos * np.exp(-4.0 * cbct)
    return predict

def primaryscatter2(inputjpeg):

    cbct, half, pdos, rt = getimages(inputjpeg)

    #predict = 1.0 * pdos * np.exp(-4.0 * cbct) + 0.15 * pdos * cbct - 0.10 * pdos * cbct * cbct
    #predict = 1.0 * pdos * np.exp(-4.0 * cbct)
    predict = pdos*(1.0 * np.exp(-4.0 * cbct) + 0.15 * cbct - 0.1 * cbct * cbct )

    w = 5.5e-6
    kernel_size=200
    muu=0
    sigma=1
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                       np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x ** 2 + y ** 2)

    # lower normal part of gaussian
    normal = 1 / (2.0 * np.pi * sigma ** 2)

    # Calculating Gaussian filter
    gauss = w*np.exp(-((dst - muu) ** 2 / (2.0 * sigma ** 2))) * normal

    #scharr = np.array([gauss])  # Gx + j*Gy
    scatter = signal.convolve2d(pdos+cbct, gauss, boundary='symm', mode='same')
    #scatter =scatter+cbct*0.1

    #predict = predict+scatter
    return predict,scatter
def primaryscatter(inputjpeg):

    cbct, half, pdos, rt = getimages(inputjpeg)

    predict = 0.9 * pdos * np.exp(-4.0 * cbct)  + 0.15 * pdos * cbct - 0.10*pdos*cbct*cbct-0.01
    #predict = 1.0 * pdos * np.exp(-4.0 * cbct)
    #predict = 1.0 * pdos * np.exp(-4.22 * cbct)

    pdosmean= np.mean(pdos)

    #scatter = np.ones(pdos.shape)*pdosmean*0.07
    scatter = 0.17 * cbct * pdos
    return predict,scatter


def primaryscatterKN(inputjpeg):
    cbct, half, pdos, rt = getimages(inputjpeg)
    predict =0.90*pdos * (1.0 * np.exp(-4.0 * cbct) + 0.15 * cbct - 0.1 * cbct * cbct)
    #predict = pdos*np.exp(-1/0.237*cbct-0.0565/0.237)
    #predict = pdos * (1.0 * np.exp(-4.0 * cbct) )
    f = (hbar*alpha/m_e/c)**2/2
    #px = np.arange(0,128,1)
    #py = np.arange(0, 128, 1)

    px, py = np.meshgrid(np.linspace(-128, 128, 256),
                       np.linspace(-128, 128, 256))

    tpix = np.sqrt(np.square(px)+np.square(py))
    theta = np.arctan(tpix*(430/256.0)/540.0)
    E = 6
    nu = E*1.0e6*e/h
    lam = c/nu
    lamp = lam+h/m_e/c *(1-np.cos(theta))
    norm = 0.1e-5
    P = lam/lamp
    out = norm*P**2*(P+1/P-np.sin(theta)**2)

    scatter = signal.convolve2d(pdos,out, boundary='symm', mode='same')

    return predict, scatter

def primaryattenold(inputjpeg):

    cbct, half, pdos, rt = getimagesold(inputjpeg)

    predict = 1.02 * pdos * np.exp(-4.0 * cbct)  + 0.15 * pdos * cbct - 0.10*pdos*cbct*cbct
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



def getpredprofilesold(inputjpeg,crossval,supinf = False):


    predimg = primaryattenold(inputjpeg)

    if (supinf):
        propred = predimg[1:256, crossval]
    else:
        propred = predimg[crossval, 1:256]

    return propred

def getratio(inputjpeg):

    cbct,half,pdos,rt = getimages(inputjpeg)
    rtmax = rt.max(0).max(-1)
    print(rtmax)
    cbctval = []
    ratio = []
    for i in range(0,256):
        for j in range(0,256):

            if(rt[i,j] > rtmax*0.5):
                cbctval = np.append(cbctval,cbct[i,j])
                ratio = np.append(ratio,rt[i,j]/pdos[i,j])

    return cbctval, ratio

def getimages(inputjpeg):
    inputpdos = np.array(inputjpeg.crop((0,0,256,256)))
    inputrt = np.array(inputjpeg.crop((256,0,512,256)))
    inputcbct = np.array(inputjpeg.crop((512,0,768,256)))
    #inputhalf = np.array(inputjpeg.crop((768,0,1024,256)))
    inputhalf = np.array(inputjpeg.crop((1536,0,1792,256)))



    #inputpdos = inputpdos * 3.448
    #inputrt = inputrt * 2.226

    #return inputcbct,inputhalf,inputpdos,inputrt
    return inputcbct/255.0,inputhalf/255.0,inputpdos/255.0,inputrt/255.0

def getimagesold(inputjpeg):
    inputpdos = np.array(inputjpeg.crop((0,0,256,256)))
    inputcbct = np.array(inputjpeg.crop((256,0,512,256)))
    inputhalf = np.array(inputjpeg.crop((512,0,768,256)))
    inputrt = np.array(inputjpeg.crop((768,0,1024,256)))

    #inputpdos = inputpdos * 3.448
    #inputrt = inputrt * 2.226

    #return inputcbct,inputhalf,inputpdos,inputrt
    return inputcbct/255.0,inputhalf/255.0,inputpdos/255.0,inputrt/255.0


def getprofiles(inputjpeg,crossval,supinf=False):

    inputpdos = np.array(inputjpeg.crop((0,0,256,256)))
    inputrt = np.array(inputjpeg.crop((256,0,512,256)))
    inputcbct = np.array(inputjpeg.crop((512,0,768,256)))
    inputhalf = np.array(inputjpeg.crop((1792, 0, 2048, 256)))
    #inputhalf = np.array(inputjpeg.crop((1024, 0, 1280, 256)))

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
    return procbct/255.0,0.85*prohalf/255.0, propdos/255.0, prort/255.0

#inputjpeg1 = PIL.Image.open('R:\TFRecords\JpegsNoNormalizationMultipleProj\\17_G0_20181008.jpeg')
#inputjpeg2 = PIL.Image.open('R:\TFRecords\JpegsNoNormalization\\17_G0_20181008.jpeg')

#inputjpeg1 = PIL.Image.open(r'/Users/caseybojechko/Documents/Image_Prediction/jpeg/phantom/9976_G0_20220413.jpeg')
#inputjpeg1 = PIL.Image.open(r'/Users/caseybojechko/Documents/Image_Prediction/jpeg/phantom/9977_G180_20220413.jpeg')
#inputjpeg1 = PIL.Image.open(r'/Users/caseybojechko/Documents/Image_Prediction/jpeg/phantom/9950_G30_20220420.jpeg')
#inputjpeg1 = PIL.Image.open(r'/Users/caseybojechko/Documents/Image_Prediction/jpeg/phantom/9951_G185_20220420.jpeg')
#inputjpeg1 = PIL.Image.open(r'/Users/caseybojechko/Documents/Image_Prediction/jpeg/phantom/9966_G0_20220413.jpeg')
inputjpeg1 = PIL.Image.open(r'/Users/caseybojechko/Documents/Image_Prediction/jpeg/phantom/9957_G60_20220420.jpeg')
#inputjpeg1 = PIL.Image.open(r'/Users/caseybojechko/Documents/Image_Prediction/jpeg/phantom/9959_G30_20220420.jpeg')
#inputjpeg1 = PIL.Image.open(r'/Users/caseybojechko/Documents/Image_Prediction/jpeg/phantom/9966_G270_20220413.jpeg')
#inputjpeg1 = PIL.Image.open(r'/Users/caseybojechko/Documents/Image_Prediction/jpeg/phantom/9952_G240_20220420.jpeg')

#inputjpeg1 = PIL.Image.open(r'/Users/caseybojechko/Documents/Image_Prediction/jpeg/fold2/34_G180_20181008.jpeg')
#inputjpeg1 = PIL.Image.open(r'/Users/caseybojechko/Documents/Image_Prediction/jpeg/otherfrac/73_G40_20200914.jpeg')
#inputjpeg1 = PIL.Image.open(r'/Users/caseybojechko/Documents/Image_Prediction/jpeg/fold2/44_G180_20210511.jpeg')
#inputjpeg1 = PIL.Image.open('R:\TFRecords\Jpegs\\76_G0_20220413.jpeg')
#inputjpeg1 = PIL.Image.open('R:\TFRecords\Jpegs\\67_G0_20220413.jpeg')

#inputjpeg1 = PIL.Image.open('R:\TFRecords\JpegsNoNormalizationMultipleProj\\7_G190_20210818.jpeg')
#inputjpeg2 = PIL.Image.open('R:\TFRecords\Jpegs\\7_G190_20210818.jpeg')

crossval = 110
supinf = False

procbct1,prohalf1,propdos1,prort1 = getprofiles(inputjpeg1,crossval,supinf)
#procbct2,prohalf2,propdos2,prort2 = getprofilesold(inputjpeg2,crossval,supinf)

propred1 = getpredprofiles(inputjpeg1,crossval,supinf)
#propred2 = getpredprofilesold(inputjpeg2,crossval,supinf)


cbctval1, ratio1 = getratio(inputjpeg1)
prim,scatter = primaryscatterKN(inputjpeg1)

rtpredict = prim+scatter
if(supinf):
    proprim = prim[1:256,crossval]
    proscatter = scatter[1:256,crossval]
    propred = rtpredict[1:256,crossval]
else:
    proprim = prim[crossval, 1:256]
    proscatter = scatter[crossval, 1:256]
    propred = rtpredict[crossval, 1:256]

#rtpredict2 = primaryatten(inputjpeg2)
#x = np.arange(-128,127,1)
#sigma, mu = 20.0, 0.0
#g = np.exp(-( (x-mu)**2 / ( 2.0 * sigma**2 ) ) )

#print(g)

#g.shape

#conv = 0.0004*np.convolve(procbct1,g,'same')
#conv = 0.004*np.convolve(propdos1,g,'same')

#conv = 0.00045*np.convolve(propdos1,procbct1,'same')
#plt.plot(conv,'g')

#propred1 = propred1 + conv
plt.figure(1)

plt.plot(0.1*procbct1,'r')
plt.plot(0.1*prohalf1,'g')
plt.plot((prort1-proprim),'b')
plt.plot(0.1*propdos1,'k--')


#rtfac = np.max(prort1)/np.max(prort2)
#pdosfac = np.max(propdos1)/np.max(propdos2)

plt.figure(2)
plt.plot(prort1,'g')
#plt.plot(propdos1,'k--')
#plt.plot(1.5*prort2,'r')
#plt.plot(prort2,'g')
#plt.plot(propdos1,'k')
#plt.plot(propdos1,'g--')
plt.plot(proprim,'r--')
#plt.plot(prort1-proprim,'b')
plt.plot(proscatter,'b--')
plt.plot(propred,'k--')
#plt.ylim([0,0.05])
#plt.plot(proscatter/prort1,'k--')
#plt.plot(propred,'g--')
#plt.plot(propred2,'g--')
#plt.plot(propred2,'r--')
#plt.plot(procbct1,'r')
#plt.plot(prort2,'g--')
#plt.plot(0.17*procbct1*propdos1,'g')
#plt.plot(procbct2,'k--')
#plt.plot(prohalf1,'r--')
#plt.plot(prohalf2,'k--')
#plt.plot(procbct1-procbct2,'g')
#plt.plot(-prort1+prort2,'b')
#plt.plot(0.1*procbct1,'r')

#plt.plot(propdos1,'k')
#plt.plot(3.5*(prort1-propred1),'g')
#plt.plot(conv,'g')
#plt.plot(prohalf2,'b--')
#plt.plot(procbct2,'r--')
#plt.plot(3.5*(prort2-propred2),'g--')

cbct1,half1,pdos1,rt1 = getimages(inputjpeg1)
#cbct2,half2,pdos2,rt2 = getimages(inputjpeg2)

#
# nx,ny = np.shape(rt1)
#
# for i in range(0,nx):
#     for j in range(0,ny):
#         if(rt1[i,j] > pdos1[i,j]):
#             print("above unity " + str(i) + " " + str(j))
#
# #
rtpredict = prim+scatter

maxrt = np.max(np.max(rt1))
diffct =[]
ratio = []
pratio = []
for i in range(0,255):
     for j in range(0,255):
         if(rt1[i,j] > 0.5*maxrt):
             diffct = np.append(diffct,cbct1[i,j])
             pratio = np.append(pratio,prim[i,j]/pdos1[i,j])
             ratio = np.append(ratio, rt1[i, j] / pdos1[i, j])

plt.figure(3)
xp = np.arange(0, 1.0, 0.01)
yp = 0.7*(1.0 * np.exp(-4.0 * xp) + 0.15 * xp - 0.1 * xp * xp)
#plt.plot(xp, yp)
plt.plot(diffct,ratio,'bo')
plt.plot(diffct,pratio,'ro')
#plt.plot(procbct2.astype(float)-procbct1.astype(float),'r')
#plt.plot(prort1.astype(float)-prort2.astype(float),'b')
# gampass = calcgamma(rt1,rtpredict)
# print(gampass)

#mask = get_HGHD(rt1)

#plt.figure(3)
#plt.imshow(mask*rt1)
#plt.subplot(2,1,1)
#plt.imshow(cbct1)
#plt.plot(procbct1,'r')
#plt.subplot(2,1,2)
#plt.imshow(rt1-rt2)
#plt.imshow(cbct2)
#plt.plot(diffct,diffrt,'bo')
#plt.subplot(3,1,1)
#plt.axhline(y=128, color='r', linestyle='-')
#plt.axvline(x=160, color='r', linestyle='-')
#plt.imshow(cbct2-cbct1)
#plt.subplot(3,1,2)
#plt.imshow(rt1-rt2)
#plt.subplot(3,1,3)
#plt.imshow(rtpredict)
plt.show()

xp = np.arange(0,256,1)
yp = np.arange(0,256,1)
X,Y = np.meshgrid(xp,yp)
plt.figure(4)
ax = plt.axes(projection='3d')
ax.plot_surface(X,Y,rt1-prim)
#plt.plot_surface(rt1-prim)
plt.show()
