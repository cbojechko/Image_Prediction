#%%
import os 
import numpy as np
import pydicom
import matplotlib.pyplot as plt
#import SimpleITK as sitk
from glob import glob
import re
import PIL
from mpl_toolkits import mplot3d



def get_HGHD(rtimage,gthres=0.05,dthres=0.5):
    #gradient = get_Grad(rtimage)

    nx, ny = np.shape(rtimage)
    mask = np.zeros((nx, ny))
    rtmax = rtimage.max(0).max(-1)
    dthres = dthres*rtmax
    print('Cutoff ' +str(dthres))
    """   
    for i in range(0, nx):
        for j in range(0, ny):
            if(gradient[i,j] < gthres and rtimage[i,j] > dthres):
             mask[i,j] = 1.0
    """
    for i in range(0, nx):
        for j in range(0, ny):
            if (rtimage[i, j] > dthres):
                mask[i, j] = 1.0


    return mask

def get_Grad(rtimage):


    nx, ny = np.shape(rtimage)
    gradient = np.zeros((nx,ny))
    pixs = 1.68

    for i in range(1,nx-1,2):
        for j in range(1,ny-1,2):
            gradient[i,j] = np.sqrt((((rtimage[i,j]-rtimage[i-1,j])/pixs)**2) + (((rtimage[i,j]-rtimage[i+1,j])/pixs)**2) + (((rtimage[i,j]-rtimage[i,j-1])/pixs)**2) + (((rtimage[i,j]-rtimage[i,j+1])/pixs)**2))

    return gradient


def getCAX(inputjpeg):

    cbct,half,pdos,rt = getimages(inputjpeg)

    cbctcax = np.mean(cbct[126:130,126:130])
    halfcax = np.mean(half[126:130,126:130])
    pdoscax =np.mean(pdos[126:130,126:130])
    rtcax = np.mean(rt[126:130,126:130])
    airgap = halfcax


    return cbctcax/255.0, halfcax/255.0,pdoscax/255.0,rtcax/255.0,airgap/255.0


def getOAX(inputjpeg):

    cbct,half,pdos,rt = getimages(inputjpeg)


    result = np.where(rt == np.amax(rt))
    #print(result)
    listOfCordinates = list(zip(result[0], result[1]))

    #print(listOfCordinates[0])
    midx = listOfCordinates[0][0]
    midy = listOfCordinates[0][1]
    cbctcax = np.mean(cbct[midx-2:midx+2,midy-2:midy+2])
    halfcax = np.mean(half[midx-2:midx+2,midy-2:midy+2])
    pdoscax =np.mean(pdos[midx-2:midx+2,midy-2:midy+2])
    rtcax = np.mean(rt[midx-2:midx+2,midy-2:midy+2])
    airgap = halfcax

    return cbctcax/255.0, halfcax/255.0,pdoscax/255.0,rtcax/255.0,airgap/255.0


def getimages(inputjpeg):
    inputpdos = np.array(inputjpeg.crop((0, 0, 256, 256)))
    inputcbct = np.array(inputjpeg.crop((512, 0, 768, 256)))
    inputhalf = np.array(inputjpeg.crop((1536, 0, 1792, 256)))
    inputrt = np.array(inputjpeg.crop((256, 0, 512, 256)))

    return inputcbct, inputhalf, inputpdos, inputrt


def plot_cax_data(path,foldlist):

    jpeg_files = glob(os.path.join(path,'*.jpeg'))
    #print(jpeg_files)
    cbctvals = []
    ratiovals = []
    airgapvals = []
    pdosvals = []
    rtvals = []
    for file in jpeg_files:
        tt = file.split('/')
        rr = tt[7].split('_')
        for idx in foldlist:
            if(int(rr[0]) == idx):


                jimage = PIL.Image.open(file)
                #print(file)
                cbctcax, halfcax,pdoscax,rtcax,airgap = getCAX(jimage)
                """  
                if(int(rr[0]) == 7):
                    print(file)
                    cbctcax, halfcax, pdoscax, rtcax, airgap = getOAX(jimage)
                if(pdoscax ==0):
                    print("ZERO")
                    continue
                """
                ratio = rtcax/pdoscax
                cbctvals = np.append(cbctvals,cbctcax)
                rtvals = np.append(rtvals,rtcax)
                pdosvals = np.append(pdosvals,pdoscax)
                airgapvals = np.append(airgapvals, airgap)
                ratiovals = np.append(ratiovals,ratio)
                #print("rtcax " + str(rtcax) + " cbct cax " + str(cbctcax))

                if (ratio < 0.26 and cbctcax < 0.36 ):
                    print("Off profile " + file)

                if(ratio >0.65):
                    print("ratio larger that 0.65 " + file)
        #if(cbctcax >125 and ratio > 0.23):
            #print("below curve " + file)
        #if(cbctcax < 46):
        #    print("low cbct " + file)




    #plt.figure()
    #plt.plot(cbctvals,ratiovals,'bo')
    #plt.show()
    return cbctvals,ratiovals,pdosvals,rtvals,airgapvals


def main():
    #path = 'R:\TFRecords\Jpegs'
    #path = 'R:\TFRecords\JpegsNoNormalization'
    #path = 'P:\Image_Prediction\half_jpeg'
    file1 = r"/Users/caseybojechko/Documents/Image_Prediction/jpeg/fold1/1_G0_20190530.jpeg"
    jpeg1 = PIL.Image.open(file1)
    cbct, half, pdos, rt = getimages(jpeg1)


    mask = get_HGHD(rt,0.10,0.1)

    masked = mask*rt
    plt.figure()
    plt.imshow(cbct)
    #plt.imshow(masked,alpha=0.5)
    #plt.contour(masked,alpha=0.1)
    #plt.imshow(rt)
    plt.contour(rt,levels=[50])

    plt.show()



if __name__ == '__main__':
    main()

