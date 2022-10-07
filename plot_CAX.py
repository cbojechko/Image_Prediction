#%%
import os 
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import SimpleITK as sitk
from glob import glob
import re
import PIL
from mpl_toolkits import mplot3d


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
        tt = file.split('\\')
        rr = tt[4].split('_')
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
    path1 = 'R:\TFRecords\JpegsNoNormalizationMultipleProj\\fold1'
    path2 =  'R:\TFRecords\JpegsNoNormalizationMultipleProj\\fold2'
    path3 = 'R:\TFRecords\JpegsNoNormalizationMultipleProj\\fold3'
    path4 = 'R:\TFRecords\JpegsNoNormalizationMultipleProj\\fold4'
    path5 = 'R:\TFRecords\JpegsNoNormalizationMultipleProj\\fold5'

    phantom = [0,1,2,3,4]
    #pats = [0, 6, 8, 10, 11, 12, 13, 15, 16] # remove 7 no cax
    #pats = np.append(pats,[1, 9, 14, 17, 20, 21, 22, 23, 24, 25])
    #pats = np.append(pats,[26, 27, 28,  32, 33, 34, 35, 44]) # remove 18 and 31 no cax
    #pats = np.append(pats,[3, 19, 29, 36, 37, 38, 39, 40, 45 ])
    #pats = np.append(pats, [4, 5, 30, 34, 41, 42, 43, 46, 47, 48,49])
    # testpat = [16]
    fold1 = [1, 2,  4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,27, 28, 29, 30 ]  # remove 3 no cax
    fold2 = [31, 32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50, 51,52,53,54,55,56 ]
    fold3 = [57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,74,75,76,77,78,79,80,81,82,83,84] # remove 73 no cax
    fold4 = [85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97 ,98, 99, 100, 101, 102 ,103, 104, 105 ,106, 107, 108, 109, 110, 111, 112, 112, 114 ]
    fold5 = [115,116,117,118,119,120,121,122,123,124,125,126,127,128, 129, 130, 131, 132, 133,134, 135, 136, 137, 138, 139]

    allpats = []
    allpats = np.concatenate((fold1,fold2,fold3,fold4,fold5))
    #allpats = np.append(allpats,fold2)
    #allpats = np.append(allpats,fold3)
    #allpats = np.append(allpats,fold4)
    #allpats = np.append(allpats,fold5)
    testpat = [116]

    #cbctvals, ratiovals = plot_cax_data(path,phantom)
    cbctvals1, ratiovals1, pdosvals1, rtvals1, airgapvals1 = plot_cax_data(path1, fold1)
    cbctvals2, ratiovals2,pdosvals2,rtvals2,airgapvals2 = plot_cax_data(path2,fold2)
    cbctvals3, ratiovals3, pdosvals3, rtvals3, airgapvals3 = plot_cax_data(path3, fold3)
    cbctvals4, ratiovals4, pdosvals4, rtvals4, airgapvals4 = plot_cax_data(path4, fold4)
    cbctvals5, ratiovals5, pdosvals5, rtvals5, airgapvals5 = plot_cax_data(path5, fold5)
    #cbcttest, ratiotest, pdostest, rttest, airgaptest = plot_cax_data(path, testpat)
    #cbctvals1, ratiovals1, pdosvals1, rtvals1, airgapvals1 = plot_cax_data(path, fold1)

    plt.figure()

    plt.plot(cbctvals1, ratiovals1, 'bo')
    plt.plot(cbctvals2, ratiovals2, 'bo')
    plt.plot(cbctvals3, ratiovals3, 'bo')
    plt.plot(cbctvals4, ratiovals4, 'bo')
    plt.plot(cbctvals5, ratiovals5, 'ro')

    #plt.plot(cbctvals1, ratiovals1, 'ro')
    #plt.plot(cbcttest,  ratiotest, 'ro')
    #plt.ylim([0,0.6])


    #plt.plot(airgapvals,ratiovals-np.exp(-4.0*cbctvals)*np.exp(-0.01*airgapvals/cbctvals),'bo')
    #plt.plot(cbcttest, ratiotest, 'ro')
    #plt.plot(airgapvals/cbctvals,ratiovals,'bo')
    #plt.plot(airgaptest ,ratiotest,'ro')
    #plt.plot(cbcttest,airgaptest, 'ro')
    #print("cbct test")
    #print(cbcttest)
    #print("half test")
    #print(airgaptest/cbcttest)

    #xp = np.arange(0.15,0.6, 0.01)
    xp = np.arange(0,1.0, 0.01)
    #xp = np.arange(0, 255, 0.01)
    # xp = np.arange(0, 1.0, 0.01)
    # #y = 1.0*np.exp(-0.015*xp)
    y = 1.0*np.exp(-4.0*xp) + 0.15 * xp -0.1*xp*xp
    #y = 1.0 * np.exp(-2.0 * xp)
    # #y = 1-4*xp+8*xp*xp-10.67*xp*xp*xp+10.67*np.power(xp,4)-8.533*np.power(xp,5)
    # #y = 1.69-0.31*xp
    #plt.plot(xp,y)

    # ax = plt.axes(projection='3d')
    # xp = np.arange(0,1.0, 0.01)
    # yp = np.arange(0, 0.7, 0.01)
    # X,Y = np.meshgrid(xp,yp)


    #ax.plot(cbctvals, ratiovals , 'r+', zdir='airgapvals/cbctvals', ys=1.0)

    #Z= -2.4*X-0.05*Y-0.2
    #Z = X*np.exp(-4.0*X)
    #Z = np.exp(-(3.25*Y+2.5) * X)-0.2
    #Z = np.exp(-4.0*X)*np.exp(-7.0*Y)+0.1*X*X
    #ax.plot_wireframe(X,Y,Z)

    #ax.scatter3D(cbctvals, airgapvals, ratiovals, c='b' )
    #ax.scatter3D(cbctvals, airgapvals, ratiovals, c='b' )
    #ax.scatter3D(cbcttest, airgaptest / cbcttest, np.log(ratiotest), c='r',s=200)
    plt.show()



    #plt.contour([cbctvals1, pdosvals1], ratiovals1)

    #x = np.array([0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.4])
    #y = 0.9*np.exp(-4.0*cbctvals1)
    #plt.plot(cbctvals1, y)

    #x = np.array([0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.4])
    #y = -1.0*np.square(1.0*cbctvals1-0.9)
    #plt.plot(cbctvals1, y)

    #plt.plot(cbctvals1, np.divide(rtvals1,pdosvals1), 'ro')
    #plt.plot(np.multiply(pdosvals1,cbctvals1), rtvals1, 'ro')
    # plt.plot(pdosvals2*cbctvals2, rtvals2, 'ro')
    # plt.plot(pdosvals3*cbctvals3, rtvals3, 'ro')
    # plt.plot(pdosvals4*cbctvals4, rtvals4, 'ro')
    # plt.plot(pdosvals5*cbctvals5, rtvals5, 'ro')


    """ 
    plt.plot(airgapvals1, ratiovals1, 'ro')
    plt.plot(airgapvals2, ratiovals2, 'bo')
    plt.plot(airgapvals3, ratiovals3, 'bo')
    plt.plot(airgapvals4, ratiovals4, 'bo')
    plt.plot(airgapvals5, ratiovals5, 'bo')

      
    plt.plot(cbctvals1,airgapvals1, 'bo')
    plt.plot(cbctvals2,airgapvals2, 'bo')
    plt.plot(cbctvals3,airgapvals3, 'bo')
    plt.plot(cbctvals4,airgapvals4, 'bo')
    plt.plot(cbctvals5,airgapvals5, 'bo')
    
    """

    """ 
    fcn1 = 0.9*pdosvals1*np.exp(-4.0*cbctvals1)
    fcn2 = 0.9*pdosvals2*np.exp(-4.0*cbctvals2)
    fcn3 = 0.9 *pdosvals3*np.exp(-4.0 * cbctvals3)
    fcn4 = 0.9 *pdosvals4* np.exp(-4.0 * cbctvals4)
    fcn5 = 0.9 *pdosvals5*np.exp(-4.0 * cbctvals5)

    #fcnt = pdostest*1.02*np.exp(-4.0*cbcttest)
    plt.plot(cbctvals1, rtvals1-fcn1, 'bo')
    plt.plot(cbctvals2, rtvals2-fcn2, 'bo')
    plt.plot(cbctvals3, rtvals3 - fcn3, 'bo')
    plt.plot(cbctvals4, rtvals4 - fcn4, 'bo')
    plt.plot(cbctvals5, rtvals5 - fcn5, 'bo')
    """
    #plt.plot(cbcttest, rttest - fcnt, 'ro')
    #plt.plot(cbctvals3,ratiovals3-1.02*np.exp(-4.0*cbctvals3),  'bo')
    #plt.plot(cbctvals4, ratiovals4-1.02*np.exp(-4.0*cbctvals4), 'bo')
    #plt.plot(cbctvals5,  ratiovals5-1.02*np.exp(-4.0*cbctvals5), 'bo')


    #plt.plot(cbcttest, ratiotest, 'ro')


    #plt.plot(cbcttest, ratiotest, 'go')



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
