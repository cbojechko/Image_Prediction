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


def getArea(inputjpeg):

    cbct,half,pdos,rt = getimages(inputjpeg)
    sig = np.where(pdos>12.0,1,0)
    out = np.sum(sig)
    area = out/(256*256)
    return area


def getCAX(inputjpeg):

    cbct,half,pdos,rt = getimages(inputjpeg)
    cbctcax = np.mean(cbct[126:130,126:130])
    halfcax = np.mean(half[126:130,126:130])
    pdoscax =np.mean(pdos[126:130,126:130])
    rtcax = np.mean(rt[126:130,126:130])



    return cbctcax , halfcax , pdoscax , rtcax, halfcax
    #return cbctcax/255.0, halfcax/255.0,pdoscax/255.0,rtcax/255.0,airgap/255.0


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
    inputcbct = np.array(inputjpeg.crop((256, 0, 512, 256)))
    inputhalf = np.array(inputjpeg.crop((512, 0, 768, 256)))
    inputrt = np.array(inputjpeg.crop((768, 0, 1024, 256)))

    return inputcbct, inputhalf, inputpdos, inputrt


def plot_cax_data(path,foldlist):

    jpeg_files = glob(os.path.join(path,'*.jpeg'))
    #print(jpeg_files)
    cbctvals = []
    ratiovals = []
    airgapvals = []
    pdosvals = []
    rtvals = []
    areavals = []
    for file in jpeg_files:
        tt = file.split('\\')
        rr = tt[3].split('_')
        for idx in foldlist:
            if(int(rr[0]) == idx):


                jimage = PIL.Image.open(file)
                #print(file)
                cbctcax, halfcax,pdoscax,rtcax,airgap = getCAX(jimage)
                area = getArea(jimage)
                if(int(rr[0]) == 7):
                    print(file)
                    cbctcax, halfcax, pdoscax, rtcax, airgap = getOAX(jimage)
                if(pdoscax ==0):
                    print("ZERO")
                    continue
                ratio = rtcax/pdoscax
                cbctvals = np.append(cbctvals,cbctcax)
                rtvals = np.append(rtvals,rtcax)
                pdosvals =  np.append(pdosvals,pdoscax)
                airgapvals = np.append(airgapvals, airgap)
                ratiovals = np.append(ratiovals,ratio)
                areavals = np.append(areavals,area)
                #print("CBCT " + str(cbctcax) + " ratio " + str(ratio))

                if (area > 0.12  ):
                    print(file)

        #if(ratio >0.7):
         #   print("ratio larger that 0.7 " + file)
        #if(cbctcax >125 and ratio > 0.23):
            #print("below curve " + file)
        #if(cbctcax < 46):
        #    print("low cbct " + file)




    #plt.figure()
    #plt.plot(cbctvals,ratiovals,'bo')
    #plt.show()
    return cbctvals,ratiovals,pdosvals,rtvals,areavals


def main():
    path = 'R:\TFRecords\Jpegs'

    pats = [0, 6,  8, 10, 11, 12, 13, 15, 16] # remove 7 no cax
    pats = np.append(pats,[1, 9, 14, 17, 20, 21, 22,23,24,25])
    pats = np.append(pats,[  26, 27, 28, 32, 33, 34, 35 ,44]) # remove 18 and 31 no cax
    pats = np.append(pats,[3, 19, 29, 36, 37, 38, 39, 40, 45 ])
    pats = np.append(pats, [4, 5, 30, 34, 41, 42, 43, 46, 47, 48,49])


    phanpats = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    phanpats = np.append(phanpats, [60, 61, 62, 63, 64, 65, 66, 67, 68, 69])
    phanpats = np.append(phanpats, [70, 71, 72, 73, 74, 75, 76, 77, 78, 79])
    phanpats = np.append(phanpats, [80, 81])
    testpat = [76]

    valid = [52,56,58,59,65,68,71,73,78]
    #cbctvals, ratiovals = plot_cax_data(path,phantom)
    cbctvals, ratiovals,pdosvals,rtvals,areavals = plot_cax_data(path,pats)
    cbctphan, ratiophan, pdosphan, rtphan, areaphan = plot_cax_data(path, phanpats)
    #cbcttest, ratiotest, pdostest, rttest, areatest = plot_cax_data(path, testpat)


    plt.figure()
    #plt.plot(cbctvals,ratiovals,'bo')
    #plt.plot(cbctvals1, ratiovals1, 'ro')
    plt.plot(areavals, cbctvals, 'ro')
    plt.plot(areaphan, cbctphan, 'bo')
    #plt.ylabel("rt/pdos CAX")
    plt.xlabel("area ")

    #ax = plt.axes(projection='3d')
    #ax.scatter3D(cbcttest, areatest, ratiotest, c='b' )

    #xp = np.arange(0, 1.0, 0.01)
    #y = 1.0*np.exp(-0.015*xp)
    #y = 1.0*np.exp(-4.0*xp) + 0.15 * xp -0.1*xp*xp
    #y = 1-4*xp+8*xp*xp-10.67*xp*xp*xp+10.67*np.power(xp,4)-8.533*np.power(xp,5)
    #y = 1.69-0.31*xp
    #plt.plot(xp,y)

    plt.show()



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
