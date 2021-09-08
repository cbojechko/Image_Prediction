
#%%
# importing neccessary libraries 
# file mangagment 
import glob
import os 
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import re 
from scipy import interpolate

from scipy import ndimage

#open a RI image
#FLUpath = os.path.join('P:\Image_Prediction','04455192','Fluence')
#FLUpath = os.path.join('P:\Image_Prediction','SingleFx')

def FluNP(Flupath):

    imsize = 256
    #Search for a numpy file 
    FLUfiles = glob.glob(str(Flupath) + '\*.optimal_fluence')

    #Flupath = os.path.join('P:\Image_Prediction','04455192','Fluence')

    for file in FLUfiles:


        f = open(file,'r')
        line = f.readlines()
        # Parse file and get the values of center, number of pixels and origin 
        dd = re.split('\n',re.split('\t',line[2])[1])  # spacing x is line 2
        sizex = int(dd[0])

        dd = re.split('\n',re.split('\t',line[3])[1])  # spacing x is line 3
        sizey = int(dd[0])

        cc = re.split('\n',re.split('\t',line[4])[1])  # spacing x is line 4
        spacex = float(cc[0])

        cc = re.split('\n',re.split('\t',line[5])[1]) # spacing x is line 5
        spacey = float(cc[0])

        oo = re.split('\n',re.split('\t',line[6])[1]) # origin x is line 6
        originx = float(oo[0])

        oo = re.split('\n',re.split('\t',line[7])[1]) # origin x is line 7
        originy = float(oo[0])
        #central pixel in x and y 
        cenpixx = int(np.ceil(np.abs(originx/spacex)))
        cenpixy = int(np.ceil(np.abs(originy/spacey)))
        
        # Format of fluence file is consistent hard code the line numbers
        fluarr = np.fromstring(line[9],dtype=float,sep ='\t')

        tt = re.search("\d+\.",file)
        gang = int(tt[0][0:-1])
        print(file)
        print("Fluence GANG " + str(tt[0][0:-1]))
        for i in range(10,len(line)-1):
            newline = np.fromstring(line[i],dtype=float,sep ='\t')
            fluarr = np.vstack([fluarr,newline])

        #pad the image so that it is centered around iso(center of image)
        xborderlow = np.abs(sizex-cenpixx)
        xborderup = np.abs(cenpixx)
        yborderlow = np.abs(sizey-cenpixy)
        yborderup = np.abs(cenpixy)
        flupad = np.pad(fluarr,((yborderlow,yborderup),(xborderlow,xborderup)),'constant',constant_values=(0,0))

        #Linear interpolation to upsample the fluence
        x = np.array(range(flupad.shape[1]))
        y = np.array(range(flupad.shape[0]))
        xx, yy = np.meshgrid(x, y)
        f = interpolate.interp2d(x, y, flupad, kind='linear')

        xnew = np.linspace(0, flupad.shape[1], 2*flupad.shape[1])
        ynew = np.linspace(0, flupad.shape[0], 2*flupad.shape[0])
        flunew = f(xnew, ynew)

        os.path.dirname(Flupath)
        RIpath = os.path.join(os.path.dirname(Flupath),'RTIMAGE')
        #Search for a dicom files
        print(RIpath)
        RIfiles = glob.glob(str(RIpath) + '\*.dcm')
        # Look for a RT dicom file with a matching gantry angle and pull the collimator angle. 
        for file in RIfiles:
            ds = pydicom.read_file(file)
            colang = int(np.rint(ds.BeamLimitingDeviceAngle))
            rigang = int(np.rint(ds.GantryAngle))
            if(rigang == 360):
                rigang = 0

            print("Fluence gantry " + str(gang) + " RI gantry " + str(rigang))
            if(rigang == gang):
                print("Gantry Angles match Col ang " + str(colang) )
                break

        print('Collimator Angle ' + str(colang))
        #Rotate by the collimator angle
        if(colang == 0):
            flurot =flunew
        else:
            flurot = ndimage.rotate(flunew, colang, reshape=True,order=1)
        
        print("Size of rotated image " + str(flurot.shape))
        #Pad the array to get 256 pixels. Use the edge which will be zeros otherwise crop 
        xcrop = (flurot.shape[0]-imsize)/2
        ycrop = (flurot.shape[1]-imsize)/2

        xcroplow = int(np.floor(xcrop))
        xcropup = int(np.ceil(xcrop))
        
        ycroplow = int(np.floor(ycrop))
        ycropup = int(np.ceil(ycrop))

        print("xcrop " + str(xcrop) + " ycrop " + str(ycrop))
        if(xcrop > 0 and ycrop > 0):
            flucrop = flurot[xcroplow:flurot.shape[0]-xcropup,ycroplow:flurot.shape[1]-ycropup]
        elif(xcrop > 0 and ycrop < 0):
            flucrop = flurot[xcroplow:flurot.shape[0]-xcropup,:]
            flucrop = np.pad(flucrop,((0,0),(np.abs(ycroplow),np.abs(ycropup))),'constant',constant_values=(0,0))
        elif(xcrop < 0 and ycrop > 0):
            flucrop = flurot[:,ycroplow:flurot.shape[1]-ycropup]
            flucrop = np.pad(flucrop,((np.abs(xcroplow),np.abs(xcropup)),(0,0)),'constant',constant_values=(0,0))
        else:
            flucrop = np.pad(flurot,((np.abs(xcroplow),np.abs(xcropup)),(np.abs(ycroplow),np.abs(ycropup))),'constant',constant_values=(0,0))
       
        print("Size of image " + str(flucrop.shape))
        #Save as numpy array
        npfileout = "Fluence" + str(gang)
        arrout = os.path.join(Flupath, npfileout)
        print("Saving RT Image "+ str(arrout))
        np.savez_compressed(arrout,flucrop)
        
        

#plt.imshow(flupad)
#plt.show()


# Main loop 
Basepath = 'P:\Image_Prediction\PatientList'
MRNs = os.listdir(Basepath)

for i in range(0,len(MRNs)):
    Flupath = os.path.join(Basepath,MRNs[i],'Fluence')
    print(Flupath)
    FluNP(Flupath)


