#%%
# open up a EPID image and save it as a 1 dim vector containing pixel information. 
# file mangagment 
import os 
import numpy as np
import pydicom

import SimpleITK as sitk


#open a RI image
RIpath = os.path.join('P:\Image_Prediction','11657988','frac1epid')
Patpath = os.path.join('P:\Image_Prediction','11657988')

for entry in os.listdir(RIpath):
    if os.path.isfile(os.path.join(RIpath, entry)):
        RIfile = os.path.join(RIpath, entry)
        print(entry)


        # read in the dicom info 
        ds = pydicom.read_file(RIfile)
        colang = ds.BeamLimitingDeviceAngle
        print("Colimator angle " + str(colang))
        rows = ds.Rows
        cols = ds.Columns
        pixspace = ds.ImagePlanePixelSpacing
        image = ds.pixel_array
        rescale = ds.RescaleSlope
        #multiply by the rescaling 
        imagescale = np.multiply(image,rescale)
        # reshape the 2d image into a 1 dim vector 
        
        imagevector = imagescale.reshape(1,rows*cols)

        npfileout = "imagevector_col" + str(int(np.round(colang))) 
        imageout = os.path.join(Patpath, npfileout)
        print("Saving Image vector "+ str(imageout))
        np.savez_compressed(imageout,imagevector)



