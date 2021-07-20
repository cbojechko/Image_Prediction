# script to test the properties of CBCT and RT Dose.  
# Need to get arrays so they have the same orgin and voxel size

# importing neccessary libraries 
# file mangagment 

#%%
import os 
import numpy as np
import pydicom
import matplotlib.pyplot as plt

Flupath = os.path.join('P:\Image_Prediction','04455192','Fluence')


for entry in os.listdir(Flupath):
    if os.path.isfile(os.path.join(Flupath, entry)):
        Flufile = os.path.join(Flupath, entry)
        print(entry)


f = open(Flufile,'r')
line = f.readlines()
# Format of fluence file is consistent hard code the line numbers
fluarr = np.fromstring(line[9],dtype=float,sep ='\t')

for i in range(10,len(line)-1):
     newline = np.fromstring(line[i],dtype=float,sep ='\t')
     fluarr = np.vstack([fluarr,newline])


#Number of pixels in fluence is 120x120 is this consistent? 
#Resample factor 
n = 1
downsample = fluarr.reshape(120//n,n,120//n,n).mean(-1).mean(1)

plt.imshow(downsample)
plt.show()



# %%
