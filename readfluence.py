# script to test the properties of CBCT and RT Dose.  
# Need to get arrays so they have the same orgin and voxel size

# importing neccessary libraries 
# file mangagment 
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

print(line[10])