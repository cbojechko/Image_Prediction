# importing neccessary libraries 

# file mangagment 
import os 
import numpy as np
import pydicom

import SimpleITK as sitk

# Get the isocenter from a RP file.  The isocenter is in the patient-based coordiante system which is a right
# handed system. 
def isofromRS(RSfile):

    ds = pydicom.read_file(RSfile)
    isocen = ds.BeamSequence[0].ControlPointSequence[0].IsocenterPosition
    return isocen

# Give the number of beams in a plan and get back the number of control points for each beam. 
def getNctrlpt(dicom,nbeams):
    ctrlpts = np.array(np.zeros((nbeams),dtype=int))
    for i in range(0,nbeams):
        ctrlpts[i] = ds.BeamSequence[i].NumberOfControlPoints
   
    return ctrlpts

# Create a long 1 dimensional Array that has 
# Collimator angle, [gantry angle_i, meter set weight_i, bankA MLCX1_i, bank B MLCX2_i ]
# Dimension is (56+58+2)*(nctrlpts-1)+1  
# Save as a np array file 
# MLCX1 is size 56 
# MLCX2 is size 58 
# When looping through skip control point 0, does not have MLC information. 
def CreateArcVector(dicom,nctrlpt,idx):
    arclist = []
    # Loop over the beams to get the properies of each. 
    if(nctrlpt < 50):
        print("Not a Treatment beam")
        return
    else:
        print("Treatment beam with " + str(nctrlpt) + " control points" )
        colang = ds.BeamSequence[idx].ControlPointSequence[0].BeamLimitingDeviceAngle
        print("Collimator Angle " + str(colang))
        arclist.append(colang)
        gangarr = np.array(np.zeros(nctrlpt-1))
        ## Create arrays for MLC leaf positions, 56 leaves on each bank  
        bankA = np.array((np.zeros(56)))
        bankB = np.array((np.zeros(56)))

    #Skip control point 0 
        for j in range(1,nctrlpt):
            gang = ds.BeamSequence[idx].ControlPointSequence[j].GantryAngle
            cmsw = ds.BeamSequence[idx].ControlPointSequence[j].CumulativeMetersetWeight
            arclist.append(gang)
            arclist.append(cmsw)
            #print ("Gantry angle " + str(gang) + " Meter Set Weight " + str(cmsw))
            bankA = ds.BeamSequence[idx].ControlPointSequence[j].BeamLimitingDevicePositionSequence[0].LeafJawPositions
            bankB = ds.BeamSequence[idx].ControlPointSequence[j].BeamLimitingDevicePositionSequence[1].LeafJawPositions
            arclist.extend(bankA)
            arclist.extend(bankB)


    print(len(arclist))

    arcvector = np.array(arclist)
    npfileout = "arcvector" + str(idx) + ".npy"
    arrout = os.path.join(RPpath, npfileout)
    print("Saving Arc vector "+ str(arrout))
    np.save(arrout,arcvector)
    return

#################################################################################################################
#open a RP plan
RPpath = os.path.join('P:\Image_Prediction','11567988','plan')

for entry in os.listdir(RPpath):
    if os.path.isfile(os.path.join(RPpath, entry)):
        RSfile = os.path.join(RPpath, entry)
        print(entry)

# Get the Isocenter
isocen = isofromRS(RSfile)
print(isocen)

# read in the dicom info 
ds = pydicom.read_file(RSfile)

# Find the length of the beam sequence, number of beams. 
nbeams = len(ds.BeamSequence)
nctrl = getNctrlpt(ds,nbeams)

# Loop over the beams and create numpy vectors for later processing. 
for i in range(0,nbeams):
    CreateArcVector(ds,nctrl[i],i)

