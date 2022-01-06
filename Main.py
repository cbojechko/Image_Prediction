

"""
This is a general writeup of the steps needed to create all of the patient data.

The steps are currently arbitrary and are likely to move later as the flow is understood
"""

import os

basepath = os.path.join('.', 'Data', 'Patient')

"""
First, create the PDOS and reduced resolution EPID images
"""
if False:
    from PreProcessingTools.CreatePDOSAndRIImages import CreatePDOS_and_RI_Images
    # Factor with which to downsample EPID images are 1280x1280
    Ndownsample = 5
    CreatePDOS_and_RI_Images(basepath, Ndownsample)

"""
Next, create the CBCT
"""
RTIpath = os.path.join(basepath, 'RTIMAGE')
CBCTpath = os.path.join(basepath, 'CT')
if True:
    #from PreProcessingTools.CreateCBCT import create_CBCT
    #create_CBCT(CBCTpath)
    from cbctprojections import MakeCBCTProjection
    MakeCBCTProjection(RIpath=RTIpath, CBCTpath=CBCTpath)