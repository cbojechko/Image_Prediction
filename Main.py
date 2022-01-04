

"""
This is a general writeup of the steps needed to create all of the patient data.

The steps are currently arbitrary and are likely to move later as the flow is understood
"""

import os

basepath = os.path.join('.', 'Data', 'Patient')

if True:
    from PreProcessingTools.CreatePDOSAndRIImages import CreatePDOS_and_RI_Images
    # Factor with which to downsample EPID images are 1280x1280
    Ndownsample = 5
    CreatePDOS_and_RI_Images(basepath, Ndownsample)

RTIpath = os.path.join(basepath,'RTIMAGE')
CBCTpath = os.path.join(basepath,'CT')
if False:
    from cbctprojections import MakeCBCTProjection
    MakeCBCTProjection(RIpath=RTIpath, CBCTpath=CBCTpath)