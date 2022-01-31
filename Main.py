

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
if False:
    #from PreProcessingTools.CreateCBCT import create_CBCT
    #create_CBCT(CBCTpath)
    from cbcthalfprojections import MakeCBCTProjection
    MakeCBCTProjection(RIpath=RTIpath, CBCTpath=CBCTpath)
"""
Lets look at the data real quick
"""
from PreProcessingTools.EvaluatingData import evaluate_data
evaluate_data()
"""
Lets create some .tfrecords from data already made
"""
data_path = r'\\ad.ucsd.edu\ahs\radon\research\Bojechko'
if True:
    from PreProcessingTools.CreateTFRecords import create_tf_records
    create_tf_records(data_path)