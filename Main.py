

"""
This is a general writeup of the steps needed to create all of the patient data.

The steps are currently arbitrary and are likely to move later as the flow is understood
"""

import os
from PreProcessingTools.Main import create_inputs, tqdm
"""
First, for preprocessing, create the padded CBCTs by registering them with the primary CT and padding
Second, create the fluence and PDOS images from DICOM handles
Third, create the DRR and half-CBCT DRR for each beam angle
Fourth, align the PDOS and fluence with the DRRs
"""
basepath = os.path.join('.', 'Data', 'Patient')
path = r'R:\Bojechko'
rewrite = False
for patient_data in ['PatientData2']:
    base_patient_path = os.path.join(path, patient_data)
    MRN_list = os.listdir(base_patient_path)
    pbar = tqdm(total=len(MRN_list), desc='Loading through patient files')
    for patient_MRN in MRN_list:
        print(patient_MRN)
        patient_path = os.path.join(base_patient_path, patient_MRN)
        create_inputs(patient_path, rewrite)
        pbar.update()
"""
First, create the PDOS and reduced resolution EPID images
"""
if False:
    from PreProcessingTools.CreatePDOSAndRIImages import CreatePDOS_and_RI_Images
    # Factor with which to downsample EPID images are 1280x1280
    Ndownsample = 5
    CreatePDOS_and_RI_Images(basepath, Ndownsample)

"""
Next, create the projection from CBCT
"""
RTIpath = os.path.join(basepath, 'RTIMAGE')
CBCTpath = os.path.join(basepath, 'CT')
if True:
    from cbctprojections import MakeCBCTProjection
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
if False:
    from PreProcessingTools.CreateTFRecords import create_tf_records
    create_tf_records(data_path)