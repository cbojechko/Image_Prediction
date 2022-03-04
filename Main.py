

"""
This is a general writeup of the steps needed to create all of the patient data.

The steps are currently arbitrary and are likely to move later as the flow is understood
"""

import os
create_patient_inputs = True
if create_patient_inputs:
    from PreProcessingTools.Main import create_inputs, tqdm
    """
    First, for preprocessing, create the padded CBCTs by registering them with the primary CT and padding
    Second, create the fluence and PDOS images from DICOM handles
    Third, create the DRR and half-CBCT DRR for each beam angle
    Fourth, align the PDOS and fluence with the DRRs
    """
    path = r'\\ad.ucsd.edu\ahs\radon\research\Bojechko'
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
Lets create some .tfrecords from data already made
"""
data_path = r'\\ad.ucsd.edu\ahs\radon\research\Bojechko'
if True:
    from PreProcessingTools.CreateTFRecords import create_tf_records
    create_tf_records(data_path)