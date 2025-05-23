

"""
This is a general writeup of the steps needed to create all of the patient data.

The steps are currently arbitrary and are likely to move later as the flow is understood
"""

import os
rewrite = True
data_path = r'\\ad.ucsd.edu\ahs\radon\research\Bojechko'
logs_file = os.path.join('.', 'errors_log.txt')
if not os.path.exists(logs_file):
    fid = open(logs_file, 'w+')
    fid.close()
if True:
    from tqdm import tqdm
    from PreProcessingTools.Main import create_inputs
    """
    First, for preprocessing, create the padded CBCTs by registering them with the primary CT and padding
    Second, create the fluence and PDOS images from DICOM handles
    Third, create the DRR and half-CBCT DRR for each beam angle
    Fourth, align the PDOS and fluence with the DRRs
    """
    for patient_data in ['phantom']: #'PatientData2',
        base_patient_path = os.path.join(data_path, patient_data)
        MRN_list = os.listdir(base_patient_path)
        # fid = open(os.path.join('.', 'PreProcessingTools', 'MRN.txt'))
        # MRN_list = fid.readlines()
        # fid.close()
        pbar = tqdm(total=len(MRN_list), desc='Loading through patient files')
        perform_on_primary_CT = True
        for patient_MRN in MRN_list:
            patient_MRN = patient_MRN.strip('\n')
            print(patient_MRN)
            patient_path = os.path.join(base_patient_path, patient_MRN)
            try:
                create_inputs(patient_path, rewrite, perform_on_primary_CT)
            except:
                fid = open(logs_file, 'a')
                fid.write("Error for {}\n".format(patient_path))
                fid.close()
            pbar.update()
"""
Lets create some .tfrecords from data already made
"""
if True:
    from PreProcessingTools.CreateTFRecords import create_tf_records
    create_tf_records(data_path, rewrite=True)
    from sort_tofolds import main
    # main()
    from DeepLearningTools.Utilities import main
    main()