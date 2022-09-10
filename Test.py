import os

path = r'K:\10yr_patients_full.txt'
fid = open(path)
MRNs = []
for line in fid:
    x = 5
    MRN = line.split('\t')[0]
    if MRN != "":
        if MRN not in MRNs:
            MRNs.append(MRN)
fid.close()
fid = open(os.path.join('K:', '10yr_patientsMRNs.txt'), 'w+')
for line in MRNs:
    fid.write(line + '\n')
fid.close()