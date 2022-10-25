import os
import shutil
from glob import glob
import pandas as pd


def sorttofolds(path):


    excel_path = r"R:\patientlist_081722.xlsx"
    data= pd.read_excel(excel_path,engine='openpyxl',sheet_name='Sheet1')

    df = pd.DataFrame(data,columns=['Index','MRN','Fx','NF','NIMG','site','Date CBCT'])
    #print(df)

    for idx,rows in df.iterrows():
        #print(rows['Index'],rows['Date CBCT'])

        print("idx " + str(rows['Index']))
        tfpath = str(path) + '\\' + str(rows['Index']) + '_*' + str(rows['Date CBCT']) + '*.tfrecord'

        pklpath = str(path) + '\\' + str(rows['Index']) + '_*' + str(rows['Date CBCT']) + '*.pkl'

        tffiles = glob(tfpath)
        pklfiles = glob(pklpath)


        for tffile in tffiles:
            #print(tffile)

            if int(rows['Index']) <=30:

                dest1 = path + "\\fold1\\" + os.path.basename(tffile)
                #print(tffile)
                print(dest1)
                shutil.move(tffile, dest1)

            elif int(rows['Index']) > 30 and int(rows['Index']) <= 56:

                dest2 = path + "\\fold2\\" + os.path.basename(tffile)
                print(dest2)
                shutil.move(tffile, dest2)

            elif int(rows['Index']) > 56 and int(rows['Index']) <= 84:

                dest3 = path + "\\fold3\\" + os.path.basename(tffile)
                print(dest3)
                shutil.move(tffile, dest3)

            elif int(rows['Index']) > 84 and int(rows['Index']) <= 114:

                dest4 = path + "\\fold4\\" + os.path.basename(tffile)
                print(dest4)
                shutil.move(tffile, dest4)

            elif int(rows['Index']) > 114 and int(rows['Index']) <= 139:

                dest5 = path + "\\fold5\\" + os.path.basename(tffile)
                print(dest5)
                shutil.move(tffile, dest5)

        for pklfile in pklfiles:

            if int(rows['Index']) <=30:

                dest1 = path + "\\fold1\\" + os.path.basename(pklfile)
                #print(tffile)
                print(dest1)
                shutil.move(pklfile, dest1)

            elif int(rows['Index']) > 30 and int(rows['Index']) <= 56:

                dest2 = path + "\\fold2\\" + os.path.basename(pklfile)
                print(dest2)
                shutil.move(pklfile, dest2)

            elif int(rows['Index']) > 56 and int(rows['Index']) <= 84:

                dest3 = path + "\\fold3\\" + os.path.basename(pklfile)
                print(dest3)
                shutil.move(pklfile, dest3)

            elif int(rows['Index']) > 84 and int(rows['Index']) <= 114:

                dest4 = path + "\\fold4\\" + os.path.basename(pklfile)
                print(dest4)
                shutil.move(pklfile, dest4)

            elif int(rows['Index']) > 114 and int(rows['Index']) <= 139:

                dest5 = path + "\\fold5\\" + os.path.basename(pklfile)
                print(dest5)
                shutil.move(pklfile, dest5)

def sortjpegs(path):

        excel_path = r"R:\patientlist_081722.xlsx"
        data = pd.read_excel(excel_path, engine='openpyxl', sheet_name='Sheet1')

        df = pd.DataFrame(data, columns=['Index', 'MRN', 'Fx', 'NF', 'NIMG', 'site', 'Date CBCT'])
        # print(df)

        for idx, rows in df.iterrows():
            # print(rows['Index'],rows['Date CBCT'])

            print("idx " + str(rows['Index']))
            jpegpath = str(path) + '\\' + str(rows['Index']) + '_*' + str(rows['Date CBCT']) + '*.jpeg'
            print(jpegpath)
            jpegfiles = glob(jpegpath)


            for jpegfile in jpegfiles:
                #print(tffile)

                if int(rows['Index']) <= 30:

                    dest1 = path + "\\fold1\\" + os.path.basename(jpegfile)
                    # print(tffile)
                    print(dest1)
                    shutil.move(jpegfile, dest1)

                elif int(rows['Index']) > 30 and int(rows['Index']) <= 56:

                    dest2 = path + "\\fold2\\" + os.path.basename(jpegfile)
                    print(dest2)
                    shutil.move(jpegfile, dest2)

                elif int(rows['Index']) > 56 and int(rows['Index']) <= 84:

                    dest3 = path + "\\fold3\\" + os.path.basename(jpegfile)
                    print(dest3)
                    shutil.move(jpegfile, dest3)

                elif int(rows['Index']) > 84 and int(rows['Index']) <= 114:

                    dest4 = path + "\\fold4\\" + os.path.basename(jpegfile)
                    print(dest4)
                    shutil.move(jpegfile, dest4)

                elif int(rows['Index']) > 114 and int(rows['Index']) <= 139:

                    dest5 = path + "\\fold5\\" + os.path.basename(jpegfile)
                    print(dest5)
                    shutil.move(jpegfile, dest5)



def sorttofoldsphan(path):
    valid = []
    train = []
    valid_idx = [52, 56, 58, 59, 65, 68, 71, 73, 78]
    train_idx = [50, 51, 53, 54, 55, 57, 60, 61, 62, 63, 64, 66, 67, 69, 70, 72, 74, 75, 76, 77, 79, 80, 81 ]
   
    for idx1 in valid_idx:
        valid.extend(glob(str(path) + '\\' + str(idx1) + '_*.tfrecord'))
        valid.extend(glob(str(path) + '\\' + str(idx1) + '_*.pkl'))
   
    for file1 in valid:
        dest1 = path + "\\phantom_valid\\" + os.path.basename(file1)
        print(file1)
        shutil.move(file1, dest1)

    for idx2 in train_idx:
        train.extend(glob(str(path) + '\\' + str(idx2) + '_*.tfrecord'))
        train.extend(glob(str(path) + '\\' + str(idx2) + '_*.pkl'))

    for file2 in train:
        dest2 = path + "\\phantom_train\\" + os.path.basename(file2)
        print(file2)
        shutil.move(file2, dest2)



def main():

    #path = r'R:\TFRecords\TrainNoNormalizationMultipleProj'
    #sorttofolds(path)

    path = r'R:\TFRecords\JpegsNoNormalizationMultipleProj'
    sortjpegs(path)

    #sorttofoldsphan(path)

if __name__ == '__main__':
    main()
