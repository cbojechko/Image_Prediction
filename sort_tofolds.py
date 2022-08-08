import os
import shutil
from glob import glob


def sorttofolds(path):
    fold1 = []
    fold2 = []
    fold3 = []
    fold4 = []
    fold5 = []


    fold1_idx = [0, 6, 7, 8, 10, 11, 12, 13, 15, 16]
    fold2_idx = [1, 9, 14, 17, 20, 21, 22,23,24,25]
    fold3_idx = [2, 18, 26, 28, 31, 32, 33, 35 ,44] # Remove 27, artifact present
    fold4_idx = [3, 19, 29, 36, 37, 38, 39, 40, 45, 48]
    fold5_idx = [4, 5, 30, 34, 41, 42, 43, 46, 47, 49] # remove 50? poor image quality 47 is feet first knee
    for idx1 in fold1_idx:
        fold1.extend(glob(str(path) + '\\' + str(idx1) + '_*.tfrecord'))
        fold1.extend(glob(str(path) + '\\' + str(idx1) + '_*.pkl'))
    for idx2 in fold2_idx:
        fold2.extend(glob(str(path) + '\\' + str(idx2) + '_*.tfrecord'))
        fold2.extend(glob(str(path) + '\\' + str(idx2) + '_*.pkl'))
    for idx3 in fold3_idx:
        fold3.extend(glob(str(path) + '\\' + str(idx3) + '_*.tfrecord'))
        fold3.extend(glob(str(path) + '\\' + str(idx3) + '_*.pkl'))
    for idx4 in fold4_idx:
        fold4.extend(glob(str(path) + '\\' + str(idx4) + '_*.tfrecord'))
        fold4.extend(glob(str(path) + '\\' + str(idx4) + '_*.pkl'))
    for idx5 in fold5_idx:
        fold5.extend(glob(str(path) + '\\' + str(idx5) + '_*.tfrecord'))
        fold5.extend(glob(str(path) + '\\' + str(idx5) + '_*.pkl'))

    for file1 in fold1:
        dest1 = path + "\\fold1\\" + os.path.basename(file1)
        print(file1)
        shutil.move(file1, dest1)
    for file2 in fold2:
        dest2 = path + "\\fold2\\" + os.path.basename(file2)
        print(file2)
        shutil.move(file2, dest2)
    for file3 in fold3:
        dest3 = path + "\\fold3\\" + os.path.basename(file3)
        print(file3)
        shutil.move(file3, dest3)
    for file4 in fold4:
        dest4 = path + "\\fold4\\" + os.path.basename(file4)
        print(file4)
        shutil.move(file4, dest4)

    for file5 in fold5:
        dest5 = path + "\\fold5\\" + os.path.basename(file5)
        print(file5)
        shutil.move(file5, dest5)


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
    path = 'R:\Bojechko\TFRecords\TrainNoNormalizationMultipleProj'
    sorttofolds(path)
    sorttofoldsphan(path)


if __name__ == '__main__':
    pass

