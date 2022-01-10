import os
import numpy as np
from DicomRTTool.ReaderWriter import DicomReaderWriter


def create_CBCT(CTPath):
    npfileout = "cbct"
    arrout = os.path.join(CTPath, npfileout)
    if os.path.exists(arrout):
        return None
    Dicom_reader = DicomReaderWriter(description='Examples',verbose=True)
    print('Read CBCT Dicom Files ......')
    Dicom_reader.walk_through_folders(CTPath) # need to define in order to use all_roi method

    Dicom_reader.set_index(0)  # This index has all the structures, corresponds to pre-RT T1-w image for patient 011
    Dicom_reader.get_images()

    dicom_sitk_handle = Dicom_reader.dicom_handle

    origin = dicom_sitk_handle.GetOrigin()
    print("Orgin " + str(origin))

    voxDim = dicom_sitk_handle.GetSpacing()
    print("voxDim " + str(voxDim))

    voxSize = dicom_sitk_handle.GetSize()
    print("VoxSize " + str(voxSize))

    voxDim = np.asarray(voxDim)
    voxSize = np.asarray(voxSize)
    origin = np.asarray(origin)

    image = Dicom_reader.ArrayDicom

    cbctnp = np.array(image)
    print("Saving CBCT vector "+ str(arrout))
    np.savez_compressed(arrout, cbct=cbctnp, origin=origin, voxDim=voxDim, voxSize=voxSize)
    return None


if __name__ == '__main__':
    pass
