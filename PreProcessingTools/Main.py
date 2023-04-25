import copy
import typing
import SimpleITK
from matplotlib import pyplot as plt
from glob import glob
from PreProcessingTools.itk_sitk_converter import *
import SimpleITK as sitk
import os
from NiftiResampler import ResampleTools
from PreProcessingTools.Pad_CBCTs_From_Digital_Phantom import update_CBCT
from PreProcessingTools.UpdatePrimaryCT import update_primary_CT
from PreProcessingTools.Tools import *
from PreProcessingTools.RegisteringImages.src.RegisterImages.WithDicomReg import registerDicom
import itk
import numpy as np
from PreProcessingTools.Dicom_RT_and_Images_to_Mask.src.DicomRTTool.ReaderWriter import DicomReaderWriter, plot_scroll_Image, pydicom

logs_file = os.path.join('.', 'errors_log.txt')
if not os.path.exists(logs_file):
    logs_fid = open(logs_file, 'w+')
    logs_fid.close()


def rotate_and_translate_image(itk_image, translations=(0, 0, 0), rotations=(0, 0, 0)):
    """
    :param itk_image:
    :param translations: translations in mm
    :param rotations: in degrees
    :return:
    """
    origin = itk_image.GetOrigin()
    imSize = itk_image.GetLargestPossibleRegion().GetSize()
    imRes = itk_image.GetSpacing()
    imDirection = itk_image.GetDirection()
    isocenter = [0, 0, 0]
    transform = itk.Euler3DTransform[itk.D].New()
    transform.SetComputeZYX(True)
    transform.SetCenter(isocenter)
    transform.SetTranslation(translations)
    transform.SetRotation(np.deg2rad(rotations[0]), np.deg2rad(rotations[1]), np.deg2rad(rotations[2]))

    interpolator = itk.LinearInterpolateImageFunction[itk.Image[itk.F, 3], itk.D].New()

    filter = itk.ResampleImageFilter[itk.Image[itk.F, 3], itk.Image[itk.F, 3]].New()
    filter.SetInterpolator(interpolator)
    filter.SetDefaultPixelValue(-1000)
    filter.SetOutputOrigin(origin)
    filter.SetOutputSpacing(imRes)
    filter.SetOutputDirection(imDirection)
    filter.SetSize(imSize)
    filter.SetTransform(transform)

    filter.SetInput(itk_image)
    filter.Update()
    output = filter.GetOutput()
    return output


def write_itk_file(output_path, itk_file):
    InputImageType = itk.Image[itk.F, 3]
    writer = itk.ImageFileWriter[InputImageType].New()
    writer.SetFileName(output_path)
    writer.SetInput(itk_file)
    writer.Update()
    return None


def create_drr(sitk_handle, sid=1000, spd=1540, gantry_angle=0, out_path=os.path.join('.', 'Output.mha'),
               translations=(0, 0, 0), distance_from_iso=None):
    """
    :param sitk_handle: handle from SimpleITK, usually from DICOMRTTool
    :param sid: source to iso-center distance, in mm
    :param spd: source to panel distance, in mm
    :param gantry_angle: angle of gantry
    :param out_path: name of out file
    :param translations: translations in x, y, z direction
    :return:
    """
    image = ConvertSimpleItkImageToItkImage(sitk_handle, itk.F)
    rotations = [0, 0, gantry_angle]  # x, y, z

    rprojection = 0.  # Projection angle in degrees
    threshold = -1000
    imRes = image.GetSpacing()

    im_sx = imRes[0]
    im_sy = imRes[1]

    imSize = image.GetLargestPossibleRegion().GetSize()
    dx = imSize[0]
    dy = imSize[1]

    Dimension = 3

    InputImageType = itk.Image[itk.F, Dimension]
    spacing = image.GetSpacing()

    final_filter = itk.ResampleImageFilter[InputImageType, InputImageType].New()
    final_filter.SetDefaultPixelValue(0)
    """
    Translate our image to put the iso-center at the center of the plan, then put the plan isocenter in the 
    absolute center of the image (just makes everything easier...
    """
    image_center = [imRes[i] * imSize[i]/2 for i in range(3)]
    shifts = [-(image.GetOrigin()[i] + image_center[i] - translations[i]) for i in range(3)]
    transformed_image = rotate_and_translate_image(image, translations=shifts, rotations=rotations)
    """
    To make things easier, shift the image so the plan iso-center is the index center, then put origin at 0, 0, 0
    """
    input_origin = [0, 0, 0]
    transformed_image.SetOrigin(input_origin)
    """
    If a half projection, just delete everything beneath the half-way point, as we're centered on iso
    """
    if distance_from_iso is not None:
        new_spot = sitk_handle.TransformPhysicalPointToIndex((0, distance_from_iso, 0))
        transformed_image[:, int(new_spot[1]):, :] = -1000
    output_origin = [transformed_image.GetOrigin()[i] for i in range(3)]
    output_origin[2] = -spd
    final_filter.SetInput(transformed_image)

    transform = itk.Euler3DTransform[itk.D].New()
    transform.SetComputeZYX(True)

    transform.SetTranslation((0, 0, 0)) # Do not change these!
    transform.SetRotation(0, 0, 0) # Do not change these!
    isocenter = [0, 0, 0]
    for i in range(3):
        isocenter[i] = input_origin[i] + imRes[i] * imSize[i] / 2 # Center of projection is the center of the image
    transform.SetCenter(isocenter)

    interpolator = itk.SiddonJacobsRayCastInterpolateImageFunction[InputImageType, itk.D].New()
    interpolator.SetProjectionAngle(np.deg2rad(rprojection))
    interpolator.SetFocalPointToIsocenterDistance(sid)
    interpolator.SetThreshold(threshold)
    interpolator.SetTransform(transform)
    interpolator.Initialize()
    final_filter.SetInterpolator(interpolator)

    final_filter.SetSize([dx, dy, 1])
    final_filter.SetOutputSpacing([spacing[0], spacing[1], 1.0])

    o2Dx = (dx - 1) / 2
    o2Dy = (dy - 1) / 2
    origin = [0, 0, 0]
    origin[0] = - im_sx * o2Dx
    origin[1] = - im_sy * o2Dy
    origin[2] = -spd

    final_filter.SetOutputOrigin(origin)
    final_filter.SetOutputDirection(image.GetDirection())
    final_filter.Update()
    filter_output = final_filter.GetOutput()
    flipFilter = itk.FlipImageFilter[InputImageType].New()
    flipFilter.SetInput(filter_output)
    flipFilter.SetFlipAxes((False, True, False))
    output = flipFilter.GetOutput()
    write_itk_file(output_path=out_path, itk_file=output)
    return None


def fix_DRR(cbct_drr: sitk.Image, ct_drr: sitk.Image):
    slice_thickness = cbct_drr.GetSpacing()[-1]
    cbct_array = sitk.GetArrayFromImage(cbct_drr)
    ct_array = sitk.GetArrayFromImage(ct_drr)
    y = np.sum(cbct_array.astype('bool')[0], axis=-1) # Flatten the array into a line, looking for the spot to move
    values = np.where(y > 0)[0]
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    dy = np.diff(y, 1)
    dx = np.ones(y.shape[0] - 1)
    yfirst = np.abs(dy / dx)
    yfirst[yfirst < 1] = 0
    yfirst_convolved = np.convolve(yfirst, kernel, mode='same')
    start_vals = np.where((yfirst_convolved[:-1] > .75) & (yfirst_convolved[1:] < .75))[0]
    start = int(start_vals[0] + 20 / slice_thickness)
    stop_vals = np.where((yfirst_convolved[:-1] < .75) & (yfirst_convolved[1:] > .75))[0]
    stop = int(stop_vals[-1] - 20 / slice_thickness)
    cbct_array[:, :start, :] = ct_array[:, :start, :]
    cbct_array[:, stop:, :] = ct_array[:, stop:, :]
    # plt.plot(axis_sum) # <-- can see the hump
    cbct_handle = array_to_sitk(cbct_array, reference_handle=ct_drr)
    return cbct_handle


def expandDRR(patient_path):
    if not os.path.exists(os.path.join(patient_path, 'Primary_CT_DRR.mha')):
        fid = open(logs_file, 'a')
        fid.write("No primary CT_DRR for {}\n".format(patient_path))
        fid.close()
        print("Could not find Primary_CT_DRR!")
        return None
    CBCT_DRR_Files = glob(os.path.join(patient_path, 'CBCT*DRR.mha'))
    CTDRRhandle = sitk.ReadImage(os.path.join(patient_path, 'Primary_CT_DRR.mha'))
    for cbct_drr_file in CBCT_DRR_Files:
        print("Writing padded CBCT for {}".format(cbct_drr_file))
        file_name = os.path.split(cbct_drr_file)[-1]
        CBCTDRRhandle = sitk.ReadImage(cbct_drr_file)
        cbct_handle = fix_DRR(cbct_drr=CBCTDRRhandle, ct_drr=CTDRRhandle)
        sitk.WriteImage(cbct_handle, os.path.join(patient_path, file_name.replace(".mha", "_Padded.mha")))
    return None


def get_outside_body_contour(annotation_handle, lowerThreshold, upperThreshold):
    connected_images = get_connected_image(annotation_handle, lowerThreshold, upperThreshold)
    outside_body = get_binary_image(connected_images, lowerThreshold=1, upperThreshold=1)
    for i in range(outside_body.GetSize()[-1]):
        connected_image = get_connected_image(outside_body[:, :, i], lowerThreshold=1, upperThreshold=1)
        binary_image = get_binary_image(connected_image, lowerThreshold=1, upperThreshold=1)
        outside_body[:, :, i] = binary_image
    return outside_body


def create_registered_cbct(patient_path, rewrite=False):
    out_folder = os.path.join(patient_path, "Niftiis")
    status_file = os.path.join(out_folder, "Finished_Reg_CBCT.txt")
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    if os.path.exists(status_file) and not rewrite:
        return None
    Dicom_reader = DicomReaderWriter(description='Examples', verbose=False)
    CT_SIUID = None
    CT_handle = None
    primary_CTSUID_path = os.path.join(out_folder, "PrimaryCTSIUD.txt")
    if not os.path.exists(primary_CTSUID_path):
        Dicom_reader.walk_through_folders(os.path.join(patient_path, 'pCT'))  # Read in the primary CT
        for index in Dicom_reader.indexes_with_contours:
            if Dicom_reader.series_instances_dictionary[index]['Description'] is not None:
                Dicom_reader.set_index(index)  # Primary CT
                Dicom_reader.get_images()
                CT_handle = Dicom_reader.dicom_handle
                sitk.WriteImage(CT_handle, os.path.join(out_folder, "Primary_CT.mha"))
                CT_SIUID = Dicom_reader.series_instances_dictionary[index]['SeriesInstanceUID']
                fid = open(primary_CTSUID_path, 'w+')
                fid.write(CT_SIUID)
                fid.close()
                break
    else:
        fid = open(primary_CTSUID_path)
        CT_SIUID = fid.readline()
        fid.close()
        CT_handle = sitk.ReadImage(os.path.join(out_folder, "Primary_CT.mha"))
    if CT_handle is None:
        fid = open(os.path.join(patient_path, "No_Primary_CT.txt"), 'w+')
        fid.close()
        return None
    Dicom_reader.walk_through_folders(os.path.join(patient_path, 'CT')) # Read in the CBCTs
    reg_path = os.path.join(patient_path, 'REG')
    if not os.path.exists(reg_path):
        fid = open(logs_file, 'a')
        fid.write("No registration folder for {}\n".format(patient_path))
        fid.close()
        print("{} does not exist! Export it".format(reg_path))
    date_time_dict = {}
    resampler = ResampleTools.ImageResampler()
    for file in os.listdir(reg_path):
        ds = pydicom.read_file(os.path.join(reg_path, file))
        for ref in ds.ReferencedSeriesSequence:
            from_uid = ref.SeriesInstanceUID
            if from_uid == CT_SIUID:
                continue
            for index in Dicom_reader.indexes_with_contours:
                if Dicom_reader.series_instances_dictionary[index]['SeriesInstanceUID'] == from_uid:
                    Dicom_reader.set_index(index)  # CBCT
                    date = Dicom_reader.return_key_info("0008|0022") #YYYYMMDD
                    time_stamp = Dicom_reader.return_key_info("0008|0032")
                    update_from_time = False
                    if date in date_time_dict:
                        time_previous = date_time_dict[date]
                        if time_stamp > time_previous:
                            update_from_time = True
                            print("Rewriting the file based on the time stamps...")
                    else:
                        date_time_dict[date] = time_stamp
                    out_reg_file = os.path.join(out_folder, "Registered_CBCT_{}.mha".format(date))
                    out_meta_file = os.path.join(out_folder, "Registered_Meta_{}.mha".format(date))
                    out_table_vert = os.path.join(out_folder, "TableHeight_CBCT_{}.txt".format(date))
                    if not os.path.exists(out_reg_file) or update_from_time or rewrite:
                        Dicom_reader.get_images()
                        cbct_handle = Dicom_reader.dicom_handle
                        sitk.WriteImage(cbct_handle, os.path.join(out_folder, "CBCT_{}.mha".format(date)))
                        couch_vert = float(Dicom_reader.reader.GetMetaData(0, "0018|1130"))
                        couch = (0, couch_vert, 0)
                        registered_handle, affine_transform = registerDicom(fixed_image=CT_handle,
                                                                            moving_image=cbct_handle,
                                                                            moving_series_instance_uid=from_uid,
                                                                            dicom_registration=ds, min_value=-1000,
                                                                            method=sitk.sitkLinear,
                                                                            return_affine=True)
                        registered_couch = affine_transform.GetInverse().TransformPoint(couch)
                        cbct_array = sitk.GetArrayFromImage(cbct_handle)
                        meta_array = np.zeros(cbct_array.shape)
                        Y, X = np.ogrid[:meta_array.shape[1], :meta_array.shape[2]]
                        dist_from_center = np.sqrt((X - meta_array.shape[1]//2) ** 2 +
                                                   (Y - meta_array.shape[2]//2) ** 2)
                        binary_images = cbct_array > -1000
                        for z in range(cbct_array.shape[0]):
                            binary_image = binary_images[z, ...]
                            total_max = np.sum(dist_from_center < meta_array.shape[1] * binary_image)  # This is the absolute max
                            upper_limit = meta_array.shape[1]
                            lower_limit = 0
                            current_guess_radii = (upper_limit - lower_limit) // 2 + lower_limit
                            previous_guess_radii = upper_limit
                            while previous_guess_radii != current_guess_radii:
                                current_sum = np.sum(dist_from_center < current_guess_radii * binary_image)
                                previous_guess_radii = current_guess_radii
                                if current_sum < total_max:
                                    lower_limit = current_guess_radii
                                    current_guess_radii = (upper_limit - lower_limit) // 2 + lower_limit
                                else:
                                    upper_limit = current_guess_radii
                                    current_guess_radii = upper_limit - (upper_limit - lower_limit) // 2
                            meta_array[z, ...] = dist_from_center < current_guess_radii
                        meta = array_to_sitk(meta_array, cbct_handle)
                        registered_meta = registerDicom(fixed_image=CT_handle,  moving_image=meta,
                                                        moving_series_instance_uid=from_uid, dicom_registration=ds,
                                                        min_value=0, method=sitk.sitkLinear)
                        sitk.WriteImage(sitk.Cast(registered_meta, sitk.sitkUInt8), out_meta_file)
                        sitk.WriteImage(registered_handle, out_reg_file)
                        isocenter = registered_handle.TransformPhysicalPointToIndex(registered_couch)
                        fid = open(out_table_vert, 'w+')
                        fid.write(str(isocenter))
                        fid.close()
    fid = open(status_file, 'w+')
    fid.close()
    return None


def create_padded_cbcts(patient_path, rewrite=False):
    patient_path = os.path.join(patient_path, "Niftiis")
    primary_CT_path = os.path.join(patient_path, "Primary_CT.mha")
    if not os.path.exists(primary_CT_path):
        print("Primary CT does not exist!")
        fid = open(logs_file, 'a')
        fid.write("No primary CT for {}\n".format(patient_path))
        fid.close()
        return None
    status_file = os.path.join(patient_path, "Finished_Padded_CBCT.txt")
    if os.path.exists(status_file) and not rewrite:
        return None
    erode_filter = sitk.BinaryErodeImageFilter()
    erode_filter.SetKernelType(sitk.sitkBall)
    CT_handle = sitk.ReadImage(os.path.join(patient_path, "Primary_CT.mha"))
    CBCT_Files = glob(os.path.join(patient_path, 'Registered_CBCT*.mha'))
    for CBCT_File in CBCT_Files:
        out_file = CBCT_File.replace("Registered_CBCT", "Padded_CBCT")
        table_file = CBCT_File.replace("Registered_", "TableHeight_").replace(".mha", ".txt")
        meta_file = CBCT_File.replace("CBCT", "Meta")
        registered_handle = sitk.ReadImage(CBCT_File)
        meta_handle = sitk.ReadImage(meta_file)
        fid = open(table_file)
        table_vert = int(fid.readline().split(', ')[1])
        fid.close()
        spacing = registered_handle.GetSpacing()
        erode_filter.SetKernelRadius((int(5/spacing[0]), int(5/spacing[1]), int(10/spacing[2])))  # x, y, z
        padded_cbct = pad_cbct(meta_handle, registered_handle, CT_handle, erode_filter, couch_start=table_vert)
        sitk.WriteImage(padded_cbct, out_file)
    fid = open(status_file, 'w+')
    fid.close()
    return None


def pad_cbct(meta_handle: sitk.Image, cbct_handle: sitk.Image, ct_handle: sitk.Image,
             erode_filter: sitk.BinaryErodeImageFilter, couch_start: int):
    """
    :param cbct_handle:
    :param ct_handle:
    :param expansion: expansion to explore, in cm
    :return:
    """
    ct_array = sitk.GetArrayFromImage(ct_handle)
    cbct_array = sitk.GetArrayFromImage(cbct_handle)
    cbct_s = cbct_array.shape
    spacing = cbct_handle.GetSpacing()
    couch_stop = couch_start + int(50 * spacing[1])
    ct_array[:, couch_stop:, :] = -1000
    cbct_array[:, couch_stop:, :] = -1000
    ct_array[:, couch_start:couch_stop, :] = cbct_array[cbct_s[0]//2, couch_start:couch_stop, cbct_s[-1]//2][None, ..., None]
    cbct_array[:, couch_start:couch_stop, :] = cbct_array[:, couch_start:couch_stop, cbct_s[-1]//2][..., None]
    binary_meta = get_binary_image(meta_handle, lowerThreshold=1, upperThreshold=2)
    eroded_meta = erode_filter.Execute(binary_meta)
    eroded_meta_array = sitk.GetArrayFromImage(eroded_meta)
    cbct_array[eroded_meta_array != 1] = ct_array[eroded_meta_array != 1]
    padded_cbct_handle = array_to_sitk(cbct_array, cbct_handle)
    return padded_cbct_handle


def shift_panel_origin(patient_path):
    fluence_files = glob(os.path.join(patient_path, "Niftiis", "Fluence_*"))
    fluence_files += glob(os.path.join(patient_path, "Niftiis", "PDOS_*"))
    # resampler = ResampleTools.ImageResampler()
    for fluence_file in fluence_files:
        fluence_handle = sitk.ReadImage(fluence_file)
        spacing = fluence_handle.GetSpacing()
        # fluence_handle = resampler.resample_image(fluence_handle, output_origin=(0, 0, -1540),
        #                                           output_spacing=spacing)
        size = fluence_handle.GetSize()
        origin = [i for i in fluence_handle.GetOrigin()]
        origin[0] -= spacing[0] * (size[0] - 1)/2
        origin[1] -= spacing[1] * (size[1] - 1)/2
        fluence_handle.SetOrigin(origin)
        sitk.WriteImage(fluence_handle, fluence_file)
    return None


def update_origin(patient_path):
    drr_files = glob(os.path.join(patient_path, "Niftiis", "DRR_*"))
    angle_dictionary = {}
    image_reader = sitk.ImageFileReader()
    image_reader.SetFileName(os.path.join(patient_path, "Niftiis", "Primary_CT.mha"))
    image_reader.ReadImageInformation()
    for drr_file in drr_files:
        image_reader.SetFileName(drr_file)
        image_reader.ReadImageInformation()
        iso_center = image_reader.GetOrigin()
        angle_date = drr_file.split('DRR_')[-1]
        angle = angle_date.split('_')[0]
        if angle not in angle_dictionary:
            angle_dictionary[angle] = iso_center
        fluence_files = glob(os.path.join(patient_path, "Niftiis", "Fluence*{}".format(angle_date)))
        for fluence_file in fluence_files:
            fluence_handle = sitk.ReadImage(fluence_file)
            fluence_handle.SetOrigin(iso_center)
            sitk.WriteImage(fluence_handle, fluence_file)
    for angle in angle_dictionary.keys():
        pdos_files = glob(os.path.join(patient_path, "Niftiis", "PDOS*{}*".format(angle)))
        for pdos_file in pdos_files:
            pdos_handle = sitk.ReadImage(pdos_file)
            pdos_handle.SetOrigin(angle_dictionary[angle])
            sitk.WriteImage(pdos_handle, pdos_file)
    return None


def createDRRs(patient_path, rewrite, perform_on_primary_CT=False):
    plan_dictionary = return_plan_dictionary(patient_path)
    padded_cbcts = glob(os.path.join(patient_path, "Niftiis", "Padded_CBCT*"))
    primary_path = os.path.join(patient_path, "Niftiis", "Primary_CT_Updated.mha")
    for padded_cbct_file in padded_cbcts:
        for beam_number in plan_dictionary:
            beam = plan_dictionary[beam_number]
            gantry_angle = beam["Gantry"]
            iso_center = beam["Iso"]
            cbct_handle = None
            description = f"G{gantry_angle}_{beam['Beam_Name']}"
            out_file = padded_cbct_file.replace("Padded_CBCT", f"DRR_{description}")
            if not os.path.exists(out_file) or rewrite:
                if not perform_on_primary_CT or not os.path.exists(primary_path):
                    cbct_handle = sitk.ReadImage(padded_cbct_file)
                else:
                    cbct_handle = sitk.ReadImage(primary_path)
                create_drr(cbct_handle, gantry_angle=gantry_angle, sid=1000, spd=1540,
                           out_path=out_file, translations=[i for i in iso_center], distance_from_iso=None)
            for height in [-50, 0, 50]:
                out_file = padded_cbct_file.replace("Padded_CBCT", f"Proj_{height//10}cm_to_iso_{description}")
                if not os.path.exists(out_file) or rewrite:
                    if cbct_handle is None:
                        cbct_handle = sitk.ReadImage(padded_cbct_file)
                    create_drr(cbct_handle, gantry_angle=gantry_angle, sid=1000, spd=1000+height,
                               out_path=out_file, translations=[i for i in iso_center], distance_from_iso=height)
                out_file = padded_cbct_file.replace("Padded_CBCT", f"Proj_{height//10}cm_from_iso_to_panel_{description}")
                if not os.path.exists(out_file) or rewrite:
                    if cbct_handle is None:
                        cbct_handle = sitk.ReadImage(padded_cbct_file)
                    create_drr(cbct_handle, gantry_angle=gantry_angle, sid=1000, spd=1540,
                               out_path=out_file, translations=[i for i in iso_center], distance_from_iso=height)
    return None


class FluenceReader(object):
    def __init__(self):
        self.reader = sitk.ImageSeriesReader()
        self.reader.MetaDataDictionaryArrayUpdateOn()
        self.reader.LoadPrivateTagsOn()
        self.reader.SetOutputPixelType(sitk.sitkFloat32)
        self.dicom_handle = None

    def set_file(self, file_name):
        self.reader.SetFileNames([file_name])
        self.dicom_handle = self.reader.Execute()

    def load_file(self):
        self.dicom_handle = self.reader.Execute()

    def return_date(self):
        return self.reader.GetMetaData(0, "0008|0022")

    def return_gantry_angle(self):
        return self.reader.GetMetaData(0, "300a|011e")

    def return_collimator_angle(self):
        return self.reader.GetMetaData(0, "300a|0120")

    def get_all_info(self):
        for key in self.reader.GetMetaDataKeys(0):
            print("{} is {}".format(key, self.reader.GetMetaData(0, key)))
        return None

    def return_image_info(self):
        return self.reader.GetMetaData(0, "0008|0008")

    def return_key_info(self, key):
        return self.reader.GetMetaData(0, key)

    def return_has_key(self, key):
        return self.reader.HasMetaDataKey(0, key)


def return_plan_dictionary(patient_path):
    ds_plan = pydicom.read_file(glob(os.path.join(patient_path, "RTPLAN", "*.dcm"))[0])
    plan_dictionary = {}
    for beam_sequence in ds_plan.BeamSequence:
        if beam_sequence.TreatmentDeliveryType == "SETUP":
            continue
        control_sequence = beam_sequence.ControlPointSequence[0]
        beam_limiting_device = control_sequence.BeamLimitingDevicePositionSequence
        plan_dictionary[beam_sequence.BeamNumber] = {"Iso": control_sequence.IsocenterPosition,
                                                     "Gantry": round(control_sequence.GantryAngle),
                                                     "X_Jaw": round(beam_limiting_device[0].LeafJawPositions[-1]-beam_limiting_device[0].LeafJawPositions[0]),
                                                     "Y_Jaw": round(beam_limiting_device[1].LeafJawPositions[-1]-beam_limiting_device[1].LeafJawPositions[0]),
                                                     "Beam_Name": beam_sequence.BeamName.replace("_", "")}
    for fraction_sequence in ds_plan.FractionGroupSequence:
        for beam_sequence in fraction_sequence.ReferencedBeamSequence:
            if beam_sequence.ReferencedBeamNumber in plan_dictionary:
                plan_dictionary[beam_sequence.ReferencedBeamNumber]["MU"] = beam_sequence.BeamMeterset
    return plan_dictionary


def create_transmission(patient_path, rewrite):
    fluence_reader = FluenceReader()
    Dicom_reader = DicomReaderWriter(description='Examples', verbose=False)
    Dicom_reader.walk_through_folders(os.path.join(patient_path, 'RTIMAGE')) # Read in the acquired images
    dicom_files = glob(os.path.join(patient_path, "RTIMAGE", "*.dcm"))
    padded_cbct = glob(os.path.join(patient_path, "Niftiis", "Padded_CBCT_*"))
    dates = [i.split("_")[-1].split('.')[0] for i in padded_cbct]
    date_dictionary = {}
    for cbct in padded_cbct:
        date_dictionary[cbct.split("_")[-1].split('.')[0]] = cbct
    plan_dictionary = return_plan_dictionary(patient_path)
    if rewrite:
        pdos_files = glob(os.path.join(patient_path, "Niftiis", "PDOS_*"))
        for pdos_file in pdos_files:
            os.remove(pdos_file)
    for file in dicom_files:
        fluence_reader.set_file(file)
        image_type = fluence_reader.return_image_info()
        date = fluence_reader.return_date()
        if image_type.find("DRR") != -1:
            continue
        gantry = round(float(fluence_reader.return_gantry_angle()))
        if gantry == 360:
            gantry = 0
        referenced_beam_number = int(fluence_reader.return_key_info("300c|0006"))
        if referenced_beam_number not in plan_dictionary:
            continue
        if fluence_reader.return_has_key("3002|0029"):
            fraction_number = int(fluence_reader.return_key_info("3002|0029"))
            description = "Fluence"
            if fraction_number == 0:
                description = "PDOS"
        elif image_type.find("CALCULATED_DOSE") != -1:
            description = "PDOS"
        elif image_type.find("ACQUIRED_DOSE") != -1:
            description = "Fluence"
            if date not in dates:
                continue
        elif image_type.find("PREDICTED") != -1:
            description = "Predicted"
        else:
            continue
        beam_name = plan_dictionary[referenced_beam_number]['Beam_Name']
        out_file = os.path.join(patient_path, "Niftiis", f"{description}_G{gantry}_{beam_name}_{date}.mha")
        if os.path.exists(out_file) and not rewrite:
            continue
        fluence_reader.load_file()
        if fluence_reader.return_has_key("3002|000d"):
            panel_shift = fluence_reader.return_key_info("3002|000d").split('\\')
            panel_shift = [float(i) for i in panel_shift]
            panel_shift[-1] = -float(fluence_reader.return_key_info("3002|0026"))
            panel_shift[:-1] = [0, 0]
            fluence_reader.dicom_handle.SetOrigin(panel_shift)
        if description == "PDOS":
            if image_type.find("CALCULATED_DOSE") != -1:
                fluence_reader.dicom_handle *= plan_dictionary[referenced_beam_number]["MU"]
                if os.path.exists(out_file):
                    # out_file = out_file.replace(".mha", "_Calc.mha")
                    continue
        """
        Now shift the origin
        """
        spacing = fluence_reader.dicom_handle.GetSpacing()
        size = fluence_reader.dicom_handle.GetSize()
        origin = [i for i in fluence_reader.dicom_handle.GetOrigin()]
        origin[0] -= spacing[0] * (size[0] - 1)/2
        origin[1] -= spacing[1] * (size[1] - 1)/2
        fluence_reader.dicom_handle.SetOrigin(origin)
        sitk.WriteImage(fluence_reader.dicom_handle, out_file)
    return None


def create_inputs(patient_path: typing.Union[str, bytes, os.PathLike], rewrite=False, perform_on_primary_CT=False):
    """
    First, for preprocessing, create the padded CBCTs by registering them with the primary CT and padding
    Second, create the fluence and PDOS images from DICOM handles
    Third, create the DRR and half-CBCT DRR for each beam angle
    Fourth, align the PDOS and fluence with the DRRs
    """
    if not os.path.exists(os.path.join(patient_path, 'pCT')):
        print("No primary CT for {}".format(patient_path))
        fid = open(logs_file, 'a')
        fid.write("No primary CT for {}\n".format(patient_path))
        fid.close()
        return None
    skip = os.path.join(patient_path, 'Inputs_made.txt')
    # if os.path.exists(skip) and not rewrite:
    #     return None
    #create_registered_cbct(patient_path=patient_path, rewrite=rewrite)
    #create_padded_cbcts(patient_path=patient_path, rewrite=rewrite)
    if patient_path.find('phantom') != -1:
        "Padding in sup-inf direction"
        # update_CBCT(os.path.join(patient_path, 'Niftiis'), rewrite=rewrite)
        update_primary_CT(patient_path, rewrite=True)
    else:
        perform_on_primary_CT = False
    createDRRs(patient_path=patient_path, rewrite=rewrite, perform_on_primary_CT=perform_on_primary_CT)
    create_transmission(patient_path=patient_path, rewrite=rewrite)
    fid = open(skip, 'w+')
    fid.close()
    return None


if __name__ == '__main__':
    pass

