from os.path import join as pjoin
import os
import pydicom as dicom
from patient import Patient
import numpy as np
from pydicom.pixel_data_handlers.util import apply_modality_lut as m_lut
import pickle


class Pathology:
    def __init__(self, _inputpath, _outputpath):
        self.inputpath = _inputpath
        self.outputpath = _outputpath

    vec_2ch = [0.6585, 6.7523, -0.0170]
    vec_2ch_inv = [-0.7194, -0.6941, -0.0234]
    vec_4ch = [0.1056, -0.6553, 0.7479]
    vec_4ch_inv = [-0.0952, 0.7712, -0.6294]
    vec_lvot = [-0.7625, -0.1435, -0.6307]
    vec_lvot_inv = [0.6704, 0.2410, 0.7017]
    patients2CH = []
    patients4CH = []
    patientsLVOT = []

    @staticmethod
    def dicom_info(file, path):
        if file.find('.dcm') != -1:
            try:
                temp = dicom.dcmread(pjoin(path, file))
                return temp[0x0010, 0x0040].value, temp[0x0010, 0x1030].value, temp[0x0028, 0x0030].value
            except Exception as e:
                print(e)
        return "NA", 0, [0, 0]

    @staticmethod
    def dicom_time(file, path):
        if file.find('.dcm') != -1:
            try:
                temp = dicom.dcmread(pjoin(path, file))
                return temp[0x0008, 0x0013].value
            except Exception as e:
                print(e)
        return 0

    def dump(self):
        for p1 in self.patients2CH:
            pic = open(pjoin(self.outputpath, p1.id, "2CH"), "wb")
            pickle.dump(p1.id, pic)
            pickle.dump(p1.gender, pic)
            pickle.dump(p1.weight, pic)
            pickle.dump(p1.images, pic)
            pic.close()
        for p2 in self.patients4CH:
            pic = open(pjoin(self.outputpath, p2.id, "4CH"), "wb")
            pickle.dump(p2.id, pic)
            pickle.dump(p2.gender, pic)
            pickle.dump(p2.weight, pic)
            pickle.dump(p2.images, pic)
            pic.close()
        for p3 in self.patientsLVOT:
            pic = open(pjoin(self.outputpath, p3.id, "LVOT"), "wb")
            pickle.dump(p3.id, pic)
            pickle.dump(p3.gender, pic)
            pickle.dump(p3.weight, pic)
            pickle.dump(p3.images, pic)
            pic.close()

    pos = {
        0: "2CH",
        1: "2CH",
        2: "4CH",
        3: "4CH",
        4: "LVOT",
        5: "LVOT"
    }

    @staticmethod
    def calculateangle(a, b, c, vec):
        c1 = a * vec[0] + b * vec[1] + c * vec[2]
        c2 = np.sqrt(pow(a, 2) + pow(b, 2) + pow(c, 2))
        c3 = np.sqrt(pow(vec[0], 2) + pow(vec[1], 2) + pow(vec[2], 2))
        angle = np.arccos(c1 / (c2 * c3))
        return angle

    def la_type(self, file):
        xx = file[0x0020, 0x0037].value[0]
        xy = file[0x0020, 0x0037].value[1]
        xz = file[0x0020, 0x0037].value[2]
        yx = file[0x0020, 0x0037].value[3]
        yy = file[0x0020, 0x0037].value[4]
        yz = file[0x0020, 0x0037].value[5]
        cross = np.cross([xx, xy, xz], [yx, yy, yz])
        angles = [self.calculateangle(cross[0], cross[1], cross[2], self.vec_2ch),
                  self.calculateangle(cross[0], cross[1], cross[2], self.vec_2ch_inv),
                  self.calculateangle(cross[0], cross[1], cross[2], self.vec_4ch),
                  self.calculateangle(cross[0], cross[1], cross[2], self.vec_4ch_inv),
                  self.calculateangle(cross[0], cross[1], cross[2], self.vec_lvot),
                  self.calculateangle(cross[0], cross[1], cross[2], self.vec_lvot_inv)]
        mini = angles.index(min(angles))
        return self.pos[mini]

    @staticmethod
    def dicom_reader(temp):
        arr = temp.pixel_array
        hu = m_lut(arr, temp)
        bottom_centile = np.percentile(hu, 1)
        top_centile = np.percentile(hu, 99)
        bottom_filtered = np.where(hu < bottom_centile, bottom_centile, hu)
        filtered = np.where(bottom_filtered > top_centile, top_centile, bottom_filtered)
        _min = np.amin(filtered)
        _max = np.amax(filtered)
        for n in range(len(filtered)):
            filtered[n] = (filtered[n] - _min) * (255 / (_max - _min))
        uint = np.uint8(filtered)
        return uint

    def create_patient(self):
        for pf in [pjoin(self.inputpath, folder_name) for folder_name in os.listdir(self.inputpath)]:
            _id = os.path.basename(pf)
            la_folder = pjoin(pf, "la")
            if os.path.isdir(la_folder) and len(os.listdir(la_folder)) != 0:
                try:
                    os.mkdir(pjoin(self.outputpath, _id))
                except OSError:
                    print("Creation of the directory failed!")
                dcm_files = sorted(os.listdir(la_folder))
                gender, weight, spacing = self.dicom_info(dcm_files[0], la_folder)
                patient2 = []
                patient4 = []
                patientl = []
                la2 = 0
                la4 = 0
                lal = 0
                time = 0
                if len(dcm_files) > 75:
                    t1 = self.dicom_time(dcm_files[0], la_folder)
                    t2 = self.dicom_time(dcm_files[75], la_folder)
                    time = max([t1, t2])
                for file in dcm_files:
                    if file.find('.dcm') != -1:
                        try:
                            temp = dicom.dcmread(pjoin(la_folder, file))
                            if time != 0 and time != temp[0x0008, 0x0013].value:
                                continue
                            la_type = self.la_type(temp)
                            if la_type == "2CH":
                                if la2 % 3 != 0:
                                    la2 += 1
                                    continue
                                la2 += 1
                                patient2.append(temp)
                            elif la_type == "4CH":
                                if la4 % 3 != 0:
                                    la4 += 1
                                    continue
                                la4 += 1
                                patient4.append(temp)
                            elif la_type == "LVOT":
                                if lal % 3 != 0:
                                    lal += 1
                                    continue
                                lal += 1
                                patientl.append(temp)
                        except Exception as e:
                            print("Error loading $s $s", _id, e)
                if len(patient2) != 0:
                    self.patients2CH.append(Patient(_id, gender, weight, spacing, patient2))
                if len(patient4) != 0:
                    self.patients4CH.append(Patient(_id, gender, weight, spacing, patient4))
                if len(patientl) != 0:
                    self.patientsLVOT.append(Patient(_id, gender, weight, spacing, patientl))


if __name__ == '__main__':
    print("Input Path:")
    ip = input()
    print("Output Path:")
    op = input()
    if os.path.isdir(ip) and os.path.isdir(op):
        pathology = Pathology(ip, op)
        pathology.create_patient()
        pathology.dump()
    else:
        if not os.path.isdir(ip):
            print("Invalid input path: ", ip)
        if not os.path.isdir(op):
            print("Invalid output path: ", op)