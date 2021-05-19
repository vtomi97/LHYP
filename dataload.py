import pickle
from os.path import join as pjoin
import os
import numpy as np
from skimage.transform import resize
from tqdm import tqdm
import random


class Dataload:
    def __init__(self, _path):
        self.path = _path

    training_data = []
    test_data = []

    LABEL = {
        "NORM": 0,
        "Normal": 0,
        "U18_m": 0,
        "U18_f": 0,
        "adult_f_sport": 0,
        "adult_m_sport": 0,
        "HCM": 1,
        "AMY": 1,
        "EMF": 1,
        "Amyloidosis": 1,
        "Fabry": 1,
        "Aortastenosis": 1
    }

    def calculate_total(self):
        i = 0
        for folder in os.listdir(self.path):
            if len(os.listdir(pjoin(self.path, folder))) == 3:
                i += 1
        return i

    @staticmethod
    def prep(data):
        temp = []
        length = len(data)
        for i in range(length):
            temp.append(resize(data[i], (100, 100)))
        for i in range(9 - length):
            r = random.randint(0, length - 1)
            temp.append(resize(data[r], (100, 100)))
        return temp

    def make_data(self, latype):
        train = round(self.calculate_total() * 0.8)
        for folder in tqdm(os.listdir(self.path)):
            if len(os.listdir(pjoin(self.path, folder))) != 3:
                continue
            pic = open(pjoin(self.path, folder, latype), "rb")
            data = pickle.load(pic)
            if len(data) < 9:
                continue
            label = pickle.load(pic)
            r_data = self.prep(data)
            if train != 0:
                self.training_data.append([r_data[0], r_data[1], r_data[2], r_data[3], r_data[4], r_data[5], r_data[6], r_data[7], r_data[8], np.eye(2)[self.LABEL[label]]])
                train -= 1
            else:
                self.test_data.append([r_data[0], r_data[1], r_data[2], r_data[3], r_data[4], r_data[5], r_data[6], r_data[7], r_data[8], np.eye(2)[self.LABEL[label]]])
        np.save(latype + "/training_data.npy", self.training_data)
        np.save(latype + "/test_data.npy", self.test_data)
        self.training_data.clear()
        self.test_data.clear()


if __name__ == '__main__':
    ds = Dataload("vtamas_data")
    ds.make_data("2CH")
    ds.make_data("4CH")
    ds.make_data("LVOT")
