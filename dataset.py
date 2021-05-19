from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image


class DataSet(Dataset):
    def __init__(self, _type, _file, transform=None):
        self.transform = transform
        self.data = np.load(_type + _file, allow_pickle=True)
        self.count = len(self.data)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        i = self.data[index]
        for j in range(9):
            i[j] = Image.fromarray(i[j])
        if self.transform is not None:
            for j in range(9):
                i[j] = self.transform(i[j])
        for j in range(9):
            i[j] = np.asarray(i[j])
        return [i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]], i[9]
