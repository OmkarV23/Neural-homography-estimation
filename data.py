import glob, sys, os, random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

def data():

    inputs_path = '/workspace/omkar_projects/WPI_CV/AutoPano/phase2/data/input/'+'*.npy'
    gts_path = '/workspace/omkar_projects/WPI_CV/AutoPano/phase2/data/gt/'+'*.npy'

    data = [(i,j) for i,j in zip(glob.glob(inputs_path),glob.glob(gts_path))]
    random.shuffle(data)

    train_data = data[:int(0.8*len(data))]
    valid_data = data[int(0.8*len(data)):int(0.9*len(data))]
    test_data = data[int(0.9*len(data)):]

    class Warper(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, index):
            input = np.load(self.data[index][0])
            gt = np.load(self.data[index][1])
            return (torch.tensor(input, dtype=torch.float32).permute(2,0,1),torch.tensor(gt, dtype=torch.float32))
                
    train_data = Warper(train_data)
    valid_data = Warper(valid_data)
    test_data = Warper(test_data)

    return train_data, valid_data, test_data