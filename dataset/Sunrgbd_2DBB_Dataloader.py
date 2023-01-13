import json
import numpy as np
import pickle
from torch.utils.data import Dataset

NYU40CLASSES = ('void',
                'wall', 'floor', 'cabinet', 'bed', 'chair',
                'sofa', 'table', 'door', 'window', 'bookshelf',
                'picture', 'counter', 'blinds', 'desk', 'shelves',
                'curtain', 'dresser', 'pillow', 'mirror', 'floor_mat',
                'clothes', 'ceiling', 'books', 'refridgerator', 'television',
                'paper', 'towel', 'shower_curtain', 'box', 'whiteboard',
                'person', 'night_stand', 'toilet', 'sink', 'lamp',
                'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop')


class SunDataset_2DBB(Dataset):
    def __init__(self, root_path='../', mode='train'):
        """
        self.root_path:    Get the data_path
        self.file_idx = []:    Get the file index, train: idx 5051 to 10335, test: idx 1 to 5050
        """
        self.root_path = root_path
        self.file_idx = []
        if mode == 'train':
            with open(self.root_path + '/data/sunrgbd/splits/train.json') as json_file:
                self.file_idx = json.load(json_file)
        elif mode == 'test':
            with open(self.root_path + '/data/sunrgbd/splits/test.json') as json_file:
                self.file_idx = json.load(json_file)

    def __len__(self):
        return len(self.file_idx)

    def __getitem__(self, idx):
        """
        Return: img, 2d bounding boxes, labels
        """
        data_pkl = pickle.load(open(self.root_path + self.file_idx[idx][1:], 'rb'))
        boxes = data_pkl['boxes']
        bbox = [[boxes['bdb2D_pos'][i][1], boxes['bdb2D_pos'][i][0], boxes['bdb2D_pos'][i][3], boxes['bdb2D_pos'][i][2]]
                for i in range(boxes['bdb2D_pos'].shape[0])]
        return np.float32(data_pkl['rgb_img'].transpose((2, 0, 1))), np.float32(bbox), boxes['size_cls']
