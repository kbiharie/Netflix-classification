import torch
import json
import os
import cv2
import numpy as np


class NetflixDataset(torch.utils.data.Dataset):

    def __init__(self, filenames, purpose):
        self.purpose = purpose
        with open(filenames) as f:
            self.data = json.load(f)[self.purpose]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        image_path = self.data[id]["path"]
        if not os.path.exists(image_path):
            print(image_path)

        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.uint8)
        return img
