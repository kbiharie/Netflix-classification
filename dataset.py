import torch
import json
import os
import cv2
import numpy as np


class NetflixDataset(torch.utils.data.Dataset):

    def __init__(self, filename, purpose):
        self.purpose = purpose
        with open(filename) as f:
            all_data = json.load(f)
            self.data = all_data[self.purpose]
            self.n_classes = len(all_data["show_dir"])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        image_path = self.data[id]["path"]
        if not os.path.exists(image_path):
            print(image_path, "does not exist")

        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.uint8)
        w = img.shape[0]
        h = img.shape[1]
        img = img[int(w/2 - 120):int(w/2 + 120), int(h/2 - 120):int(h/2 + 120)]

        img = img.astype(np.float32) / 255.
        img = torch.from_numpy(img)

        img = img.permute(2,0,1)
        label = self.data[id]["label"]
        label = np.array(label).astype(np.longlong)

        return img, torch.from_numpy(label)
