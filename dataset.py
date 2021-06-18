import torch
import json
import os
import numpy as np
from torchvision import transforms
from PIL import Image


class NetflixDataset(torch.utils.data.Dataset):

    def __init__(self, filename, purpose, return_org=False):
        """
        :param filename: Name with the file paths and labels of the images
        :param purpose: Determines the image set from {"train", "val", "test"}
        """
        self.purpose = purpose
        with open(filename) as f:
            all_data = json.load(f)
            self.data = all_data[self.purpose]
            self.n_classes = len(all_data["show_dir"])
            self.shows = all_data["show_dir"]
        self.return_org = return_org

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        # Transforms used on every image
        tfs = transforms.Compose([transforms.RandomResizedCrop(240), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        tfs_standard = transforms.Compose([transforms.Resize(size=(720, 1280)), transforms.ToTensor()])

        image_path = self.data[id]["path"]
        if not os.path.exists(image_path):
            print(image_path, "does not exist")

        img = Image.open(image_path)
        original_image= tfs_standard(img)

        img = tfs(img)

        # Integer identifying the show
        label = self.data[id]["label"]
        label = np.array(label).astype(np.longlong)

        if self.return_org:
            return img, torch.from_numpy(label), original_image

        return img, torch.from_numpy(label)
