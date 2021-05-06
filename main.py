from dataset import NetflixDataset
import cv2
import random

if __name__ == "__main__":

    dataset = NetflixDataset("../dataset/filenames.json", "train")
    print(len(dataset))

    ids = [x for x in range(len(dataset))]
    random.shuffle(ids)

    for id in ids:
        img = dataset.__getitem__(id)
        cv2.imshow("Image", img)
        cv2.waitKey(0)
