import cv2
import math
import glob
import os
import random
import json


def process_episode(episode_path, show_name, dataset_folder, next_image, timeframe=120):
    cap = cv2.VideoCapture(episode_path)
    fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(frames // (timeframe * fps))
    for i in range(1, frames // (timeframe * fps)):
        frame_number = i * fps * timeframe
        if frame_number < frames:
            cap.set(1, frame_number - 1)
            res, frame = cap.read()
            if res:
                name = dataset_folder + show_name + "\\" + str(i + next_image) + ".jpg"
                cv2.imwrite(name, frame)

    cap.release()
    # cv2.destroyAllWindows()
    return next_image + frames // (timeframe * fps) - 1


def process_show(show_path, dataset_folder):
    show_name = show_path.split("\\")[-1]
    next_image = 0
    os.mkdir(dataset_folder + show_name)
    for path in glob.glob(show_path + "\\**"):
        print(path)
        next_image = process_episode(path, show_name, dataset_folder, next_image)


def create_images(shows_folder, dataset_folder):
    for path in glob.glob(shows_folder + "**"):
        process_show(path, dataset_folder)


def create_json(dataset_folder, n_train=80, n_val=10, n_test=10):

    show_dir = {}

    image_paths = {}
    for folder in glob.glob(dataset_folder + "**"):
        if not os.path.isdir(folder):
            continue
        show_name = folder.split("\\")[-1]
        image_paths[show_name] = []
        for image_path in glob.glob(folder + "\\*.jpg"):
            image_paths[show_name].append(image_path.replace("\\", "/"))
        print(show_name)
        show_dir[show_name] = len(show_dir)

    dataset = {"train": [], "val": [], "test": []}
    for show_name in image_paths:
        random.shuffle(image_paths[show_name])
        for i in range(n_train + n_val + n_test):
            entry = {"show": show_name, "path": image_paths[show_name][i], "label": show_dir[show_name]}
            if i < n_train:
                dataset["train"].append(entry)
            elif i < n_train + n_val:
                dataset["val"].append(entry)
            else:
                dataset["test"].append(entry)

    dataset["show_dir"] = show_dir

    with open(dataset_folder + "filenames.json", "w") as w:
        json.dump(dataset, w)


if __name__ == '__main__':
    show_folder = "../shows/"
    dataset_folder = "../dataset/"
    # create_images(show_folder, dataset_folder)
    create_json(dataset_folder)
