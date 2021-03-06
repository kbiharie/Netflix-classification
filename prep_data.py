import cv2
import math
import glob
import os
import random
import json


def process_episode(episode_path, show_name, dataset_folder, next_image, timeframe=35):
    """
    Create images from one episode
    :param episode_path: Path to the video file
    :param show_name: Name of the show
    :param dataset_folder: Folder to the shows folder
    :param next_image: Image index to start with
    :param timeframe: Seconds between two images
    :return: Index of last image
    """
    cap = cv2.VideoCapture(episode_path)
    fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Number of images taken from this episode
    print(frames // (timeframe * fps))
    for i in range(1, frames // (timeframe * fps)):
        frame_number = i * fps * timeframe

        if frame_number < frames:
            cap.set(1, frame_number - 1)
            res, frame = cap.read()

            # Save the image
            if res:
                name = dataset_folder + show_name + "\\" + str(i + next_image) + ".jpg"
                cv2.imwrite(name, frame)

    cap.release()
    # cv2.destroyAllWindows()
    return next_image + frames // (timeframe * fps) - 1


def process_show(show_path, dataset_folder):
    """
    Process all episodes from one show
    :param show_path: Path to the show folder
    :param dataset_folder: Path to the dataset folder
    """
    show_name = show_path.split("\\")[-1]
    next_image = 0
    os.mkdir(dataset_folder + show_name)
    for path in glob.glob(show_path + "\\**"):
        print(path)
        next_image = process_episode(path, show_name, dataset_folder, next_image)


def create_images(shows_folder, dataset_folder):
    """
    Process all episodes from all shows
    :param shows_folder: Path to the shows folder
    :param dataset_folder: Path to the dataset folder
    """
    for path in glob.glob(shows_folder + "**"):
        process_show(path, dataset_folder)


def create_json(dataset_folder, n_train=400, n_val=50, n_test=50):
    """
    Create a json file for images in the dataset
    :param dataset_folder: Path to the dataset folder
    :param n_train: Number of train images per show
    :param n_val: Number of validation images per show
    :param n_test: Number of test images per show
    :return:
    """

    # Aggregate all image paths
    show_dir = {}
    image_paths = {}
    for folder in glob.glob(dataset_folder + "**"):
        if not os.path.isdir(folder):
            continue
        show_name = folder.split("\\")[-1]
        image_paths[show_name] = []
        for image_path in glob.glob(folder + "\\*.jpg"):
            image_paths[show_name].append(image_path.replace("\\", "/"))
        show_dir[show_name] = len(show_dir)

    # Split into train, validation and test
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

    # Save to a json file
    with open(dataset_folder + "filenames_animated.json", "w") as w:
        json.dump(dataset, w)


if __name__ == '__main__':
    show_folder = "../shows/"
    dataset_folder = "../dataset/"
    # create_images(show_folder, dataset_folder)
    create_json(dataset_folder)
