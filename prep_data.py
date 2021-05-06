import cv2
import math
import glob
import os


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
    show_name = (show_path.split("\\")[-1])
    next_image = 0
    os.mkdir(dataset_folder + show_name)
    for path in glob.glob(show_path + "\\**"):
        print(path)
        next_image = process_episode(path, show_name, dataset_folder, next_image)


def create_images(shows_folder, dataset_folder):
    for path in glob.glob(shows_folder + "**"):
        process_show(path, dataset_folder)


if __name__ == '__main__':
    create_images("..\\shows\\", "..\\dataset\\")
