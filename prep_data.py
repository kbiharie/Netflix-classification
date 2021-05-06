import cv2

def create_images(show, video_path, save_path, timeframe=240):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    currentframe = 0

    while True:
        ret, frame = video.read()

        if ret:
            if currentframe + 1 % (timeframe * fps) == 0:
                name = str(save_path) + str(show) + 'frame' + str(currentframe) + '.jpg'
                cv2.imwrite(name, frame)
            currentframe += 1
        else:
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    show = "st"
    path = "shows/st/ep1"
    save_path = "./images/"
    create_images(show, path, save_path)
