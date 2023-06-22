import getopt
import os
import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt

image_folder = "resources/persons"
img_indexes = [36, 37, 38, 210, 211, 212, 213, 214, 215, 382, 483, 484, 485, 510, 511, 512, 513, 514, 515, 631, 632,
               633, 634, 711, 712, 713, 714, 905, 906, 907, 976, 977, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1115]


def are_any_objects(stats):
    return stats.shape[0] > 1


def if_human_proportions(stat):
    return stat[3] / stat[2] > 2


def is_below(main_stat, sub_stat):
    return 0 < sub_stat[1] - main_stat[1] - main_stat[3] < 100


def is_between_edges(main_stat, sub_stat):
    return sub_stat[0] + sub_stat[2] > main_stat[0] and sub_stat[0] < main_stat[0] + main_stat[2]


def update_right_bottom(right_bottom, sub_stat):
    return ((sub_stat[0] + sub_stat[2]) if (right_bottom[0] < sub_stat[0] + sub_stat[2]) else right_bottom[0]), (
        (sub_stat[1] + sub_stat[3]) if (right_bottom[1] < sub_stat[1] + sub_stat[3]) else right_bottom[1])


def update_left_top(left_top, sub_stat):
    return sub_stat[0] if left_top[0] > sub_stat[0] else left_top[0], left_top[1]


def update_rectangle_tops(left_top, right_bottom, sub_stat):
    return update_left_top(left_top, sub_stat), update_right_bottom(right_bottom, sub_stat)


def get_object_coordinates(stats, stat):
    left_top, right_bottom = (stat[0], stat[1]), (stat[0] + stat[2], stat[1] + stat[3])
    for (j, sub_stat) in enumerate(stats):
        if j == 0:
            continue
        if is_below(stat, sub_stat) and is_between_edges(stat, sub_stat):
            left_top, right_bottom = update_rectangle_tops(left_top, right_bottom, sub_stat)
    return left_top, right_bottom


def process_image(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, frame = cv2.threshold(frame, 230, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    frame = cv2.medianBlur(frame, 3)
    frame = cv2.erode(frame, kernel)
    frame = cv2.dilate(frame, kernel)
    return frame


def human_coordinates(image):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(image)

    if are_any_objects(stats):
        for (i, stat) in enumerate(stats):
            if i == 0:
                continue
            if if_human_proportions(stat):
                return get_object_coordinates(stats, stat)
    return None, None


def save_human_silhouette_to_file():
    os.makedirs("resources/persons", exist_ok=True)
    os.makedirs("resources/frames", exist_ok=True)
    video_capture = cv2.VideoCapture('resources/vid1_IR.avi')
    iPedestrian = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        image_processed = process_image(frame)
        left_top, right_bottom = human_coordinates(image_processed)
        if left_top is not None:
            ROI = image_processed[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]]
            cv2.imwrite(image_folder + '/sample_%06d.png' % iPedestrian, ROI)
            iPedestrian = iPedestrian + 1
            if iPedestrian == 3090:
                cv2.imwrite('resources/frames/frame_%06d.png' % iPedestrian, frame)
    video_capture.release()


def create_silhouette_pattern():
    silhouette_pattern = np.zeros((192, 64))

    for iPedestrian in img_indexes:
        filename = 'sample_%06d.png' % iPedestrian
        I = cv2.imread(os.path.join(image_folder, filename))
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        I = cv2.resize(I, (64, 192))
        I = I / np.max(I)
        silhouette_pattern += I
    silhouette_pattern = silhouette_pattern / 40
    cv2.imshow('silhouette_pattern', silhouette_pattern)
    cv2.waitKey(0)
    cv2.imwrite('resources/pattern.png', silhouette_pattern * 255)


def detect_object():
    I = cv2.imread('resources/pattern.png')
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    I = I.astype('uint8')
    I = I / np.max(I)
    plt.imshow(I, 'gray')

    PDM1 = I
    PDM0 = np.ones((192, 64)) - I

    def IoU(rect1, rect2):
        x1, y1, w1, h1 = rect1[1], rect1[0], 64, 192
        x2, y2, w2, h2 = rect2[1], rect2[0], 64, 192
        left = max([x1, x2])
        right = min([x1 + w1, x2 + w2])
        top = max([y1, y2])
        bottom = min([y1 + h1, y2 + h2])
        area1 = max([(right - left), 0]) * max([(bottom - top), 0])
        area2 = (w1 * h1) + (w2 * h2) - area1
        IoU = area1 / area2
        return IoU

    test_frame = cv2.imread('resources/frames/frame_%06d.png' % 3090)
    test_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
    (T, I) = cv2.threshold(test_frame, 35, 255, cv2.THRESH_BINARY)
    result = np.zeros((360, 480), np.float32)
    for i in range(192 // 2, test_frame.shape[0] - (192 // 2)):
        for j in range(64 // 2, test_frame.shape[1] - 64 // 2):
            prob = 0
            prob = np.sum(I[i - 192 // 2:i + 192 // 2, j - 64 // 2:j + 64 // 2] * PDM1) + np.sum(
                (1 - I[i - 192 // 2:i + 192 // 2, j - 64 // 2:j + 64 // 2]) * PDM0)
            result[i, j] = prob
    result = result / np.max(np.max(result))
    result_uint8 = np.uint8(result * 255)
    counter = 1
    Rect1 = []
    for i in range(192 // 2, test_frame.shape[0] - (192 // 2)):
        for j in range(64 // 2, test_frame.shape[1] - 64 // 2):
            if result_uint8[i, j] > 100:
                notFound = True
                for rect_index in range(len(Rect1)):
                    rect = Rect1[rect_index]
                    if IoU(rect[1], (i - 192 // 2, j - 64 // 2, i + 192 // 2, j + 64 // 2)) > 0.0:
                        notFound = False
                        if result_uint8[i, j] > result_uint8[rect[1][0] + 192 // 2, rect[1][1] + 64 // 2]:
                            Rect1[rect_index] = (
                                result_uint8[i, j], (i - 192 // 2, j - 64 // 2, i + 192 // 2, j + 64 // 2))
                        break
                if notFound:
                    Rect1.append((result_uint8[i, j], (i - 192 // 2, j - 64 // 2, i + 192 // 2, j + 64 // 2)))
            fr = test_frame.copy()

    for rect in Rect1:
        cv2.rectangle(test_frame, (rect[1][1], rect[1][0]), (rect[1][3], rect[1][2]), (255, 0, 0), 2)
    cv2.imshow('frame', test_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(argv):
    opts, args = getopt.getopt(argv, "hscd:", ["save", "create", "detect"])
    for opt, arg in opts:
        if opt == '-h':
            print('probabilistic_detector.py <option>')
            print('Options:')
            print('-save, save silhouettes to files')
            print('-create, create silhouette pattern')
            print('-detect, detect objects')
            sys.exit()
        elif opt in ("-s", "--save"):
            print('Saving silhouettes to files')
            save_human_silhouette_to_file()
        elif opt in ("-c", "--create"):
            print('Creating silhouette pattern')
            create_silhouette_pattern()
        elif opt in ("-d", "--detect"):
            print('Detecting objects on example frame')
            detect_object()


if __name__ == '__main__':
    main(sys.argv[1:])
