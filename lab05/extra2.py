import math
import statistics

import cv2
import numpy as np
import matplotlib.pyplot as plt


def I__median_and_morfology(I):
    I_median = cv2.medianBlur(I, 3)
    I_median = cv2.medianBlur(I_median, 3)

    kernel = np.ones((5, 5), np.uint8)
    # I_result = cv2.erode(I_median,kernel)
    I_result = cv2.dilate(I_median, kernel)
    I_result = cv2.dilate(I_median, kernel)

    return I_result


def vis_flow(flow):
    hsv = np.ones((flow[..., 0].shape[0], flow[..., 0].shape[1], 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[:, :, 0] = (ang * 90 / np.pi)
    hsv[:, :, 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[:, :, 2] = 255
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    cv2.imshow('rgb', rgb)


def I_gray(filename):
    I = cv2.imread(filename)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    return I


def vis_flow(flow):
    hsv = np.ones((flow[..., 0].shape[0], flow[..., 0].shape[1], 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[:, :, 0] = (ang * 90 / np.pi)
    hsv[:, :, 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[:, :, 2] = 255
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    cv2.imshow('rgb', rgb)


# initializing subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False, varThreshold=35)

prev = I_gray('../common/pedestrian/input/in000299.jpg')
fgmask_prev = fgbg.apply(prev)
fgmask_prev = I__median_and_morfology(fgmask_prev)
for i in range(300, 1100, 3):
    next = I_gray('../common/pedestrian/input/in%06d.jpg' % i)

    # applying on each frame
    fgmask_next = fgbg.apply(next)
    fgmask_next = I__median_and_morfology(fgmask_next)

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask_next)
    I_VIS = cv2.imread('../common/pedestrian/input/in%06d.jpg' % i)  # copy of the input image
    if (stats.shape[0] > 1):  # are there any objects
        mag_values = []
        ang_values = []
        for z in range(int(np.amax(labels)) + 1):
            mag_values.append([])
            ang_values.append([])
        flow = cv2.calcOpticalFlowFarneback(fgmask_prev, fgmask_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        for row in range(fgmask_next.shape[0]):
            for col in range(fgmask_next.shape[1]):
                if labels[row, col] != 0:
                    if mag[row, col] > 1:
                        mag_values[labels[row, col]].append(mag[row, col])
                        ang_values[labels[row, col]].append(math.atan2(flow[row, col, 1], flow[row, col, 0]))
        mean_mags = [0] * (np.amax(labels) + 1)
        mean_angs = [0] * (np.amax(labels) + 1)
        std_mags = [0] * (np.amax(labels) + 1)
        std_angs = [0] * (np.amax(labels) + 1)
        for j in range(len(mag_values)):
            if len(mag_values[j]) > 1 and len(ang_values[j]) > 1:
                mean_mags[j] = (statistics.mean(mag_values[j]))
                mean_angs[j] = (statistics.mean(ang_values[j]))
                std_mags[j] = (np.std(mag_values[j]))
                std_angs[j] = (np.std(ang_values[j]))
        for (j, stat) in enumerate(stats):
            if j == 0 or stats[j, 4] < 1000:
                continue
            left_top, right_bottom = (stat[0], stat[1]), (stat[0] + stat[2], stat[1] + stat[3])
            # drawing a bbox
            cv2.rectangle(I_VIS, left_top, right_bottom, (255, 255, 0), 2)
            cv2.putText(I_VIS, "mean mag: %f" % mean_mags[j], left_top, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0))
            cv2.putText(I_VIS, "mean ang: %f" % mean_angs[j], (int(centroids[j, 0]), int(centroids[j, 1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0))
            cv2.putText(I_VIS, "std mag: %f" % std_mags[j], (stats[j, 0], stats[j, 1] + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0))
            cv2.putText(I_VIS, "std ang: %f" % std_angs[j], (int(centroids[j, 0]), int(centroids[j, 1]) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0))

    cv2.imshow('fgmask', I_VIS)
    # cv2.imshow('labels', labels)
    if (i == 471):
        cv2.waitKey(0)
    else:
        cv2.waitKey(10)

    fgmask_prev = fgmask_next

cv2.destroyAllWindows()

# PL:
# 1. średnia jest mniej więcej stała, jeśli chodzi o odchylenie standardowe to dla pedestrians jest duże
# 2. Dla zbioru pedestrian średni kierunek nie odpowiada rzeczywistości, natomiast dla highway moim zdaniem odpowiada
# 3. Odchylenie standardowe jest większe dla zbioru pedestrian, ponieważ więcej części dla obiektu porusza się w różnych stronach,
#    w przypadku highway jest mniejsze, ponieważ samochód porusza się jako zwarta bryła

# ENG:
# 1. the mean is more or less constant, the standard deviation for pedestrians is large
# 2. For the pedestrian set, the average direction does not correspond to reality, while for the highway, in my opinion, it does
# 3. The standard deviation is greater for the pedestrian set because more parts of the object move in different directions,
# for highway is smaller because the car moves as a compact body
