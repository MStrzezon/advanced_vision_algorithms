import math

import cv2
import numpy as np
import scipy
from matplotlib import pyplot as plt


def gradient(im):
    dx = scipy.ndimage.convolve1d(np.int32(im), np.array([-1, 0, 1]), 1)
    dy = scipy.ndimage.convolve1d(np.int32(im), np.array([-1, 0, 1]), 0)

    magnitude = np.sqrt(dx ** 2 + dy ** 2)

    max_B = np.logical_and(magnitude[:, :, 1] < magnitude[:, :, 0],
                           magnitude[:, :, 2] < magnitude[:, :, 0])

    max_G = np.logical_and(magnitude[:, :, 0] < magnitude[:, :, 1],
                           magnitude[:, :, 2] < magnitude[:, :, 1])

    max_R = np.logical_and(magnitude[:, :, 0] < magnitude[:, :, 2],
                           magnitude[:, :, 1] < magnitude[:, :, 2])

    max_magnitude = magnitude[:, :, 2] * max_R + magnitude[:, :, 1] * max_G + magnitude[:, :, 0] * max_B

    dy_max = dy[:, :, 2] * max_R + dy[:, :, 1] * max_G + dy[:, :, 0] * max_B
    dx_max = dx[:, :, 2] * max_R + dx[:, :, 1] * max_G + dx[:, :, 0] * max_B

    orientation = np.arctan2(dy_max, dx_max)
    orientation = orientation * 180 / np.pi
    final_magnitude = np.zeros((im.shape[0], im.shape[1]))
    final_magnitude[1:-1, 1:-1] = max_magnitude[1:-1, 1:-1]
    final_orientation = np.zeros((im.shape[0], im.shape[1]))
    final_orientation[1:-1, 1:-1] = orientation[1:-1, 1:-1]
    return np.int32(final_magnitude), np.int32(final_orientation)


def histograms(image, cellSize=8):
    magnitude, orientation = gradient(image)
    YY, XX, ZZ = image.shape
    YY_cell = np.int32(YY / cellSize)
    XX_cell = np.int32(XX / cellSize)
    histogram_container = np.zeros((YY_cell, XX_cell, 9))
    for i in range(YY_cell):
        for j in range(XX_cell):
            magnitude_cell = magnitude[i * cellSize:(i + 1) * cellSize, j * cellSize:(j + 1) * cellSize]
            orientation_cell = orientation[i * cellSize:(i + 1) * cellSize, j * cellSize:(j + 1) * cellSize]
            negative_values = orientation_cell < 0
            orientation_cell[negative_values] = orientation_cell[negative_values] + 180
            for k in range(cellSize):
                for l in range(cellSize):
                    single_magnitude = magnitude_cell[k, l]
                    single_orientation = orientation_cell[k, l]
                    lower_range_number = np.int32(np.floor((single_orientation - 10) / 20))
                    upper_range_number = lower_range_number + 1
                    if lower_range_number == -1:
                        lower_range_number = 8
                    if upper_range_number == 9:
                        upper_range_number = 0
                    middleRange = np.floor((single_orientation - 10) / 20) * 20 + 10
                    d = min(abs(single_orientation - middleRange), 180 - abs(single_orientation - middleRange)) / 20

                    histogram_container[i, j, lower_range_number] = histogram_container[
                                                                        i, j, lower_range_number] + single_magnitude * (
                                                                                1 - d)
                    histogram_container[i, j, upper_range_number] = histogram_container[
                                                                        i, j, upper_range_number] + single_magnitude * (
                                                                        d)

    return histogram_normalization(histogram_container, YY_cell, XX_cell)


def histogram_normalization(hist, YY_cell, XX_cell):
    e = math.pow(0.00001, 2)
    F = []
    for jj in range(0, YY_cell - 1):
        for ii in range(0, XX_cell - 1):
            H0 = hist[jj, ii, :]
            H1 = hist[jj, ii + 1, :]
            H2 = hist[jj + 1, ii, :]
            H3 = hist[jj + 1, ii + 1, :]
            H = np.concatenate((H0, H1, H2, H3))
            n = np.linalg.norm(H)
            Hn = H / np.sqrt(math.pow(n, 2) + e)
            F = np.concatenate((F, Hn))
    return F


def hog(image, cellSize=8):
    return histograms(image, cellSize)


def HOGpicture(w, bs):  # w - histograms, bs - cell size (8)
    bim1 = np.zeros((bs, bs))
    bim1[np.round(bs // 2):np.round(bs // 2) + 1, :] = 1
    bim = np.zeros(bim1.shape + (9,))
    bim[:, :, 0] = bim1
    for i in range(0, 9):  # 2:9,
        bim[:, :, i] = scipy.misc.imrotate(bim1, -i * 20, 'nearest') / 255
    Y, X, Z = w.shape
    w[w < 0] = 0
    im = np.zeros((bs * Y, bs * X))
    for i in range(Y):
        iisl = i * bs
        iisu = (i + 1) * bs
        for j in range(X):
            jjsl = j * bs
            jjsu = (j + 1) * bs
            for k in range(9):
                im[iisl:iisu, jjsl:jjsu] += bim[:, :, k] * w[i, j, k]
    return im


if __name__ == '__main__':
    img = cv2.imread('images/train/pos/per00060.ppm')

    plt.imshow(img, 'gray')
    plt.show()

    hog = histograms(img)
