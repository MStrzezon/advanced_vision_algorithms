import cv2
import scipy.ndimage as filters
import numpy as np
from matplotlib import pyplot as plt


def read_gray(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def normalize(img):
    img = img.astype(np.float32)
    img -= img.min()
    img /= img.max()
    return img

def H(img, mask_size):
    sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=mask_size)
    sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=mask_size)
    Ix = cv2.GaussianBlur(sobelx * sobelx, (mask_size, mask_size), 0)
    Iy = cv2.GaussianBlur(sobely * sobely, (mask_size, mask_size), 0)
    Ixy = cv2.GaussianBlur(sobelx * sobely, (mask_size, mask_size), 0)
    det = Ix * Iy - Ixy * Ixy
    trace = Ix + Iy
    H = det - 0.05 * trace * trace
    return normalize(H)


def find_max(image, size, threshold):  # size - maximum filter mask size
    data_max = filters.maximum_filter(image, size)
    maxima = (image == data_max)
    diff = image > threshold
    maxima[diff == 0] = 0
    return np.nonzero(maxima)


def plot_points(img, points):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.plot(points[1], points[0], '*')
    plt.show()

if __name__ == "__main__":

    # Fontanna

    fontanna1 = read_gray('resources/fontanna1.jpg')
    fontanna2 = read_gray('resources/fontanna2.jpg')
    fontanna1_max = find_max(H(fontanna1, 7), 7, 0.4)
    fontanna2_max = find_max(H(fontanna2, 7), 7, 0.4)

    plot_points(fontanna1, fontanna1_max)
    plot_points(fontanna2, fontanna2_max)
    #
    # # Budynek
    # budynek1 = read_gray('resources/budynek1.jpg')
    # budynek2 = read_gray('resources/budynek2.jpg')
    # budynek1_max = find_max(H(budynek1, 7), 7, 0.2)
    # budynek2_max = find_max(H(budynek2, 7), 7, 0.2)
    #
    # plot_points(budynek1, budynek1_max)
    # plot_points(budynek2, budynek2_max)
