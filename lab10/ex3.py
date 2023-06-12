import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import hamming
from lab10.ex1 import H, read_gray, plot_points
from lab10.utils.pm import plot_matches


def non_maximal_suppresion(keypoints):
    keypoints_supressed = np.zeros(keypoints.shape, dtype=np.float32)
    for i in range(1, keypoints.shape[0] - 1):
        for j in range(1, keypoints.shape[1] - 1):
            roi = keypoints[i - 1:i + 2, j - 1:j + 2]
            if roi[1, 1] == 0:
                continue
            ind = np.unravel_index(np.argmax(roi, axis=None), roi.shape)
            maximum_value = np.amax(roi)
            keypoints_supressed[i - 1 + ind[0], j - 1 + ind[1]] = maximum_value
    return keypoints_supressed

def get_keypoints_with_31_neighbors(keypoints):
    result = []
    for i in range(31, keypoints.shape[0] - 31):
        for j in range(31, keypoints.shape[1] - 31):
            if keypoints[i, j] == 0:
                continue
            result.append(((i, j), keypoints[i, j]))
    return result

def sort_keypoints(keypoints):
    return sorted(keypoints, key=lambda x: x[1], reverse=True)

def add_centroid_and_orientation(image, keypoints):
    all_keypoints_info = []
    for keypoint in keypoints:
        m00 = 0
        m10 = 0
        m01 = 0
        m11 = 0
        index = keypoint[0]
        for i in range(-3, 3):
            for j in range(-3, 3):
                if round(np.sqrt(i ** 2 + j ** 2)) > 3:
                    continue
                m00 += image[index[0] + i, index[1] + j]
                m10 += (index[0] + i) * image[index[0] + i, index[1] + j]
                m01 += (index[1] + j) * image[index[0] + i, index[1] + j]
                m11 += (index[0] + i) * (index[1] + j) * image[index[0] + i, index[1] + j]
        x = m10 / m00
        y = m01 / m00
        c = (x, y)
        theta = np.arctan2(m01, m10)
        all_keypoints_info.append((keypoint[0], keypoint[1], c, theta))
    return all_keypoints_info


def brief_descriptor(image, keypoints):
    positions = np.loadtxt("resources/orb_descriptor_positions.txt").astype(np.int8)
    positions_0 = positions[:, :2]
    positions_1 = positions[:, 2:]
    image = cv2.GaussianBlur(image, (5, 5), 0)

    descriptor = np.zeros((len(keypoints), 256), dtype=np.uint16)
    for i, keypoint in enumerate(keypoints):
        x, y = keypoint[0]
        harris_value = keypoint[1]
        c = keypoint[2]
        theta = keypoint[3]
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        # Wykonanie testu binarnego dla par punktów
        for j in range(256):
            position_row_0 = positions_0[j, 0]
            position_col_0 = positions_0[j, 1]
            position_row_1 = positions_1[j, 0]
            position_col_1 = positions_1[j, 1]
            position_row_0_rotated = int(np.round(position_row_0 * sin_theta + position_col_0 * cos_theta))
            position_col_0_rotated = int(np.round(position_col_0 * cos_theta - position_row_0 * sin_theta))
            position_row_1_rotated = int(np.round(position_row_1 * sin_theta + position_col_1 * cos_theta))
            position_col_1_rotated = int(np.round(position_col_1 * cos_theta - position_row_1 * sin_theta))
            if image[y + position_row_0_rotated, x + position_col_0_rotated] < image[y + position_row_1_rotated, x + position_col_1_rotated]:
                descriptor[i, j] = 1

    return keypoints, descriptor

def get_hamming_distance(arr1, arr2):
    return hamming(arr1, arr2) * len(arr1)

def fast_detector_harris(image):
    threshold = 40
    n = 3

    k = 0.05
    window_size = 3

    keypoints = np.zeros(image.shape, dtype=np.float32)
    harris_matrix = H(image, 7)
    for i in range(n, image.shape[0] - n):
        for j in range(n, image.shape[1] - n):
            center_pixel = image[i, j]
            consecutive_pixels = []

            for dx, dy in [(0, n), (n, 0), (0, -n), (-n, 0), (n, n), (-n, -n), (n, -n), (-n, n)]:
                if abs(int(center_pixel) - image[i + dy, j + dx]) > threshold:
                    consecutive_pixels.append(True)
                else:
                    consecutive_pixels.append(False)

            if sum(consecutive_pixels) >= 7:
                keypoints[i, j] = harris_matrix[i, j]

    return keypoints


def orb(image):
    keypoints = fast_detector_harris(image)
    supressed_keypoints = non_maximal_suppresion(keypoints)
    keypoints_with_31_neighbors = get_keypoints_with_31_neighbors(supressed_keypoints)
    keypoints_with_31_neighbors.sort(key=lambda x: x[1], reverse=True)
    keypoints = keypoints_with_31_neighbors[:200]
    fontanna1_max = [[point[0][0] for point in keypoints], [point[0][1] for point in keypoints]]
    plot_points(image, fontanna1_max)

    keypoints_with_centroid_and_orientation = add_centroid_and_orientation(image, keypoints)
    return brief_descriptor(image, keypoints_with_centroid_and_orientation)

def get_matches(descriptor1, descriptor2):
    matches = []
    keypoinst1, descriptor1 = descriptor1
    keypoinst2, descriptor2 = descriptor2
    for i, desc1 in enumerate(descriptor1):
        matches_for_pt = []
        for j, desc2 in enumerate(descriptor2):
            matches_for_pt.append((get_hamming_distance(desc1, desc2), keypoinst2[j][0]))
        matches_for_pt.sort(key=lambda x: x[0])
        matches.append((matches_for_pt[0][0], keypoinst1[i][0], matches_for_pt[0][1]))
    matches.sort(key=lambda x: x[0])
    return matches[:20]

if __name__ == "__main__":
    fontanna1 = read_gray('resources/fontanna1.jpg')
    fontanna2 = read_gray('resources/fontanna2.jpg')
    fontanna1_orb = orb(fontanna1)
    fontanna2_orb = orb(fontanna2)
    matches = get_matches(fontanna1_orb, fontanna2_orb)
    plot_matches(fontanna1, fontanna2, matches)
    plt.show()
