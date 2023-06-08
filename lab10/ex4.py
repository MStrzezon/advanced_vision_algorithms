import cv2
import numpy as np
from matplotlib import pyplot as plt

from lab10.ex1 import read_gray

left_panorama = read_gray('resources/left_panorama.jpg')
right_panorama = read_gray('resources/right_panorama.jpg')


def show_characteristic_points(img):
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(img, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    # draw only keypoints location,not size and orientation
    img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
    plt.imshow(img)
    plt.show()


def match(img1, img2):
    orb = cv2.ORB_create()
    keypointsL, desL = orb.detectAndCompute(img1, None)
    keypointsR, desR = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desL, desR)
    matches = sorted(matches, key=lambda x: x.distance)
    best_matches = matches[:10]
    # matched_image = cv2.drawMatches(img1,
    #                                 kp1, img2, kp2, best_matches, None,
    #                                 matchColor=(0, 255, 0), matchesMask=None,
    #                                 singlePointColor=(255, 0, 0), flags=0)
    # plt.imshow(matched_image)
    # plt.show()
    keypointsL = np.float32([kp.pt for kp in keypointsL])
    keypointsR = np.float32([kp.pt for kp in keypointsR])
    ptsA = np.float32([keypointsL[m.queryIdx] for m in matches])
    ptsB = np.float32([keypointsR[m.trainIdx] for m in matches])
    H, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 5.0)
    width = img1.shape[1] + img2.shape[1]
    height = img1.shape[0] + img2.shape[0]
    result = cv2.warpPerspective(img1, H, (width, height))
    result[0:img2.shape[0], 0:img2.shape[1]] = img2
    plt.imshow(result, cmap='gray')
    plt.show()

# show_characteristic_points(left_panorama)
# show_characteristic_points(right_panorama)
match(left_panorama, right_panorama)