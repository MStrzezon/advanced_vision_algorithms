import cv2
import numpy as np

img = cv2.imread("resources/trybik.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh, bin_image = cv2.threshold(gray_img, 230, 255, cv2.THRESH_BINARY)
not_image = cv2.bitwise_not(bin_image)

contours, hierarchy = cv2.findContours(not_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img, contours, 0, (0, 255, 0), 2)
cv2.imshow("Contours", img)
cv2.waitKey(0)

sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)

grad_amplitude = np.sqrt(np.square(sobelx) + np.square(sobely))
grad_amplitude = grad_amplitude / np.amax(grad_amplitude)
grad_orientation = np.arctan2(sobely, sobelx)

moments = cv2.moments(not_image, 1)
cX = moments['m10'] / moments['m00']
cY = moments['m01'] / moments['m00']
# cv2.circle(img, (int(cX), int(cY)), 2, (0, 0, 255))
# cv2.imshow("Center", img)
# cv2.waitKey(0)

Rtable = [[] for _ in range(360)]
for i in contours[0]:
    v_length = np.sqrt(np.square(i[0][0] - cX) + np.square(i[0][1] - cY))
    v_angle = np.arctan2(i[0][1] - cY, i[0][0] - cX)
    v_angle = v_angle * 180 / np.pi + 180
    grad_orientation_in_point = grad_orientation[i[0][0], i[0][1]] * 180 / np.pi + 180
    index = round(grad_orientation_in_point)
    if index == 360:
        index = 0
    Rtable[index].append([v_length, v_angle])

##################################################################################
# trybiki2
img_trybiki2 = cv2.imread("resources/trybiki2.jpg")
gray_trybiki2 = cv2.cvtColor(img_trybiki2, cv2.COLOR_BGR2GRAY)

sobelx_trybiki2 = cv2.Sobel(gray_trybiki2, cv2.CV_64F, 1, 0, ksize=5)
sobely_trybiki2 = cv2.Sobel(gray_trybiki2, cv2.CV_64F, 0, 1, ksize=5)

grad_amplitude_trybiki_2 = np.sqrt(np.square(sobelx_trybiki2) + np.square(sobely_trybiki2))
grad_amplitude_trybiki_2 = grad_amplitude_trybiki_2 / np.amax(grad_amplitude_trybiki_2)

grad_orientation_trybiki_2 = np.arctan2(sobely_trybiki2, sobelx_trybiki2)

height, width = grad_amplitude_trybiki_2.shape

accumulator = np.zeros([2 * height, 2 * width])

for x in range(height):
    for y in range(width):
        if grad_amplitude_trybiki_2[x, y] > 0.5:
            orient = grad_orientation_trybiki_2[x, y]
            orient_degree = orient * 180 / np.pi + 180
            index = round(orient_degree)
            if index == 360:
                index = 0
            for k in Rtable[index]:
                v_length = k[0]
                v_angle = k[1]
                x1 = (v_length * np.cos(v_angle * np.pi / 180)) + x
                y1 = (v_length * np.sin(v_angle * np.pi / 180)) + y
                accumulator[round(y1), round(x1)] += 1
accumulator = accumulator / np.amax(accumulator) * 255
cv2.imshow("Accumulator", accumulator)
cv2.waitKey(0)

M = np.where(accumulator == np.amax(accumulator))
cv2.circle(img_trybiki2, (int(M[0]), int(M[1])), 2, (0, 0, 255))
dx = cX - int(M[0])
dy = int(M[1]) - cY
cv2.drawContours(img_trybiki2, contours, -1, (0, 255, 0), offset=(int(-dx), int(dy)))
cv2.imshow("Contours", img_trybiki2)
cv2.waitKey(0)
