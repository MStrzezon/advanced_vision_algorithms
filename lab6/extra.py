import cv2
import matplotlib.pyplot as plt
import numpy as np

img_l = cv2.imread('resources/aloes/aloeL.jpg')
img_r = cv2.imread('resources/aloes/aloeR.jpg')

def census_transform(gray_left, gray_right, block_size, dmax):
    # rozmiar obrazów
    rows, cols = gray_left.shape
    # rozmiar bloku NxN
    N = block_size
    # połowa rozmiaru bloku
    R = N // 2
    # inicjalizacja obrazu transformacji Censusa
    census = np.zeros((rows, cols, dmax), dtype=np.uint8)

    # iteracja po każdym pikselu w obrazie
    for r in range(R, rows - R):
        for c in range(R, cols - R):
            print(r, c)
            # pobieramy blok NxN wokół piksela (r,c) z lewego obrazu
            block = np.array(gray_left[r - R:r + R + 1, c - R:c + R + 1])
            threshold = block[R, R]
            # binaryzacja bloku
            block[block <= threshold] = 0
            block[block > threshold] = 1
            # zamiana wartości binarnej na liczbową
            block = block.astype(np.uint8)
            block = block.flatten()
            # obliczenie transformacji Censusa dla bloku
            for i in range(dmax):
                if (c - i - R) < 0:
                    break
                # przesuwamy blok o d pikseli w kierunku prawo w prawym obrazie
                block2 = np.array(gray_right[r - R:r + R + 1, c - i - R:c - i + R + 1])
                threshold2 = block2[R, R]
                # binaryzacja bloku
                block2[block2 <= threshold2] = 0
                block2[block2 > threshold2] = 1
                # zamiana wartości binarnej na liczbową
                block2 = block2.astype(np.uint8)
                block2 = block2.flatten()
                # obliczenie różnic między blokami
                diff = np.bitwise_xor(block, block2)
                census[r, c, i] = np.sum(diff)

    # znajdujemy wartość d dla każdego piksela
    dmap = np.argmin(census, axis=2)
    return dmap


gray_left = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
scale_percent = 30  # percent of original size
width = int(img_l.shape[1] * scale_percent / 100)
height = int(img_l.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
img_l = cv2.resize(gray_left, dim, interpolation=cv2.INTER_AREA)
img_r = cv2.resize(gray_right, dim, interpolation=cv2.INTER_AREA)
census_result = census_transform(img_l, img_r, 5, 25)

census_result_ = census_result.astype(np.uint8)

plt.imshow(census_result_, 'gray')
plt.show()


# Block matching
# stereo_BM = cv2.StereoBM_create(numDisparities=96, blockSize=35)
# BM_nocalib = stereo_BM.compute(gray_left, gray_right)
# BM_nocalib = cv2.normalize(BM_nocalib, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
# plt.imshow(BM_nocalib, 'gray')
# plt.show()
#
#
# # Semi-global matching
# stereo_SGBM = cv2.StereoSGBM_create(numDisparities=100, blockSize=25)
# SGBM_nocalib = stereo_SGBM.compute(gray_left, gray_right)
# SGBM_nocalib = cv2.normalize(SGBM_nocalib, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
# plt.imshow(SGBM_nocalib, 'gray')
# plt.show()