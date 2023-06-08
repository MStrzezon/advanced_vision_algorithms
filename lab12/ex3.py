import cv2
import numpy as np
from joblib import load

from lab12.ex1 import hog

clf = load('svm.joblib')
for k in range(1, 2):
    image = cv2.imread('images/test/testImage%d.png' % k)
    image = cv2.resize(image, (int(image.shape[1] * 0.8), int(image.shape[0] * 0.8)))
    result = np.copy(image)

    cv2.rectangle(image, (0, 0), (64, 128), (0, 255, 0), 2)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    clip_cols = 64
    clip_rows = 128
    img_rows, img_cols, _ = image.shape
    for i in range(0, img_cols-clip_cols, 16):
        for j in range(0, img_rows - clip_rows, 16):
            print(i, j)
            clip = image[j:j+clip_rows, i:i+clip_cols]
            hisogram_hog = hog(clip)
            if clf.predict(hisogram_hog.reshape(1, -1)) == 1:
                cv2.rectangle(result,(i, j), (i+clip_cols, j+clip_rows), (0, 255, 0), 2)
    cv2.imshow('result', result)
    cv2.waitKey(0)

