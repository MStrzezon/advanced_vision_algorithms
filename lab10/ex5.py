import cv2

def test_shift(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()

    img1_kp, img1_des = sift.detectAndCompute(img1, None)
    img2_kp, img2_des = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(img1_des, img2_des, k=2)
    best_matches = [[m] for m, n in matches if m.distance < 0.2 * n.distance]
    matched_image = cv2.drawMatchesKnn(img1, img1_kp, img2, img2_kp, best_matches, None, flags=2)
    cv2.imshow('matched', matched_image)
    cv2.waitKey(0)


def test_fontanna_normal():
    fontanna1 = cv2.imread('resources/fontanna1.jpg', cv2.IMREAD_GRAYSCALE)
    fontanna2 = cv2.imread('resources/fontanna2.jpg', cv2.IMREAD_GRAYSCALE)
    test_shift(fontanna1, fontanna2)

def test_fontanna_pow():
    fontanna1 = cv2.imread('resources/fontanna1.jpg', cv2.IMREAD_GRAYSCALE)
    fontanna_pow = cv2.imread('resources/fontanna_pow.jpg', cv2.IMREAD_GRAYSCALE)
    test_shift(fontanna1, fontanna_pow)

if __name__ == '__main__':
    test_fontanna_pow()