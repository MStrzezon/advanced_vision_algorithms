import cv2
import numpy as np

path = 'resources/'
image_dir = path + "aloes/"
number_of_images = 50
img_l = cv2.imread(image_dir + "aloeL.jpg")
img_r = cv2.imread(image_dir + "aloeR.JPG")
img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)


def calibrate(img_l, img_r):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
    # inner size of chessboard
    width = 9
    height = 6
    square_size = 0.025  # 0.025 meters
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height * width, 1, 3), np.float64)
    objp[:, 0, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    objp = objp * square_size  # Create real world coords. Use your metric.
    # Arrays to store object points and image points from all the images.
    objpoints_left = []  # 3d point in real world space
    objpoints_right = []  # 3d point in real world space
    imgpoints_left = []  # 2d points in image plane.
    imgpoints_right = []  # 2d points in image plane.
    img_width = 640
    img_height = 480
    image_size = (img_width, img_height)

    for i in range(1, number_of_images):
        # read image
        img_left = cv2.imread(image_dir + "left_%02d.png" % i)
        img_right = cv2.imread(image_dir + "right_%02d.png" % i)
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, (width, height), cv2.
                                                           CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.
                                                           CALIB_CB_NORMALIZE_IMAGE)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, (width, height), cv2.
                                                             CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.
                                                             CALIB_CB_NORMALIZE_IMAGE)
        Y_left, X_left, channels_left = img_left.shape
        Y_right, X_right, channels_right = img_right.shape
        # skip images where the corners of the chessboard are too close to the
        if (ret_left == True):
            minRx = corners_left[:, :, 0].min()
            maxRx = corners_left[:, :, 0].max()
            minRy = corners_left[:, :, 1].min()
            maxRy = corners_left[:, :, 1].max()
            border_threshold_x = X_left / 12
            border_threshold_y = Y_left / 12
            x_thresh_bad = False
            if (minRx < border_threshold_x):
                x_thresh_bad = True
            y_thresh_bad = False
            if (minRy < border_threshold_y):
                y_thresh_bad = True
            if (y_thresh_bad == True) or (x_thresh_bad == True):
                continue
        if ret_left == True and ret_right == True:
            objpoints_left.append(objp)
            corners2_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            imgpoints_left.append(corners2_left)
            cv2.drawChessboardCorners(img_left, (7, 6), corners2_left, ret_left)
            objpoints_right.append(objp)
            corners2_r = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
            imgpoints_right.append(corners2_r)
            cv2.drawChessboardCorners(img_right, (7, 6), corners2_r, ret_right)
    cv2.destroyAllWindows()

    def get_parameters(objpoints, imgpoints):
        N_OK = len(objpoints)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        ret, K, D, _, _ = \
            cv2.fisheye.calibrate(
                objpoints,
                imgpoints,
                image_size,
                K,
                D,
                rvecs,
                tvecs,
                calibration_flags,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )
        return K, D

    K_left, D_left = get_parameters(objpoints_left, imgpoints_left)
    K_right, D_right = get_parameters(objpoints_right, imgpoints_right)
    # Let’s rectify our results

    imgpoints_left = np.asarray(imgpoints_left, dtype=np.float64)
    imgpoints_right = np.asarray(imgpoints_right, dtype=np.float64)
    (RMS, _, _, _, _, rotationMatrix, translationVector) = cv2.fisheye.stereoCalibrate(objpoints_left, imgpoints_left,
                                                                                       imgpoints_right, K_left, D_left,
                                                                                       K_right, D_right, image_size,
                                                                                       None, None,
                                                                                       cv2.CALIB_FIX_INTRINSIC, (
                                                                                           cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                                                                                           30, 0.01))
    R2 = np.zeros([3, 3])
    P1 = np.zeros([3, 4])
    P2 = np.zeros([3, 4])
    Q = np.zeros([4, 4])
    # Rectify calibration results
    (leftRectification, rightRectification, leftProjection, rightProjection,
     dispartityToDepthMap) = cv2.fisheye.stereoRectify(K_left, D_left, K_right, D_right, image_size, rotationMatrix,
                                                       translationVector, 0, R2, P1, P2, Q, cv2.CALIB_ZERO_DISPARITY,
                                                       (0, 0), 0, 0)
    map1_left, map2_left = cv2.fisheye.initUndistortRectifyMap(K_left, D_left, leftRectification, leftProjection,
                                                               image_size, cv2.CV_16SC2)
    map1_right, map2_right = cv2.fisheye.initUndistortRectifyMap(K_right, D_right, rightRectification, rightProjection,
                                                                 image_size, cv2.CV_16SC2)
    dst_L = cv2.remap(img_l, map1_left, map2_left, cv2.INTER_LINEAR)
    dst_R = cv2.remap(img_r, map1_right, map2_right, cv2.INTER_LINEAR)

    dst_L = cv2.cvtColor(dst_L, cv2.COLOR_BGR2GRAY)
    dst_R = cv2.cvtColor(dst_R, cv2.COLOR_BGR2GRAY)
    return dst_L, dst_R


# Block matching
stereo_BM = cv2.StereoBM_create(numDisparities=128, blockSize=15)
BM_nocalib = stereo_BM.compute(img_l, img_r)
# dst_L, dst_R = calibrate(img_l, img_r)
# BM_calib = stereo_BM.compute(dst_L, dst_R)
BM_nocalib = cv2.normalize(BM_nocalib, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
# BM_calib = cv2.normalize(BM_calib, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
cv2.imshow('no_calib', BM_nocalib)
# cv2.imshow('calib', BM_calib)
cv2.waitKey(0)
cv2.destroyAllWindows()
# plt.imshow(BM_nocalib, 'gray')
# plt.show()


# Semi-global matching
stereo_SGBM = cv2.StereoSGBM_create(numDisparities=100, blockSize=15)
SGBM_nocalib = stereo_SGBM.compute(img_l, img_r)
# dst_L, dst_R = calibrate(img_l, img_r)
# SGBM_calib = stereo_SGBM.compute(dst_L, dst_R)
SGBM_nocalib = cv2.normalize(SGBM_nocalib, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
# SGBM_calib = cv2.normalize(SGBM_calib, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
cv2.imshow('no_calib', SGBM_nocalib)
# cv2.imshow('calib', SGBM_calib)
cv2.waitKey(0)
cv2.destroyAllWindows()
# plt.imshow(SGBM_nocalib, cmap='gray')
# plt.show()
