import numpy as np
import cv2
import glob
import pickle


objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

objpoints = []  # 3d points in real world space
imgpoints = []  # 2d pints in image plane

# List of calibration images
images = glob.glob('./camera_cal/calibration*.jpg')

for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    # if found, add object points, image points
    if ret == True:
        print('Working on', fname)
        objpoints.append(objp)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        write_name = './corners_found/corners found' + str(idx)+".jpg"
        cv2.imwrite(write_name, img)

# Load image for reference
img = cv2.imread('./camera_cal/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_size, None, None)

dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open("./calibration_pickle.p", "wb"))

# Undistort example
img = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('./corners_found/undistort.jpg', img)
