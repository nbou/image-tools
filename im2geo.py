# script which outputs a ray in 3D space from an image point, from a given vehicle pose and camera calibration

import cv2
import numpy as np
import matplotlib.pyplot as plt
import calParser

# function to give the ray origin and direction, takes calibration filepath and line from readInf in the format
#  [index, imname(no filetype), box midpoint)]
def im2geo(calfile, posefile, line):
    cam,dist = calParser.parseCal(calfile)
    # cam,dist = calParser.parseCal('/home/nader/scratch/ng2_scaled_baseline.calib')

    # point(s) of interest
    # u,v = 1360/2,1024/2
    u,v = line[2]
    pt = np.array([[[u,v]]], dtype=np.float32)

    # undistort poi(s)
    pt_undist = cv2.undistortPoints(pt,cam,dist, P=cam)
    # print(pt_undist)


    posepth = posefile#'/home/nader/scratch/stereo_pose_est.data'
    poseim = line[1] #'PR_20100604_080817_570_LC16'
    pose = calParser.parsePose(posepth,poseim)
    # print(np.shape(pose))
    trns = np.array(pose[4:7])
    angs = np.array(pose[7:11])

    # Generate rotation matrix, R, from euler angles in stereo_pose file (stored in angs)
    R1 = np.array([[np.cos(angs[0]), np.sin(angs[0]), 0],
                   [-np.sin(angs[0]), np.cos(angs[0]),0],
                   [0,                0,              1]])

    R2 = np.array([[1,  0,               0],
                   [0,  np.cos(angs[1]), np.sin(angs[1])],
                   [0, -np.sin(angs[1]), np.cos(angs[1])]])

    R3 = np.array([[ np.cos(angs[2]), np.sin(angs[2]), 0],
                   [-np.sin(angs[2]), np.cos(angs[2]), 0],
                   [0,                0,               1]])

    # R = np.matmul(np.matmul(R1,R2), R3)
    R = np.matmul(np.matmul(R3,R2),R1)
    # project point of interest into 3D (cam-1 * pt)
    hom_pt = np.array([pt_undist[0][0][0], pt_undist[0][0][1], 1])
    hom_pt = np.matmul(np.linalg.inv(cam),hom_pt)

    # transform ray from camera to vehicle reference frame (using rotation matrix R)
    direction = np.matmul(R,hom_pt)
    origin = trns

    # print("dir: ", np.shape(direction), '\n', "origin: ", np.shape(origin))
    # print("dir: ", direction, '\n', "origin: ", origin)
    return direction,origin

# Example .calib file
# 1360 1024
#
# Cam1_mat
# 1736.49233331 0.00000000 687.23531391
# 0.00000000 1733.74525406 501.08026641
# 0.00000000 0.00000000 1.00000000
#
# Cam 1 dist
# 0.15808590 0.76137626 0.00569993 -0.00067913
#
# Cam1 transformation
# 1.00000000 0.00000000 0.00000000 0.00000000
# 1.00000000 0.00000000 0.00000000 0.00000000
# 1.00000000 0.00000000 0.00000000 0.00000000
#
#
# 1360 1024
# 1738.62666794 0.00000000 682.48926624
# 0.00000000 1736.66673076 510.35526868
# 0.00000000 0.00000000 1.00000000
#
# 0.17427338 0.66559118 0.00355058 -0.00255854
#
# 0.99996769 0.00018500 0.00803655 -0.00019401
# 0.99999935 0.00112026 -0.00803633 -0.00112178
# 0.99996708 -0.06995887 -0.00023571 0.00052106


# camera distortion parameters and intrinsic matrix (from calibration file)
# dist = np.array([0.15808590, 0.76137626, 0.00569993, -0.00067913])
# cam = np.array([[1736.49233331, 0.00000000, 687.23531391],[0.00000000, 1733.74525406, 501.08026641],[0.00000000, 0.00000000, 1.00000000]])