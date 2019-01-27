# script which outputs a ray in 3D space from an image point, from a given vehicle pose and camera calibration

import cv2
import numpy as np
import matplotlib.pyplot as plt




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
dist = np.array([0.15808590, 0.76137626, 0.00569993, -0.00067913])
cam = np.array([[1736.49233331, 0.00000000, 687.23531391],[0.00000000, 1733.74525406, 501.08026641],[0.00000000, 0.00000000, 1.00000000]])

# point(s) of interest
u,v = 10,10
pt = np.array([[[u,v]]], dtype=np.float32)

# undistort poi(s)
pt_undist = cv2.undistortPoints(pt,cam,dist, P=cam)
# print(pt_undist)


# im = cv2.imread('/home/nader/PR_20090320_061617_702_LC16.pgm', -1)#'/home/nader/SWAP_storage/TAS_LOBS/FRAMES/PR_20100604_084020_800_LC16.png')
# cv2.circle(im,(int(u),int(v)),5,(0,0,255),-1)
# plt.imshow(im)
# plt.show()
# im_undist = cv2.undistort(im,cam,dist)
#
#
#
# im2 = np.concatenate((im,im_undist), axis=1)
# plt.imshow(im2)
# plt.show()
# cv2.imwrite('im.png', im2)

# extract vehicle translation and rotation (from stereo_pose_est.data file)
# pose 18776 	1275640820.8008968830108643 	-43.6173105794598683 	147.2208324550808243 	-90.0693583031992944 	-788.1200980122725923 	69.1765022273584265 	-0.0051188729680123 	-0.0125153269463330 	-2.8263936263024938 	PR_20100604_084020_800_LC16.png 	PR_20100604_084020_800_RM16.png 	2.2570000000000001 	0.9545548668309964 	0
#      poseID, timestamp,                   lat                 long                    X  (north)            Y (east)          Z (depth)           X euler (rad)       Y-euler             Z-euler
pose = [18776,1275640820.8008968830108643,-43.6173105794598683,147.2208324550808243,-90.0693583031992944,-788.1200980122725923,69.1765022273584265,-0.0051188729680123,-0.0125153269463330,-2.8263936263024938]#,'PR_20100604_084020_800_LC16.png 	PR_20100604_084020_800_RM16.png 	2.2570000000000001 	0.9545548668309964 	0]
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

R = np.matmul(np.matmul(R1,R2), R3)

# project point of interest into 3D (cam-1 * pt)
hom_pt = np.array([pt_undist[0][0][0], pt_undist[0][0][1], 1])
hom_pt = np.matmul(np.linalg.inv(cam),hom_pt)

# transform ray from camera to vehicle reference frame (using rotation matrix R)
direction = np.matmul(R,hom_pt)
origin = trns

print("dir: ", np.shape(direction), '\n', "origin: ", np.shape(origin))
print("dir: ", direction, '\n', "origin: ", origin)