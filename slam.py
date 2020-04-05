#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Initiate STAR detector
orb = cv2.ORB_create()
def compute_orb(img):
    # find the keypoints with ORB
    kp = orb.detect(img, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    return kp, des

img1 = cv2.imread("data/highway1.jpg")
img2 = cv2.imread("data/highway2.jpg")

# find points of interest in points

img1_kp, img1_des = compute_orb(img1)
img2_kp, img2_des = compute_orb(img2)

if False:
    # draw only keypoints location,not size and orientation
    img1_keypoints = cv2.drawKeypoints(img1, kp, cv2.DRAW_MATCHES_FLAGS_DEFAULT, (0,255,0))
    plt.imshow(img1_keypoints)
    plt.show()

# match points in images

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# TODO not all the matches here are good
matches = bf.match(img1_des, img2_des)
img3 = cv2.drawMatches(img1,img1_kp,img2,img2_kp,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
if True:
    plt.imshow(img3)
    plt.show()

matched_points_img1 = np.asarray([img1_kp[match.queryIdx].pt for match in matches])
matched_points_img2 = np.asarray([img2_kp[match.trainIdx].pt for match in matches])
essential_matrix, mask = cv2.findEssentialMat(matched_points_img1, matched_points_img2)

points, rotation, translation, mask = cv2.recoverPose(essential_matrix, matched_points_img1, matched_points_img2)
# translation is a unit vector of length 1, X pointing to right, Y pointing into image, and Z pointing upwards
