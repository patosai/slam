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

img1_kp, img1_des = compute_orb(img1)
img2_kp, img2_des = compute_orb(img2)

# draw only keypoints location,not size and orientation
#img1_keypoints = cv2.drawKeypoints(img1, kp, cv2.DRAW_MATCHES_FLAGS_DEFAULT, (0,255,0))
#plt.imshow(img1_keypoints)
#plt.show() # not needed for Jupyter notebook

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(img1_des, img2_des)
img3 = cv2.drawMatches(img1,img1_kp,img2,img2_kp,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3)
plt.show()
