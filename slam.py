#!/usr/bin/env python3

import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from src import eight_point_algorithm

# Initiate ORB detector
orb = cv2.ORB_create()


def compute_orb(img):
    """find the keypoints with ORB"""
    kp = orb.detect(img, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    return kp, des


img1 = cv2.imread("data/img1.png")
img2 = cv2.imread("data/img2.png")

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
# matches is of type DMatch and have the properties:
# - DMatch.distance - Distance between descriptors. The lower, the better it is.
# - DMatch.trainIdx - Index of the descriptor in train descriptors
# - DMatch.queryIdx - Index of the descriptor in query descriptors
# - DMatch.imgIdx - Index of the train image.
matches = sorted(matches, key = (lambda x: x.distance))

if True:
    matches_to_draw = matches
    img3 = cv2.drawMatches(img1,img1_kp,img2,img2_kp,matches_to_draw,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.show()

selected_matches = matches
matched_points = [[[img1_kp[match.trainIdx].pt[1], img1_kp[match.trainIdx].pt[0]],
                   [img2_kp[match.queryIdx].pt[1], img2_kp[match.queryIdx].pt[0]]]
                  for match in selected_matches]
E = eight_point_algorithm.calculate_essential_matrix(matched_points)
print(E)
singular_values = np.linalg.svd(E, compute_uv=False)
print(singular_values)
