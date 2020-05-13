#!/usr/bin/env python3

import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from src import eight_point_algorithm, essential

# Initiate ORB detector
orb = cv2.ORB_create()


def compute_orb(img):
    """find the keypoints with ORB"""
    kp = orb.detect(img, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    return kp, des


img1 = cv2.imread("data/road1.jpg")
img2 = cv2.imread("data/road2.jpg")
intrinsic_camera_matrix = np.asarray([[9.842439e+02, 0.000000e+00, 6.900000e+02],
                                      [0.000000e+00, 9.808141e+02, 2.331966e+02],
                                      [0.000000e+00, 0.000000e+00, 1.000000e+00]])

# find points of interest in points
img1_kp, img1_des = compute_orb(img1)
img2_kp, img2_des = compute_orb(img2)

if False:
    # draw only keypoints location,not size and orientation
    img1_keypoints = cv2.drawKeypoints(img1, img1_kp, cv2.DRAW_MATCHES_FLAGS_DEFAULT, (0,255,0))
    plt.imshow(img1_keypoints)
    plt.show()

# match points in images

bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
# TODO not all the matches here are good
matches = bf.match(img1_des, img2_des)
# matches is of type DMatch and have the properties:
# - DMatch.distance - Distance between descriptors. The lower, the better it is.
# - DMatch.trainIdx - Index of the descriptor in train descriptors
# - DMatch.queryIdx - Index of the descriptor in query descriptors
# - DMatch.imgIdx - Index of the train image.
matches = sorted(matches, key = (lambda x: x.distance))
selected_matches = [match for match in matches if match.distance < 32]

if False:
    matches_to_draw = selected_matches
    img3 = cv2.drawMatches(img1,img1_kp,img2,img2_kp,matches_to_draw,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.show()

# opencv images store y coord first, then x coord
matched_points = np.asarray([[[img1_kp[match.trainIdx].pt[1], img1_kp[match.trainIdx].pt[0]],
                              [img2_kp[match.queryIdx].pt[1], img2_kp[match.queryIdx].pt[0]]]
                             for match in selected_matches])
fundamental_matrix = eight_point_algorithm.calculate_fundamental_matrix(matched_points)

if False:
    matches_to_draw = selected_matches[:8]

    # draw epipolar lines on left image
    line_equations = [np.asarray([img2_kp[match.queryIdx].pt[1], img2_kp[match.queryIdx].pt[0], 1]) @ fundamental_matrix
                      for match in matches_to_draw]
    left_img = img1.copy()
    for a, b, c in line_equations:
        # ax + by + c = 0
        # points are again in (y, x) format
        cv2.line(left_img,
                 (int(-c / b), 0),
                 (int(-(c + a * left_img.shape[1]) / b), left_img.shape[1]),
                 (255, 0, 0),
                 thickness=2)

    # draw epipolar lines on the right image
    line_equations = [fundamental_matrix @ np.asarray([img1_kp[match.trainIdx].pt[1], img1_kp[match.trainIdx].pt[0], 1]).transpose()
                      for match in matches_to_draw]
    right_img = img2.copy()
    for a, b, c in line_equations:
        cv2.line(right_img,
                 (int(-c / b), 0),
                 (int(-(c + a * right_img.shape[1]) / b), right_img.shape[1]),
                 (255, 0, 0),
                 thickness=2)

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(left_img)

    fig.add_subplot(1, 2, 2)
    plt.imshow(right_img)

    plt.show()

essential_matrix = np.transpose(intrinsic_camera_matrix) @ fundamental_matrix @ intrinsic_camera_matrix
rotation, translation = essential.essential_matrix_to_rotation_translation(essential_matrix, matched_points)
