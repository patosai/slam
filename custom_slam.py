#!/usr/bin/env python3

import OpenGL.GL as gl

import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from src import display, fundamental, essential, plot

# Initiate ORB detector
orb = cv2.ORB_create()


def compute_orb(img):
    """find the keypoints with ORB"""
    kp = orb.detect(img, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    return kp, des


def find_initial_position(img1, img2):
    """takes the first two images and finds the relative locations of the cameras"""


img1 = cv2.imread("data/road1.jpg")
img2 = cv2.imread("data/road2.jpg")
intrinsic_camera_matrix = np.asarray([[9.842439e+02, 0.000000e+00, 6.900000e+02],
                                      [0.000000e+00, 9.808141e+02, 2.331966e+02],
                                      [0.000000e+00, 0.000000e+00, 1.000000e+00]])
image_size = [img1.shape[1], img1.shape[0]]

# find points of interest in points
img1_kp, img1_des = compute_orb(img1)
img2_kp, img2_des = compute_orb(img2)

if False:
    plot.plot_image_keypoints(img1, img1_kp)

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

if True:
    matches_to_draw = selected_matches
    img3 = cv2.drawMatches(img1,img1_kp,img2,img2_kp,matches_to_draw,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.show()

# opencv images store y coord first, then x coord
matched_points = np.asarray([[np.float32(img1_kp[match.trainIdx].pt),
                              np.float32(img2_kp[match.queryIdx].pt)]
                             for match in selected_matches])
fundamental_matrix = fundamental.calculate_fundamental_matrix(matched_points)

if False:
    matches_to_draw = selected_matches[:8]

    # draw epipolar lines on left image
    line_equations = [np.hstack(np.float32(img2_kp[match.queryIdx].pt), [1]) @ fundamental_matrix
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
    line_equations = [fundamental_matrix @ np.hstack(np.float32(img2_kp[match.queryIdx].pt), [1]).transpose()
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

essential_matrix = fundamental.fundamental_to_essential_matrix(fundamental_matrix, intrinsic_camera_matrix)
rotation, translation = essential.essential_matrix_to_rotation_translation(essential_matrix, matched_points)
camera_vector = rotation @ np.asarray([0, 0, 1])
print("camera vector after: ", camera_vector)
print("translation: ", translation)

print("my fundamental matrix")
print(fundamental_matrix)
display.setup_pangolin()

while False:#not display.should_quit():
    display.init_frame()

    # Draw Point Cloud
    points = np.random.random((100000, 3)) * 10
    gl.glPointSize(2)
    gl.glColor3f(1.0, 0.0, 0.0)
    display.draw_points(points)

    # Draw camera
    pose = np.identity(4)
    pose[:3, 3] = np.random.randn(3)
    gl.glLineWidth(1)
    gl.glColor3f(0.0, 0.0, 1.0)
    display.draw_camera(pose, 0.5, 0.75, 0.8)

    display.finish_frame()
