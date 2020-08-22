#!/usr/bin/env python3

import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from src import display, plot


intrinsic_camera_matrix = np.asarray([[9.842439e+02, 0.000000e+00, 6.900000e+02],
                                      [0.000000e+00, 9.808141e+02, 2.331966e+02],
                                      [0.000000e+00, 0.000000e+00, 1.000000e+00]])


# Initiate ORB detector
orb = cv2.ORB_create()


def compute_orb(img):
    """find the keypoints with ORB"""
    kp = orb.detect(img, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    return kp, des


# key: point in image frame; tuple of (frame_num, 2D X, 2D Y)
# value: triangulated point; tuple of (X, Y, Z)
TRIANGULATED_POINT_LOOKUP = {}

# index: frame num
# value: 3x4 camera pose matrix
CAMERA_POSES = []


def find_initial_position(img1, img2):
    """takes the first two images and finds the relative locations of the cameras.
    the first camera is assumed to be at [0, 0, 0]"""
    # find points of interest in points
    img1_kp, img1_des = compute_orb(img1)
    img2_kp, img2_des = compute_orb(img2)

    # get closest 2 matches per point
    bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING)
    matches = bf.knnMatch(img1_des, img2_des, k=2)

    good_matches = []
    pts1 = []
    pts2 = []
    # Lowe's ratio test
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append(m)
            pts1.append(img1_kp[m.queryIdx].pt)
            pts2.append(img2_kp[m.trainIdx].pt)

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    # essential matrix gives the motion of the points
    # to get motion of the camera, flip the inputs between pts1 and pts2
    essential_matrix, e_mask = cv2.findEssentialMat(pts2, pts1, intrinsic_camera_matrix)

    # select only inlier points as per the RANSAC method
    pts1 = pts1[e_mask.ravel() == 1]
    pts2 = pts2[e_mask.ravel() == 1]

    _, rotation, translation, mask, triangulated_points = cv2.recoverPose(essential_matrix, pts2, pts1, intrinsic_camera_matrix, distanceThresh=50)
    triangulated_points = np.asarray([np.divide(triangulated_points[0], triangulated_points[3]),
                                      np.divide(triangulated_points[1], triangulated_points[3]),
                                      np.divide(triangulated_points[2], triangulated_points[3])]).transpose()

    CAMERA_POSES.clear()
    CAMERA_POSES.append(np.hstack((np.identity(3), np.array([[0], [0], [0]]))))
    CAMERA_POSES.append(np.hstack((rotation, translation)))
    return rotation, translation, triangulated_points


def pangolin_draw(points):
    """points is a Nx3 matrix"""
    display.setup()

    while not display.should_quit():
        display.init_frame()
        for idx, pose in enumerate(CAMERA_POSES):
            is_last_pose = idx == len(CAMERA_POSES) - 1
            homogenous_pose = np.vstack((pose, [0, 0, 0, 1]))
            display.draw_camera(homogenous_pose, (0.0, 1.0, 1.0 if is_last_pose else 0.0))

        display.draw_points(points)

        display.finish_frame()


if __name__ == "__main__":
    plt.ion()

    intrinsic_camera_matrix = np.asarray([[1000, 0.000000e+00, 640/2],
                                          [0.000000e+00, 1000, 360/2],
                                          [0.000000e+00, 0.000000e+00, 1.000000e+00]])
    def get_next_frame(cap):
        for _ in range(10):
            cap.grab()
        ret, img = cap.retrieve()
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return ret, grayscale


    cap = cv2.VideoCapture('data/sanfrancisco-cut.mp4')
    ret, img0 = get_next_frame(cap)
    ret, img1 = get_next_frame(cap)
    rotation, translation, triangulated_points = find_initial_position(img0, img1)

    pangolin_draw(triangulated_points)

    # while cap.isOpened():
    #     img0 = img1
    #     ret, image = get_next_frame(cap)
    #     img1 = image
    #     rotation, translation, triangulated_points = find_initial_position(img0, img1)
    #     pangolin_draw(triangulated_points)
    #     plt.draw()
    #     plt.pause(0.1)
    # cap.release()

    while True:
        plt.draw()
        plt.pause(0.1)
