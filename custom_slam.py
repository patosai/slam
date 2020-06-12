#!/usr/bin/env python3

import OpenGL.GL as gl

import cv2
import matplotlib
matplotlib.use('TkAgg')
import math
import matplotlib.pyplot as plt
import numpy as np

from src import display, fundamental, essential, plot

# Initiate ORB detector
orb = cv2.ORB_create()


def compute_orb(img):
    """find the keypoints with ORB"""
    keypoints = orb.detect(img, None)
    # compute the descriptors with ORB
    keypoints, descriptors = orb.compute(img, keypoints)
    return keypoints, descriptors


def match_knn(img1_descriptors, img2_descriptors, k=2):
    """Finds k nearest descriptors in img2 for each descriptor in img1 """
    matches = []
    for img1_idx, img1_descriptor in enumerate(img1_descriptors):
        possible_matches = []
        for img2_idx, img2_descriptor in enumerate(img2_descriptors):
            if img1_idx != img2_idx:
                hamming_distance = np.count_nonzero(img1_descriptor != img2_descriptor)
                possible_matches.append({"distance": hamming_distance, "index": img2_idx})
        possible_matches.sort(key=lambda x: x["distance"])
        limited_matches = possible_matches[:k]
        matches.append(limited_matches)
    return matches


def find_initial_position(img1, img2):
    """takes the first two images and finds the relative locations of the cameras"""
    img1_kp, img1_des = compute_orb(img1)
    img2_kp, img2_des = compute_orb(img2)
    matches = match_knn(img1_des, img2_des, k=2)
    good_img1_points = []
    good_img2_points = []
    for idx, (m, n) in enumerate(matches):
        # Lowe's ratio test
        first_match_much_better = m["distance"] < 0.7 * n["distance"]
        if first_match_much_better:
            img1_point = np.asarray(img1_kp[idx].pt)
            img2_point = np.asarray(img2_kp[m["index"]].pt)
            # my own test to choose points that move a lot
            # more parallax helps reduce off-by-one-pixel errors
            point_moved_a_lot = np.linalg.norm(img1_point - img2_point) > 10
            if point_moved_a_lot:
                good_img1_points.append(img1_point)
                good_img2_points.append(img2_point)
    good_img1_points = np.asarray(good_img1_points)
    good_img2_points = np.asarray(good_img2_points)

    # Hartley's coordinate normalization
    # origin of points should be at (0, 0); average distance to center should be sqrt(2)
    img1_centroid = np.average(good_img1_points, axis=0)
    img2_centroid = np.average(good_img2_points, axis=0)

    img1_scale = math.sqrt(2) / np.average(np.linalg.norm(good_img1_points - img1_centroid, axis=1))
    img2_scale = math.sqrt(2) / np.average(np.linalg.norm(good_img2_points - img2_centroid, axis=1))

    img1_transform_matrix = np.array([[img1_scale, 0, -1*img1_scale*img1_centroid[0]],
                                      [0, img1_scale, -1*img1_scale*img1_centroid[1]],
                                      [0, 0, 1]])
    img2_transform_matrix = np.array([[img2_scale, 0, -1*img2_scale*img2_centroid[0]],
                                      [0, img2_scale, -1*img2_scale*img2_centroid[1]],
                                      [0, 0, 1]])

    good_img1_points = (good_img1_points - img1_centroid) * img1_scale
    good_img2_points = (good_img2_points - img2_centroid) * img2_scale

    # fundamental_matrix = fundamental.calculate_fundamental_matrix(good_img1_points, good_img2_points)
    fundamental_matrix = fundamental.calculate_fundamental_matrix_with_ransac(good_img1_points, good_img2_points)

    # unscale fundamental matrix
    fundamental_matrix = img2_transform_matrix.transpose() @ fundamental_matrix @ img1_transform_matrix

    essential_matrix = fundamental.fundamental_to_essential_matrix(fundamental_matrix, intrinsic_camera_matrix)
    rotation, translation = essential.essential_matrix_to_rotation_translation(essential_matrix, good_img1_points, good_img2_points)


img1 = cv2.imread("data/road1.jpg")
img2 = cv2.imread("data/road2.jpg")
intrinsic_camera_matrix = np.asarray([[9.842439e+02, 0.000000e+00, 6.900000e+02],
                                      [0.000000e+00, 9.808141e+02, 2.331966e+02],
                                      [0.000000e+00, 0.000000e+00, 1.000000e+00]])

find_initial_position(img1, img2)

