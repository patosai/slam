#!/usr/bin/env python3

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
    return [keypoint.pt for keypoint in keypoints], descriptors


def match_knn(img1_descriptors, img2_descriptors, k=2):
    """Finds k nearest descriptors in img2 for each descriptor in img1 """
    img1_descriptors = np.unpackbits(np.uint8(img1_descriptors), axis=1)
    img2_descriptors = np.unpackbits(np.uint8(img2_descriptors), axis=1)
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
    img1_pts, img1_des = compute_orb(img1)
    img2_pts, img2_des = compute_orb(img2)
    matches = match_knn(img1_des, img2_des, k=2)

    point_movements = []
    good_img1_points = []
    good_img2_points = []

    for idx, (m, n) in enumerate(matches):
        # Lowe's ratio test
        first_match_much_better = m["distance"] < 0.75 * n["distance"]
        if first_match_much_better:
            img1_point = np.asarray(img1_pts[idx])
            img2_point = np.asarray(img2_pts[m["index"]])

            good_img1_points.append(img1_point)
            good_img2_points.append(img2_point)

            point_movements.append(np.linalg.norm(img1_point - img2_point))

    good_img1_points = np.asarray(good_img1_points)
    good_img2_points = np.asarray(good_img2_points)

    point_movements = [(idx, val) for idx, val in enumerate(point_movements)]
    point_movements.sort(key=lambda x: -x[1])
    point_movements = [x[0] for x in point_movements]
    top_most_moved_point_indices = point_movements[-20:]
    top_img1_points = good_img1_points[top_most_moved_point_indices]
    top_img2_points = good_img1_points[top_most_moved_point_indices]

    # Hartley's coordinate normalization
    # origin of points should be at (0, 0); average distance to center should be sqrt(2)
    img1_centroid = np.average(good_img1_points, axis=0)
    img2_centroid = np.average(good_img2_points, axis=0)

    img1_scale = math.sqrt(2) / np.average(np.linalg.norm(top_img1_points - img1_centroid, axis=1))
    img2_scale = math.sqrt(2) / np.average(np.linalg.norm(top_img2_points - img2_centroid, axis=1))

    img1_transform_matrix = np.array([[img1_scale, 0, -1*img1_scale*img1_centroid[0]],
                                      [0, img1_scale, -1*img1_scale*img1_centroid[1]],
                                      [0, 0, 1]])
    img2_transform_matrix = np.array([[img2_scale, 0, -1*img2_scale*img2_centroid[0]],
                                      [0, img2_scale, -1*img2_scale*img2_centroid[1]],
                                      [0, 0, 1]])

    scaled_top_img1_points = (top_img1_points - img1_centroid) * img1_scale
    scaled_top_img2_points = (top_img2_points - img2_centroid) * img2_scale

    # fundamental_matrix = fundamental.calculate_fundamental_matrix(top_img1_points, top_img2_points)
    fundamental_matrix = fundamental.calculate_fundamental_matrix_with_ransac(scaled_top_img1_points,
                                                                              scaled_top_img2_points,
                                                                              iterations=190)

    # unscale fundamental matrix
    fundamental_matrix = img2_transform_matrix.transpose() @ fundamental_matrix @ img1_transform_matrix

    essential_matrix = fundamental.fundamental_to_essential_matrix(fundamental_matrix, intrinsic_camera_matrix)
    rotation, translation, triangulated_points = essential.calculate_pose(essential_matrix,
                                                                          good_img1_points,
                                                                          good_img2_points,
                                                                          intrinsic_camera_matrix)
    return rotation, translation, triangulated_points


img1 = cv2.imread("data/road1.jpg")
img2 = cv2.imread("data/road2.jpg")
intrinsic_camera_matrix = np.asarray([[9.842439e+02, 0.000000e+00, 6.900000e+02],
                                      [0.000000e+00, 9.808141e+02, 2.331966e+02],
                                      [0.000000e+00, 0.000000e+00, 1.000000e+00]])

rotation, translation, points = find_initial_position(img1, img2)

np.set_printoptions(suppress=True)
print("num triangulated points", len(points))
print("camera direction", rotation @ [0, 0, 1])
print("translation", translation)

display.setup_pangolin()

while not display.should_quit():
    display.init_frame()
    camera_1_pose = np.identity(4)
    camera_2_pose = np.identity(4)
    camera_2_pose[:3, :3] = rotation
    camera_2_pose[:3, 3] = translation

    display.draw_camera(camera_1_pose, (0.0, 1.0, 0.0))
    display.draw_camera(camera_2_pose, (0.0, 1.0, 1.0))

    display.draw_points(points)

    display.finish_frame()

# img3 = cv2.drawKeypoints(img1, plot_keypoints, outImage=None, color=(0, 255, 0))
# plt.ion()
# plt.imshow(img3)
# plt.draw()
# plt.pause(0.001)
