#!/usr/bin/env python3

import cv2
import matplotlib
matplotlib.use('TkAgg')
import math
import matplotlib.pyplot as plt
import numpy as np
import threading
import time


from src import display, fundamental, essential, plot

# Initiate ORB detector
orb = cv2.ORB_create()


def compute_orb(img):
    """find the keypoints with ORB"""
    keypoints = orb.detect(img, None)
    # compute the descriptors with ORB
    keypoints, descriptors = orb.compute(img, keypoints)
    descriptor_bits = np.unpackbits(np.uint8(descriptors), axis=1)
    return [keypoint.pt for keypoint in keypoints], descriptor_bits


def match_knn(img0_descriptors, img1_descriptors, k=2):
    """Finds k nearest descriptors in img1 for each descriptor in img0 """
    matches = []
    for img0_idx, img0_descriptor in enumerate(img0_descriptors):
        possible_matches = []
        for img1_idx, img1_descriptor in enumerate(img1_descriptors):
            if img0_idx != img1_idx:
                hamming_distance = np.count_nonzero(img0_descriptor != img1_descriptor)
                possible_matches.append({"distance": hamming_distance, "index": img1_idx})
        possible_matches.sort(key=lambda x: x["distance"])
        limited_matches = possible_matches[:k]
        matches.append(limited_matches)
    return matches


def find_matches_between_images(img0, img1, show_keypoints=False):
    img0_pts, img0_des = compute_orb(img0)
    img1_pts, img1_des = compute_orb(img1)
    matches = match_knn(img0_des, img1_des, k=2)

    point_movements = []
    good_img0_points = []
    good_img1_points = []

    for idx, (m, n) in enumerate(matches):
        # Lowe's ratio test
        first_match_much_better = m["distance"] < 0.75 * n["distance"]
        if first_match_much_better:
            img0_point = np.asarray(img0_pts[idx])
            img1_point = np.asarray(img1_pts[m["index"]])

            good_img0_points.append(img0_point)
            good_img1_points.append(img1_point)

            point_movements.append(np.linalg.norm(img0_point - img1_point))

    good_img0_points = np.asarray(good_img0_points)
    good_img1_points = np.asarray(good_img1_points)

    if show_keypoints:
        fig, ax = plt.subplots()
        keypoints_image = img1.copy()
        for point in good_img1_points:
            circle = plt.Circle(point, radius=5, color='#00FF00', fill=False)
            ax.add_artist(circle)
        plt.imshow(keypoints_image)

    point_movements = [{"index": idx, "movement": val} for idx, val in enumerate(point_movements)]
    point_movements.sort(key=lambda x: -x["movement"])
    point_movements = [x["index"] for x in point_movements]
    top_most_moved_point_indices = point_movements[-20:]

    return [good_img0_points, good_img1_points], top_most_moved_point_indices


def find_initial_position(img0, img1, show_keypoints=False):
    """takes the first two images and finds the relative locations of the cameras"""
    good_points, top_point_indices = find_matches_between_images(img0, img1, show_keypoints=show_keypoints)
    top_points = [image_points[top_point_indices] for image_points in good_points]

    # Hartley's coordinate normalization
    # origin of points should be at (0, 0); average distance to center should be sqrt(2)
    centroids = [np.average(points, axis=0)
                 for points in good_points]
    scale = [math.sqrt(2) / np.average(np.linalg.norm(good_points[idx] - centroids[idx], axis=1))
             for idx in range(len(good_points))]

    transform_matrices = [np.array([[scale[idx], 0, -1*scale[idx]*centroids[idx][0]],
                                    [0, scale[idx], -1*scale[idx]*centroids[idx][1]],
                                    [0, 0, 1]])
                          for idx in range(len(good_points))]
    scaled_top_points = [(top_points[idx] - centroids[idx]) * scale[idx]
                         for idx in range(len(good_points))]

    # fundamental_matrix = fundamental.calculate_fundamental_matrix(top_img0_points, top_img1_points)
    fundamental_matrix = fundamental.calculate_fundamental_matrix_with_ransac(scaled_top_points[0],
                                                                              scaled_top_points[1],
                                                                              iterations=190)

    # unscale fundamental matrix
    fundamental_matrix = transform_matrices[1].transpose() @ fundamental_matrix @ transform_matrices[0]

    essential_matrix = fundamental.fundamental_to_essential_matrix(fundamental_matrix, intrinsic_camera_matrix)
    rotation, translation, triangulated_points = essential.calculate_pose(essential_matrix,
                                                                          good_points[0],
                                                                          good_points[1],
                                                                          intrinsic_camera_matrix)
    return rotation, translation, triangulated_points


def run_pangolin(threading_event, camera_poses, points):
    display.setup_pangolin()

    while not threading_event.is_set() and not display.should_quit():
        display.init_frame()

        for pose in camera_poses[:-1]:
            display.draw_camera(pose, (0.0, 1.0, 0.0))
        display.draw_camera(camera_poses[-1], (0.0, 1.0, 1.0))

        display.draw_points(points)

        display.finish_frame()


if __name__ == "__main__":
    plt.ion()

    img0 = cv2.imread("data/road1.jpg")
    img1 = cv2.imread("data/road2.jpg")
    intrinsic_camera_matrix = np.asarray([[9.842439e+02, 0.000000e+00, 6.900000e+02],
                                          [0.000000e+00, 9.808141e+02, 2.331966e+02],
                                          [0.000000e+00, 0.000000e+00, 1.000000e+00]])

    rotation, translation, points = find_initial_position(img0, img1, show_keypoints=True)

    camera_1_pose = np.identity(4)
    camera_2_pose = np.identity(4)
    camera_2_pose[:3, :3] = rotation
    camera_2_pose[:3, 3] = translation
    camera_poses = [camera_1_pose, camera_2_pose]

    np.set_printoptions(suppress=True)
    print("num triangulated points in front: ", len([point for point in points if point[2] > 0]), "/", len(points))
    print("camera direction", rotation @ [0, 0, 1])
    print("translation", translation)

    threading_event = threading.Event()
    try:
        thread = threading.Thread(target=run_pangolin, args=(threading_event, camera_poses, points))
        thread.start()
        # no need to join the thread

        while True:
            plt.draw()
            plt.pause(0.1)
    except (KeyboardInterrupt, SystemExit):
        print("keyboard interrupt")
        threading_event.set()
