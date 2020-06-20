#!/usr/bin/env python3

import collections
import cv2
import matplotlib
matplotlib.use('TkAgg')
import math
import matplotlib.pyplot as plt
import numpy as np
import threading


from src import display, fundamental, essential, plot, triangulation

# Initiate ORB detector
ORB_DETECTOR = cv2.ORB_create()
PANGOLIN_LOCK = threading.Lock()

CAMERA_POSES = []
TRIANGULATED_POINT_LOOKUP = {}
ALL_TRIANGULATED_POINTS = None


def get_triangulated_point(frame_num, point):
    global TRIANGULATED_POINT_LOOKUP
    return TRIANGULATED_POINT_LOOKUP.get(frame_num, {}).get(tuple(point))


def set_triangulated_point(frame_num, point, triangulated_point):
    global TRIANGULATED_POINT_LOOKUP
    TRIANGULATED_POINT_LOOKUP[frame_num] = TRIANGULATED_POINT_LOOKUP[frame_num] or {}
    TRIANGULATED_POINT_LOOKUP[frame_num][tuple(point)] = triangulated_point


def compute_orb(img):
    """find the keypoints with ORB"""
    keypoints = ORB_DETECTOR.detect(img, None)
    # compute the descriptors with ORB
    keypoints, descriptors = ORB_DETECTOR.compute(img, keypoints)
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

    good_img0_points = []
    good_img1_points = []

    for idx, (m, n) in enumerate(matches):
        # Lowe's ratio test
        first_match_much_better = m["distance"] < 0.5 * n["distance"]
        if first_match_much_better:
            img0_point = np.asarray(img0_pts[idx])
            img1_point = np.asarray(img1_pts[m["index"]])

            good_img0_points.append(img0_point)
            good_img1_points.append(img1_point)

    good_img0_points = np.asarray(good_img0_points)
    good_img1_points = np.asarray(good_img1_points)

    if show_keypoints:
        fig, ax = plt.subplots()
        keypoints_image = img1.copy()
        for point in good_img1_points:
            circle = plt.Circle(point, radius=5, color='#00FF00', fill=False)
            ax.add_artist(circle)
        plt.imshow(keypoints_image)

    return good_img0_points, good_img1_points


def find_initial_position(img0, img1, show_keypoints=False):
    """takes the first two images and finds the relative locations of the cameras"""
    good_points = find_matches_between_images(img0, img1, show_keypoints=show_keypoints)

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
    scaled_good_points = [(good_points[idx] - centroids[idx]) * scale[idx]
                          for idx in range(len(good_points))]

    # fundamental_matrix = fundamental.calculate_fundamental_matrix(good_img0_points, good_img1_points)
    fundamental_matrix = fundamental.calculate_fundamental_matrix_with_ransac(scaled_good_points[0],
                                                                              scaled_good_points[1],
                                                                              iterations=500)

    # unscale fundamental matrix
    fundamental_matrix = transform_matrices[1].transpose() @ fundamental_matrix @ transform_matrices[0]

    essential_matrix = fundamental.fundamental_to_essential_matrix(fundamental_matrix, intrinsic_camera_matrix)
    rotation, translation, triangulated_points = essential.calculate_pose(essential_matrix,
                                                                          good_points[0],
                                                                          good_points[1],
                                                                          intrinsic_camera_matrix)
    return rotation, translation, triangulated_points, good_points


def run_pangolin(stop_event, lock):
    global PANGOLIN_LOCK, CAMERA_POSES, ALL_TRIANGULATED_POINTS
    display.setup_pangolin()

    while not stop_event.is_set() and not display.should_quit():
        display.init_frame()

        with lock:
            for pose in camera_poses[:-1]:
                display.draw_camera(pose, (0.0, 1.0, 0.0))
            display.draw_camera(camera_poses[-1], (0.0, 1.0, 1.0))

            if ALL_TRIANGULATED_POINTS is not None:
                display.draw_points(ALL_TRIANGULATED_POINTS)

        display.finish_frame()


if __name__ == "__main__":
    lock = threading.Lock()
    plt.ion()

    img0 = cv2.imread("data/0000000002.png")
    img1 = cv2.imread("data/0000000003.png")
    intrinsic_camera_matrix = np.asarray([[9.842439e+02, 0.000000e+00, 6.900000e+02],
                                          [0.000000e+00, 9.808141e+02, 2.331966e+02],
                                          [0.000000e+00, 0.000000e+00, 1.000000e+00]])

    rotation, translation, triangulated_points, image_points = find_initial_position(img0, img1, show_keypoints=True)

    camera_1_pose = np.identity(4)
    camera_2_pose = np.identity(4)
    camera_2_pose[:3, :3] = rotation
    camera_2_pose[:3, 3] = translation
    camera_poses = [camera_1_pose, camera_2_pose]

    good_triangulated_points_mask = triangulated_points[:, 2] > 0
    good_triangulated_points = triangulated_points[good_triangulated_points_mask]

    with lock:
        CAMERA_POSES = camera_poses
        for idx, is_good in enumerate(good_triangulated_points_mask):
            if is_good:
                image_0_point = tuple(image_points[0][idx])
                image_1_point = tuple(image_points[1][idx])
                set_triangulated_point(0, image_0_point, good_triangulated_points[idx])
                set_triangulated_point(1, image_1_point, good_triangulated_points[idx])

        if ALL_TRIANGULATED_POINTS is None:
            ALL_TRIANGULATED_POINTS = good_triangulated_points
        else:
            ALL_TRIANGULATED_POINTS = np.unique(np.concatenate((ALL_TRIANGULATED_POINTS, good_triangulated_points), axis=0), axis=0)

    # -------------------

    img2 = cv2.imread("data/0000000005.png")
    img1_points, img2_points = find_matches_between_images(img1, img2, show_keypoints=True)
    known_points = []
    matched_image_2_points = []
    for idx, image_1_point in enumerate(img1_points):
        known_point = get_triangulated_point(1, image_1_point)
        if known_point is not None:
            known_points.append(known_point)
            matched_image_2_points.append(img2_points[idx])
    # TODO
    triangulation.triangulate_pose_from_points(known_points, matched_image_2_points, intrinsic_camera_matrix)

    thread_stop_event = threading.Event()
    try:
        thread = threading.Thread(target=run_pangolin, args=(thread_stop_event, lock))
        thread.start()
        # no need to join the thread

        while True:
            plt.draw()
            plt.pause(0.1)
    except (KeyboardInterrupt, SystemExit):
        thread_stop_event.set()
