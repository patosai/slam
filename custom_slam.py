#!/usr/bin/env python3

import cv2
import matplotlib
matplotlib.use('TkAgg')
import math
import matplotlib.pyplot as plt
import numpy as np
import threading

from src import display, fundamental, essential, logger, plot, triangulation, util



class Slam:
    def __init__(self, intrinsic_camera_matrix, show_image_keypoints=True, show_3d_visualization=True):
        self.visualization_lock = threading.Lock()
        self.orb_detector = cv2.ORB_create()
        self.intrinsic_camera_matrix = intrinsic_camera_matrix
        self.show_image_keypoints = show_image_keypoints
        self.pyplot_subplots = None
        self.visualization_stop_event = None
        self.visualization_thread = None

        self.camera_poses = None
        self.triangulated_point_lookup = None
        self.all_triangulated_points = None

        self.reset()

        if show_3d_visualization:
            self.start_3d_visualization()

    def reset(self):
        self.camera_poses = []
        self.triangulated_point_lookup = {}
        self.all_triangulated_points = None

    def start_3d_visualization(self):
        thread_stop_event = threading.Event()
        thread = threading.Thread(target=self.run_visualization, args=[thread_stop_event])
        thread.start()
        self.visualization_stop_event = thread_stop_event
        self.visualization_thread = thread

    def stop_3d_visualization(self):
        if self.visualization_stop_event:
            self.visualization_stop_event.set()

    def get_triangulated_point(self, frame_num, point):
        return self.triangulated_point_lookup.get(frame_num, {}).get(tuple(point))

    def set_triangulated_point(self, frame_num, point, triangulated_point):
        self.triangulated_point_lookup[frame_num] = self.triangulated_point_lookup.get(frame_num, {})
        self.triangulated_point_lookup[frame_num][tuple(point)] = triangulated_point

    def add_camera_pose(self, pose):
        with self.visualization_lock:
            self.camera_poses.append(pose)

    def get_latest_camera_pose(self):
        return self.camera_poses[-1]

    def compute_orb(self, img):
        """find the keypoints with ORB"""
        keypoints = self.orb_detector.detect(img, None)
        # compute the descriptors with ORB
        keypoints, descriptors = self.orb_detector.compute(img, keypoints)
        descriptor_bits = np.unpackbits(np.uint8(descriptors), axis=1)
        return [keypoint.pt for keypoint in keypoints], descriptor_bits

    def match_knn(self, img0_descriptors, img1_descriptors, k=2):
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

    def find_matches_between_images(self, img0, img1):
        img0_pts, img0_des = self.compute_orb(img0)
        img1_pts, img1_des = self.compute_orb(img1)
        matches = self.match_knn(img0_des, img1_des, k=2)

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

        if self.show_image_keypoints:
            self.pyplot_subplots = self.pyplot_subplots or plt.subplots()
            figure, axes = self.pyplot_subplots
            axes.clear()
            keypoints_image = img1.copy()
            for point in good_img1_points:
                circle = plt.Circle(point, radius=5, color='#00FF00', fill=False)
                axes.add_artist(circle)
            axes.imshow(keypoints_image)
            plt.draw()
            plt.pause(0.001)

        return good_img0_points, good_img1_points

    def find_initial_pose(self, img0, img1):
        """takes the first two images and finds the relative locations of the cameras"""
        logger.info("Finding initial pose...")

        self.reset()

        good_points = self.find_matches_between_images(img0, img1)

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
                                                                                  iterations=8)

        # unscale fundamental matrix
        fundamental_matrix = transform_matrices[1].transpose() @ fundamental_matrix @ transform_matrices[0]

        essential_matrix = fundamental.fundamental_to_essential_matrix(fundamental_matrix, self.intrinsic_camera_matrix)
        camera_2_pose, triangulated_points = essential.calculate_pose(essential_matrix,
                                                                      good_points[0],
                                                                      good_points[1],
                                                                      self.intrinsic_camera_matrix)

        logger.info("Initial pose - camera direction: ", util.pose_to_rotation(camera_2_pose) @ [0, 0, 1],
                    " translation: ", util.pose_to_translation(camera_2_pose))
        camera_1_pose = util.rotation_translation_to_pose(np.identity(3), np.zeros(3))
        camera_poses = [camera_1_pose, camera_2_pose]

        good_triangulated_points_mask = triangulated_points[:, 2] > 0
        good_triangulated_points = triangulated_points[good_triangulated_points_mask]

        with self.visualization_lock:
            self.camera_poses = camera_poses
            self.all_triangulated_points = good_triangulated_points
        for idx, is_good in enumerate(good_triangulated_points_mask):
            if is_good:
                image_0_point = tuple(good_points[0][idx])
                image_1_point = tuple(good_points[1][idx])
                self.set_triangulated_point(0, image_0_point, triangulated_points[idx])
                self.set_triangulated_point(1, image_1_point, triangulated_points[idx])

    def find_next_pose(self, img0, img1):
        logger.info("Finding next pose...")
        img0_points, img1_points = self.find_matches_between_images(img0, img1)
        known_3d_points = []
        matched_img0_points = []
        matched_img1_points = []
        for idx, img0_point in enumerate(img0_points):
            known_point = self.get_triangulated_point(1, img0_point)
            if known_point is not None and known_point[2] < 100:
                known_3d_points.append(known_point)
                matched_img0_points.append(img0_point)
                matched_img1_points.append(img1_points[idx])

        camera_pose, triangulated_points = triangulation.triangulate_pose_from_points_with_ransac(self.get_latest_camera_pose(),
                                                                                                  known_3d_points,
                                                                                                  matched_img0_points,
                                                                                                  matched_img1_points,
                                                                                                  intrinsic_camera_matrix)
        next_pose = self.get_latest_camera_pose() @ camera_pose
        self.add_camera_pose(next_pose)

    def run_visualization(self, stop_event):
        display.setup()

        while not stop_event.is_set() and not display.should_quit():
            display.init_frame()

            with self.visualization_lock:
                if len(self.camera_poses) > 0:
                    for idx, pose in enumerate(self.camera_poses):
                        color = (0.0, 1.0, 1.0) if idx == len(self.camera_poses)-1 else (0.0, 1.0, 0.0)
                        display.draw_camera(pose, color)

                if self.all_triangulated_points is not None:
                    display.draw_points(self.all_triangulated_points)

            display.finish_frame()


if __name__ == "__main__":
    plt.ion()

    intrinsic_camera_matrix = np.asarray([[9.842439e+02, 0.000000e+00, 6.900000e+02],
                                          [0.000000e+00, 9.808141e+02, 2.331966e+02],
                                          [0.000000e+00, 0.000000e+00, 1.000000e+00]])
    slam = Slam(intrinsic_camera_matrix,
                show_image_keypoints=True,
                show_3d_visualization=True)

    img0 = cv2.imread("data/0000000002.png")
    img1 = cv2.imread("data/0000000003.png")

    slam.find_initial_pose(img0, img1)

    img2 = cv2.imread("data/0000000004.png")
    slam.find_next_pose(img1, img2)

    try:
        while True:
            plt.draw()
            plt.pause(0.1)
    except (KeyboardInterrupt, SystemExit):
        slam.stop_3d_visualization()
