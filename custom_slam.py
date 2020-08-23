#!/usr/bin/env python3

import cv2
import matplotlib
matplotlib.use('TkAgg')
import math
import matplotlib.pyplot as plt
import numpy as np
import threading

from src import display, epipolar, logger, plot, triangulation, util



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
        with self.visualization_lock:
            self.camera_poses = [np.identity(4)]
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
        assert len(point) == 2
        return self.triangulated_point_lookup.get(frame_num, {}).get(tuple(point))

    def set_triangulated_point(self, frame_num, point, triangulated_point):
        assert len(point) == 2
        self.triangulated_point_lookup[frame_num] = self.triangulated_point_lookup.get(frame_num, {})
        self.triangulated_point_lookup[frame_num][tuple(point)] = triangulated_point

    def current_frame_num(self):
        return len(self.camera_poses)

    def add_camera_pose(self, pose):
        with self.visualization_lock:
            self.camera_poses.append(pose)

    def add_triangulated_points(self, points):
        with self.visualization_lock:
            if self.all_triangulated_points is None:
                self.all_triangulated_points = np.asarray(points)
            else:
                self.all_triangulated_points = np.append(self.all_triangulated_points, points, axis=0)

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

        img0_points = []
        img1_points = []
        good_point_indices = []

        for idx, (m, n) in enumerate(matches):
            # Lowe's ratio test
            first_match_much_better = m["distance"] < 0.75 * n["distance"]
            if first_match_much_better:
                img0_point = np.asarray(img0_pts[idx])
                img1_point = np.asarray(img1_pts[m["index"]])
                distance = np.linalg.norm(img0_point - img1_point)
                # more than 16px parallax
                # this seems to make things worse TODO figure out why
                if True: #distance > 16:
                    good_point_indices.append(len(img0_points))
                img0_points.append(img0_point)
                img1_points.append(img1_point)
        assert len(img0_points) == len(img1_points)
        img0_points = np.asarray(img0_points)
        img1_points = np.asarray(img1_points)
        good_point_indices = np.asarray(good_point_indices)

        if self.show_image_keypoints:
            self.pyplot_subplots = self.pyplot_subplots or plt.subplots()
            plot.plot_image_matches(img0, img0_points, img1, img1_points, subplots=self.pyplot_subplots, show=False)
            figure, axes = self.pyplot_subplots
            figure.canvas.draw_idle()
            plt.pause(0.001)

        return [img0_points, img1_points], good_point_indices

    def retrieve_pose_and_triangulated_points(self, img0, img1):
        point_matches, good_point_indices = self.find_matches_between_images(img0, img1)
        good_point_matches = [points[good_point_indices]
                              for points in point_matches]

        # Hartley's coordinate normalization
        # origin of points should be at (0, 0); average distance to center should be sqrt(2)
        centroids = [np.average(points, axis=0)
                     for points in good_point_matches]
        scale = [math.sqrt(2) / np.average(np.linalg.norm(good_point_matches[idx] - centroids[idx], axis=1))
                 for idx in range(len(good_point_matches))]

        transform_matrices = [np.array([[scale[idx], 0, -1*scale[idx]*centroids[idx][0]],
                                        [0, scale[idx], -1*scale[idx]*centroids[idx][1]],
                                        [0, 0, 1]])
                              for idx in range(len(good_point_matches))]
        scaled_good_points = [(good_point_matches[idx] - centroids[idx]) * scale[idx]
                              for idx in range(len(good_point_matches))]

        # fundamental_matrix = fundamental.calculate_fundamental_matrix(good_img0_points, good_img1_points)
        fundamental_matrix = epipolar.calculate_fundamental_matrix_with_ransac(scaled_good_points[0],
                                                                               scaled_good_points[1],
                                                                               iterations=256)

        # unscale fundamental matrix
        fundamental_matrix = transform_matrices[1].transpose() @ fundamental_matrix @ transform_matrices[0]

        essential_matrix = epipolar.fundamental_to_essential_matrix(fundamental_matrix, self.intrinsic_camera_matrix)
        camera_2_pose, triangulated_points = epipolar.calculate_pose_from_essential_matrix(essential_matrix,
                                                                                           good_point_matches[0],
                                                                                           good_point_matches[1],
                                                                                           self.intrinsic_camera_matrix)

        good_triangulated_points_mask = triangulated_points[:, 2] > 0

        return camera_2_pose, good_point_matches, triangulated_points, good_triangulated_points_mask

    def find_initial_pose(self, img0, img1):
        """takes the first two images and finds the relative locations of the cameras"""
        logger.info("Finding initial pose...")

        self.reset()

        camera_pose, points2d, points3d, good_points_mask = self.retrieve_pose_and_triangulated_points(img0, img1)

        logger.info("First pose - translation: ", util.pose_to_translation(camera_pose), ", camera vector: ", util.pose_to_rotation(camera_pose) @ [0, 0, 1])

        for idx, is_good in enumerate(good_points_mask):
            if is_good:
                image_0_point = tuple(points2d[0][idx])
                image_1_point = tuple(points2d[1][idx])
                self.set_triangulated_point(0, image_0_point, points3d[idx])
                self.set_triangulated_point(1, image_1_point, points3d[idx])
        self.add_triangulated_points(points3d[good_points_mask])
        self.add_camera_pose(camera_pose)

    def find_next_pose(self, img0, img1):
        logger.info("Finding next pose...")

        camera_pose, points2d, points3d, good_points_mask = self.retrieve_pose_and_triangulated_points(img0, img1)

        # retrieve scale
        # known_3d_points = []
        # triangulated_3d_points = []
        # for idx, img0_point in enumerate(points2d[0]):
        #     already_triangulated_point = self.get_triangulated_point(self.current_frame_num()-1, img0_point)
        #     if already_triangulated_point is not None:
        #         known_3d_points.append(already_triangulated_point)
        #         triangulated_3d_points.append(points3d[idx])
        # known_3d_points = np.asarray(known_3d_points)
        # triangulated_3d_points = np.asarray(triangulated_3d_points)
        # assert len(known_3d_points) == len(triangulated_3d_points)
        # assert len(known_3d_points) >= 2
        # known_distances = np.linalg.norm(np.diff(known_3d_points, axis=0), axis=1)
        # calculated_distances = np.linalg.norm(np.diff(triangulated_3d_points, axis=0), axis=1)
        # scale = np.average(known_distances) / np.average(calculated_distances)
        # pose_to_scale = util.rotation_translation_to_pose(util.pose_to_rotation(camera_pose),
        #                                                   util.pose_to_translation(camera_pose)*scale,)
        pose_to_scale = camera_pose

        next_pose = self.get_latest_camera_pose() @ pose_to_scale
        logger.info("Next pose - translation:", util.pose_to_translation(next_pose),
                    ", camera vector:", util.pose_to_rotation(next_pose) @ [0, 0, 1])
        for idx, is_good in enumerate(good_points_mask):
            if is_good:
                image_0_point = tuple(points2d[0][idx])
                image_1_point = tuple(points2d[1][idx])
                self.set_triangulated_point(self.current_frame_num(), image_0_point, points3d[idx])
                self.set_triangulated_point(self.current_frame_num(), image_1_point, points3d[idx])
        self.add_triangulated_points(points3d[good_points_mask])
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

    intrinsic_camera_matrix = np.asarray([[1000, 0.000000e+00, 640/2],
                                          [0.000000e+00, 1000, 360/2],
                                          [0.000000e+00, 0.000000e+00, 1.000000e+00]])
    slam = Slam(intrinsic_camera_matrix,
                show_image_keypoints=True,
                show_3d_visualization=True)

    def get_next_frame(cap):
        for _ in range(15):
            cap.grab()
        ret, img = cap.retrieve()
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return ret, grayscale


    cap = cv2.VideoCapture('data/sanfrancisco-cut.mp4')
    ret, img0 = get_next_frame(cap)
    ret, img1 = get_next_frame(cap)
    slam.find_initial_pose(img0, img1)
    while cap.isOpened():
        img0 = img1
        ret, image = get_next_frame(cap)
        if ret is False:
            break
        img1 = image
        slam.find_next_pose(img0, img1)
        plt.draw()
        plt.pause(0.1)
    cap.release()


    # intrinsic_camera_matrix = np.asarray([[9.842439e+02, 0.000000e+00, 6.900000e+02],
    #                                       [0.000000e+00, 9.808141e+02, 2.331966e+02],
    #                                       [0.000000e+00, 0.000000e+00, 1.000000e+00]])
    # img0 = cv2.imread("data/0000000000.png")
    # img1 = cv2.imread("data/0000000002.png")
    #
    # slam.find_initial_pose(img0, img1)
    #
    # # img2 = cv2.imread("data/0000000002.png")
    # # slam.find_next_pose(img1, img2)
    # for val in range(4, 28, 2):
    #     img0 = img1
    #     image_str = f"data/{val:0>10}.png"
    #     img1 = cv2.imread(image_str)
    #     slam.find_next_pose(img0, img1)
    #     plt.draw()
    #     plt.pause(0.1)

    try:
        while True:
            plt.draw()
            plt.pause(0.1)
    except (KeyboardInterrupt, SystemExit):
        slam.stop_3d_visualization()
