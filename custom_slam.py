#!/usr/bin/env python3

import cv2
import itertools
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import threading

from src import display, epipolar, logger, plot, triangulation, util


def match_knn_brute_force(img0_descriptors, img1_descriptors, num_matches):
    """Finds k nearest descriptors in img1 for each descriptor in img0, using brute force"""
    # TODO no O(n^2) algo pls
    matches = []
    for img0_idx, img0_descriptor in enumerate(img0_descriptors):
        possible_matches = []
        for img1_idx, img1_descriptor in enumerate(img1_descriptors):
            hamming_distance = np.count_nonzero(img0_descriptor != img1_descriptor)
            possible_matches.append({"distance": hamming_distance, "index": img1_idx})
        possible_matches.sort(key=lambda x: x["distance"])
        limited_matches = possible_matches[:num_matches]
        matches.append(limited_matches)
    return matches


def match_knn_flann(img0_descriptors, img1_descriptors, flann_matcher, num_matches=2):
    # TODO why is this slow?
    matches = flann_matcher.knnMatch(img0_descriptors, img1_descriptors, k=num_matches)
    return [[{"distance": m.distance, "index": m.trainIdx},
             {"distance": n.distance, "index": n.trainIdx}]
            for m, n in matches]


def match_knn(img0_descriptors, img1_descriptors, num_matches=2):
    """Finds k nearest descriptors in img1 for each descriptor in img0 """
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, trees=5)
    search_params = dict(checks=50)
    flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
    return match_knn_flann(img0_descriptors, img1_descriptors, flann_matcher, num_matches)


def compute_orb(img, orb_detector):
    """find the keypoints with ORB"""
    keypoints = orb_detector.detect(img, None)
    # compute the descriptors with ORB
    keypoints, descriptors = orb_detector.compute(img, keypoints)
    descriptor_bits = np.unpackbits(np.uint8(descriptors), axis=1)
    return np.asarray([keypoint.pt for keypoint in keypoints]), descriptor_bits


_pyplot_subplots = None
def find_matches_between_images(img0, img1, orb_detector, show_image_keypoints=False):
    global _pyplot_subplots
    img0_pts, img0_des = compute_orb(img0, orb_detector)
    img1_pts, img1_des = compute_orb(img1, orb_detector)
    matches = match_knn(img0_des, img1_des, num_matches=2)

    matched_img0_points = []
    matched_img1_points = []

    for idx, (m, n) in enumerate(matches):
        # Lowe's ratio test
        first_match_much_better = m["distance"] < 0.5 * n["distance"]
        if first_match_much_better:
            matched_img0_points.append(img0_pts[idx])
            matched_img1_points.append(img1_pts[m["index"]])
    assert len(matched_img0_points) == len(matched_img1_points)
    matched_img0_points = np.asarray(matched_img0_points)
    matched_img1_points = np.asarray(matched_img1_points)

    if show_image_keypoints:
        _pyplot_subplots = _pyplot_subplots or plt.subplots()
        plot.plot_image_matches(img0, matched_img0_points, img1, matched_img1_points, subplots=_pyplot_subplots, show=False)
        figure, axes = _pyplot_subplots
        figure.canvas.draw_idle()
        plt.pause(0.001)

    return matched_img0_points, matched_img1_points


def normalize_points(points):
    average = np.average(points, axis=0)
    zeroed_points = points - average
    distance = np.average(np.linalg.norm(zeroed_points, axis=1))
    scale_factor = distance/np.sqrt(2)
    translation_matrix = np.asarray([[scale_factor, 0, -1*scale_factor*average[0]],
                                     [0, scale_factor, -1*scale_factor*average[1]],
                                     [0, 0, 1]])
    return zeroed_points/scale_factor, translation_matrix


def normalize_points_and_find_fundamental_matrix(img0_points, img1_points):
    normalized_img0_points, img0_translation_matrix = normalize_points(img0_points)
    normalized_img1_points, img1_translation_matrix = normalize_points(img1_points)

    fundamental_matrix = epipolar.calculate_fundamental_matrix_with_ransac(normalized_img0_points, normalized_img1_points)
    fundamental_matrix = img1_translation_matrix.T @ fundamental_matrix @ img0_translation_matrix
    return fundamental_matrix


def find_pose_and_triangulated_points(img0, img1, intrinsic_camera_matrix, orb_detector, show_image_keypoints=False):
    img0_points, img1_points = find_matches_between_images(img0, img1, orb_detector, show_image_keypoints=show_image_keypoints)
    fundamental_matrix = normalize_points_and_find_fundamental_matrix(img0_points, img1_points)
    essential_matrix = epipolar.fundamental_to_essential_matrix(fundamental_matrix, intrinsic_camera_matrix)
    camera_2_pose, triangulated_points = epipolar.calculate_pose_from_essential_matrix(essential_matrix,
                                                                                       img0_points,
                                                                                       img1_points,
                                                                                       intrinsic_camera_matrix)

    good_triangulated_points_mask = triangulated_points[:, 2] > 0
    return camera_2_pose, [img0_points, img1_points], triangulated_points, good_triangulated_points_mask


class Slam:
    def __init__(self, intrinsic_camera_matrix, show_image_keypoints=True, show_3d_visualization=True):
        self.visualization_lock = threading.Lock()
        self.orb_detector = cv2.ORB_create()
        self.flann_matcher = None
        self.intrinsic_camera_matrix = intrinsic_camera_matrix
        self.show_image_keypoints = show_image_keypoints
        self.pyplot_subplots = None
        self.visualization_stop_event = None
        self.visualization_thread = None

        self.latest_image_2d_points = None

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

    def add_triangulated_points_to_viz(self, points):
        with self.visualization_lock:
            if self.all_triangulated_points is None:
                self.all_triangulated_points = np.asarray(points)
            else:
                self.all_triangulated_points = np.append(self.all_triangulated_points, points, axis=0)

    def get_latest_camera_pose(self):
        return self.camera_poses[-1]

    def run_visualization(self, stop_event):
        display.setup()

        while not stop_event.is_set() and not display.should_quit():
            display.init_frame()

            with self.visualization_lock:
                if len(self.camera_poses) > 0:
                    for idx, pose in enumerate(self.camera_poses):
                        color = (0.0, 1.0, 0.0) if idx == len(self.camera_poses)-1 else (0.0, 1.0, 1.0)
                        display.draw_camera(pose, color)

                if self.all_triangulated_points is not None:
                    display.draw_points(self.all_triangulated_points)

            display.finish_frame()

    def log_camera_pose(self, camera_pose):
        logger.info("Translation: ", util.pose_to_translation(camera_pose), ", camera vector: ", util.pose_to_rotation(camera_pose) @ [0, 0, 1])

    def find_initial_pose(self, prev_image, new_image):
        """takes two images, finds the relative locations of the cameras (up to a scale factor), and finds triangulated points"""
        logger.info("-----------------------")
        logger.info("Finding initial pose...")

        self.reset()

        current_frame_num = self.current_frame_num()
        camera_pose, points2d, points3d, good_points_mask = find_pose_and_triangulated_points(prev_image,
                                                                                              new_image,
                                                                                              self.intrinsic_camera_matrix,
                                                                                              self.orb_detector,
                                                                                              show_image_keypoints=self.show_image_keypoints)

        self.log_camera_pose(camera_pose)

        for idx, is_good in enumerate(good_points_mask):
            #_if is_good:
            prev_image_point = tuple(points2d[0][idx])
            new_image_point = tuple(points2d[1][idx])
            self.set_triangulated_point(current_frame_num, prev_image_point, points3d[idx])
            self.set_triangulated_point(current_frame_num + 1, new_image_point, points3d[idx])
        self.add_triangulated_points_to_viz(points3d[good_points_mask])
        self.add_camera_pose(camera_pose)
        self.latest_image_2d_points = points2d[1]

    def find_next_pose(self, prev_image, new_image):
        logger.info("-----------------------")
        logger.info("Finding next pose...")
        current_frame_num = self.current_frame_num()
        latest_camera_pose = self.get_latest_camera_pose()
        prev_image_points, new_image_points = find_matches_between_images(prev_image,
                                                                          new_image,
                                                                          self.orb_detector,
                                                                          show_image_keypoints=self.show_image_keypoints)

        matches_with_3d_points = []
        for prev_point, new_point in zip(prev_image_points, new_image_points):
            point_3d = self.get_triangulated_point(current_frame_num, prev_point)
            if point_3d is not None:
                matches_with_3d_points.append({"prev_point": prev_point,
                                               "new_point": new_point,
                                               "3d_point": point_3d})
        pose, points = triangulation.triangulate_pose_from_points_with_ransac(latest_camera_pose,
                                                                              np.array([match["prev_point"] for match in matches_with_3d_points]),
                                                                              np.array([match["new_point"] for match in matches_with_3d_points]),
                                                                              np.array([match["3d_point"] for match in matches_with_3d_points]),
                                                                              prev_image_points,
                                                                              new_image_points,
                                                                              self.intrinsic_camera_matrix)
        self.log_camera_pose(pose)
        self.add_camera_pose(pose)


def main():
    plt.ion()

    intrinsic_camera_matrix = np.asarray([[9.842439e+02, 0.000000e+00, 6.900000e+02],
                                          [0.000000e+00, 9.808141e+02, 2.331966e+02],
                                          [0.000000e+00, 0.000000e+00, 1.000000e+00]])
    slam = Slam(intrinsic_camera_matrix,
                show_image_keypoints=True,
                show_3d_visualization=True)
    img0 = cv2.imread("data/0000000000.png")
    img1 = cv2.imread("data/0000000002.png")

    slam.find_initial_pose(img0, img1)

    img2 = cv2.imread("data/0000000004.png")
    slam.find_next_pose(img1, img2)


    # def get_next_frame(cap):
    #     for _ in range(15):
    #         cap.grab()
    #     ret, img = cap.retrieve()
    #     grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     return ret, grayscale
    #
    #
    # cap = cv2.VideoCapture('data/sanfrancisco-cut.mp4')
    # ret, img0 = get_next_frame(cap)
    # ret, img1 = get_next_frame(cap)
    # slam.find_initial_pose(img0, img1)
    # while cap.isOpened():
    #     img0 = img1
    #     ret, image = get_next_frame(cap)
    #     if ret is False:
    #         break
    #     img1 = image
    #     slam.find_next_pose(img0, img1)
    #     plt.draw()
    #     plt.pause(0.1)
    # cap.release()

    # for val in range(4, 28, 2):
    #     img0 = img1
    #     image_str = f"data/{val:0>10}.png"
    #     img1 = cv2.imread(image_str)
    #     slam.find_next_pose(img0, img1)
    #     plt.draw()
    #     plt.pause(0.1)
    #
    try:
        while True:
            plt.draw()
            plt.pause(0.1)
    except (KeyboardInterrupt, SystemExit):
        slam.stop_3d_visualization()


if __name__ == "__main__":
    main()