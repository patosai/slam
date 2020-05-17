import cv2
import numpy as np


def find_fundamental_mat(matches, img1_keypoints, img2_keypoints):
    pts1 = np.asarray([[img1_keypoints[match.trainIdx].pt[1], img1_keypoints[match.trainIdx].pt[0]] for match in matches])
    pts2 = np.asarray([[img2_keypoints[match.trainIdx].pt[1], img2_keypoints[match.trainIdx].pt[0]] for match in matches])
    fundamental_matrix, mask = cv2.findFundamentalMat(pts1, pts2)
    return fundamental_matrix


def pose_and_points_from_matches(matches, img1_keypoints, img2_keypoints, intrinsic_camera_matrix):
    pts1 = np.asarray([[img1_keypoints[match.trainIdx].pt[1], img1_keypoints[match.trainIdx].pt[0]] for match in matches])
    pts2 = np.asarray([[img2_keypoints[match.trainIdx].pt[1], img2_keypoints[match.trainIdx].pt[0]] for match in matches])
    E, _ = cv2.findEssentialMat(pts1, pts2, intrinsic_camera_matrix)
    _, R, t, _, triangulated_points = cv2.recoverPose(E, pts1, pts2, intrinsic_camera_matrix, distanceThresh=50)
    return R, t, triangulated_points
