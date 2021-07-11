# The eight-point algorithm by Christopher Longuet-Higgins
#
# Given a set of eight matching points in two images, calculates the essential matrix

import math
import numpy as np
import random
import sympy as sp

from . import logger, triangulation, util


def two_d_to_three_d(pt):
    """Given a 2D point in (X, Y), returns a 3D projection of it"""
    return np.asarray([pt[0], pt[1], 1])


def calculate_fundamental_matrix(img0_points, img1_points):
    """Sets up a homogenous system of eight linear equations and attempts to solve the 3x3 essential matrix."""
    img0_points = np.asarray(img0_points)
    img1_points = np.asarray(img1_points)
    assert img0_points.shape == img1_points.shape
    assert len(img0_points) >= 8

    # Let ax, ay be the xy-coordinates of a. Let bx, be the xy-coordinates of b.
    # Let A = [ax  and B = [bx
    #          ay           by
    #          1]           1]
    # Let the essential matrix (E) be [[e11 e12 e13]
    #                                  [e21 e22 e23]
    #                                  [e31 e32 e33]]
    # Each set of two images must satisfy the constraint (B')^T E A = 0
    #
    # Put another way, e∙c = 0, where e = [e11 e12 e13 e21 e22 e23 e31 e32 e33],
    # and c = [bx*ax, bx*ay, bx, by*ax, by*ay, by, ax, ay, 1]
    # e C = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    C = np.asarray([[b[0]*a[0], b[0]*a[1], b[0], b[1]*a[0], b[1]*a[1], b[1], a[0], a[1], 1] for a, b in zip(img0_points, img1_points)])
    # singular value decomposition: C = USV
    u, s, vh = np.linalg.svd(C)
    estimated_f = vh[-1]
    estimated_f = estimated_f.reshape((3, 3))

    normalized_estimated_f = estimated_f / np.linalg.norm(estimated_f)

    # enforce that F has two singular values that are 1, and the third is 0
    # Tsai and Huang method of setting 3rd singular value of SVD to 0 minimizes the Froebius norm
    u, s, vh = np.linalg.svd(normalized_estimated_f)
    s[2] = 0
    final_estimated_f = u @ np.diag(s) @ vh

    return final_estimated_f


def calculate_fundamental_matrix_with_ransac(img0_points, img1_points, iterations=1000):
    """Calculates the fundamental matrix with points pseudorandomly selected from the given points.
    The matrix with the most matches (inliers) is returned."""
    img0_points = np.asarray(img0_points)
    img1_points = np.asarray(img1_points)
    assert img0_points.shape == img1_points.shape
    assert len(img0_points) >= 8

    all_indices = range(len(img0_points))
    inlier_error_threshold = 0.005
    img0_points_with_z = np.hstack((img0_points, np.ones((len(img0_points), 1))))
    img1_points_with_z = np.hstack((img1_points, np.ones((len(img1_points), 1))))
    max_inliers = -1
    total_inlier_error = None
    max_inlier_matrix = None
    for iteration in range(min(iterations, math.comb(len(img0_points), 8))):
        random.seed(0x1337BEEF + iteration)
        chosen_indices = random.sample(all_indices, 8)
        fundamental_matrix = calculate_fundamental_matrix(img0_points[chosen_indices], img1_points[chosen_indices])
        errors = np.square([(img1_point.T @ fundamental_matrix) @ img0_point
                             for img0_point, img1_point in zip(img0_points_with_z, img1_points_with_z)])
        inlier_mask = errors < inlier_error_threshold
        inlier_error = np.sum(errors[inlier_mask])
        num_inliers = np.count_nonzero(inlier_mask)
        if num_inliers > max_inliers or (num_inliers == max_inliers and inlier_error < total_inlier_error):
            max_inliers = num_inliers
            max_inlier_matrix = fundamental_matrix
            total_inlier_error = inlier_error
    logger.info("fundamental matrix ransac - num inliers: ", max_inliers, "/", len(img0_points),
                " error: ", total_inlier_error)
    return max_inlier_matrix


def fundamental_to_essential_matrix(fundamental_matrix, intrinsic_camera_matrix):
    """The fundamental matrix is the essential matrix, but with intrinsic camera parameters embedded.
    If the camera matrix is known, the essential matrix can be calculated."""
    essential_matrix = np.transpose(intrinsic_camera_matrix) @ fundamental_matrix @ intrinsic_camera_matrix
    normalized_essential_matrix = essential_matrix / np.linalg.norm(essential_matrix)
    return normalized_essential_matrix


def calculate_pose_from_essential_matrix(essential_matrix, img0_points, img1_points, intrinsic_camera_matrix):
    """given an essential matrix, calculates the translation vector (normalized) and the rotation matrix"""
    u, s, v = np.linalg.svd(essential_matrix)
    # http://igt.ip.uca.fr/~ab/Classes/DIKU-3DCV2/Handouts/Lecture16.pdf
    w = np.asarray([[0, -1, 0],
                    [1, 0, 0],
                    [0, 0, 1]])
    possible_rotations = [u @ w @ v, u @ np.transpose(w) @ v]
    rotation_multiplier = 1 if np.linalg.det(possible_rotations[0]) > 0 else -1
    possible_rotations = [rotation_multiplier * rotation for rotation in possible_rotations]

    # translation X points right, Y points up, Z points into the image
    possible_translations = [u[:, -1], -1*u[:, -1]]

    winning_num_points = -1
    winning_rotation = None
    winning_translation = None
    winning_triangulated_points = None
    for rotation in possible_rotations:
        for translation in possible_translations:
            # The camera matrix describes how the WORLD is transformed relative to the CAMERA
            # To see the motion of the camera, rotation/translation need to be inverted
            img0_camera_matrix = intrinsic_camera_matrix @ util.rotation_translation_to_pose(np.identity(3), np.zeros(3))[:3]
            img1_camera_matrix = intrinsic_camera_matrix @ util.rotation_translation_to_pose(rotation, translation)[:3]
            triangulated_points = triangulation.triangulate_points_from_pose(img0_camera_matrix,
                                                                             img1_camera_matrix,
                                                                             img0_points,
                                                                             img1_points)
            camera_2_vector = rotation @ np.asarray([0, 0, 1])

            points_in_front_of_camera_1 = triangulated_points[:, 2] > 0
            points_in_front_of_camera_2 = np.dot((triangulated_points - translation), camera_2_vector) > 0
            num_points_in_front_of_camera = np.count_nonzero(np.multiply(points_in_front_of_camera_1, points_in_front_of_camera_2))

            if num_points_in_front_of_camera > winning_num_points:
                winning_rotation = rotation
                winning_translation = translation
                winning_num_points = num_points_in_front_of_camera
                winning_triangulated_points = triangulated_points
    logger.debug("essential matrix - winning num points: %d/%d" % (winning_num_points, len(img0_points)))

    return util.rotation_translation_to_pose(winning_rotation, winning_translation), winning_triangulated_points
