# The eight-point algorithm by Christopher Longuet-Higgins
#
# Given a set of eight matching points in two images, calculates the essential matrix

import numpy as np
import random


def two_d_to_three_d(pt):
    """Given a 2D point in (X, Y), returns a 3D projection of it"""
    return np.asarray([pt[0], pt[1], 1])


def calculate_fundamental_matrix(img1_points, img2_points):
    "Sets up a homogenous system of eight linear equations and attempts to solve the 3x3 essential matrix."
    img1_points = np.asarray(img1_points)
    img2_points = np.asarray(img2_points)
    assert img1_points.shape == img2_points.shape
    assert len(img1_points) >= 8

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
    C = np.transpose(np.asarray([[b[0]*a[0], b[0]*a[1], b[0], b[1]*a[0], b[1]*a[1], b[1], a[0], a[1], 1] for a, b in zip(img1_points, img2_points)]))
    # singular value decomposition: C = USV
    u, s, vh = np.linalg.svd(C)
    # if 8 points given, select the left singular vector associated with the 0 singular value
    # otherwise, select the one associated with the minimum singular value
    idx_to_select = -1 if len(img1_points) == 8 else np.argmin(s)
    estimated_f = u[:, idx_to_select]
    estimated_f = estimated_f.reshape((3, 3))

    # enforce that F has two singular values that are 1, and the third is 0
    u, s, vh = np.linalg.svd(estimated_f)
    estimated_f = u @ np.diag([1, 1, 0]) @ vh

    normalized_estimated_f = estimated_f / np.linalg.norm(estimated_f)

    return normalized_estimated_f


def calculate_fundamental_matrix_with_ransac(img1_points, img2_points, iterations=500):
    """Calculates the fundamental matrix with pseudorandomly selected points.
    The matrix with the most matches (inliers) is returned. This is the RANSAC algorithm."""
    img1_points = np.asarray(img1_points)
    img2_points = np.asarray(img2_points)
    assert img1_points.shape == img2_points.shape
    assert len(img1_points) >= 8

    all_indices = range(len(img1_points))
    inlier_error_threshold = 0.005
    img1_points_with_z = np.hstack((img1_points, np.ones((len(img1_points), 1))))
    img2_points_with_z = np.hstack((img2_points, np.ones((len(img2_points), 1))))
    max_inliers = 0
    total_inlier_error = None
    max_inlier_matrix = None
    for iteration in range(iterations):
        random.seed(0x1337BEEF + iteration)
        chosen_indices = random.sample(all_indices, 8)
        fundamental_matrix = calculate_fundamental_matrix(img1_points[chosen_indices], img2_points[chosen_indices])
        errors = np.square([img2_point @ fundamental_matrix @ img1_point.transpose()
                            for img1_point, img2_point in zip(img1_points_with_z, img2_points_with_z)])
        inlier_mask = errors < inlier_error_threshold
        inlier_error = np.sum(errors[inlier_mask])
        num_inliers = np.count_nonzero(inlier_mask)
        if num_inliers > max_inliers or (num_inliers == max_inliers and inlier_error < total_inlier_error):
            max_inliers = num_inliers
            max_inlier_matrix = fundamental_matrix
            total_inlier_error = inlier_error
    return max_inlier_matrix


def fundamental_to_essential_matrix(fundamental_matrix, intrinsic_camera_matrix):
    """The fundamental matrix is the essential matrix, but with intrinsic camera parameters embedded.
    If the camera matrix is known, the essential matrix can be calculated."""
    essential_matrix = np.transpose(intrinsic_camera_matrix) @ fundamental_matrix @ intrinsic_camera_matrix
    normalized_essential_matrix = essential_matrix / np.linalg.norm(essential_matrix)
    return normalized_essential_matrix
