# The eight-point algorithm by Christopher Longuet-Higgins
#
# Given a set of eight matching points in two images, calculates the essential matrix

import numpy as np


def two_d_to_three_d(pt):
    """Given a 2D point in (X, Y), returns a 3D projection of it"""
    return np.asarray([pt[0], pt[1], 1])


def calculate_fundamental_matrix(matched_points):
    "Sets up a homogenous system of eight linear equations and attempts to solve the 3x3 essential matrix."
    assert len(matched_points) >= 8
    for point_set in matched_points:
        assert len(point_set) == 2
        for point in point_set:
            assert len(point) == 2
            assert isinstance(point[0], float)
            assert isinstance(point[1], float)

    first_point_average = np.average(np.asarray([point_set[0] for point_set in matched_points]), axis=0)
    second_point_average = np.average(np.asarray([point_set[1] for point_set in matched_points]), axis=0)

    # scale by a factor
    matched_points = [[[point_set[0][0] / first_point_average[0], point_set[0][1] / first_point_average[1]],
                       [point_set[1][0] / second_point_average[0], point_set[1][1] / second_point_average[1]]]
                      for point_set in matched_points]

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
    C = np.transpose(np.asarray([[b[0]*a[0], b[0]*a[1], b[0], b[1]*a[0], b[1]*a[1], b[1], a[0], a[1], 1] for a, b in matched_points]))
    # singular value decomposition: C = USV
    u, s, v = np.linalg.svd(C)
    # if 8 points given, select the left singular vector associated with the 0 singular value
    # otherwise, select the one associated with the minimum singular value
    idx_to_select = -1 if len(matched_points) == 8 else np.argmin(s)
    estimated_f = u[:, idx_to_select]
    estimated_f = estimated_f.reshape((3, 3))

    # unscale by the same factor
    first_image_scale_matrix = np.asarray([[first_point_average[0], 0, 0],
                                            [0, first_point_average[1], 0],
                                            [0, 0, 1]])
    second_image_scale_matrix = np.asarray([[second_point_average[0], 0, 0],
                                            [0, second_point_average[1], 0],
                                            [0, 0, 1]])
    estimated_f = np.transpose(second_image_scale_matrix) @ estimated_f @ first_image_scale_matrix

    normalized_estimated_f = estimated_f / np.linalg.norm(estimated_f)

    return normalized_estimated_f


def fundamental_to_essential_matrix(fundamental_matrix, intrinsic_camera_matrix):
    essential_matrix = np.transpose(intrinsic_camera_matrix) @ fundamental_matrix @ intrinsic_camera_matrix
    normalized_essential_matrix = essential_matrix / np.linalg.norm(essential_matrix)
    return normalized_essential_matrix
