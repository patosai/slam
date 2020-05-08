# The eight-point algorithm by Christopher Longuet-Higgins
#
# Given a set of eight matching points in two images, calculates the essential matrix

import numpy as np


def calculate_estimated_essential_matrix(matched_points):
    "Sets up a homogenous system of eight linear equations and attempts to solve the 3x3 essential matrix."
    assert len(matched_points) == 8
    for point_set in matched_points:
        assert len(point_set) == 2
        for point in point_set:
            assert len(point) == 2
            assert isinstance(point[0], float)
            assert isinstance(point[1], float)

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
    C = np.transpose(np.array([[b[0]*a[0], b[0]*a[1], b[0], b[1]*a[0], b[1]*a[1], b[1], a[0], a[1], 1] for a, b in matched_points]))
    # singular value decomposition: C = USV
    u, s, v = np.linalg.svd(C)
    index_of_minimum_singular_value = np.argmin(s)
    estimated_e = u[:, index_of_minimum_singular_value]
    estimated_e = np.reshape(estimated_e, (3, 3))
    print(estimated_e)
    return estimated_e


def calculate_essential_matrix(matched_points):
    calculate_estimated_essential_matrix(matched_points)

