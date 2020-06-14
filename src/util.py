import numpy as np


def to_homogenous_coordinates(point):
    """takes a point and adds 1 to make it homogenous coordinates"""
    return np.hstack((point, [[1]]))


def vector_to_cross_product_matrix(vector):
    assert vector.shape == (3,)
    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])
