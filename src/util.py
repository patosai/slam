import numpy as np


def pixel_to_camera_coords(xy_point):
    "takes a 2d point and adds a z-coordinate set to 1"
    return np.asarray([xy_point[0], xy_point[1], 1])


def vector_to_cross_product_matrix(vector):
    assert vector.shape == (3,)
    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])
