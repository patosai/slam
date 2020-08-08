import numpy as np


def to_homogenous_coordinates(point):
    """takes a point and adds 1 to make it homogenous coordinates"""
    return np.hstack((point, [[1]]))


def vector_to_cross_product_matrix(vector):
    assert vector.shape == (3,)
    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])


def pose_to_rotation(pose):
    """Returns rotation of the camera, with respect to world coordinates"""
    assert pose.shape == (4, 4)
    return pose[:3, :3].transpose()


def pose_to_translation(pose):
    """Returns translation of the camera, with respect to world coordinates"""
    assert pose.shape == (4, 4)
    return pose[:3, 3] * -1


def rotation_translation_to_pose(rotation, translation):
    """Takes rotation/translation of camera with respect to world coordinates, and returns a pose that has
    the world coordinates with respect to the camera coordinates"""
    assert rotation.shape == (3, 3)
    assert translation.shape == (3,)
    return np.vstack((np.hstack((rotation.transpose(), -1*translation.reshape((3, 1)))),
                      [[0, 0, 0, 1]]))
