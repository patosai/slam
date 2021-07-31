import numpy as np
import operator as op
from functools import reduce


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
    return -1 * pose[:3, :3].transpose() @ pose[:3, 3]


def rotation_translation_to_pose(rotation, translation):
    """Takes rotation/translation of camera with respect to world coordinates, and returns a pose that has
    the world coordinates with respect to the camera coordinates"""
    assert rotation.shape == (3, 3)
    assert translation.shape == (3,)
    transposed_rotation = rotation.transpose()
    return np.vstack((np.hstack((transposed_rotation, (-1*transposed_rotation @ translation).reshape((3, 1)))),
                      [0, 0, 0, 1]))


def point_distance(pt1, pt2):
    """Accepts two points of any dimension and computes their Euclidean distance"""
    return np.linalg.norm(pt1 - pt2)


def collect_symbolic_equation_with_respect_to_vars(eq, vars):
    assert isinstance(vars, list)
    eq = eq.expand()
    if len(vars) == 0:
        return {1: eq}
    var_map = eq.collect(vars[0], evaluate=False)
    final_var_map = {}
    for var_power in var_map:
        sub_expression = var_map[var_power]
        sub_var_map = collect_symbolic_equation_with_respect_to_vars(sub_expression, vars[1:])
        for sub_var_power in sub_var_map:
            final_var_map[var_power*sub_var_power] = sub_var_map[sub_var_power]
    return final_var_map


def rq_decomposition(m):
    matrix_reverser = np.rot90(np.diag(np.ones(m.shape[0])))
    m_prime = matrix_reverser @ m
    q_prime, r_prime = np.linalg.qr(m_prime)
    q = matrix_reverser @ q_prime.T
    r = matrix_reverser @ r_prime.T @ matrix_reverser
    return r, q
