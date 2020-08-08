import numpy as np

from . import triangulation, util


def calculate_pose(essential_matrix, img0_points, img1_points, intrinsic_camera_matrix):
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

    winning_num_points = 0
    winning_rotation = None
    winning_translation = None
    winning_triangulated_points = None
    for rotation in possible_rotations:
        for translation in possible_translations:
            # The camera matrix describes how the WORLD is transformed relative to the CAMERA
            # To see the motion of the camera, rotation/translation need to be inverted
            img0_camera_matrix = intrinsic_camera_matrix @ np.hstack((np.identity(3), np.zeros((3, 1))))
            img1_camera_matrix = intrinsic_camera_matrix @ np.hstack((rotation.transpose(), -1*translation.reshape((3, 1))))
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

    return util.rotation_translation_to_pose(winning_rotation, winning_translation), winning_triangulated_points
