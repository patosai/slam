import numpy as np


def essential_matrix_to_rotation_translation(essential_matrix, matched_points):
    "given an essential matrix, calculates the translation vector (normalized) and the rotation matrix"
    u, s, v = np.linalg.svd(essential_matrix)
    # http://igt.ip.uca.fr/~ab/Classes/DIKU-3DCV2/Handouts/Lecture16.pdf
    w = np.asarray([[0, -1, 0],
                    [1, 0, 0],
                    [0, 0, 1]])
    possible_rotations = [u @ w @ v, u @ np.transpose(w) @ v]
    rotation_multiplier = 1 if np.linalg.det(possible_rotations[0]) > 0 else -1
    possible_rotations = [rotation_multiplier * rotation for rotation in possible_rotations]

    # translation X points right, Y points down, Z points into the image
    possible_translations = [u[:, -1], -1*u[:, -1]]

    winning_num_points = 0
    winning_rotation = None
    winning_translation = None
    for rotation in possible_rotations:
        for translation in possible_translations:
            points = [np.asarray([point_set[0][0], point_set[0][1], 1]) for point_set in matched_points]
            points_in_front_of_camera = [(rotation[-1] @ (translation - point) > 0) for point in points]
            num_points_in_front_of_camera = sum(points_in_front_of_camera)
            if num_points_in_front_of_camera > winning_num_points:
                winning_rotation = rotation
                winning_translation = translation
                winning_num_points = num_points_in_front_of_camera

    return winning_rotation, winning_translation
