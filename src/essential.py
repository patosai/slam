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

    for rotation in possible_rotations:
        for translation in possible_translations:
            point_a = matched_points[0][0]
            #print([point_a; 1] - translation)


    # TODO need to test points to determine the correct solution out of the four
    return rotation, translation
