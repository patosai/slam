import numpy as np


def essential_matrix_to_rotation_translation(e):
    "given an essential matrix, calculates the translation vector (normalized) and the rotation matrix"
    u, s, v = np.linalg.svd(e)
    # http://igt.ip.uca.fr/~ab/Classes/DIKU-3DCV2/Handouts/Lecture16.pdf
    w = np.asarray([[0, -1, 0],
                    [1, 0, 0],
                    [0, 0, 1]])
    rotation = u @ w @ v
    # translation X points right, Y points down, Z points into the image
    translation = u[:, 2]
    # TODO need to test points to determine the correct solution out of the four
    return rotation, translation
