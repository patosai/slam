import numpy as np

from . import plot, util


def essential_matrix_to_rotation_translation(essential_matrix, img1_points, img2_points):
    """given an essential matrix, calculates the translation vector (normalized) and the rotation matrix"""
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
            # TODO
            print("rotation")
            print(rotation)
            print("translation")
            print(translation)
            camera_vector = rotation @ np.asarray([0, 0, 1])

            print("camera vector")
            print(camera_vector)
            print("------")

            position = translation
            num_points_in_front_of_camera = 0
            for img1_pt, img2_pt in zip(img1_points, img2_points):
                first_point = util.pixel_to_camera_coords(img1_pt)
                vector_to_point = first_point - position
                camera_in_same_direction = np.dot(vector_to_point, camera_vector) >= 0
                if camera_in_same_direction:
                    num_points_in_front_of_camera = num_points_in_front_of_camera + 1
                else:
                    num_points_in_front_of_camera = -1
                    break

            if num_points_in_front_of_camera > winning_num_points:
                winning_rotation = rotation
                winning_translation = translation
                winning_num_points = num_points_in_front_of_camera

    return winning_rotation, winning_translation
