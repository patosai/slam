import numpy as np

from . import plot, util


def triangulate_points(rotation, translation, img1_points, img2_points, intrinsic_camera_matrix):
    """Triangulate points using linear least squares, where rotation and translation are with respect to camera 1"""
    # Given a rotation and translation between cameras, the projective matrix for each camera can be calculated
    # and used in a linear system of equations to triangulate all points in the two images.
    # Let M be the projective matrix for image 1, and M' the projective matrix for image 2.
    #
    # Let p = [u which is the 2D homogenous coordinates of a point in image 1.
    #          w
    #          1]
    # Let p' = [u' which is the 2D homogenous coordinates of a point in image 2.
    #           w'
    #           1]
    # Let P = [X which is the homogenous coordinates of the 3D location of the point
    #          Y
    #          Z
    #          W]
    #
    # p = MP
    # p' = M'P
    #
    # Because the two sides are equal, the cross product of the two sides should equal zero.
    # Then SVD can be applied to solve for P, similarly to the fundamental matrix.
    # Take the vector associated with the singular value of 0
    # ([p]_x)MP = 0
    # ([p']_x)M'P = 0

    img1_projective_matrix = intrinsic_camera_matrix @ np.hstack((np.identity(3), np.zeros((3, 1))))
    img2_projective_matrix = intrinsic_camera_matrix @ np.hstack((rotation, translation.reshape((3, 1))))
    all_projective_matrices = np.vstack((img1_projective_matrix, img2_projective_matrix))

    img1_homogenous_coordinates = np.hstack((img1_points, np.ones((len(img1_points), 1))))
    img2_homogenous_coordinates = np.hstack((img2_points, np.ones((len(img2_points), 1))))

    triangulated_points = []
    for img1_pt, img2_pt in zip(img1_homogenous_coordinates, img2_homogenous_coordinates):
        # A has shape 6x4
        a = np.vstack((util.vector_to_cross_product_matrix(img1_pt) @ img1_projective_matrix,
                       util.vector_to_cross_product_matrix(img2_pt) @ img2_projective_matrix))
        assert a.shape == (6, 4)
        u, s, vh = np.linalg.svd(a, full_matrices=False)
        print("------")
        print("s")
        print(s)
        print("------")
        assert u.shape[1] == 4
        point = u[:, -1]
        triangulated_points.append(point)

    dimension_per_row = np.asarray(triangulated_points).transpose()
    triangulated_points = np.asarray([dimension_per_row[0] / dimension_per_row[3],
                                      dimension_per_row[1] / dimension_per_row[3],
                                      dimension_per_row[2] / dimension_per_row[3]]).transpose()

    assert len(img1_points) == len(triangulated_points)

    print("triangulated points")
    np.set_printoptions(suppress=True)
    print(triangulated_points)

    return triangulated_points


def essential_matrix_to_rotation_translation(essential_matrix, img1_points, img2_points, intrinsic_camera_matrix):
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
    winning_triangulated_points = None
    for rotation in possible_rotations:
        for translation in possible_translations:
            triangulated_points = triangulate_points(rotation,
                                                     translation,
                                                     img1_points,
                                                     img2_points,
                                                     intrinsic_camera_matrix)
            camera_2_vector = rotation @ np.asarray([0, 0, 1])
            print("camera vector")
            print(camera_2_vector)
            print("translation")
            print(translation)

            points_in_front_of_camera_1 = triangulated_points[:, 2] > 0
            points_in_front_of_camera_2 = np.dot((triangulated_points - translation), camera_2_vector) > 0
            num_points_in_front_of_camera = np.count_nonzero(np.multiply(points_in_front_of_camera_1, points_in_front_of_camera_2))
            print(num_points_in_front_of_camera)
            print('----')

            if num_points_in_front_of_camera > winning_num_points:
                winning_rotation = rotation
                winning_translation = translation
                winning_num_points = num_points_in_front_of_camera
                winning_triangulated_points = triangulated_points

    print("==========")
    print("winning rotation")
    print(winning_rotation)
    print("winning translation")
    print(winning_translation)
    print("camera_vector")
    print(winning_rotation @ np.asarray([0, 0, 1]))
    print("num points in front: ", winning_num_points)
    print("------")

    return winning_rotation, winning_translation, winning_triangulated_points
