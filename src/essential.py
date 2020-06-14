import numpy as np

from . import plot, util


def triangulate_points(rotation, translation, img1_points, img2_points, intrinsic_camera_matrix):
    """Triangulate points using linear least squares, where rotation and translation are with respect to camera 1"""
    # Given a rotation and translation between cameras, the projective matrix for each camera can be calculated
    # and used in a linear system of equations to triangulate all points in the two images.
    # Let M be the projective matrix for image 1, and M' the projective matrix for image 2.
    #
    # Let p = [u which is the 2D homogenous coordinates of a point in image 1.
    #          v
    #          1]
    # Let p' = [u' which is the 2D homogenous coordinates of a point in image 2.
    #           v'
    #           1]
    # Let P = [X which is the homogenous coordinates of the 3D location of the point
    #          Y
    #          Z
    #          W]
    #
    # p = MP
    # p' = M'P
    #
    # Since the two sides are equal, their cross product should be 0
    # pxMP = 0
    # p'xM'P = 0
    #
    # Three equations come from each of the equations, but only two are linearly independent.
    # -M_1P + vM_2P = 0, where M_1 is the 1st row of M (zero-indexed)
    # M_0P - uM_2P = 0
    # -vM_0P + uM_1P = 0   <- -1*(u*first line + v*second line)
    #
    # -M'_1P + vM'_2P = 0
    # M'_0P - u'M'_2P = 0
    # -v'M'_0P + u'M'_1P = 0   <- -1*(u'*first line + v'*second line)
    #
    # Let D = [-M_1 + vM_2,
    #          M_0 - uM_2,
    #          -M'_1 + vM'_2,
    #          M'_0 - u'M'_2]
    # DP = 0. SVD can be applied to solve for P.
    # letting SVD(D) = USV, P can be found by taking the vector of U associated with the smallest singular value in S.

    # the camera matrix describes how the WORLD is transformed relative to the CAMERA
    # the rotation and translation are given as the movement of the camera
    # so the rotation/translation need to be inverted
    img1_camera_matrix = intrinsic_camera_matrix @ np.hstack((np.identity(3), np.zeros((3, 1))))
    img2_camera_matrix = intrinsic_camera_matrix @ np.hstack((rotation.transpose(), -translation.reshape((3, 1))))

    triangulated_points = []
    for img1_pt, img2_pt in zip(img1_points, img2_points):
        d = np.asarray([-img1_camera_matrix[1] + img1_pt[1]*img1_camera_matrix[2],
                        img1_camera_matrix[0] - img1_pt[0]*img1_camera_matrix[2],
                        -img2_camera_matrix[1] + img2_pt[1]*img2_camera_matrix[2],
                        img2_camera_matrix[0] - img2_pt[0]*img2_camera_matrix[2]])
        assert d.shape == (4, 4)

        # Scale all elements in D to less than 1 to reduce error
        # The resulting P needs to be scaled up by the same amount
        max_of_rows = np.abs(d).max(axis=1)
        scale_factor = np.diag(max_of_rows)
        scale_factor_inverse = np.diag(1/max_of_rows)

        d_scaled = d @ scale_factor_inverse

        u, s, vh = np.linalg.svd(d_scaled)
        v = vh.transpose()
        assert v.shape == (4, 4)
        point = v[:, -1]
        scaled_point = point @ scale_factor
        triangulated_points.append(scaled_point)


    dimension_per_row = np.asarray(triangulated_points).transpose()
    triangulated_points = np.asarray([dimension_per_row[0] / dimension_per_row[3],
                                      dimension_per_row[1] / dimension_per_row[3],
                                      dimension_per_row[2] / dimension_per_row[3]]).transpose()
    np.set_printoptions(suppress=False)
    print("camera vector")
    print(rotation @ np.asarray([0, 0, 1]))
    print("translation")
    print(translation)
    print("points")
    print(triangulated_points)
    print("====")

    assert len(img1_points) == len(triangulated_points)

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

    # translation X points right, Y points up, Z points into the image
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

            points_in_front_of_camera_1 = triangulated_points[:, 2] > 0
            points_in_front_of_camera_2 = np.dot((triangulated_points - translation), camera_2_vector) > 0
            num_points_in_front_of_camera = np.count_nonzero(np.multiply(points_in_front_of_camera_1, points_in_front_of_camera_2))

            if num_points_in_front_of_camera > winning_num_points:
                winning_rotation = rotation
                winning_translation = translation
                winning_num_points = num_points_in_front_of_camera
                winning_triangulated_points = triangulated_points

    return winning_rotation, winning_translation, winning_triangulated_points
