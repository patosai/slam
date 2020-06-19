import numpy as np

from . import plot, util


def triangulate_points_from_pose(rotation, translation, img0_points, img1_points, intrinsic_camera_matrix):
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
    img0_camera_matrix = intrinsic_camera_matrix @ np.hstack((np.identity(3), np.zeros((3, 1))))
    img1_camera_matrix = intrinsic_camera_matrix @ np.hstack((rotation.transpose(), -1*translation.reshape((3, 1))))

    triangulated_points = []
    for img0_pt, img1_pt in zip(img0_points, img1_points):
        d = np.asarray([-1*img0_camera_matrix[1] + img0_pt[1]*img0_camera_matrix[2],
                        img0_camera_matrix[0] - img0_pt[0]*img0_camera_matrix[2],
                        -1*img1_camera_matrix[1] + img1_pt[1]*img1_camera_matrix[2],
                        img1_camera_matrix[0] - img1_pt[0]*img1_camera_matrix[2]])
        assert d.shape == (4, 4)

        # Scale all elements in D to less than 1 to reduce error
        # The resulting P needs to be scaled up by the same amount
        max_of_rows = np.abs(d).max(axis=1)
        scale_factor = np.diag(max_of_rows)
        scale_factor_inverse = np.diag(1/max_of_rows)

        d_scaled = d @ scale_factor_inverse

        u, s, vh = np.linalg.svd(d_scaled)
        assert vh.shape == (4, 4)
        point = vh[-1]
        scaled_point = scale_factor @ point
        triangulated_points.append(scaled_point)

    dimension_per_row = np.asarray(triangulated_points).transpose()
    triangulated_points = np.asarray([dimension_per_row[0] / dimension_per_row[3],
                                      dimension_per_row[1] / dimension_per_row[3],
                                      dimension_per_row[2] / dimension_per_row[3]]).transpose()
    assert len(img0_points) == len(triangulated_points)
    return triangulated_points


def triangulate_pose_from_points(known_points, image_points):
    # TODO
    pass