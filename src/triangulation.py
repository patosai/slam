import numpy as np
import random

from . import logger, plot, util


# TODO delete
import cv2
import matplotlib.pyplot as plt


def triangulate_points_from_pose(img0_camera_matrix, img1_camera_matrix, img0_points, img1_points):
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


def triangulate_pose_from_points(known_3d_points, image_points, intrinsic_camera_matrix):
    """Triangulate camera pose from known 3D points and 2D matches in the camera image"""
    assert len(known_3d_points) == len(image_points)
    # Similar to above, p = MP, where
    #   - p is the 3x1 2D homogenous point, p = [u, v, 1]
    #   - M is the 3x4 camera matrix (UNKNOWN)
    #   - P is the 4x1 3D homogenous point, P = [X, Y, Z, 1]
    # Applying the same cross product thing gives (once again, matrices/vectors are zero-indexed to look like the code)
    #   (-M_1 + vM_2)P = 0
    #   (M_0 - uM_2)P = 0
    # Breaking down these equations gives
    #   X(-M_1,0 + vM_2,0) + Y(-M_1,1 + vM_2,1) + Z(-M_1,2 + vM_2,3) + (-M_1,3 + vM_2,3) = 0
    #   X(M_0,0 - uM_2,0) + Y(M_0,1 - uM_2,1) + Z(M_0,2 - uM_2,2) + (M_0,3 - uM_2,3) = 0
    # M has 11 unknowns (12 variables - 1 scale factor)
    # Each 2D/3D correspondence gives 2 equations
    # 11/2 = 6 correspondences required
    #
    # Letting x equal the unknowns of M ([M_0,0, M_0,1, ..., M_3,3, M_3,4]), Ax = 0
    equations = [[[0, 0, 0, 0, -x, -y, -z, -1, v*x, v*y, v*z, v],
                  [x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z, -u]]
                 for [u, v], [x, y, z] in zip(image_points, known_3d_points)]
    a = np.concatenate(equations, axis=0)
    # once again, use SVD to find x; it's the column in V corresponding to the smallest singular value
    u, s, vh = np.linalg.svd(a)
    matrix_variables = vh[-1]
    camera_matrix = matrix_variables.reshape((3, 4))

    result = np.linalg.inv(intrinsic_camera_matrix) @ camera_matrix
    determinant = np.linalg.det(result[:3, :3])
    scale_factor = np.cbrt(determinant)
    result = result / scale_factor
    rotation = result[:3, :3]
    translation = result[:3, 3]
    return util.rotation_translation_to_pose(rotation, translation)

    # # decompose left 3x3 matrix with SVD to recover rotation matrix and scale factor
    # u, s, vh = np.linalg.svd(result[:, :3])
    # scale_factor = s[0]
    # rotation = u @ vh
    # translation = result[:, 3] / scale_factor
    # return util.rotation_translation_to_pose(rotation, translation)


def triangulate_pose_from_points_with_ransac(previous_camera_pose,
                                             known_3d_points,
                                             previous_image_points,
                                             image_points,
                                             intrinsic_camera_matrix,
                                             iterations=8):
    assert len(known_3d_points) == len(image_points)
    all_indices = range(len(image_points))
    known_3d_points = np.asarray(known_3d_points)
    image_points = np.asarray(image_points)

    good_point_distance_cutoff = 10

    winning_pose = None
    winning_num_good_points = -1
    winning_error = None
    winning_triangulated_points = None

    for iteration in range(iterations):
        random.seed(0x1337BEEF + iteration)
        chosen_indices = random.sample(all_indices, 6)
        next_camera_pose = triangulate_pose_from_points(known_3d_points[chosen_indices],
                                                        image_points[chosen_indices],
                                                        intrinsic_camera_matrix)
        triangulated_points = triangulate_points_from_pose(previous_camera_pose,
                                                           next_camera_pose,
                                                           previous_image_points,
                                                           image_points)
        distances_from_known = np.linalg.norm(known_3d_points - triangulated_points, axis=1)
        num_good_points = np.count_nonzero(distances_from_known < good_point_distance_cutoff)
        error = np.sum(distances_from_known)
        if (num_good_points > winning_num_good_points) or (num_good_points == winning_num_good_points and error < winning_error):
            winning_pose = next_camera_pose
            winning_num_good_points = num_good_points
            winning_error = error
            winning_triangulated_points = triangulated_points

    logger.info("Triangulated pose from points - translation:", util.pose_to_translation(winning_pose),
                "camera vector:", util.pose_to_rotation(winning_pose) @ [0, 0, 1],
                "num points matched:", winning_num_good_points)
    return winning_pose, winning_triangulated_points

# https://www-users.cs.umn.edu/~hspark/CSci5980/Lec15_PnP.pdf


if __name__ == "__main__":
    known_3d_points = np.asarray([[1.58053041e+00, -3.20156923e+00, 5.03916598e+01],
                                  [-8.09583849e+00, -1.05173228e+00, 6.12705667e+01],
                                  [-2.16507803e+00, -3.10635558e+00, 7.60814226e+01],
                                  [-6.34364941e-02, -5.63441545e-01, 5.80920451e+01],
                                  [-2.82575971e+00, -3.58970662e+00, 8.17548681e+01],
                                  [2.37329213e+00, -2.21337026e+00, 7.51611333e+01],
                                  [-3.38914767e+00, -2.63351425e+00, 7.51221960e+01],
                                  [-7.37504688e+00, 2.69810112e+00, 1.66449735e+01],
                                  [-1.02491303e+01, -1.56004211e+00, 6.09047941e+01],
                                  [-1.90565982e+00, 2.43619355e-01, 1.48406014e+01],
                                  [-2.88958131e+00, -1.75010805e+00, 6.54434401e+01],
                                  [-1.33287248e+01, -6.45483875e-01, 3.70017434e+01],
                                  [-1.50247316e+01, -8.67126947e+00, 5.73489786e+01],
                                  [1.39182190e+00, -8.09629071e-01, 5.34166778e+01],
                                  [3.10346764e+00, -1.18874630e+00, 7.46361407e+01],
                                  [3.10363168e+00, -1.05783545e+00, 7.65939931e+01],
                                  [-2.50051654e+00, -2.05662026e+00, 5.21600525e+01],
                                  [8.09005253e-01, -2.07296511e-01, 2.05331117e+01],
                                  [9.88787974e-01, -3.47755336e-01, 2.55688033e+01]])
    previous_image_points = np.asarray([[720., 171.],
                                        [557., 217.],
                                        [661., 194.],
                                        [688., 224.],
                                        [655., 191.],
                                        [721., 205.],
                                        [644.40002441, 199.20001221],
                                        [222.00001526, 404.40002441],
                                        [520.80004883, 208.80000305],
                                        [548.64001465, 252.00001526],
                                        [645.11999512, 207.36001587],
                                        [324.        , 217.44000244],
                                        [426.24002075,  84.96000671],
                                        [715.39208984, 219.45602417],
                                        [730.9440918 , 217.72802734],
                                        [729.9072876 , 219.80163574],
                                        [640.74249268, 194.91842651],
                                        [726.58959961, 223.94885254],
                                        [727.38592529, 222.15727234]])
    image_points = np.asarray([[720,         166],
                               [553,         213],
                               [660,         190],
                               [687,         220],
                               [654,         186],
                               [721,         200],
                               [643.20001221, 194.40000916],
                               [142.56001282, 427.68002319],
                               [513.60003662, 204.00001526],
                               [504.5760498,  250.56002808],
                               [643.68005371, 203.04000854],
                               [299.52001953, 213.12001038],
                               [413.2800293,   73.44000244],
                               [715.39208984, 214.27201843],
                               [730.9440918,  214.27201843],
                               [729.9072876,  215.6544342, ],
                               [640.74249268, 192.84483337],
                               [726.58959961, 221.46052551],
                               [727.38592529, 218.57409668]])
    previous_pose = np.asarray([[9.99985692e-01, 5.28993613e-03, 7.94646417e-04, -9.19542164e-02],
                                [-5.28994719e-03, 9.99986008e-01, 1.18131948e-05, 3.80877809e-02],
                                [-7.94572807e-04, -1.60166633e-05,  9.99999684e-01, -9.95034544e-01]])
    intrinsic_camera_matrix = np.asarray([[9.842439e+02, 0.000000e+00, 6.900000e+02],
                                          [0.000000e+00, 9.808141e+02, 2.331966e+02],
                                          [0.000000e+00, 0.000000e+00, 1.000000e+00]])

    chosen_indices = [1, 2, 7, 8, 11, 12]

    pose = triangulate_pose_from_points(known_3d_points[chosen_indices],
                                        image_points[chosen_indices],
                                        intrinsic_camera_matrix)
    print(known_3d_points[chosen_indices])
    print(image_points[chosen_indices])
    print("camera vector:", util.pose_to_rotation(pose) @ [0, 0, 1], "translation: ", util.pose_to_translation(pose))

    show_image = True
    if show_image:
        all_indices = range(len(image_points))
        figure, axes = plt.subplots()
        axes.clear()
        for point in image_points:
            circle = plt.Circle(point, radius=5, color='#FF0000', fill=False)
            axes.add_artist(circle)
        for point in image_points[chosen_indices]:
            circle = plt.Circle(point, radius=5, color='#00FF00', fill=False)
            axes.add_artist(circle)
        axes.imshow(cv2.imread("data/0000000005.png"))
        plt.show()

