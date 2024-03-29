import math
import numpy as np
import random

from . import logger, plot, util


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
    # letting SVD(D) = U*S*V, P can be found by taking the vector of U associated with the smallest singular value in S.

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


def triangulate_pose_from_points(image_points, known_3d_points, intrinsic_camera_matrix):
    """Triangulate camera pose from known 3D points and 2D matches in the camera image"""
    # TODO this might not work well because triangulated points might not be that accurate
    assert len(known_3d_points) == len(image_points)
    assert len(known_3d_points) >= 6
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

    # https://www.uio.no/studier/emner/matnat/its/nedlagte-emner/UNIK4690/v17/forelesninger/lecture_5_2_pose_from_known_3d_points.pdf
    # P = K[R|t]
    #   = K[R|-RC]
    # P = [M|-MC]
    # Calculate camera center C by taking SVD of P, since PC = 0
    # To calculate rotation, do RQ decomposition. R is a right triangular matrix (intrinsic camera matrix), Q is an orthogonal matrix (rotation matrix)
    # To calculate translation, t = -RC

    a = np.concatenate(equations, axis=0)
    # once again, use SVD to find x; it's the column in V corresponding to the smallest singular value
    u, s, vh = np.linalg.svd(a)
    matrix_variables = vh[-1]
    camera_matrix = matrix_variables.reshape((3, 4))
    camera_matrix = camera_matrix * np.sign(np.linalg.det(camera_matrix[:3, :3]))

    u, s, vh = np.linalg.svd(camera_matrix)
    camera_center = vh[-1, :3] / vh[-1, -1]

    r, q = util.rq_decomposition(camera_matrix[:3, :3])
    # enforce positive diagonal of K
    diagonal = np.diag(np.sign(np.diag(r)))
    calculated_intrinsic_camera_matrix = r @ diagonal
    rotation = diagonal @ q
    translation = -1 * rotation @ camera_center
    print("calculated intrinsic camera matrix")
    print(calculated_intrinsic_camera_matrix)
    print("rotation")
    print(rotation)
    print("translation")
    print(translation)
    return util.rotation_translation_to_pose(rotation, translation)


def triangulate_pose_from_points_with_ransac(previous_camera_pose,
                                             matching_previous_image_points_with_3d_position,
                                             matching_next_image_points_with_3d_position,
                                             matching_3d_points,
                                             matching_previous_image_points,
                                             matching_next_image_points,
                                             intrinsic_camera_matrix,
                                             iterations=100):
    assert len(matching_previous_image_points_with_3d_position) == len(matching_next_image_points_with_3d_position) == len(matching_3d_points)
    assert len(matching_3d_points) >= 6
    max_iterations = math.comb(len(matching_3d_points), 6)
    iterations = min(iterations, max_iterations)
    all_indices = range(len(matching_next_image_points_with_3d_position))

    winning_pose = None
    winning_num_good_points = -1
    winning_error = None

    for iteration in range(iterations):
        random.seed(0x1337BEEF + iteration)
        chosen_indices = random.sample(all_indices, 6)
        next_camera_pose = triangulate_pose_from_points(matching_next_image_points_with_3d_position[chosen_indices],
                                                        matching_3d_points[chosen_indices],
                                                        intrinsic_camera_matrix)
        triangulated_points = triangulate_points_from_pose(intrinsic_camera_matrix @ previous_camera_pose[:3],
                                                           intrinsic_camera_matrix @ next_camera_pose[:3],
                                                           matching_previous_image_points_with_3d_position,
                                                           matching_next_image_points_with_3d_position)
        print(triangulated_points)
        rotation = util.pose_to_rotation(next_camera_pose)
        translation = util.pose_to_translation(next_camera_pose)
        camera_2_vector = rotation @ np.asarray([0, 0, 1])
        points_in_front_of_camera_1 = triangulated_points[:, 2] > 0
        points_in_front_of_camera_2 = np.dot((triangulated_points - translation), camera_2_vector) > 0
        num_points_in_front_of_cameras = np.count_nonzero(np.multiply(points_in_front_of_camera_1, points_in_front_of_camera_2))
        error = np.sum(np.linalg.norm(matching_3d_points - triangulated_points, axis=1))

        if num_points_in_front_of_cameras > winning_num_good_points \
                or (num_points_in_front_of_cameras == winning_num_good_points and winning_error > error):
            winning_pose = next_camera_pose
            winning_num_good_points = num_points_in_front_of_cameras
            winning_error = error
    logger.info("Pose from points, num good points: ", winning_num_good_points, "/", len(matching_3d_points))
    assert winning_pose.shape == (4, 4)
    return winning_pose, triangulate_points_from_pose(previous_camera_pose,
                                                      winning_pose,
                                                      matching_previous_image_points,
                                                      matching_next_image_points)

# https://www-users.cs.umn.edu/~hspark/CSci5980/Lec15_PnP.pdf


if __name__ == "__main__":
    known_3d_points = np.asarray([[ 6.03787985, -4.89579876, 62.66137844],
                                  [-8.39317134, 3.03552779, 18.92342537],
                                  [-15.53855632, -2.46611132, 92.2252502],
                                  [-2.58636809, 0.3066047, 20.0799362],
                                  [-16.17911693, -0.85989838, 44.88294597],
                                  [-18.83353488, -10.9979917, 71.92970211],
                                  [2.28097053, -0.6059205, 58.0935008],
                                  [3.27491554, -1.18588676, 85.08488668]])
    image_points = np.asarray([[783.60003662, 152.40000916],
                               [142.56001282, 427.68002319],
                               [513.60003662, 204.00001526],
                               [504.5760498, 250.56002808],
                               [299.52001953, 213.12001038],
                               [413.2800293, 73.44000244],
                               [726.58959961, 221.46052551],
                               [727.38592529, 218.57409668]])
    intrinsic_camera_matrix = np.asarray([[9.842439e+02, 0.000000e+00, 6.900000e+02],
                                          [0.000000e+00, 9.808141e+02, 2.331966e+02],
                                          [0.000000e+00, 0.000000e+00, 1.000000e+00]])
    pose = triangulate_pose_from_points(image_points, known_3d_points, intrinsic_camera_matrix)
    print(pose)
    print(util.pose_to_translation(pose))
    print(util.pose_to_rotation(pose))
