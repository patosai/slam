import numpy as np


def pixel_to_camera_coords(xy_point):
    "takes a 2d point and adds a z-coordinate set to 1"
    return np.asarray([xy_point[0], xy_point[1], 1])
