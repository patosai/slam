import sys
sys.path.append("./lib")
import pangolin
import numpy as np
import OpenGL.GL as gl
from . import util


scam = None
dcam = None


def setup():
    global dcam, scam
    pangolin.CreateWindowAndBind('Main', 640, 480)
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Define Projection and initial ModelView matrix
    width = 640
    height = 480
    z_near = 0.2
    z_far = 1000
    scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(width, height, 420, 420, 320, 240, z_near, z_far),
        # Negative Y direction is up since top left corner of images is (0,0)
        pangolin.ModelViewLookAt(-2, -2, -4, 0, 0, 0, pangolin.AxisDirection.AxisNegY))
    handler = pangolin.Handler3D(scam)

    # Create Interactive View in window
    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
    dcam.SetHandler(handler)


def should_quit():
    return pangolin.ShouldQuit()


def draw_points(points):
    gl.glPointSize(2)
    gl.glColor3f(1.0, 0.0, 0.0)
    pangolin.DrawPoints(points)


def draw_camera(pose, color=(0.0, 1.0, 0.0)):
    assert pose.shape == (4, 4)
    assert len(color) == 3
    width = 0.2
    height_ratio = 0.75
    z_ratio = 0.8
    gl.glLineWidth(1)
    gl.glColor3f(color[0], color[1], color[2])

    # convert pose to reflect camera position with respect to world coordinate origin

    # TODO wtf why doesn't this work?
    # new_pose = util.rotation_translation_to_pose(util.pose_to_rotation(pose).transpose(),
    #                                              util.pose_to_translation(pose)*-1)

    # do this instead of the above
    new_pose = np.identity(4)
    new_pose[:3, :3] = pose[:3, :3].transpose()
    new_pose[:3, 3] = pose[:3, 3] * -1

    pangolin.DrawCamera(new_pose, width, height_ratio, z_ratio)


def init_frame():
    global dcam, scam
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glClearColor(0.0, 0.0, 0.0, 1.0)
    dcam.Activate(scam)


def finish_frame():
    pangolin.FinishFrame()


if __name__ == "__main__":
    import math
    from . import util

    rotation = np.identity(3)
    translation = np.array([0, 1, 1])
    pose = util.rotation_translation_to_pose(rotation, translation)

    # counterclockwise
    angle = math.pi/4
    # next_rotation = np.asarray([[math.cos(angle), -math.sin(angle), 0],
    #                             [math.sin(angle), math.cos(angle), 0],
    #                             [0, 0, 1]])
    next_rotation = np.asarray([[1, 0, 0],
                                [0, math.cos(angle), -math.sin(angle)],
                                [0, math.sin(angle), math.cos(angle)]])
    next_translation = np.array([0, 1, 0])
    next_pose = util.rotation_translation_to_pose(next_rotation, next_translation)
    next_pose = pose @ next_pose


    angle = math.pi/4
    next2_rotation = np.asarray([[1, 0, 0],
                                [0, math.cos(angle), -math.sin(angle)],
                                [0, math.sin(angle), math.cos(angle)]])
    next2_translation = np.array([0, 1, 0])
    next2_pose = util.rotation_translation_to_pose(next2_rotation, next2_translation)
    next2_pose = next_pose @ next2_pose

    setup()
    while True:
        init_frame()
        draw_points([[0, 0, 0]])
        draw_camera(pose, (1, 0, 0))
        draw_camera(next_pose, (0, 1, 0))
        draw_camera(next2_pose, (0, 0, 1))
        finish_frame()
