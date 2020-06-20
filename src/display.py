import sys
sys.path.append("./lib")
import pangolin
import numpy as np
import OpenGL.GL as gl


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
    width = 0.2
    height_ratio = 0.75
    z_ratio = 0.8
    gl.glLineWidth(1)
    gl.glColor3f(color[0], color[1], color[2])
    homogenous_pose = np.vstack((pose, [0, 0, 0, 1]))
    pangolin.DrawCamera(homogenous_pose, width, height_ratio, z_ratio)


def init_frame():
    global dcam, scam
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glClearColor(0.0, 0.0, 0.0, 1.0)
    dcam.Activate(scam)


def finish_frame():
    pangolin.FinishFrame()
