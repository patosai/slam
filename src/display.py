import sys
sys.path.append("./lib")
import pangolin
import OpenGL.GL as gl


scam = None
dcam = None


def setup_pangolin():
    global dcam, scam
    pangolin.CreateWindowAndBind('Main', 640, 480)
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Define Projection and initial ModelView matrix
    scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
        pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
    handler = pangolin.Handler3D(scam)

    # Create Interactive View in window
    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
    dcam.SetHandler(handler)


def should_quit():
    return pangolin.ShouldQuit()


def draw_points(points):
    pangolin.DrawPoints(points)


def draw_camera(pose, x, y, z):
    pangolin.DrawCamera(pose, x, y, z)


def init_frame():
    global dcam
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glClearColor(1.0, 1.0, 1.0, 1.0)
    dcam.Activate(scam)


def finish_frame():
    pangolin.FinishFrame()
