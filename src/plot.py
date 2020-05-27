import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_image_keypoints(image, image_keypoints):
    # draw only keypoints location,not size and orientation
    drawn_image = cv2.drawKeypoints(image, image_keypoints, cv2.DRAW_MATCHES_FLAGS_DEFAULT, (0,255,0))
    plt.imshow(drawn_image)
    plt.show()


def plot_vectors(vector_list):
    vector_map = [dict([("start_point", [0, 0, 0]), ("vector", vector)]) for vector in vector_list]
    plot_vectors_with_starting_point(vector_map)


def plot_vectors_with_starting_point(vector_map):
    for item in vector_map:
        assert("start_point" in item)
        assert("vector" in item)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for item in vector_map:
        start_point = item["start_point"]
        vector = item["vector"]
        v = np.array(vector)
        vlength = np.linalg.norm(v)
        ax.quiver(start_point[0],
                  start_point[1],
                  start_point[2],
                  vector[0],
                  vector[1],
                  vector[2],
                  pivot='tail',
                  length=vlength,
                  arrow_length_ratio=0.3/vlength)
    ax.set_xlim([-4,4])
    ax.set_ylim([-4,4])
    ax.set_zlim([0,4])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


