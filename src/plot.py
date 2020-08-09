import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_image_keypoints(image, image_keypoints):
    # draw only keypoints location,not size and orientation
    drawn_image = cv2.drawKeypoints(image, image_keypoints, cv2.DRAW_MATCHES_FLAGS_DEFAULT, (0,255,0))
    plt.imshow(drawn_image)
    plt.show()


def plot_image_matches(image1, image1_points, image2, image2_points):
    figure, axes = plt.subplots()
    axes.clear()
    plt.imshow(image1, alpha=0.5)
    plt.imshow(image2, alpha=0.5)
    for point in image1_points:
        circle = plt.Circle(point, radius=5, color='#ff245c', fill=False)
        axes.add_artist(circle)
    for point in image2_points:
        circle = plt.Circle(point, radius=5, color='#24d2ff', fill=False)
        axes.add_artist(circle)
    for image1_pt, image2_pt in zip(image1_points, image2_points):
        line = plt.Line2D([image1_pt[0], image2_pt[0]],
                          [image1_pt[1], image2_pt[1]],
                          linewidth=1,
                          color='#e7ed46')
        axes.add_artist(line)
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


if __name__ == "__main__":
    orb_detector = cv2.ORB_create()
    def compute_orb(img):
        """find the keypoints with ORB"""
        keypoints = orb_detector.detect(img, None)
        # compute the descriptors with ORB
        keypoints, descriptors = orb_detector.compute(img, keypoints)
        descriptor_bits = np.unpackbits(np.uint8(descriptors), axis=1)
        return [keypoint.pt for keypoint in keypoints], descriptor_bits

    def match_knn(img0_descriptors, img1_descriptors, k=2):
        """Finds k nearest descriptors in img1 for each descriptor in img0 """
        matches = []
        for img0_idx, img0_descriptor in enumerate(img0_descriptors):
            possible_matches = []
            for img1_idx, img1_descriptor in enumerate(img1_descriptors):
                if img0_idx != img1_idx:
                    hamming_distance = np.count_nonzero(img0_descriptor != img1_descriptor)
                    possible_matches.append({"distance": hamming_distance, "index": img1_idx})
            possible_matches.sort(key=lambda x: x["distance"])
            limited_matches = possible_matches[:k]
            matches.append(limited_matches)
        return matches

    img0 = cv2.imread("data/0000000002.png")
    img1 = cv2.imread("data/0000000003.png")

    img0_pts, img0_des = compute_orb(img0)
    img1_pts, img1_des = compute_orb(img1)
    matches = match_knn(img0_des, img1_des, k=2)

    good_img0_points = []
    good_img1_points = []

    for idx, (m, n) in enumerate(matches):
        # Lowe's ratio test
        first_match_much_better = m["distance"] < 0.5 * n["distance"]
        if first_match_much_better:
            img0_point = np.asarray(img0_pts[idx])
            img1_point = np.asarray(img1_pts[m["index"]])

            good_img0_points.append(img0_point)
            good_img1_points.append(img1_point)

    good_img0_points = np.asarray(good_img0_points)
    good_img1_points = np.asarray(good_img1_points)
    plot_image_matches(img0, good_img0_points, img1, good_img1_points)