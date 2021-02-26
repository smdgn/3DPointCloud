# Class PointMatcher:
# Class performing brute-force point matching, given left and right keypoints.
# Keypoints and descriptors are obtained via ORB.
# Written by Samed Dogan, University of applied sciences

import numpy as np
import cv2
from scipy.spatial import distance


class PointMatcher:

    def __init__(self):

        self.matches = None
        self.l_kp = None
        self.r_kp = None
        self.l_des = None
        self.r_des = None

    def __find_pairs__(self, l_des, r_des):

        """Find corresponding pairs of left and right keypoints
        Input: left-keypoints, right_keypoints, left_descriptor, right_descriptor
        Return: DMatch Obj for proper use with opencv"""

        # Brute-force Approach: Measure the distance to each point, take the smallest distance
        # either hamming distance for ORB pairs, or L2 for SIFT, SURF

        keypoints = []
        for l_index, l_d in enumerate(l_des):
            min_distance = np.inf
            min_index = 0
            for r_index, r_d in enumerate(r_des):
                # dis = np.sqrt(np.sum(np.power(l_point - r_point, 2)))
                dis = distance.hamming(l_d.flatten(), r_d.flatten())
                if dis < min_distance:
                    min_distance = dis
                    min_index = r_index
            # save found point-pair
            keypoints.append(cv2.DMatch(_imgIdx=0, _queryIdx=l_index,
                                        _trainIdx=min_index, _distance=min_distance))
        return sorted(keypoints, key=lambda x: x.distance)

    def __check_inputs__(self, left_image, right_image):
        if len(np.shape(left_image)) == 3:
            left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        if len(np.shape(right_image)) == 3:
            right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        return left_image, right_image

    def match(self, left_image, right_image, n=1000):
        """Retrieve Image Keypoints with Orb detector and perform point-matching
        Input: Query-Image, Train-Image, number of features to
        Output: DMatch Obj-Array"""
        left_image, right_image = self.__check_inputs__(left_image, right_image)
        det = cv2.ORB_create(nfeatures=n)
        # TODO: Check image format assert...

        l_kp, l_des = det.detectAndCompute(left_image, None)
        r_kp, r_des = det.detectAndCompute(right_image, None)

        matches = self.__find_pairs__(l_des, r_des)
        self.matches = matches
        self.l_kp = l_kp
        self.l_des = l_des
        self.r_kp = r_kp
        self.r_des = r_des

        return matches

    def get_coordinates(self):
        """Return image coordinates of matched keypoint-pairs as n_array((x,y))"""

        if self.matches is None:
            raise ValueError("Matches cant be None. Perform matching first")

        # points are ordered in y,x
        left_points = [(np.ceil(self.l_kp[match.queryIdx].pt)).astype(int) for match in self.matches]
        right_points = [(np.ceil(self.r_kp[match.trainIdx].pt)).astype(int) for match in self.matches]

        # reverse order , add 1
        left_points = [(points[0], points[1], 1) for points in left_points]
        right_points = [(points[0], points[1], 1) for points in right_points]

        return np.array(left_points), np.array(right_points)



