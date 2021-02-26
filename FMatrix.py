# Class FMatrix:
# FMatrix estimates the optimal fundamentalmatrix given point correspondences
# via normalized 8-point algorithm and levenberg-marquardt optimisation
# Robust estimation is done via RANSAC, calculating the minimal sampson distance of given points.
#
# Written by Samed Dogan, University of applied sciences

import numpy as np
import cv2
from scipy.optimize import least_squares
import scipy.optimize


class FMatrix:
    """Compute the Fundamental Matrix from Point Correspondences"""

    def __init__(self):
        self.F = np.zeros((3, 3))
        self.el = np.zeros(3)
        self.er = np.zeros(3)

    def __coeff__(self, p1, p2):
        """return coefficients for two corresponding points p1, p1 with
        p1 = (x1,y1) """
        x1, y1, _ = p1
        x2, y2, _ = p2
        return np.array([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1])

    def __coeffM__(self, p1, p2):
        """return  coeffMatrix for correspondence-points of shape (n, 2)"""
        return np.array([self.__coeff__(point1, point2) for point1, point2 in zip(p1, p2)])

    def __compN__(self, points):
        """ compute normalization Matrix for points xi,yi  according to
        and multiple view geometry by hartley and zisserman"""

        # compute the mean or centroid of the points (x,y) -> avg(x) and avg(y)
        pm = np.mean(points.astype(np.float64), axis=0, dtype=np.float64)

        # assure distance between centroid and point to be sqrt(2)
        # explicit implementation acc to 3D-Computer Vision by C. Wöhler p. 25f.
        # and Revisiting Hartley’sNormalized Eight-Point Algorithm by W. Chonacki et al.
        distance = [np.power(points[:, i] - pm[i], 2) for i in range(2)]
        s = np.sqrt(2) / np.sqrt(np.mean(np.sum(distance, axis=0)))

        # explicit implementation acc to Numercial Stabilty of the 8 - point algortihm by P. Lamoureux
        # distance = np.sqrt(np.sum(distance, axis=0))
        # s = np.sqrt(2) / np.mean(distance)

        # build the normalization matrix
        return np.array([[s, 0, -pm[0] * s],
                         [0, s, -pm[1] * s],
                         [0, 0,         1]])

    def __cost__(self, points_l, points_r, F):
        """Implements the sampson distance, used as cost function for ransac
        according to 'Multiple View Geometry', Zisserman, p.287 Equ. 11.9"""

        epsilon = 1e-20
        cost = []
        for p1, p2 in zip(points_l, points_r):
            nominator = np.power(p2.T @ F @ p1, 2)
            denominator = np.sum(np.power((F @ p1)[:2], 2) + np.power((F.T @ p2)[:2], 2)) + epsilon
            cost.append(nominator / denominator)
        return np.array(cost)

    def __compute_epipoles__(self, F):
        # Compute the epipoles -> Nullspace of F and F'
        U, S, V = np.linalg.svd(F)
        el = V[-1] / V[-1][2]
        er = U[:, -1] / U[:, -1][2]
        return el, er

    def __estimateF__(self, p1, p2):
        """Computes the Fundamental Matrix and
        corresponding epipoles el, er via Point Correspondences P1 and P2"""

        # Get normalisation Matrix N and normalize correspondences
        Nl = self.__compN__(p1)
        Nr = self.__compN__(p2)
        p1 = np.array([Nl @ p for p in p1])
        p2 = np.array([Nr @ p for p in p2])

        coefficients = self.__coeffM__(p1, p2)

        # Compute the Nullspace of Ax = b via SVD. The Last Column of V Represents the Nullspace of the linear Equation
        U, S, V = np.linalg.svd(coefficients)
        F = V[-1].reshape(3, 3)

        # A Further Constraint must be applied, ensuring Rank2 of the Matrix
        # Set the lowest singular value to 0 and recalculate the Matrix see J.P.Siebert [43f.]
        U, S, V = np.linalg.svd(F)
        S[2] = 0
        F = np.matmul(U, np.matmul(np.diag(S), V))

        # denormalize
        F = Nr.T @ F @ Nl

        # get epipoles
        el, er = self.__compute_epipoles__(F)

        return F, el, er

    def __ransac__(self, points_l, points_r):

        """Register outliers and compute the most fitting F-Matrix
        according to 'Multiple View Geometry', Zisserman, p.291 Alg. 11.4"""
        # set ransac parameters
        t = 1.25       # threshold, can be 1.25
        samples = 8
        epsilon = 0.4   # conservative outlier estimate, will be changed in runtime
        p = 0.99        # probability that at least one subset is free from outliers
        N = np.inf      # iterates n times, estimated adaptively

        # set algorithm parameter
        total_count = len(points_l)
        final_inliers = np.zeros(total_count, dtype=bool)
        final_F = None
        final_el = None
        final_er = None
        i = 0

        while i < N:
            # randomly select 8 point-correspondences
            rand_index = np.random.randint(total_count, size=8)
            p_l = points_l[rand_index]
            p_r = points_r[rand_index]

            # estimate F
            F, el, er = self.__estimateF__(p_l, p_r)

            # calculate the sampson distance of every point
            # choose points within threshold t to be inliners, and create a subset of points
            d = self.__cost__(points_l, points_r, F)
            inliers = np.abs(d) < t
            inlier_count = np.sum(inliers)
            final_inlier_count = np.sum(final_inliers)

            # choose F with the largest number of inliers
            # in case of ties, choose the solution with smallest std of inliers
            if inlier_count > final_inlier_count:
                final_inliers = inliers
                final_F = F
                final_el = el
                final_er = er
                # recalculate N and epsilon
                epsilon = 1 - (inlier_count / total_count)
                N = np.log(1 - p) / np.log(1 - np.power(1 - epsilon, samples))
            elif inlier_count == final_inlier_count:
                current_subset = points_l[inliers], points_r[inliers]
                previous_subset = points_l[final_inliers], points_r[final_inliers]
                curr_d = self.__cost__(current_subset[0], current_subset[1], F)
                prev_d = self.__cost__(previous_subset[0], previous_subset[1], final_F)
                curr_std = np.std(curr_d)
                prev_std = np.std(prev_d)
                if curr_std < prev_std:
                    final_inliers = inliers
                    final_F = F
                    final_el = el
                    final_er = er
            i = i+1
        return final_inliers, final_F, final_el, final_er

    def compute(self, p1, p2):
        """Computes the Fundamental Matrix and
        corresponding epipoles el, er via Point Correspondences P1 and P2
        Zissermann 291, alg. 11.4"""

        # Get inliers of point correspondences and the best approximation for F
        inliers, F, *_ = self.__ransac__(p1, p2)
        p1, p2 = p1[inliers], p2[inliers]

        def custom_cost(x, p1, p2):
            x = np.reshape(x, (3, 3))
            return self.__cost__(p1, p2, x)

        # perform Levenberg-Marquardt optimisation with respect to the cost function
        res = least_squares(custom_cost, F.flatten(), args=(p1, p2), method='lm')\
                             #verbose=1)
        self.F = np.reshape(res.x, (3, 3))
        check1, check2, check3 = self.check(p1,p2)
        self.el, self.er = self.__compute_epipoles__(self.F)

        return self.F, self.el, self.er, p1, p2, inliers

    def check(self, p1, p2):
        """Check quality of estimated F-Matrix and epipoles given the point-correspondences p1 and p2.
        Values near zero indicate good estimates.
        Returns: Score for F-matrix, score for left epipole, score for right epipole"""

        check1 = np.array([pr.T @ self.F @ pl for pl, pr in zip(p1, p2)])
        check2 = self.F @ self.el
        check3 = self.er.T @ self.F

        return check1, check2, check3

    def createProjectionmatrix(self, e=None, F=None):
        """create Projectionmatrix based on a Fundamental-matrix and right epipole (homogenous coordinates)
        Input: epipole e of right image, Fundamentalmatrix F
        Ouput: projectionmatrix p1 (left), p2 (right)"""
        if e is None:
            e = self.er
        if F is None:
            F = self.F

        # basic projectionmatrix P1 with no Rotation nor Translation
        p1 = np.hstack((np.identity(3), np.zeros((3, 1))))

        # second projectionmatrix, create Skew-symmetric first
        skew = np.array([[0, -e[2], e[1]],
                         [e[2], 0, -e[0]],
                         [-e[1], e[0], 0]])

        # Compute matrix, 3D-Computer Vision by C. Wöhler p. 12
        p2 = np.hstack((skew @ F, np.reshape(e, (3, 1))))
        return p1, p2


def draw_epilines(img1, img2, p1, p2, F):
    # get shape of the images
    s1 = np.shape(img1)
    s2 = np.shape(img2)
    # compute epiline for the right image
    epi_r = [F @ point for point in p1]
    # compute epiline for left image
    epi_l = [F.T @ point for point in p2]

    # compute the line points via ax + bx + c = 0
    r_y1 = [int(-l[2] / l[1]) for l in epi_r]
    r_y2 = [int(-(s2[1] * l[0] + l[2]) / l[1]) for l in epi_r]

    l_y1 = [int(-l[2] / l[1]) for l in epi_l]
    l_y2 = [int(-(s1[1] * l[0] + l[2]) / l[1]) for l in epi_l]

    for ry1, ry2, ly1, ly2, pt1, pt2 in zip(r_y1, r_y2, l_y1, l_y2, p1, p2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        img1 = cv2.line(img1, (0, ly1), (s1[1], ly2), color)
        img1 = cv2.circle(img1, tuple(pt1[:2]), 3, color, -1)
        img2 = cv2.line(img2, (0, ry1), (s2[1], ry2), color)
        img2 = cv2.circle(img2, tuple(pt2[:2]), 3, color, -1)
    return img1, img2