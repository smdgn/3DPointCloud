import numpy as np

class Triangulation:
    """for each point correspondence xi <--> xi', compute the point Xi in space that projects to these two image
    points"""

    def __TriangulatePoints__(self, proj1, proj2, p1, p2):

        TriangulatedPoints = []

        for i in range(len(p1)):
            A = np.array([p1[i, 0] * proj1[2] - proj1[0],
                          p1[i, 1] * proj1[2] - proj1[1],
                          p2[i, 0] * proj2[2] - proj2[0],
                          p2[i, 1] * proj2[2] - proj2[1]])
            u, s, vh = np.linalg.svd(A, full_matrices=True)
            V = vh. T
            X = V[:, -1] #x is the last column of V, where A = usvh is the SVD of A
            erg = X / X[-1] #Get homogenized 3D Points
            TriangulatedPoints.append(erg)
        return np.array(TriangulatedPoints)

    def __TriangulationCheck_p1__(self, p1, proj1, X_in_Space):

        TriangulationCheck_P1 = []

        for i in range(len(p1)):
            p1_check = proj1 @ X_in_Space[i]
            p1_check = p1_check / p1_check[-1]
            TrianulationError_P1 = np.absolute(p1[i] - p1_check)
            TriangulationCheck_P1.append(TrianulationError_P1)
        return np.array(TriangulationCheck_P1)

    def __TriangulationCheck_p2__(self, p2, proj2, X_in_Space):

        TriangulationCheck_P2 = []

        for i in range(len(p2)):
            p2_check = proj2 @ X_in_Space[i]
            p2_check = p2_check / p2_check[-1]
            TrianulationError_P2 = np.absolute(p2[i] - p2_check)
            TriangulationCheck_P2.append(TrianulationError_P2)
        return np.array(TriangulationCheck_P2)