#Ablaufprogramm für die 3D Rekonstruktion
#Written by Samed Dogan & Joshua Meiser, University of Applied Sciences Munich.

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from FMatrix import FMatrix, draw_epilines
from PointMatcher import PointMatcher
from Triangulation import Triangulation
from ColorReconstruction import ColorofPoint

# Load Input Images and Rescale to appropriate Size
left_image = cv2.resize(cv2.imread('media/l-image.jpg'), None, fx=0.2, fy=0.2)
right_image = cv2.resize(cv2.imread('media/r-image.jpg'), None, fx=0.2, fy=0.2)

# create Instance of PointMatcher Class and match 500 Keypoints
# return a sorted Cv2Matcher Obj
matcher = PointMatcher()
matches = matcher.match(left_image, right_image, 500)
# Retrieve Coordinates of matches points
left, right = matcher.get_coordinates()
# Show best 200 Matches
img = cv2.drawMatches(left_image, matcher.l_kp, right_image, matcher.r_kp, matches[:200], None, flags=2)
cv2.imshow("Found Keypoints", img)
cv2.imwrite('media/pointmatches.jpg', img)

# Create Fmatrix instance, Compute the Fundamental Matrix from best 200 matches
# Ransac is performed, choosing the Set of inliers closest to line defined by the Sampson-Distance
# The returned Fundamental matrix is fitting to the chosen inliers via levenberg-marqardt optimisation
# Returns: Fundamentalmatrix, Left-Epipole, Right-Epipole, Set of Inliers P1 and P2, Inlier Bool Array
matrix = FMatrix()
F, el, er, p1, p2, inliers = matrix.compute(left[:200], right[:200])
# compute the projection matrices
proj1, proj2 = matrix.createProjectionmatrix()
# check the quality of estimated Fundamentalmatrix
# where check1 returns the summed error over all points with respect to the epipolar constraint
# check2 returns the quality of the left-epipole and check3 the quality of the right epipole
# values close to zero indicate good estimation, for Debugging
#check1, check2, check3 = matrix.check(p1, p2)

# draw the best matches after outlier detection
matches = matches[:200]
matches = [match for (match, value) in zip(matches, inliers) if value]
img2 = cv2.drawMatches(left_image, matcher.l_kp, right_image, matcher.r_kp, matches, None, flags=2)
cv2.imshow("Sorted Keypoints", img2)
cv2.imwrite('media/pointmatches-ransac.jpg', img2)

# draw the epipolar lines, derived from F. Check visual quality of the epipolar constraint
left_image_epi, right_image_epi = draw_epilines(left_image, right_image, p1,p2,F)
cv2.imshow("Epipolar lines corresponding to right keypoints", left_image)
cv2.imshow("Epipolar lines corresponding to left keypoints", right_image)
cv2.imwrite('media/epipolar-left.jpg', left_image_epi)
cv2.imwrite('media/epipolar-right.jpg', right_image_epi)

#normalize left, right points, if needed
#Nl = matrix.__compN__(left[:200])
#Nr = matrix.__compN__(right[:200])
#normalized_left = np.array([Nl @ p for p in left[:200]])
#normalized_right = np.array([Nr @ p for p in right[:200]])

# Triangulate the points, unsorted points with outliers are used, to establish more points.
# In theory, more points can be established trough guided matching with F
TriangulatePoints = Triangulation()
X_in_Space = TriangulatePoints.__TriangulatePoints__(proj1, proj2, left[:200], right[:200])

#Denormalize XinSpace, if needed
#X_in_Space = X_in_Space[:, 0:3]
#X_in_Space = np.array([Nr. T @ p @ Nl for p in X_in_Space])
#N = len(X_in_Space)
#X_in_Space = np.c_[X_in_Space, np.ones(N)]

# ========== for Debugging =============================
#check4 = TriangulatePoints.__TriangulationCheck_p1__(left[:200], proj1, X_in_Space)
#check5 = TriangulatePoints.__TriangulationCheck_p2__(right[:200], proj2, X_in_Space)

#print('check F', check1)
#print('check el', check2)
#print('check3 er', check3)
#print('check4 p1 = proj1X', check4)
#print('check5 p2 = proj2X', check5)
# =========================================================#

#Get color values
left_image = cv2.resize(cv2.imread('media/l-image.jpg'), None, fx=0.2, fy=0.2)
imgRGB = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
ColorReconstruction = ColorofPoint()
color_of_X = ColorReconstruction.__ColorofPointCorrespondence__(imgRGB, left[:200])

# plot 3D points
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_in_Space[:,0], X_in_Space[:,1], X_in_Space[:,2], c=color_of_X/255.0, cmap='hot')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.savefig('media/3D-Pointcloud.png')
plt.show()
cv2.waitKey()

