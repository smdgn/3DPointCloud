# 3DPointCloud

This Projects aims to create an accurate dense 3D pointcloud reconstruction given a series of images. 
The current version only works with stereo view images and will be updated shortly.

Point-correspondences are searched via ORB and matched via brute-force approach without cross checking. 
Outliers are removed using ransac, customized for the normalized 8 point algorithm.

<p float = "left">
<img src="https://github.com/smdgn/images/blob/master/pointmatches-ransac.jpg" width="224" height="224">
<p float = "left">
  <img src="https://github.com/smdgn/images/blob/master/epipolar-left.jpg" width="224" height="224">
  <img src="https://github.com/smdgn/images/blob/master/epipolar-right.jpg" width="224" height="224">
</p>
<img src="https://github.com/smdgn/images/blob/master/3D-Pointcloud.png" width="224" height="224">
