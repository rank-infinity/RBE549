# RBE549
WPI Computer Vision Course Assignments 
Outputs and Code is linked via jupyter notebook as links attached to project name. Please keep scrolling, each notebook has image outputs.

HW  0 - [PbLite edge detector implementation]([url](https://github.com/rank-infinity/RBE549/blob/main/Nehal_hw0/Phase1/Code/trial.ipynb))
        Boundary detection by classification of pixel based on difference in histogram distribution of different classes present in neighborhood found in half disk masks of particular pixel. Classes are decided by by forming different filter banks, congregating output of each filter bank and clustering all pixels.

HW 1 - [[Camera Calibration using Zhang's method]([url](https://github.com/rank-infinity/RBE549/blob/main/Nehal_hw1/trial.ipynb))]

P1 - [Image Stitching]([url](https://github.com/rank-infinity/RBE549/blob/main/Nehal_p1/Phase1/Code/trial.ipynb))
      Features are matched within consecutive images, homography estimated using RANSAC and then pictures transformed and blended. 

P2 - [Structure from Motion]([url](https://github.com/rank-infinity/RBE549/blob/main/Nehal_p2/Phase1/trials.ipynb))
      Features are matched withing i mage. Two images are selected and Fundamental and Essential matrix is derived through epipolar geomety. Then 4 possible pose of the two cameras with relation to each other is derived trough essentiall matrix. Chierality check gives us the correct pose which is then used to get an estimate of where the world point coordinates of features are through linear triangulation using where a feature is in image1  and image2 and the relative pose betweeen them. This initial world point is further optimized by performing least square on the point obtained by reprojecting the world point onto the 2d image frames and the ground truth feature point.
