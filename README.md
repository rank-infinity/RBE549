# RBE549 – Computer Vision

**Worcester Polytechnic Institute (WPI)**
Course Repository for Computer Vision Assignments

---

## 📌 Overview

This repository contains implementations, experiments, and results for assignments and projects completed as part of **RBE 549 – Computer Vision** at WPI.

🔗 **All code and outputs are provided via Jupyter Notebooks.**
Each project title below links directly to its corresponding notebook. Please **scroll through the notebooks** to view image outputs, visualizations, and explanations.

---

## 📘 Homework Assignments

### **HW 0 – PbLite Edge Detector**

🔗 **[PbLite Edge Detector Implementation](https://github.com/rank-infinity/RBE549/blob/main/Nehal_hw0/Phase1/Code/trial.ipynb)**

* Implements the **PbLite edge detection algorithm**.
* Performs boundary detection by classifying pixels based on differences in **histogram distributions**.
* Uses **half-disk masks** to analyze pixel neighborhoods.
* Pixel classes are derived by:

  * Constructing multiple **filter banks**
  * Aggregating filter responses
  * Clustering pixel features

---

### **HW 1 – Camera Calibration**

🔗 **[Camera Calibration using Zhang’s Method](https://github.com/rank-infinity/RBE549/blob/main/Nehal_hw1/trial.ipynb)**

* Implements **Zhang’s camera calibration technique**.
* Estimates intrinsic and extrinsic camera parameters.
* Uses checkerboard images for calibration and validation.

---

## 🧩 Projects

### **Project 1 – Image Stitching**

🔗 **[Image Stitching](https://github.com/rank-infinity/RBE549/blob/main/Nehal_p1/Phase1/Code/trial.ipynb)**

* Detects and matches features across **consecutive images**.
* Estimates **homography using RANSAC**.
* Warps images into a common reference frame.
* Produces a final panorama via **image blending**.

---

### **Project 2 – Structure from Motion (SfM)**

🔗 **[Structure from Motion](https://github.com/rank-infinity/RBE549/blob/main/Nehal_p2/Phase1/trials.ipynb)**

* Matches features across multiple images.
* Selects image pairs to compute:

  * **Fundamental Matrix**
  * **Essential Matrix** using epipolar geometry
* Recovers **four possible camera poses** from the essential matrix.
* Applies **cheirality check** to determine the correct camera pose.
* Estimates 3D world points via **linear triangulation**.
* Refines 3D points using **least-squares optimization** by minimizing reprojection error.

---

## 📎 Notes

* Each notebook includes **visual outputs, plots, and intermediate results**.
* Code is written with clarity and experimentation in mind rather than library abstraction.

---

✨ Feel free to explore the notebooks and reach out if you have questions or suggestions.
