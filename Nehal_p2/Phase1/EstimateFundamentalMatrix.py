import numpy as np
import GetInlierRANSAC

def normalize_points(points):
    """Normalize points to have zero mean and average distance sqrt(2) from origin"""
    centroid = np.mean(points, axis=0)
    shifted = points - centroid
    avg_dist = np.mean(np.sqrt(np.sum(shifted**2, axis=1)))
    scale = np.sqrt(2) / avg_dist
    
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])
    return T

def normalize_points_his(points):
    # Calculate mean and standard deviation of coordinates
    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)
    mean_u, mean_v = mean
    std_u, std_v = std
    # Translation matrix to center the coordinates
    T = np.array([
        [1 / std_u, 0, -mean_u / std_u],
        [0, 1 / std_v, -mean_v / std_v],
        [0, 0, 1]
    ])

    # Apply the transformation to the coordinates
    # u_normalized, v_normalized, _ = np.dot(T, np.column_stack((u, v, np.ones_like(u))).T)

    return  T


# Then in calc_fundamental_matrix, normalize before computing


#  New approach -> theshold in pixel space. Rest everything is same
def estimate_fundamental_matrix_pixel(match_list, threshold=2):
    x1_s, y1_s, x2_s, y2_s = match_list
    print(f"Number of matches received: {x1_s.shape[0]}")

    points1 = np.vstack((x1_s, y1_s)).T
    points2 = np.vstack((x2_s, y2_s)).T
    T1 = normalize_points(points1)
    T2 = normalize_points(points2)
   
    F, inliers1, inliers2, outliers1, outliers2 = GetInlierRANSAC.getInliersRANSAC_Pixel(x1_s, y1_s, x2_s, y2_s, T1, T2, threshold=threshold)
    # F  = GetInlierRANSAC.calc_fundamental_matrix(inliers1[:,0], inliers1[:,1], inliers2[:,0], inliers2[:,1])

    return F, inliers1, inliers2, outliers1, outliers2


# WRONG
def estimate_fundamental_matrix(match_list, threshold=0.01):
    x1_s, y1_s, x2_s, y2_s = match_list
    print(f"Number of matches received: {x1_s.shape[0]}")

    points1 = np.vstack((x1_s, y1_s)).T
    points2 = np.vstack((x2_s, y2_s)).T
    T1 = normalize_points(points1)
    T2 = normalize_points(points2)
    norm_points1 = (T1 @ np.vstack((x1_s, y1_s, np.ones_like(x1_s))))[:2].T
    norm_points2 = (T2 @ np.vstack((x2_s, y2_s, np.ones_like(x2_s))))[:2].T
    x1_s_norm, y1_s_norm = norm_points1[:,0], norm_points1[:,1]
    x2_s_norm, y2_s_norm = norm_points2[:,0], norm_points2[:,1]
    F, inliers1, inliers2, outliers1, outliers2 = GetInlierRANSAC.getInliersRANSAC(x1_s_norm, y1_s_norm, x2_s_norm, y2_s_norm, threshold=threshold)

    F  = GetInlierRANSAC.calc_fundamental_matrix(inliers1[:,0], inliers1[:,1], inliers2[:,0], inliers2[:,1])
    F = T2.T @ F @ T1

    inliers1 = (np.linalg.inv(T1) @ np.vstack((inliers1.T, np.ones(inliers1.shape[0])))) [:2].T
    inliers2 = (np.linalg.inv(T2) @ np.vstack((inliers2.T, np.ones(inliers2.shape[0])))) [:2].T
    outliers1 = (np.linalg.inv(T1) @ np.vstack((outliers1.T, np.ones(outliers1.shape[0])))) [:2].T
    outliers2 = (np.linalg.inv(T2) @ np.vstack((outliers2.T, np.ones(outliers2.shape[0])))) [:2].T

    # If you want ransac in unnormalized space directly, uncomment below
    # F  = GetInlierRANSAC.calc_fundamental_matrix(inliers1[:,0], inliers1[:,1], inliers2[:,0], inliers2[:,1])

    return F, inliers1, inliers2, outliers1, outliers2
