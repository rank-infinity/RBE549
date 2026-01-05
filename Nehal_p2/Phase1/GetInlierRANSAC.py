import numpy as np

# np.random.seed(42)

def calc_fundamental_matrix(x1_s, y1_s, x2_s, y2_s):
    A = []
    for i in range(x1_s.shape[0]):
        x1 = x1_s[i]
        y1 = y1_s[i]
        x2 = x2_s[i]
        y2 = y2_s[i]
        A.append([x1*x2, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])

    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    U, S, Vt = np.linalg.svd(F)
    S[2] = 0  
    F = U @ np.diag(S) @ Vt

    return F

def getInliersRANSAC_Pixel(x1_s, y1_s, x2_s, y2_s, T1, T2, threshold=2, total_iterations=2000):
    F = np.eye(3)
    tot_points = x1_s.shape[0]
    num_inliers = 0
    best_inliers1 = []
    best_inliers2 = []
    best_outliers1 = []
    best_outliers2 = []

    # Normalize points
    points1 = np.vstack((x1_s, y1_s)).T
    points2 = np.vstack((x2_s, y2_s)).T
    norm_points1 = (T1 @ np.vstack((x1_s, y1_s, np.ones_like(x1_s))))[:2].T
    norm_points2 = (T2 @ np.vstack((x2_s, y2_s, np.ones_like(x2_s))))[:2].T
    x1_s_norm, y1_s_norm = norm_points1[:,0], norm_points1[:,1]
    x2_s_norm, y2_s_norm = norm_points2[:,0], norm_points2[:,1]

    point_chooser = np.arange(tot_points)
    for i in range(total_iterations):
        indices = np.random.choice(point_chooser, 8, replace=False)
        x1_sample = x1_s_norm[indices]
        y1_sample = y1_s_norm[indices]
        x2_sample = x2_s_norm[indices]
        y2_sample = y2_s_norm[indices]
        F_temp_normalized = calc_fundamental_matrix(x1_sample, y1_sample, x2_sample, y2_sample)
        F_temp = T2.T @ F_temp_normalized @ T1

        # Count inliers
        inliers1 = []
        inliers2 = []

        outliers1= []
        outliers2 = []
        for j in range(tot_points):
            # Sampson distance
            num_error = abs(np.array([x2_s[j], y2_s[j], 1]).reshape(3,1).T @ F_temp @ np.array([x1_s[j], y1_s[j], 1]).reshape(3,1))**2
            temp1 = F_temp.T @ np.array([x2_s[j], y2_s[j], 1]).reshape(3,1)
            temp2 = F_temp @ np.array([x1_s[j], y1_s[j], 1]).reshape(3,1)
            denom_error  = temp1[0]**2 + temp1[1]**2 + temp2[0]**2 + temp2[1]**2
            error = num_error / denom_error

            # Algebraic distance- Symmetric epipolar distance
            # image1_num = (np.array([x1_s[j], y1_s[j], 1]).reshape(3,1).T @ F_temp @ np.array([x2_s[j], y2_s[j], 1]).reshape(3,1))**2
            # image1_denom = (F_temp.T @ np.array([x1_s[j], y1_s[j], 1]).reshape(3,1))[0]**2 + (F_temp.T @ np.array([x1_s[j], y1_s[j], 1]).reshape(3,1))[1]**2
            # image2_denom = (F_temp @ np.array([x2_s[j], y2_s[j], 1]).reshape(3,1))[0]**2 + (F_temp @ np.array([x2_s[j], y2_s[j], 1]).reshape(3,1))[1]**2
            # error = np.sqrt((image1_num / image1_denom) + (image1_num / image2_denom))

            if error < threshold:
                inliers1.append((x1_s[j], y1_s[j]))
                inliers2.append((x2_s[j], y2_s[j]))
            else:
                outliers1.append((x1_s[j], y1_s[j]))
                outliers2.append((x2_s[j], y2_s[j]))

        # Update F if we found more inliers
        if len(inliers1) > num_inliers:
            best_inliers1 = inliers1
            best_inliers2 = inliers2
            best_outliers1 = outliers1
            best_outliers2 = outliers2
            F = F_temp
            num_inliers = len(inliers1)
    return F, np.array(best_inliers1), np.array(best_inliers2), np.array(best_outliers1), np.array(best_outliers2)

# Totally WRONG
# Considers x1.T @ F @ x2 instead of x2.T @ F @ x1
def getInliersRANSAC(x1_s, y1_s, x2_s, y2_s, threshold=0.01, total_iterations=2000):
    F = np.eye(3)
    tot_points = x1_s.shape[0]
    num_inliers = 0
    best_inliers1 = []
    best_inliers2 = []
    best_outliers1 = []
    best_outliers2 = []

    point_chooser = np.arange(tot_points)
    for i in range(total_iterations):
        indices = np.random.choice(point_chooser, 8, replace=False)
        x1_sample = x1_s[indices]
        y1_sample = y1_s[indices]
        x2_sample = x2_s[indices]
        y2_sample = y2_s[indices]
        F_temp = calc_fundamental_matrix(x1_sample, y1_sample, x2_sample, y2_sample)

        # Count inliers
        inliers1 = []
        inliers2 = []

        outliers1= []
        outliers2 = []
        for j in range(tot_points):
            # Sampson distance
            # num_error = abs(np.array([x1_s[j], y1_s[j], 1]).reshape(3,1).T @ F_temp @ np.array([x2_s[j], y2_s[j], 1]).reshape(3,1))**2
            # denom_error  = np.linalg.norm(F_temp.T @ np.array([x1_s[j], y1_s[j], 1]).reshape(3,1))**2 + np.linalg.norm(F_temp @ np.array([x2_s[j], y2_s[j], 1]).reshape(3,1))**2
            # error = num_error / denom_error

            # Algebraic distance- Symmetric epipolar distance
            image1_num = (np.array([x1_s[j], y1_s[j], 1]).reshape(3,1).T @ F_temp @ np.array([x2_s[j], y2_s[j], 1]).reshape(3,1))**2
            image1_denom = (F_temp.T @ np.array([x1_s[j], y1_s[j], 1]).reshape(3,1))[0]**2 + (F_temp.T @ np.array([x1_s[j], y1_s[j], 1]).reshape(3,1))[1]**2
            image2_denom = (F_temp @ np.array([x2_s[j], y2_s[j], 1]).reshape(3,1))[0]**2 + (F_temp @ np.array([x2_s[j], y2_s[j], 1]).reshape(3,1))[1]**2
            error = np.sqrt((image1_num / image1_denom) + (image1_num / image2_denom))

            if error < threshold:
                inliers1.append((x1_s[j], y1_s[j]))
                inliers2.append((x2_s[j], y2_s[j]))
            else:
                outliers1.append((x1_s[j], y1_s[j]))
                outliers2.append((x2_s[j], y2_s[j]))

        # Update F if we found more inliers
        if len(inliers1) > num_inliers:
            best_inliers1 = inliers1
            best_inliers2 = inliers2
            best_outliers1 = outliers1
            best_outliers2 = outliers2
            F = F_temp
            num_inliers = len(inliers1)
    return F, np.array(best_inliers1), np.array(best_inliers2), np.array(best_outliers1), np.array(best_outliers2)
