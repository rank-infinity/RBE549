from scipy.optimize import least_squares
import numpy as np

# point_3D is not in homogeneous coordinates
def reprojection_error(points_3D_flattened, P1, P2, points1, points2):
    # Reshape params back to (N, 3)
    N = len(points1)
    points_3D = points_3D_flattened.reshape(N, 3)
    
    # Convert to homogeneous coordinates
    points_3D_hom = np.hstack([points_3D, np.ones((N, 1))])  # (N, 4)
    
    # Project to camera 1
    proj1 = (P1 @ points_3D_hom.T).T  # (N, 3)
    proj1_2d = proj1[:, :2] / proj1[:, 2:3]  # (N, 2)
    
    # Project to camera 2
    proj2 = (P2 @ points_3D_hom.T).T  # (N, 3)
    proj2_2d = proj2[:, :2] / proj2[:, 2:3]  # (N, 2)
    
    # Compute residuals (difference between projected and observed)
    residuals1 = proj1_2d - points1  # (N, 2)
    residuals2 = proj2_2d - points2  # (N, 2)
    
    # Flatten to 1D residual vector
    residuals = np.hstack([residuals1.flatten(), residuals2.flatten()])  # (N*4,)
    
    return residuals

# Make it homogeneous in error function or scipy optimize will have to guess one more parameter for each point
def nonlinearTriangulation(pose, points1, points2, initial_points3d, K1, K2=None):
    if K2 is None:
        K2 = K1

    R, t = pose

    # In pixel coordinates
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K2 @ np.hstack((R, t.reshape(3, 1)))

    x0 = initial_points3d.flatten()

    result = least_squares(reprojection_error, x0, args=(P1, P2, points1, points2), method ='trf', loss='huber', verbose=0)
    optimized_points = result.x.reshape(-1, 3)

    return optimized_points
