import numpy as np


def calculate_camera_pose(E):
    # Decompose the Essential matrix to get possible rotations and translations
    #  R, t  is pose relative to world frame i.e camera at world frame
    #  Whichever camera is on left of F, that camera is at world frame
    
    U, S, Vt = np.linalg.svd(E)
    
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]

    if np.linalg.det(R1) < 0:
        R1 = -R1

    if np.linalg.det(R2) < 0:
        R2 = -R2

    # Four possible solutions
    possible_poses = [
        (R1, t),
        (R1, -t),
        (R2, t),
        (R2, -t)
    ]

    return possible_poses
