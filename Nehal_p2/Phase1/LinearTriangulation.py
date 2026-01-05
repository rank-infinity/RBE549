import numpy as np

# Points 3d are in camera1 frame
def linearTriangulation(P, points1, points2):
    
    P1, P2 = P
    num_points = points1.shape[0]
    points_3d = []

    for i in range(num_points):
        x1, y1 = points1[i]
        x2, y2 = points2[i]

        A = np.array([ 
            x1 * P1[2] - P1[0],
            y1 * P1[2] - P1[1],
            x2 * P2[2] - P2[0],
            y2 * P2[2] - P2[1]
        ])

        U, S, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[3]  
        points_3d.append(X[:3])

    return np.array(points_3d)
