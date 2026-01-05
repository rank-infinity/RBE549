import numpy as np
from LinearTriangulation import linearTriangulation
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_world_coords(
    world_coords,
    camera_poses=None,
    colors=None,
    save_path=None,
    hold=False,
    cam_size=0.8
):
    """
    Parameters
    ----------
    world_coords : list of (N,3) arrays
        3D points in world coordinates
    camera_poses : list of (R, t), optional
        R: (3,3), t: (3,)
    colors : list of colors, optional
        One color per world_coords entry
    cam_size : float
        Size of camera triangle
    """

    if colors is None:
        colors = ['k'] * len(world_coords)

    fig = plt.figure()

    for i, coords in enumerate(world_coords):
        c = np.asarray(coords)
        x = c[:, 0]
        z = c[:, 2]

        plt.plot(x, z, '.', markersize=1, color=colors[i])

        # --- Draw camera ---
        if camera_poses is not None:
            R, t = camera_poses[i]

            # Camera center in world coordinates
            C = -R.T @ t

            # Camera forward (Z), right (X) directions in world frame
            z_dir = R.T[:, 2]
            x_dir = R.T[:, 0]

            # Triangle vertices (XZ plane)
            p0 = C
            p1 = C + cam_size * ( z_dir + 0.5 * x_dir)
            p2 = C + cam_size * ( z_dir - 0.5 * x_dir)

            tri_x = [p0[0], p1[0], p2[0], p0[0]]
            tri_z = [p0[2], p1[2], p2[2], p0[2]]

            plt.plot(tri_x, tri_z, '-', color=colors[i], linewidth=2)
            plt.scatter(C[0], C[2], color=colors[i], s=35)

    plt.xlabel("X")
    plt.ylabel("Z")
    plt.axis([-20, 20, -10, 25])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        if not hold:
            plt.close()
    else:
        plt.show()


def disambiguate_camera_pose_old(possible_poses, points1, points2, K1, K2=None):
    if K2 is None:
        K2 = K1

    max_positive_depths = 0
    best_pose = None
    best_3d_points = None
    all_world_points = []
    for pose in possible_poses:
        R, t = pose
        P2 = K2 @ np.hstack((R, t.reshape(3, 1)))
        P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
        
        points_3d = linearTriangulation((P1, P2), points1, points2)
        # num_positive_depths = np.sum(R[2, :] @ (points_3d.T - t.reshape(3, 1)) > 0)
        all_world_points.append(points_3d)
        X = points_3d.T
        Z1 = X[2, :]
        Z2 = (R @ X + t.reshape(3,1))[2, :]
        num_positive_depths = np.sum((Z1 > 0) & (Z2 > 0))


        if num_positive_depths > max_positive_depths:
            max_positive_depths = num_positive_depths
            best_pose = pose
            best_3d_points = points_3d
        print("Pose:", R, t, "Positive depths:", num_positive_depths)
        print("\tZ1+", np.sum(Z1>0), "Z2+", np.sum(Z2>0))

    colors = ['r', 'g', 'b', 'm']
    plot_world_coords(all_world_points, camera_poses=possible_poses, colors=colors)
    return best_pose, best_3d_points


# points given are in image coordinates
def disambiguate_camera_pose(possible_poses, points1, points2, K1, K2=None):
    if K2 is None:
        K2 = K1

    # camera coordinates
    points1_cam = (np.linalg.inv(K1) @ np.vstack((points1.T, np.ones((1, points1.shape[0]))))).T[:,:2]
    points2_cam = (np.linalg.inv(K2) @ np.vstack((points2.T, np.ones((1, points2.shape[0]))))).T[:,:2]

    max_positive_depths = 0
    best_pose = None
    best_3d_points = None
    best_idx=None
    all_world_points = []
    for i, pose in enumerate(possible_poses):
        R, t = pose
        P2 = np.hstack((R, t.reshape(3, 1)))
        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        
        points_3d = linearTriangulation((P1, P2), points1_cam, points2_cam)
        # num_positive_depths = np.sum(R[2, :] @ (points_3d.T - t.reshape(3, 1)) > 0)
        all_world_points.append(points_3d)
        X = points_3d.T
        Z1 = X[2, :]
        Z2 = (R @ X + t.reshape(3,1))[2, :]
        num_positive_depths = np.sum((Z1 > 0) & (Z2 > 0))


        if num_positive_depths > max_positive_depths:
            max_positive_depths = num_positive_depths
            best_pose = pose
            best_3d_points = points_3d
            best_idx = i

        print("Pose:", R, t, "Positive depths:", num_positive_depths)
        print("\tZ1+", np.sum(Z1>0), "Z2+", np.sum(Z2>0))
    
    R,t = possible_poses[best_idx]
    X = best_3d_points.T
    Z1 = X[2, :]
    Z2 = (R @ X + t.reshape(3,1))[2, :]
    mask = (Z1 > 0) & (Z2 > 0)
    valid_world_points = best_3d_points[mask]
    valid_points1 = points1[mask]
    valid_points2 = points2[mask]

    colors = ['r', 'g', 'b', 'm']
    plot_world_coords(all_world_points, camera_poses=possible_poses, colors=colors)
    return best_pose, valid_world_points, valid_points1, valid_points2

