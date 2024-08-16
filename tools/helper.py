import numpy as np
from scipy.spatial.transform import Rotation


def compute_diver_body_frame(pose_3d):
    """
    Compute the diver body frame given the 3D pose.

    Args:
        pose_3d : numpy.ndarray (12, 3)
            The 3D pose of the diver.

    Returns:
        x_hat : numpy.ndarray (3,)
            The x-axis unit vector of the diver's body frame.
        y_hat : numpy.ndarray (3,)
            The y-axis unit vector of the diver's body frame.
        z_hat : numpy.ndarray (3,)
            The z-axis unit vector of the diver's body frame.
    """
    # COCO format:
    # [r_shoulder, l_shoulder, r_elbow, l_elbow, r_wrist, l_wrist,
    #  r_hip, l_hip, r_knee, l_knee, r_ankle, l_ankle]
    r_shoulder, l_shoulder = pose_3d[1], pose_3d[0]
    r_hip, l_hip = pose_3d[7], pose_3d[6]

    center_mass = np.mean(
        [r_shoulder, l_shoulder, r_hip, l_hip], axis=0)

    rhs = r_shoulder - r_hip
    rhls = l_shoulder - r_hip
    lhs = l_shoulder - l_hip
    lhrs = r_shoulder - l_hip

    r_cross = np.cross(rhls, rhs)
    l_cross = np.cross(lhs, lhrs)

    z = (r_cross + l_cross) / 2
    z_hat = z / (np.linalg.norm(z) + 1e-5)

    hip_midpt = (r_hip + l_hip) / 2
    y = hip_midpt - center_mass
    y_hat = y / (np.linalg.norm(y) + 1e-5)

    # x is the cross product of y_hat and z_hat
    x_hat = np.cross(y_hat, z_hat)

    return x_hat, y_hat, z_hat


def compute_angle_difference(prev, curr):
    """Compute the angle difference between two coordinates.

    Args:
        prev : list[numpy.ndarray (3,) * 3]
            The previous coordinate in the format of [x, y, z].
        curr : list[numpy.ndarray (3,) * 3]
            The current coordinate in the format of [x, y, z].

    Returns:
        angle : list[float, float, float]
            The angle difference in radian.
    """
    x0, y0, z0 = prev
    x1, y1, z1 = curr

    R0 = np.array([x0, y0, z0]).T
    R1 = np.array([x1, y1, z1]).T

    # R1 = R @ R0
    # Compute the rotation matrix
    rotation = Rotation.from_matrix(R1 @ R0.T)

    # Convert the rotation matrix to Euler angles [roll, pitch, yaw]
    angle_diff = rotation.as_euler('xyz', degrees=False)

    return angle_diff

def feat_noise_mask(L, lm, masking_ratio):
    mask = np.ones(L, dtype=bool)
    p_m = 1 / lm
    p_u = p_m * masking_ratio / (1 - masking_ratio)
    p = [p_m, p_u]

    state = int(np.random.rand() > masking_ratio)
    for i in range(L):
        mask[i] = state
        if np.random.rand() < p[state]:
            state = 1 - state

    return mask


def noise_mask(ts, lm, r):
    L = ts.shape[0]

    mask = np.ones(ts.shape, dtype=bool)
    for feature in range(ts.shape[1]):
        mask[:, feature] = feat_noise_mask(L, lm, r)
    return mask
