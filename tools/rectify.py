import cv2
import yaml
import numpy as np


class Rectificator:
    def __init__(self, calib_file_path):
        self.meta, self.cam_params = \
            self._parse_calibration_data(calib_file_path)

    def _gen_intrinsic(self, camera_intrinsic):
        K = np.array([
            [camera_intrinsic[0], 0, camera_intrinsic[2]],
            [0, camera_intrinsic[1], camera_intrinsic[3]],
            [0, 0, 1]
        ], dtype=np.float32)

        return K

    def _parse_calibration_data(self, calibration_file_path):
        # Load calibration data
        with open(calibration_file_path, 'r') as f:
            calib_data = yaml.safe_load(f)

        # Extract camera parameters
        left_camera_intrinsic = \
            self._gen_intrinsic(calib_data['cam0']['intrinsics'])
        left_dist_coeffs = np.array(calib_data['cam0']['distortion_coeffs'])
        right_camera_intrinsic = \
            self._gen_intrinsic(calib_data['cam1']['intrinsics'])
        right_dist_coeffs = np.array(calib_data['cam1']['distortion_coeffs'])
        trans_mat = np.array(calib_data['cam1']['T_cn_cnm1']).reshape(4, 4)

        width, height = calib_data['cam0']['resolution']

        # Prepare rectify parameters
        # trans_mat[:3, 3:] * 1000 -> is to convert the translation
        # from meters to mm
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            left_camera_intrinsic, left_dist_coeffs,
            right_camera_intrinsic, right_dist_coeffs,
            (width, height), trans_mat[:3, :3], trans_mat[:3, 3:] * 1000)

        map1_left, map2_left = cv2.initUndistortRectifyMap(
            left_camera_intrinsic, left_dist_coeffs,
            R1, P1, (width, height), cv2.CV_32FC1)
        map1_right, map2_right = cv2.initUndistortRectifyMap(
            right_camera_intrinsic, right_dist_coeffs,
            R2, P2, (width, height), cv2.CV_32FC1)

        cam_params = {
            # rectified intrinsic matrix of left camera
            'K1': P1[:, :3].tolist(),
            # rectified intrinsic matrix of right camera
            'K2': P2[:, :3].tolist(),
            # projection matrix that projects points given in the rectified
            # first camera coordinate system into the rectified first
            # camera's image
            'P1': P1.tolist(),
            # projection matrix that projects points given in the rectified
            # first camera coordinate system into the rectified second
            # camera's image.
            'P2': P2.tolist(),
            # baseline (translation between the two cameras)
            'baseline': -P2[0][-1] / P1[0][0],
            # transformation matrix that takes points in left camera coordinate
            # to right camera coordinate
            'T_cr_cl': [[1, 0, 0, P2[0][-1] / P1[0][0]],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]]
        }

        meta = {
            'left': {
                'camera_intrinsic': left_camera_intrinsic,
                'dist_coeffs': left_dist_coeffs,
                'R': R1,
                'P': P1,
                'map1': map1_left,
                'map2': map2_left
            },
            'right': {
                'camera_intrinsic': right_camera_intrinsic,
                'dist_coeffs': right_dist_coeffs,
                'R': R2,
                'P': P2,
                'map1': map1_right,
                'map2': map2_right
            },
        }

        return meta, cam_params

    def rectify_images(self, side, img):
        """
        Rectify the input image

        Args:
            side (str): 'left' or 'right'
            img (np.ndarray): input image

        Returns:
            np.ndarray: rectified image
        """
        map1 = self.meta[side]['map1']
        map2 = self.meta[side]['map2']
        img_rectified = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

        return img_rectified

    def rectify_annots(self, side, kpts, bbox=None):
        """
        Rectify the annotations so that they are consistent with the
        rectified image

        Args:
            side (str): 'left' or 'right'
            kpts (np.ndarray): keypoints
            bbox (np.ndarray): bounding box

        Returns:
            np.ndarray: rectified keypoints
            np.ndarray: rectified bounding box
        """
        intrinsics = self.meta[side]['camera_intrinsic']
        dist_coeffs = self.meta[side]['dist_coeffs']
        R = self.meta[side]['R']
        P = self.meta[side]['P']

        kpts = kpts.reshape(-1, 3)
        kpts_rectified = cv2.undistortPoints(
            kpts[:, :2].reshape((-1, 1, 2)), intrinsics, dist_coeffs, R=R, P=P)
        kpts_rectified = kpts_rectified.reshape(-1, 2)
        kpts_rectified = np.hstack([kpts_rectified, kpts[:, 2].reshape(-1, 1)])
        kpts_rectified = kpts_rectified.flatten()

        bbox_rectified = None
        if bbox is not None:
            bbox_rectified = cv2.undistortPoints(
                bbox.reshape((2, 1, 2)), intrinsics, dist_coeffs, R=R, P=P)
            bbox_rectified = bbox_rectified.flatten()

        return kpts_rectified, bbox_rectified

    def get_cam_params(self):
        return self.cam_params
