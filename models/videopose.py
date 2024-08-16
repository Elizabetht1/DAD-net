import numpy as np

from tools.utils import load_onnx_model


class VideoPose:
    def __init__(self, model_path):
        self.model = load_onnx_model(model_path)

        self.base_joint = 6
        self.base_pose = 13

    def _to_scale_invariant(self, pts):
        """
        Convert the given points to scale-invariant coordinates.

        Args:
            pts : numpy.ndarray (N, 12, 3)
                The input points.

        Returns:
            numpy.ndarray (N, 12, 3) :
                The scale-invariant coordinates of the points.
        """
        scale_invariant_pts = pts - \
            pts[[self.base_pose], :][:, [self.base_joint]]
        scale_invariant_pts = scale_invariant_pts / \
            np.linalg.norm(scale_invariant_pts[[self.base_pose]],
                           ord=2,
                           axis=(-1, -2),
                           keepdims=True)
        return scale_invariant_pts

    def inference(self, x):
        x_si = self._to_scale_invariant(x)

        x_si = np.expand_dims(x_si, 0)
        x_si = np.ascontiguousarray(x_si)
        x_si = np.asarray(x_si, dtype=np.float32)

        inp = {"joints": x_si}
        pose3d = self.model.run(None, inp)[0] * 1000
        pose3d = np.squeeze(pose3d, axis=(0, 1))

        return pose3d
