import math
import torch
import numpy as np


# temporary version of 3DGS ViewpointCamera for testing
class ViewpointCamera:
    def __init__(
        self,
        position: list | np.ndarray,
        target: list | np.ndarray,
        up: list | np.ndarray,
        fov_y: float,
        image_width: int,
        image_height: int,
        z_near: float = 0.01,
        z_far: float = 100.0,
    ):
        self.image_width = image_width
        self.image_height = image_height
        self.FoVy = fov_y
        self.FoVx = 2.0 * math.atan(
            math.tan(fov_y / 2.0) * (image_width / image_height)
        )
        self.znear = z_near
        self.zfar = z_far

        w2c = self._look_at(
            np.asarray(position, dtype=np.float32),
            np.asarray(target, dtype=np.float32),
            np.asarray(up, dtype=np.float32),
        )
        proj = self._projection_matrix(self.FoVx, self.FoVy, z_near, z_far)

        self.world_view_transform = torch.tensor(
            w2c, dtype=torch.float32, device="cuda"
        )
        self.projection_matrix = torch.tensor(
            proj, dtype=torch.float32, device="cuda"
        )
        self.full_proj_transform = self.projection_matrix @ self.world_view_transform

    @staticmethod
    def _look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
        f = center - eye
        f /= np.linalg.norm(f)
        up = up / np.linalg.norm(up)
        s = np.cross(f, up)
        s /= np.linalg.norm(s)
        u = np.cross(s, f)

        m = np.eye(4, dtype=np.float32)
        m[0, :3] = s
        m[1, :3] = u
        m[2, :3] = -f
        m[0, 3] = -np.dot(s, eye)
        m[1, 3] = -np.dot(u, eye)
        m[2, 3] = np.dot(f, eye)
        return m

    @staticmethod
    def _projection_matrix(
        fov_x: float, fov_y: float, z_near: float, z_far: float
    ) -> np.ndarray:
        tan_half_fov_y = math.tan(fov_y / 2.0)
        tan_half_fov_x = math.tan(fov_x / 2.0)

        top = tan_half_fov_y * z_near
        bottom = -top
        right = tan_half_fov_x * z_near
        left = -right

        p = np.zeros((4, 4), dtype=np.float32)
        p[0, 0] = 2.0 * z_near / (right - left)
        p[1, 1] = 2.0 * z_near / (top - bottom)
        p[0, 2] = (right + left) / (right - left)
        p[1, 2] = (top + bottom) / (top - bottom)
        p[2, 2] = z_far / (z_far - z_near)
        p[2, 3] = -(z_far * z_near) / (z_far - z_near)
        p[3, 2] = 1.0
        return p
