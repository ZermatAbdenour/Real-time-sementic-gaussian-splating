import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from gsplat import rendering

from imgui_bundle import imgui

from RTSGS.GaussianSplatting.PointCloud import PointCloud

def _to_torch_f32(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x)).to(device=device, dtype=torch.float32)


def _extract_camera_rt_from_view(view_col_major: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Matches your Camera.look_at() implementation where translation lives in the last ROW:
      view[3,0:3] = translation
    And basis is in columns.

    We interpret it as row-vector convention:
      p_cam = p_world @ R + t
    """
    V = np.asarray(view_col_major, dtype=np.float32)
    R = V[:3, :3].copy()   # [3,3]
    t = V[3, :3].copy()    # [3]
    return R, t

class GaussianSplatting:
    """
    This class is ONLY responsible for training / optimization.
    For now it's a placeholder (pass), as requested.
    """

    def __init__(self, pcd: PointCloud,):
        self.pcd = pcd

    def training_step(self):
        # TODO: implement online optimization of gaussian params using new frames
        pass
