import time
from typing import Optional, Tuple

import numpy as np
import torch
from gsplat import rendering

from imgui_bundle import imgui

from RTSGS.GaussianSplatting.PointCloud import PointCloud
from RTSGS.GaussianSplatting.Renderer.Camera import Camera
from RTSGS.GUI.ImageWidget import ImageWidget


def _to_torch_f32(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x)).to(device=device, dtype=torch.float32)


def _extract_camera_rt_from_view(view_col_major: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Matches your Camera.look_at() implementation where translation lives in the last ROW:
      view[3,0:3] = translation
    And basis is in columns.

    We interpret it as row-vector convention:
      p_cam = p_world @ R + t

    Returns:
      R_row: (3,3)
      t_row: (3,)
    """
    V = np.asarray(view_col_major, dtype=np.float32)
    R = V[:3, :3].copy()
    t = V[3, :3].copy()
    return R, t


def _row_rt_to_viewmat_col_major(R_row: torch.Tensor, t_row: torch.Tensor) -> torch.Tensor:
    """
    Convert your row-vector (p_cam = p_world @ R_row + t_row) into a standard
    column-vector view matrix usable by gsplat (world -> camera):

      p_cam_col = V @ p_world_col

    For a row-form affine:
      p_cam_row = p_world_row @ [R_row  0;
                                 t_row  1]
    The equivalent column-form matrix is the transpose:

      V = [[R_row^T, t_row^T],
           [0      , 1    ]]

    Returns:
      viewmat: (4,4) torch.float32, column-vector convention.
    """
    V = torch.eye(4, device=R_row.device, dtype=torch.float32)
    V[:3, :3] = R_row.t()
    V[:3, 3] = t_row

    # flip Y axis to match gsplat/OpenGL convention
    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], device=R_row.device))
    V = flip @ V
    return V


class GaussianSplattingWindow:
    """
    Rendering + ImGui window using gsplat.rendering.rasterization API.

    Assumptions / mapping:
      - PointCloud provides:
          all_points: [N,3] world
          all_colors: [N,3] in [0,1]
          all_covariances: optional [N,3,3] world covariance (not directly used by this API)
          all_alpha: optional [N,1] or [N] opacity
      - gsplat.rendering.rasterization expects (typical):
          means  : [N,3]
          quats  : [N,4] (xyzw)
          scales : [N,3] (positive)
          opacities: [N]
          colors : [N,3]
          viewmats: [C,4,4]
          Ks     : [C,3,3] (pinhole intrinsics)
    """

    def __init__(self, pcd: PointCloud, camera: Camera, width: int = 960, height: int = 540):
        self.pcd = pcd
        self.camera = camera

        self.W = int(width)
        self.H = int(height)

        self.device = getattr(pcd, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # intrinsics from pcd (torch tensors)
        self.fx = float(pcd.fx.item()) if torch.is_tensor(pcd.fx) else float(pcd.fx)
        self.fy = float(pcd.fy.item()) if torch.is_tensor(pcd.fy) else float(pcd.fy)
        self.cx = float(pcd.cx.item()) if torch.is_tensor(pcd.cx) else float(pcd.cx)
        self.cy = float(pcd.cy.item()) if torch.is_tensor(pcd.cy) else float(pcd.cy)

        blank = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        self.image_widget = ImageWidget(blank)

        self._xyz: Optional[torch.Tensor] = None
        self._rgb: Optional[torch.Tensor] = None
        self._cov: Optional[torch.Tensor] = None
        self._alpha: Optional[torch.Tensor] = None

        self._last_frame_time = time.time()
        self.is_open = True

        # default gaussian shape if you don't yet train quats/scales
        self.default_scale = float(getattr(pcd.config, "get", lambda k, d=None: d)("default_scale", 0.01)) if hasattr(pcd, "config") else 0.01

    def _pull_latest_buffers(self) -> bool:
        xyz = getattr(self.pcd, "all_points", None)
        rgb = getattr(self.pcd, "all_colors", None)
        if xyz is None or rgb is None:
            return False
        if xyz.numel() == 0:
            return False

        self._xyz = xyz.to(self.device).float()
        self._rgb = rgb.to(self.device).float().clamp(0.0, 1.0)

        self._cov = getattr(self.pcd, "all_covariances", None)
        if self._cov is not None:
            self._cov = self._cov.to(self.device).float()

        self._alpha = getattr(self.pcd, "all_alpha", None)
        if self._alpha is not None:
            self._alpha = self._alpha.to(self.device).float()

        return True

    @torch.no_grad()
    def _render_with_gsplat(self) -> np.ndarray:
        assert self._xyz is not None and self._rgb is not None

        # ---- camera -> gsplat viewmats + K ----
        self.camera.update_view()
        R_np, t_np = _extract_camera_rt_from_view(self.camera.view)
        R_row = _to_torch_f32(R_np, self.device)
        t_row = _to_torch_f32(t_np, self.device)

        # convert to column-vector world->cam 4x4
        viewmat = _row_rt_to_viewmat_col_major(R_row, t_row)  # [4,4]
        viewmats = viewmat.unsqueeze(0)  # [1,4,4]

        K = torch.tensor(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)  # [1,3,3]

        # ---- gaussian params (means/quats/scales/opacities/colors) ----
        means = self._xyz  # [N,3]

        # No orientation training yet -> identity quaternion
        # gsplat generally uses (x, y, z, w)
        N = means.shape[0]
        quats = torch.zeros((N, 4), device=self.device, dtype=torch.float32)
        quats[:, 3] = 1.0

        # No scale training yet -> constant isotropic scale
        scales = torch.full((N, 3), float(self.default_scale), device=self.device, dtype=torch.float32)

        # opacity
        if self._alpha is None:
            opacities = torch.ones((N,), device=self.device, dtype=torch.float32)
        else:
            a = self._alpha
            if a.ndim == 2 and a.shape[1] == 1:
                a = a[:, 0]
            opacities = a.clamp(0.0, 1.0).contiguous()

        colors = self._rgb.contiguous()  # [N,3] in [0,1]

        # ---- gsplat rasterization ----
        # returns:
        #   image: [C,H,W,3] float
        #   depth: [C,H,W,1] or [C,H,W]
        #   meta : dict
        image, depth, meta = rendering.rasterization(
            means=means,
            quats=quats,
            scales=scales.abs() + 1e-6,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=K,
            width=self.W,
            height=self.H,
            render_mode="RGB",
        )

        # image -> uint8 HWC for ImageWidget
        # handle either [1,H,W,3] or [H,W,3] depending on version
        if image.ndim == 4:
            image0 = image[0]
        else:
            image0 = image

        rgb_u8 = (image0.clamp(0.0, 1.0) * 255.0).to(torch.uint8).detach().cpu().numpy()
        return rgb_u8

    def draw(self):
        now = time.time()
        _dt = now - self._last_frame_time
        self._last_frame_time = now

        if not self.is_open:
            return

        self.is_open = imgui.begin("Gaussian Splatting Window", self.is_open)[1]

        if not self._pull_latest_buffers():
            imgui.text("No point cloud buffers available yet (pcd.all_points/all_colors empty).")
            imgui.end()
            return

        rgb = self._render_with_gsplat()
        imgui.text("Renderer: gsplat.rendering.rasterization")

        self.image_widget.set_image_rgb(rgb)
        self.image_widget.draw(fit_to_window=True)

        imgui.end()