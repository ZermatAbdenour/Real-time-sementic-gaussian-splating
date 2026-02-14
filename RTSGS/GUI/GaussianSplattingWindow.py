import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rendering

from imgui_bundle import imgui

from RTSGS.GaussianSplatting.PointCloud import PointCloud
from RTSGS.GaussianSplatting.Renderer.Camera import Camera
from RTSGS.GUI.ImageWidget import ImageWidget


def _to_torch_f32(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x)).to(device=device, dtype=torch.float32)


def _extract_camera_rt_from_view(view_col_major: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    V = np.asarray(view_col_major, dtype=np.float32)
    R = V[:3, :3].copy()
    t = V[3, :3].copy()
    return R, t


def _row_rt_to_viewmat_col_major(R_row: torch.Tensor, t_row: torch.Tensor) -> torch.Tensor:
    V = torch.eye(4, device=R_row.device, dtype=torch.float32)
    V[:3, :3] = R_row.t()
    V[:3, 3] = t_row
    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], device=R_row.device))
    V = flip @ V
    return V


class GaussianSplattingWindow:
    def __init__(self, pcd: PointCloud, camera: Camera, width: int = 960, height: int = 540):
        self.pcd = pcd
        self.camera = camera

        self.W = int(width)
        self.H = int(height)

        self.device = getattr(pcd, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Intrinsics from pcd
        self.fx = float(pcd.fx.item()) if torch.is_tensor(pcd.fx) else float(pcd.fx)
        self.fy = float(pcd.fy.item()) if torch.is_tensor(pcd.fy) else float(pcd.fy)
        self.cx = float(pcd.cx.item()) if torch.is_tensor(pcd.cx) else float(pcd.cx)
        self.cy = float(pcd.cy.item()) if torch.is_tensor(pcd.cy) else float(pcd.cy)

        blank = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        self.image_widget = ImageWidget(blank)

        # Buffers for trained parameters
        self._xyz: Optional[torch.Tensor] = None
        self._rgb: Optional[torch.Tensor] = None
        self._quats: Optional[torch.Tensor] = None
        self._scales: Optional[torch.Tensor] = None
        self._alpha: Optional[torch.Tensor] = None

        self._last_frame_time = time.time()
        self.is_open = True
        self.default_scale = 0.01

    def _pull_latest_buffers(self) -> bool:
        """Fetches the latest trained tensors from the PointCloud object."""
        xyz = getattr(self.pcd, "all_points", None)
        rgb = getattr(self.pcd, "all_colors", None)
        
        if xyz is None or rgb is None or xyz.numel() == 0:
            return False

        self._xyz = xyz.detach()
        self._rgb = rgb.detach()  

        # Pull the optimized parameters
        self._quats = getattr(self.pcd, "all_quaternions", None)
        self._scales = getattr(self.pcd, "all_scales", None)
        self._alpha = getattr(self.pcd, "all_alpha", None)

        return True

    @torch.no_grad()
    def _render_with_gsplat(self) -> np.ndarray:
        # 1. Camera logic: Convert GUI camera to gsplat-compatible viewmats
        self.camera.update_view()
        R_np, t_np = _extract_camera_rt_from_view(self.camera.view)
        R_row = _to_torch_f32(R_np, self.device)
        t_row = _to_torch_f32(t_np, self.device)
        viewmat = _row_rt_to_viewmat_col_major(R_row, t_row)
        viewmats = viewmat.unsqueeze(0)

        # Build Intrinsics Matrix
        K = torch.tensor([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], device=self.device, dtype=torch.float32).unsqueeze(0)

        # Safety check for empty pointclouds
        if self._xyz is None or self._xyz.numel() == 0:
            return np.zeros((self.H, self.W, 3), dtype=np.uint8)

        N = self._xyz.shape[0]

        # 2. Quaternions: Must be unit length to represent valid rotations
        if self._quats is not None:
            quats = torch.nn.functional.normalize(self._quats.detach(), p=2, dim=-1)
        else:
            quats = torch.zeros((N, 4), device=self.device)
            quats[:, 3] = 1.0

        # 3. Scales: Optimized in log-space, so we apply Exp to get physical size
        if self._scales is not None:
            scales = torch.exp(self._scales.detach())
        else:
            scales = torch.full((N, 3), self.default_scale, device=self.device)

        # 4. Opacity: Optimized as logits, apply Sigmoid to squash to [0, 1]
        if self._alpha is not None:
            opacities = torch.sigmoid(self._alpha.detach()).squeeze(-1)
        else:
            opacities = torch.ones((N,), device=self.device)

        # 5. Color: Apply Sigmoid to map logit parameters back to RGB [0, 1]
        # This is the fix that prevents the visualization from looking flat/dark
        if self._rgb is not None:
            colors = torch.sigmoid(self._rgb.detach())
        else:
            colors = torch.ones((N, 3), device=self.device)

        # 6. Rasterization Call
        # Note: render_mode="RGB" ensures we get back a 3-channel image
        image, _, _ = rendering.rasterization(
            means=self._xyz.detach(),
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors.contiguous(),
            viewmats=viewmats,
            Ks=K,
            width=self.W,
            height=self.H,
            render_mode="RGB",
        )

        # 7. Post-process to CPU NumPy for ImGui display
        # Squeeze batch dimension: [1, H, W, 3] -> [H, W, 3]
        image0 = image.squeeze(0) if image.ndim == 4 else image
        
        # Clamp just in case of float precision errors, then map to 0-255
        rgb_u8 = (image0.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
        
        return rgb_u8
    
    def draw(self):
        if not self.is_open: return
        self.is_open = imgui.begin("Gaussian Splatting Window", self.is_open)[1]

        if not self._pull_latest_buffers():
            imgui.text("Waiting for PointCloud data...")
            imgui.end()
            return

        rgb = self._render_with_gsplat()
        self.image_widget.set_image_rgb(rgb)
        self.image_widget.draw(fit_to_window=True)
        imgui.end()