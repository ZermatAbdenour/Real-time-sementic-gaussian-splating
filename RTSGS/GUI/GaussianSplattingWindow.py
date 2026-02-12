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
        self._rgb = rgb.detach().clamp(0.0, 1.0)

        # Pull the optimized parameters
        self._quats = getattr(self.pcd, "all_quaternions", None)
        self._scales = getattr(self.pcd, "all_scales", None)
        self._alpha = getattr(self.pcd, "all_alpha", None)

        return True

    @torch.no_grad()
    def _render_with_gsplat(self) -> np.ndarray:
        # 1. Camera logic - Keep as is
        self.camera.update_view()
        R_np, t_np = _extract_camera_rt_from_view(self.camera.view)
        R_row = _to_torch_f32(R_np, self.device)
        t_row = _to_torch_f32(t_np, self.device)
        viewmat = _row_rt_to_viewmat_col_major(R_row, t_row)
        viewmats = viewmat.unsqueeze(0)

        K = torch.tensor([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], device=self.device, dtype=torch.float32).unsqueeze(0)

        if self._xyz is None or self._xyz.numel() == 0:
            return np.zeros((self.H, self.W, 3), dtype=np.uint8)

        N = self._xyz.shape[0]

        # 2. FIX: Quaternions - Normalize to ensure valid rotation
        if self._quats is not None:
            quats = torch.nn.functional.normalize(self._quats.detach(), p=2, dim=-1)
        else:
            quats = torch.zeros((N, 4), device=self.device)
            quats[:, 3] = 1.0

        # 3. FIX: Scales - Apply Exponential activation
        # This is critical because the optimizer works in log-space.
        # Without exp(), negative log-scales make Gaussians explode or disappear.
        if self._scales is not None:
            scales = torch.exp(self._scales.detach())
        else:
            scales = torch.full((N, 3), self.default_scale, device=self.device)

        # 4. FIX: Opacity - Apply Sigmoid activation
        # This converts optimized logits back into a 0.0 - 1.0 range.
        if self._alpha is not None:
            opacities = torch.sigmoid(self._alpha.detach()).squeeze(-1)
        else:
            opacities = torch.ones((N,), device=self.device)

        # 5. Render - Use activated values
        # We also ensure colors are detached to prevent any graph buildup
        image, _, _ = rendering.rasterization(
            means=self._xyz.detach(),
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=self._rgb.detach().contiguous(),
            viewmats=viewmats,
            Ks=K,
            width=self.W,
            height=self.H,
            render_mode="RGB",
        )

        # 6. Post-process to CPU NumPy
        # gsplat returns [Batch, H, W, 3]. We squeeze to get [H, W, 3].
        image0 = image.squeeze(0) if image.ndim == 4 else image
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