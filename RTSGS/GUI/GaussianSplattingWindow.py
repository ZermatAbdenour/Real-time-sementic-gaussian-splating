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
    def __init__(self, pcd: PointCloud, camera: Camera, title: str = "GSplat Renderer"):
        self.pcd = pcd
        self.camera = camera
        self.title = title
        self.device = getattr(pcd, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.image_widget = ImageWidget(np.zeros((camera.height, camera.width, 3), dtype=np.uint8))
        self.is_open = True
       
        self.fx = float(pcd.fx.item()) if torch.is_tensor(pcd.fx) else float(pcd.fx)
        self.fy = float(pcd.fy.item()) if torch.is_tensor(pcd.fy) else float(pcd.fy)
        self.default_scale = 0.01
        self._xyz = self._rgb = self._quats = self._scales = self._alpha = None

    def _pull_latest_buffers(self) -> bool:
        xyz, rgb = getattr(self.pcd, "all_points", None), getattr(self.pcd, "all_colors", None)
        if xyz is None or xyz.numel() == 0: return False
        self._xyz, self._rgb = xyz.detach(), rgb.detach()
        self._quats, self._scales, self._alpha = getattr(self.pcd, "all_quaternions", None), getattr(self.pcd, "all_scales", None), getattr(self.pcd, "all_alpha", None)
        return True

    @torch.no_grad()
    def _render_with_gsplat(self) -> np.ndarray:

        self.camera.update_view()
        R_np, t_np = _extract_camera_rt_from_view(self.camera.view)
        viewmat = _row_rt_to_viewmat_col_major(_to_torch_f32(R_np, self.device), _to_torch_f32(t_np, self.device))
      
        K = torch.tensor([[self.fx, 0, self.camera.width/2], [0, self.fy, self.camera.height/2], [0, 0, 1]], 
                         device=self.device, dtype=torch.float32).unsqueeze(0)

        N = self._xyz.shape[0]
        q = F.normalize(self._quats.detach(), p=2, dim=-1) if self._quats is not None else torch.tensor([[0,0,0,1.]], device=self.device).repeat(N, 1)
        s = torch.exp(self._scales.detach()) if self._scales is not None else torch.full((N, 3), self.default_scale, device=self.device)
        o = torch.sigmoid(self._alpha.detach()).squeeze(-1) if self._alpha is not None else torch.ones((N,), device=self.device)
        c = torch.sigmoid(self._rgb.detach()) if self._rgb is not None else torch.ones((N, 3), device=self.device)

        img, _, _ = rendering.rasterization(
            means=self._xyz, quats=q, scales=s, opacities=o, colors=c.contiguous(),
            viewmats=viewmat.unsqueeze(0), Ks=K, width=self.camera.width, height=self.camera.height, render_mode="RGB"
        )
        return (img.squeeze().clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()

    def draw(self, delta_time: float):
        if not self.is_open: return
        expanded,self.is_open = imgui.begin(self.title, self.is_open)
        if(not expanded):
            imgui.end()
            return 
        avail = imgui.get_content_region_avail()
        if int(avail.x) != self.camera.width or int(avail.y) != self.camera.height:
            self.camera.update_resolution(int(avail.x), int(avail.y))

        self.camera.process_window_input(imgui.is_window_hovered(), imgui.is_window_focused(), delta_time)

        # Render and Display
        if self._pull_latest_buffers():
            rgb = self._render_with_gsplat()
            self.image_widget.set_image_rgb(rgb)
            self.image_widget.draw(fit_to_window=True)
        else:
            imgui.text("Waiting for PointCloud data...")
            
        imgui.end()