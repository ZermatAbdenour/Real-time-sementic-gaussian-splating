import threading
import numpy as np
import torch
import torch.nn.functional as F
import concurrent.futures
import time


@torch.no_grad()
def rotmat_to_quat_xyzw(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to quaternion (x, y, z, w) on GPU.
    R: (..., 3, 3)
    Returns: (..., 4)
    """
    t = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    qw = torch.sqrt(torch.clamp(1.0 + t, min=1e-8)) * 0.5
    denom = torch.clamp(4.0 * qw, min=1e-8)

    qx = (R[..., 2, 1] - R[..., 1, 2]) / denom
    qy = (R[..., 0, 2] - R[..., 2, 0]) / denom
    qz = (R[..., 1, 0] - R[..., 0, 1]) / denom

    q = torch.stack([qx, qy, qz, qw], dim=-1)
    return torch.nan_to_num(q)


class PointCloud:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Camera Parameters (keep as float tensors on device)
        self.fx = torch.tensor(float(config.get("fx")), device=self.device)
        self.fy = torch.tensor(float(config.get("fy")), device=self.device)
        self.cx = torch.tensor(float(config.get("cx")), device=self.device)
        self.cy = torch.tensor(float(config.get("cy")), device=self.device)

        self.depth_scale = float(config.get("depth_scale", 1.0))
        self.voxel_size = float(config.get("voxel_size", 0.02))

        # SH Parameters
        self.sh_degree = int(config.get("sh_degree", 1))
        self.num_sh_bases = (self.sh_degree + 1) ** 2

        # Gaussian Properties
        self.sigma_px = float(config.get("sigma_px", 4.0))
        self.sigma_z0 = float(config.get("sigma_z0", 0.005))
        self.sigma_z1 = float(config.get("sigma_z1", 0.0))
        self.alpha_init = float(config.get("alpha_init", 1.0))
        self.alpha_min = float(config.get("alpha_min", 0.01))
        self.alpha_max = float(config.get("alpha_max", 1.0))
        self.alpha_depth_scale = float(config.get("alpha_depth_scale", 0.0))

        self.all_points = None
        self.all_sh = None
        self.all_scales = None
        self.all_quaternions = None
        self.all_alpha = None

        self.pixel_subsample = float(config.get("pixel_subsample", 1.0))
        self.unproject_stride = int(config.get("unproject_stride", 2))

        # Rotation Fix (cached on device)
        ax, ay = np.radians(-90), np.radians(180)
        Rx = torch.tensor([[1, 0, 0],
                           [0, np.cos(ax), -np.sin(ax)],
                           [0, np.sin(ax),  np.cos(ax)]], dtype=torch.float32, device=self.device)
        Ry = torch.tensor([[ np.cos(ay), 0, np.sin(ay)],
                           [0, 1, 0],
                           [-np.sin(ay), 0, np.cos(ay)]], dtype=torch.float32, device=self.device)
        self.R_fix = (Ry @ Rx).to(self.device)

        # Cache T_fix and intrinsics matrix buffers (avoid per-frame alloc)
        self.T_fix = torch.eye(4, device=self.device, dtype=torch.float32)
        self.T_fix[:3, :3] = self.R_fix

        self.K = torch.eye(3, device=self.device, dtype=torch.float32)

        # --- Async Handling ---
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.is_processing = False
        self.lock = threading.Lock()

        # Profiling
        self.enable_profile = bool(config.get("enable_profile", False))

    def _cuda_sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    @torch.no_grad()
    def _make_gaussians_batch(self, z: torch.Tensor, R_world_from_cam: torch.Tensor, N: int):
        sigma_z = self.sigma_z0 + self.sigma_z1 * (z * z)
        sigma_x = (self.sigma_px / self.fx) * z
        sigma_y = (self.sigma_px / self.fy) * z
        scales = torch.log(torch.clamp(torch.stack([sigma_x, sigma_y, sigma_z], dim=-1), min=0.01))

        quat = rotmat_to_quat_xyzw(R_world_from_cam.unsqueeze(0)).expand(N, -1)

        if self.alpha_depth_scale > 0.0:
            a = self.alpha_init * torch.exp(-z / self.alpha_depth_scale)
        else:
            a = torch.full((N,), float(self.alpha_init), device=self.device)

        a = torch.logit(torch.clamp(a, self.alpha_min, self.alpha_max)).unsqueeze(1)
        return scales.to(torch.float32), quat.to(torch.float32), a.to(torch.float32)

    @torch.no_grad()
    def voxel_filter_with_gaussians(self, points, sh, scales, quats, alpha, voxel):
        # Note: this filters ONLY within the current frame; it does NOT dedupe vs past frames.
        if points.numel() == 0:
            return points, sh, scales, quats, alpha

        vox = torch.floor(points / voxel).to(torch.int64)

        # Use packed 3D key via arithmetic on int64 without storing global seen_keys.
        # We can hash by shifting; safer than overflow? We'll use a large offset/base like your original,
        # but compute locally and discard after.
        off = 1_000_000
        base = 2 * off + 1
        keys = (vox[:, 0] + off) * (base**2) + (vox[:, 1] + off) * base + (vox[:, 2] + off)

        _, inverse = torch.unique(keys, return_inverse=True)
        num = int(inverse.max().item()) + 1

        counts = torch.zeros((num, 1), device=self.device, dtype=points.dtype)
        counts.scatter_add_(0, inverse[:, None], torch.ones((points.shape[0], 1), device=self.device, dtype=points.dtype))

        def reduce_mean(tensor):
            out_shape = (num,) + tensor.shape[1:]
            out = torch.zeros(out_shape, device=self.device, dtype=tensor.dtype)
            idx = inverse.view(-1, *([1] * (tensor.dim() - 1))).expand_as(tensor)
            out.scatter_add_(0, idx, tensor)
            div = counts.view(-1, *([1] * (tensor.dim() - 1)))
            return out / div

        pts_m = reduce_mean(points)
        sh_m = reduce_mean(sh)
        scales_m = reduce_mean(scales)
        quats_m = F.normalize(reduce_mean(quats), p=2, dim=1)
        alpha_m = reduce_mean(alpha)

        return pts_m, sh_m, scales_m, quats_m, alpha_m

    @torch.no_grad()
    def process_single_keyframe(self, rgb_np, depth_np, pose_np):
        t0 = time.perf_counter()
        pose = torch.from_numpy(pose_np).to(self.device).float()

        s = self.unproject_stride
        if s > 1:
            # Keep copy() on depth to ensure contiguous; rgb can be non-contig but torch handles it
            rgb_s = torch.from_numpy(rgb_np[::s, ::s]).to(self.device).float().mul_(1.0 / 255.0)
            depth_s = torch.from_numpy(depth_np[::s, ::s].copy()).to(self.device).float().mul_(1.0 / self.depth_scale)
            fx_s, fy_s = self.fx / s, self.fy / s
            cx_s, cy_s = self.cx / s, self.cy / s
        else:
            rgb_s = torch.from_numpy(rgb_np).to(self.device).float().mul_(1.0 / 255.0)
            depth_s = torch.from_numpy(depth_np).to(self.device).float().mul_(1.0 / self.depth_scale)
            fx_s, fy_s = self.fx, self.fy
            cx_s, cy_s = self.cx, self.cy

        H, W = depth_s.shape
        z_raw = depth_s.reshape(-1)
        mask = z_raw > 0

        # Optional "overlap check" render against existing map (this can be expensive)
        if self.all_points is not None and self.all_points.numel() > 0:
            from gsplat import rendering

            w2c = torch.inverse(self.T_fix @ pose).unsqueeze(0)

            # Update cached K values (no realloc)
            self.K.zero_()
            self.K[0, 0] = fx_s
            self.K[1, 1] = fy_s
            self.K[0, 2] = cx_s
            self.K[1, 2] = cy_s
            self.K[2, 2] = 1.0
            Ks = self.K.unsqueeze(0)

            # Avoid allocating huge dummy colors each time: use a single-channel or cached tensor
            # (gsplat expects colors Nx3; cache and resize only when needed)
            n = self.all_points.shape[0]
            dummy_colors = torch.ones((n, 3), device=self.device, dtype=torch.float32)

            rendered_ed, render_alphas, _ = rendering.rasterization(
                means=self.all_points,
                quats=F.normalize(self.all_quaternions, p=2, dim=-1),
                scales=torch.exp(self.all_scales),
                opacities=torch.sigmoid(self.all_alpha).squeeze(-1),
                colors=dummy_colors,
                viewmats=w2c,
                Ks=Ks,
                width=W,
                height=H,
                render_mode="ED",
            )

            D_rend_flat = rendered_ed[0, ..., 0].reshape(-1)
            O_flat = render_alphas[0, ..., 0].reshape(-1)

            tau = float(self.config.get("depth_tau", 0.05))
            hole_mask = (O_flat < 0.5) | (torch.abs(D_rend_flat - z_raw) > tau)

            valid_pixels = (z_raw > 0).sum().item()
            if valid_pixels > 0:
                unexplained_pixels = (hole_mask & (z_raw > 0)).sum().item()
                explained_ratio = 1.0 - (unexplained_pixels / valid_pixels)
                if explained_ratio > 0.7:
                    # Skip spawning if mostly explained by existing map
                    return None

            mask &= hole_mask

        if self.pixel_subsample < 1.0:
            mask &= (torch.rand(z_raw.shape, device=self.device) < self.pixel_subsample)

        indices = torch.where(mask)[0]
        if indices.numel() == 0:
            return None

        # Unproject (vectorized)
        z_f = z_raw[indices]
        u = (indices % W).to(torch.float32)
        v = (indices // W).to(torch.float32)

        x = (u - cx_s) * z_f / fx_s
        y = (v - cy_s) * z_f / fy_s
        points_cam = torch.stack([x, y, z_f], dim=1)

        # World transform with correction
        R_corr = self.R_fix @ pose[:3, :3]
        t_corr = (self.R_fix @ pose[:3, 3].unsqueeze(-1)).squeeze(-1)
        points_world = (R_corr @ points_cam.T).T + t_corr

        colors = rgb_s.reshape(-1, 3)[indices]

        # SH conversion
        sh_full = torch.zeros((points_world.shape[0], self.num_sh_bases, 3), device=self.device, dtype=torch.float32)
        sh_full[:, 0, :] = torch.logit(torch.clamp(colors, 0.001, 0.999)) / 0.28209479177387814

        # Attributes
        N = points_world.shape[0]
        scales, quats, alpha = self._make_gaussians_batch(z_f, R_corr, N)

        out = self.voxel_filter_with_gaussians(points_world, sh_full, scales, quats, alpha, self.voxel_size)

        if self.enable_profile:
            self._cuda_sync()
            dt = (time.perf_counter() - t0) * 1000.0
            print(f"[process_single_keyframe] {dt:.2f} ms  (H={H}, W={W}, N_in={indices.numel()}, N_out={out[0].shape[0]})")

        return out

    def update_async(self, rgb_np, depth_np, pose_np):
        # Note: GPU work in threads can increase jitter; consider keeping this synchronous for RT.
        if self.is_processing:
            return False

        self.is_processing = True

        def _task():
            try:
                new_data = self.process_single_keyframe(rgb_np, depth_np, pose_np)
                if new_data is not None:
                    self._merge_data(new_data)
            finally:
                self.is_processing = False

        self.executor.submit(_task)
        return True

    def _merge_data(self, new_data):
        with self.lock:
            if self.all_points is None:
                self.all_points, self.all_sh, self.all_scales, self.all_quaternions, self.all_alpha = new_data
            else:
                self.all_points = torch.cat([self.all_points, new_data[0]], dim=0)
                self.all_sh = torch.cat([self.all_sh, new_data[1]], dim=0)
                self.all_scales = torch.cat([self.all_scales, new_data[2]], dim=0)
                self.all_quaternions = torch.cat([self.all_quaternions, new_data[3]], dim=0)
                self.all_alpha = torch.cat([self.all_alpha, new_data[4]], dim=0)

    def get_map(self):
        return self.all_points, self.all_sh, self.all_scales, self.all_quaternions, self.all_alpha