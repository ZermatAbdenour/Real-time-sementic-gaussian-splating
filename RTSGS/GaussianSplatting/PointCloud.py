import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as Rot

class PointCloud:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.fx = torch.tensor(float(config.get("fx")), device=self.device)
        self.fy = torch.tensor(float(config.get("fy")), device=self.device)
        self.cx = torch.tensor(float(config.get("cx")), device=self.device)
        self.cy = torch.tensor(float(config.get("cy")), device=self.device)

        self.depth_scale = float(config.get("depth_scale", 1.0))
        self.voxel_size = float(config.get("voxel_size", 0.02))
        self.novelty_voxel = float(config.get("novelty_voxel", self.voxel_size))

        self.sigma_px = float(config.get("sigma_px", 4.0))
        self.sigma_z0 = float(config.get("sigma_z0", 0.005))
        self.sigma_z1 = float(config.get("sigma_z1", 0.0))
        self.alpha_init = float(config.get("alpha_init", 1.0))
        self.alpha_min = float(config.get("alpha_min", 0.01))
        self.alpha_max = float(config.get("alpha_max", 1.0))
        self.alpha_depth_scale = float(config.get("alpha_depth_scale", 0.0))

        self.all_points = None
        self.all_colors = None
        self.all_scales = None
        self.all_quaternions = None
        self.all_alpha = None

        self._pack_offset = int(config.get("pack_offset", 1_000_000))
        self._pack_base = 2 * self._pack_offset + 1
        self.seen_keys = torch.empty((0,), dtype=torch.int64, device=self.device)
        self.pixel_subsample = float(config.get("pixel_subsample", 1.0))

        ax, ay = np.radians(-90), np.radians(180)
        Rx = torch.tensor([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
        Ry = torch.tensor([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
        self.R_fix = (Ry @ Rx).to(self.device).float()

    def _pack_voxels(self, vox_xyz: torch.Tensor) -> torch.Tensor:
        off, base = self._pack_offset, self._pack_base
        return (vox_xyz[:, 0] + off) * (base**2) + (vox_xyz[:, 1] + off) * base + (vox_xyz[:, 2] + off)

    @torch.no_grad()
    def _make_gaussians_batch(self, points_cam, R_world_from_cam, z):
        N = points_cam.shape[0]
        sigma_z = self.sigma_z0 + self.sigma_z1 * (z * z)
        sigma_x, sigma_y = (self.sigma_px / self.fx) * z, (self.sigma_px / self.fy) * z
        scales = torch.log(torch.clamp(torch.stack([sigma_x, sigma_y, sigma_z], dim=-1), min=0.01))

        quats = torch.from_numpy(Rot.from_matrix(R_world_from_cam.cpu().numpy()).as_quat()).to(self.device).float()

        a = self.alpha_init * torch.exp(-z / self.alpha_depth_scale) if self.alpha_depth_scale > 0.0 else torch.full((N,), float(self.alpha_init), device=self.device)
        a = torch.logit(torch.clamp(a, self.alpha_min, self.alpha_max)).unsqueeze(1)
        return scales, quats, a

    @torch.no_grad()
    def novelty_filter_fast_with_gaussians(self, points, colors, voxel):
        if points.numel() == 0: return None
        vox = torch.floor(points / voxel).to(torch.int64)
        keys = self._pack_voxels(vox)
        keys_unique, inv_indices = torch.unique(keys, return_inverse=True)
        
        first_idx = torch.full(keys_unique.shape, points.shape[0], dtype=torch.long, device=self.device)
        first_idx.scatter_reduce_(0, inv_indices, torch.arange(points.shape[0], device=self.device), "amin", include_self=False)

        is_new = torch.ones_like(keys_unique, dtype=torch.bool) if self.seen_keys.numel() == 0 else ~torch.isin(keys_unique, self.seen_keys)
        if not is_new.any(): return None

        self.seen_keys = torch.cat([self.seen_keys, keys_unique[is_new]])
        return points[first_idx[is_new]], colors[first_idx[is_new]], first_idx[is_new]

    @torch.no_grad()
    def voxel_filter_with_gaussians(self, points, colors, scales, quats, alpha, voxel):
        if points.numel() == 0: return points, colors, scales, quats, alpha
        vox = torch.floor(points / voxel).to(torch.int64)
        _, inverse = torch.unique(vox, dim=0, return_inverse=True)
        num = int(inverse.max().item()) + 1
        counts = torch.zeros((num, 1), device=self.device, dtype=points.dtype)
        counts.scatter_add_(0, inverse[:, None], torch.ones((points.shape[0], 1), device=self.device))

        def reduce_mean(tensor, dim_size):
            out = torch.zeros((num, dim_size), device=self.device, dtype=tensor.dtype)
            out.scatter_add_(0, inverse[:, None].expand(-1, dim_size), tensor)
            return out / counts

        return reduce_mean(points, 3), reduce_mean(colors, 3), reduce_mean(scales, 3), \
               F.normalize(reduce_mean(quats, 4), p=2, dim=1), reduce_mean(alpha, 1)

    @torch.no_grad()
    def process_keyframes(self, rgb_list, depth_list, pose_list):
        if not rgb_list or len(rgb_list) == 0: return None

        rgb = torch.from_numpy(np.stack(rgb_list)).to(self.device).float() / 255.0
        depth = torch.from_numpy(np.stack(depth_list)).to(self.device).float() / self.depth_scale
        poses = torch.from_numpy(np.stack(pose_list)).to(self.device).float()

        Kf, H, W = depth.shape
        z_raw = depth.reshape(-1)
        mask = z_raw > 0
        if self.pixel_subsample < 1.0:
            mask &= (torch.rand(z_raw.shape, device=self.device) < self.pixel_subsample)
        
        indices = torch.where(mask)[0]
        if indices.numel() == 0: return None

        z_f = z_raw[indices]
        points_cam = torch.stack([((indices % W).float() - self.cx) * z_f / self.fx,
                                  ((indices // W % H).float() - self.cy) * z_f / self.fy,
                                  z_f], dim=1)

        R, t = poses[indices // (H * W), :3, :3], poses[indices // (H * W), :3, 3]
        R_corr, t_corr = self.R_fix @ R, (self.R_fix @ t.unsqueeze(-1)).squeeze(-1)

        points_world = torch.bmm(R_corr, points_cam.unsqueeze(-1)).squeeze(-1) + t_corr
        colors = rgb.reshape(-1, 3)[indices]

        res = self.novelty_filter_fast_with_gaussians(points_world, colors, self.novelty_voxel)
        if res is None: return None
        
        pts, cols, k_idx = res
        scales, quats, alpha = self._make_gaussians_batch(points_cam[k_idx], R_corr[k_idx], z_f[k_idx])
        return self.voxel_filter_with_gaussians(pts, torch.logit(torch.clamp(cols, 0.001, 0.999)), scales, quats, alpha, self.voxel_size)

    def update_full_pointcloud(self, rgb_keyframes, depth_keyframes, poses):
        if not rgb_keyframes: return self.get_map()
        new_data = self.process_keyframes(rgb_keyframes, depth_keyframes, poses)
        if new_data is None: return self.get_map()

        if self.all_points is None:
            self.all_points, self.all_colors, self.all_scales, self.all_quaternions, self.all_alpha = new_data
        else:
            self.all_points = torch.cat([self.all_points, new_data[0]])
            self.all_colors = torch.cat([self.all_colors, new_data[1]])
            self.all_scales = torch.cat([self.all_scales, new_data[2]])
            self.all_quaternions = torch.cat([self.all_quaternions, new_data[3]])
            self.all_alpha = torch.cat([self.all_alpha, new_data[4]])
        return self.get_map()

    def get_map(self):
        return self.all_points, self.all_colors, self.all_scales, self.all_quaternions, self.all_alpha