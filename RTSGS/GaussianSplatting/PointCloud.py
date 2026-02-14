import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as Rot

class PointCloud:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Intrinsics
        self.fx = torch.tensor(float(config.get("fx")), device=self.device)
        self.fy = torch.tensor(float(config.get("fy")), device=self.device)
        self.cx = torch.tensor(float(config.get("cx")), device=self.device)
        self.cy = torch.tensor(float(config.get("cy")), device=self.device)

        self.depth_scale = float(config.get("depth_scale", 1.0))
        self.voxel_size = float(config.get("voxel_size", 0.02))
        self.novelty_voxel = float(config.get("novelty_voxel", self.voxel_size))

        # Noise and Alpha params
        self.sigma_px = float(config.get("sigma_px", 4.0))
        self.sigma_z0 = float(config.get("sigma_z0", 0.005))
        self.sigma_z1 = float(config.get("sigma_z1", 0.0))
        self.alpha_init = float(config.get("alpha_init", 1.0))
        self.alpha_min = float(config.get("alpha_min", 0.01))
        self.alpha_max = float(config.get("alpha_max", 1.0))
        self.alpha_depth_scale = float(config.get("alpha_depth_scale", 0.0))

        # Global Map (Persistent Training Variables)
        self.all_points = None
        self.all_colors = None
        self.all_scales = None
        self.all_quaternions = None
        self.all_alpha = None

        # Image Grid
        H, W = int(config.get("height")), int(config.get("width"))
        v, u = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing="ij",
        )
        self.v, self.u = v.float(), u.float()

        # Novelty/Packing state - MUST BE ON DEVICE
        self._pack_offset = int(config.get("pack_offset", 1_000_000))
        self._pack_base = 2 * self._pack_offset + 1
        self.seen_keys = torch.empty((0,), dtype=torch.int64, device=self.device)
        self.pixel_subsample = float(config.get("pixel_subsample", 1.0))

    def _pack_voxels(self, vox_xyz: torch.Tensor) -> torch.Tensor:
        """ Packs 3D voxel coordinates into a single 64-bit integer key on GPU. """
        off = self._pack_offset
        base = self._pack_base
        return (vox_xyz[:, 0] + off) * (base**2) + (vox_xyz[:, 1] + off) * base + (vox_xyz[:, 2] + off)

    @torch.no_grad()
    def _make_gaussians_batch(self, points_cam, R_world_from_cam, z):
        device = self.device
        N = points_cam.shape[0]

        # 1. Scale Calculation (Matches voxel density)
        sigma_z = self.sigma_z0 + self.sigma_z1 * (z * z)
        sigma_x = (self.sigma_px / self.fx) * z
        sigma_y = (self.sigma_px / self.fy) * z
        scales = torch.stack([sigma_x, sigma_y, sigma_z], dim=-1)
        
        # LOG SPACE: Clamp to prevent -inf or tiny dots
        # Ensuring scales are at least 1/4 of voxel size to prevent gaps
        scales = torch.log(torch.clamp(scales, min=0.01)) 

        # 2. Quaternions
        r_np = R_world_from_cam.cpu().numpy()
        if r_np.ndim == 2:
            q_single = Rot.from_matrix(r_np).as_quat()
            quats = torch.from_numpy(q_single).to(device).float().expand(N, 4)
        else:
            quats = torch.from_numpy(Rot.from_matrix(r_np).as_quat()).to(device).float()

        # 3. Alpha (Opacity)
        if self.alpha_depth_scale > 0.0:
            a = self.alpha_init * torch.exp(-z / self.alpha_depth_scale)
        else:
            a = torch.full((N,), float(self.alpha_init), device=device)
        
        # LOGIT SPACE: Clamp to [0.1, 0.99] to keep gradients healthy
        a = torch.clamp(a, self.alpha_min, self.alpha_max).unsqueeze(1)
        a = torch.logit(a)

        return scales, quats, a

    @torch.no_grad()
    def novelty_filter_fast_with_gaussians(self, points, colors, voxel):
        """ GPU-only novelty filter using packed voxel keys. """
        vox = torch.floor(points / voxel).to(torch.int64)
        keys = self._pack_voxels(vox)
        
        keys_unique, inv_indices = torch.unique(keys, return_inverse=True)
        
        # Get first occurrence index for each unique key on GPU
        first_idx = torch.full(keys_unique.shape, points.shape[0], dtype=torch.long, device=self.device)
        first_idx.scatter_reduce_(0, inv_indices, torch.arange(points.shape[0], device=self.device), "amin", include_self=False)

        if self.seen_keys.numel() == 0:
            is_new = torch.ones_like(keys_unique, dtype=torch.bool)
        else:
            is_new = ~torch.isin(keys_unique, self.seen_keys)

        if not is_new.any(): 
            return None

        new_keys = keys_unique[is_new]
        rep_idx = first_idx[is_new]
        
        # Update global state on GPU
        self.seen_keys = torch.cat([self.seen_keys, new_keys])
        
        return points[rep_idx], colors[rep_idx], rep_idx

    @torch.no_grad()
    def voxel_filter_with_gaussians(self, points, colors, scales, quats, alpha, voxel):
        """ Downsamples the final novel points using voxel grid averaging. """
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

        f_points = reduce_mean(points, 3)
        f_colors = reduce_mean(colors, 3)
        f_scales = reduce_mean(scales, 3)
        f_quats = F.normalize(reduce_mean(quats, 4), p=2, dim=1)
        f_alpha = reduce_mean(alpha, 1)

        return f_points, f_colors, f_scales, f_quats, f_alpha

    @torch.no_grad()
    def process_keyframes(self, rgb_list, depth_list, pose_list):
        device = self.device
        rgb = torch.from_numpy(np.stack(rgb_list)).to(device, non_blocking=True).float() / 255.0
        depth = torch.from_numpy(np.stack(depth_list)).to(device, non_blocking=True).float() / self.depth_scale
        poses = torch.from_numpy(np.stack(pose_list)).to(device, non_blocking=True).float()

        Kf, H, W = depth.shape
        z = depth.reshape(-1)

        # 1. Faster Masking & Subsampling
        mask = z > 0
        if self.pixel_subsample < 1.0:
            mask &= (torch.rand(z.shape, device=device) < self.pixel_subsample)
        
        indices = torch.where(mask)[0]
        if indices.numel() == 0: return None

        z_f = z[indices]
        u_f = (indices % W).float()
        v_f = ((indices // W) % H).float()

        # 2. Backprojection to Camera Space
        x = (u_f - self.cx) * z_f / self.fx
        y = (v_f - self.cy) * z_f / self.fy
        points_cam = torch.stack((x, y, z_f), dim=1)

        # 3. Transformation to World Space
        frame_ids = indices // (H * W)
        R = poses[frame_ids, :3, :3]
        t = poses[frame_ids, :3, 3]
        points_world = torch.bmm(R, points_cam.unsqueeze(-1)).squeeze(-1) + t
        colors = rgb.reshape(-1, 3)[indices]

        # 4. Filter Novelty BEFORE expensive Gaussian creation
        res = self.novelty_filter_fast_with_gaussians(points_world, colors, self.novelty_voxel)
        if res is None: return None
        
        filtered_pts, filtered_cols, kept_indices = res
        
        # 5. Create Gaussians ONLY for novel points
        scales, quats, alpha = self._make_gaussians_batch(
            points_cam[kept_indices], 
            R[kept_indices], 
            z_f[kept_indices]
        )


        filtered_cols = torch.clamp(filtered_cols, min=0.001, max=0.999)
        filtered_cols = torch.logit(filtered_cols)
        # 6. Final Voxel Downsampling
        return self.voxel_filter_with_gaussians(filtered_pts, filtered_cols, scales, quats, alpha, self.voxel_size)

    def update_full_pointcloud(self, rgb_keyframes, depth_keyframes, poses):
        """ Main entry point: Integrates new frames into the persistent map. """
        if not depth_keyframes: return None

        new_data = self.process_keyframes(rgb_keyframes, depth_keyframes, poses)

        if new_data is None:
            return self.all_points, self.all_colors, self.all_scales, self.all_quaternions, self.all_alpha

        new_pts, new_cols, new_scales, new_quats, new_alphas = new_data
        
        if self.all_points is None:
            self.all_points, self.all_colors = new_pts, new_cols
            self.all_scales, self.all_quaternions, self.all_alpha = new_scales, new_quats, new_alphas
        else:
            # Concatenate along the Gaussian dimension
            self.all_points = torch.cat([self.all_points, new_pts], dim=0)
            self.all_colors = torch.cat([self.all_colors, new_cols], dim=0)
            self.all_scales = torch.cat([self.all_scales, new_scales], dim=0)
            self.all_quaternions = torch.cat([self.all_quaternions, new_quats], dim=0)
            self.all_alpha = torch.cat([self.all_alpha, new_alphas], dim=0)

        return self.all_points, self.all_colors, self.all_scales, self.all_quaternions, self.all_alpha