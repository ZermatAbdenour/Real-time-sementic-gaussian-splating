import numpy as np
import torch

from RTSGS.DataLoader.DataLoader import DataLoader


class PointCloud:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # intrinsics
        self.fx = torch.tensor(float(config.get("fx")), device=self.device, dtype=torch.float32)
        self.fy = torch.tensor(float(config.get("fy")), device=self.device, dtype=torch.float32)
        self.cx = torch.tensor(float(config.get("cx")), device=self.device, dtype=torch.float32)
        self.cy = torch.tensor(float(config.get("cy")), device=self.device, dtype=torch.float32)

        self.depth_scale = float(config.get("depth_scale", 1.0))

        # voxels
        self.voxel_size = float(config.get("voxel_size", 0.02))
        self.novelty_voxel = float(config.get("novelty_voxel", self.voxel_size))

        # frame skipping
        self.frame_skip = int(config.get("frame_skip", 1))
        self.frame_count = 0

        # -----------------------------
        # Gaussian params (NEW)
        # -----------------------------
        # pixel noise (in pixels)
        self.sigma_px = float(config.get("sigma_px", 1.0))

        # depth noise model: sigma_z = sigma_z0 + sigma_z1 * z^2  (meters)
        self.sigma_z0 = float(config.get("sigma_z0", 0.002))
        self.sigma_z1 = float(config.get("sigma_z1", 0.0))

        # alpha parameters
        self.alpha_init = float(config.get("alpha_init", 1.0))
        self.alpha_min = float(config.get("alpha_min", 0.01))
        self.alpha_max = float(config.get("alpha_max", 1.0))
        # if > 0: alpha = alpha_init * exp(-z / alpha_depth_scale)
        self.alpha_depth_scale = float(config.get("alpha_depth_scale", 0.0))

        # storage (CPU)
        self.points = []
        self.colors = []
        self.gaussian_covariances = []  # list of [Ni, 3, 3] CPU
        self.gaussian_alpha = []        # list of [Ni, 1] CPU

        self.all_points = None
        self.all_colors = None
        self.all_covariances = None
        self.all_alpha = None

        # image grid
        H, W = int(config.get("height")), int(config.get("width"))
        self.v, self.u = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing="ij",
        )

        # packing parameters
        self._pack_offset = int(config.get("pack_offset", 1_000_000))
        self._pack_base = 2 * self._pack_offset + 1

        # FAST novelty state (CPU tensor)
        self.seen_keys = torch.empty((0,), dtype=torch.int64, device="cpu")
        self._dedup_every = int(config.get("dedup_every", 20))  # frames
        self._frame_since_dedup = 0

        # optional pixel subsampling
        self.pixel_subsample = float(config.get("pixel_subsample", 1.0))

    # ---------------------------------------------------------
    # NEW: build covariance + alpha for each point
    # ---------------------------------------------------------
    @torch.no_grad()
    def _make_gaussians(self, points_cam: torch.Tensor, R_world_from_cam: torch.Tensor, z: torch.Tensor):
        """
        points_cam: [N,3] in camera frame
        R_world_from_cam: [3,3] (single-frame) rotation
        z: [N] depth in meters for each point (same order as points_cam)

        Returns:
          cov_world: [N,3,3]
          alpha: [N,1]
        """
        device = points_cam.device
        N = points_cam.shape[0]

        # depth noise
        sigma_z = self.sigma_z0 + self.sigma_z1 * (z * z)  # [N]

        # convert pixel noise into metric noise at depth z:
        # sigma_x ≈ (sigma_px / fx) * z
        # sigma_y ≈ (sigma_px / fy) * z
        sigma_x = (self.sigma_px / self.fx) * z
        sigma_y = (self.sigma_px / self.fy) * z

        # camera-frame covariance (diagonal)
        cov_cam = torch.zeros((N, 3, 3), device=device, dtype=torch.float32)
        cov_cam[:, 0, 0] = sigma_x * sigma_x
        cov_cam[:, 1, 1] = sigma_y * sigma_y
        cov_cam[:, 2, 2] = sigma_z * sigma_z

        # rotate to world: cov_world = R * cov_cam * R^T
        # Here R is constant for this frame.
        R = R_world_from_cam.to(device=device, dtype=torch.float32)
        cov_world = R.unsqueeze(0) @ cov_cam @ R.t().unsqueeze(0)

        # alpha
        if self.alpha_depth_scale > 0.0:
            a = self.alpha_init * torch.exp(-z / self.alpha_depth_scale)
        else:
            a = torch.full((N,), float(self.alpha_init), device=device, dtype=torch.float32)

        a = torch.clamp(a, min=self.alpha_min, max=self.alpha_max).unsqueeze(1)  # [N,1]
        return cov_world, a

    @torch.no_grad()
    def add_frame(self, rgb: np.ndarray, depth: np.ndarray, pose: np.ndarray):
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return

        depth = depth.astype(np.float32) / self.depth_scale

        img = torch.from_numpy(rgb).to(self.device).float() / 255.0
        depth_t = torch.from_numpy(depth).to(self.device)
        pose_t = torch.from_numpy(pose).to(self.device).float()

        R = pose_t[:3, :3]
        t = pose_t[:3, 3]

        z_full = depth_t
        mask = z_full > 0

        if self.pixel_subsample < 1.0:
            rnd = torch.rand_like(z_full, dtype=torch.float32)
            mask = mask & (rnd < self.pixel_subsample)

        if mask.sum() == 0:
            return

        indices = torch.where(mask)
        v_indices = indices[0]
        u_indices = indices[1]

        u = u_indices.float()
        v = v_indices.float()
        z = depth_t[indices]  # [N]

        # back-project
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        points_cam = torch.stack((x, y, z), dim=1)  # [N,3]

        # to world
        points_world = (R @ points_cam.T).T + t

        # colors
        points_colors = img[v_indices, u_indices]  # [N,3]

        # NEW: gaussians (before filtering so we can filter them consistently)
        cov_world, alpha = self._make_gaussians(points_cam, R, z)

        # novelty gating
        points_world, points_colors, cov_world, alpha = self.novelty_filter_fast_with_gaussians(
            points_world, points_colors, cov_world, alpha, voxel=self.novelty_voxel
        )
        if points_world is None:
            return

        # voxel averaging / merging
        points_world, points_colors, cov_world, alpha = self.voxel_filter_with_gaussians(
            points_world, points_colors, cov_world, alpha, voxel=self.voxel_size
        )

        # store CPU
        self.points.append(points_world.detach().cpu())
        self.colors.append(points_colors.detach().cpu())
        self.gaussian_covariances.append(cov_world.detach().cpu())
        self.gaussian_alpha.append(alpha.detach().cpu())

        if torch.cuda.is_available() and (self.frame_count % 10 == 0):
            torch.cuda.empty_cache()

    @torch.no_grad()
    def add_keyframes_batch_gpu(self, rgb_list, depth_list, pose_list):
        """
        True GPU batch processing of keyframes.
        """
        device = self.device
        K = len(rgb_list)
        assert K > 0

        rgb = torch.from_numpy(np.stack(rgb_list)).to(device).float() / 255.0
        depth = torch.from_numpy(np.stack(depth_list)).to(device).float() / self.depth_scale
        poses = torch.from_numpy(np.stack(pose_list)).to(device).float()

        Kf, H, W = depth.shape

        u = self.u[None].expand(Kf, -1, -1).reshape(-1).float()
        v = self.v[None].expand(Kf, -1, -1).reshape(-1).float()

        z = depth.reshape(-1)
        mask = z > 0
        if self.pixel_subsample < 1.0:
            mask &= (torch.rand_like(z) < self.pixel_subsample)

        if mask.sum() == 0:
            return

        u = u[mask]
        v = v[mask]
        z = z[mask]

        # Backprojection (camera space)
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        points_cam = torch.stack((x, y, z), dim=1)  # [N,3]

        # Pose expansion
        frame_ids = (
            torch.arange(Kf, device=device)
            .view(Kf, 1, 1)
            .expand(Kf, H, W)
            .reshape(-1)[mask]
        )

        R = poses[frame_ids, :3, :3]  # [N,3,3]
        t = poses[frame_ids, :3, 3]   # [N,3]

        points_world = torch.bmm(R, points_cam.unsqueeze(-1)).squeeze(-1) + t

        # Colors
        colors = rgb.reshape(-1, 3)[mask]

        # NEW: Gaussians for batch
        cov_world, alpha = self._make_gaussians_batch(points_cam, R, z)

        # Novelty + voxel filters (ONCE)
        points_world, colors, cov_world, alpha = self.novelty_filter_fast_with_gaussians(
            points_world, colors, cov_world, alpha, voxel=self.novelty_voxel
        )
        if points_world is None:
            return

        points_world, colors, cov_world, alpha = self.voxel_filter_with_gaussians(
            points_world, colors, cov_world, alpha, voxel=self.voxel_size
        )

        # Store
        self.points.append(points_world.detach().cpu())
        self.colors.append(colors.detach().cpu())
        self.gaussian_covariances.append(cov_world.detach().cpu())
        self.gaussian_alpha.append(alpha.detach().cpu())

    @torch.no_grad()
    def _make_gaussians_batch(self, points_cam: torch.Tensor, R_world_from_cam: torch.Tensor, z: torch.Tensor):
        """
        points_cam: [N,3]
        R_world_from_cam: [N,3,3] per-point rotation (from frame_ids)
        z: [N]

        Returns:
          cov_world: [N,3,3]
          alpha: [N,1]
        """
        device = points_cam.device
        N = points_cam.shape[0]

        sigma_z = self.sigma_z0 + self.sigma_z1 * (z * z)
        sigma_x = (self.sigma_px / self.fx) * z
        sigma_y = (self.sigma_px / self.fy) * z

        cov_cam = torch.zeros((N, 3, 3), device=device, dtype=torch.float32)
        cov_cam[:, 0, 0] = sigma_x * sigma_x
        cov_cam[:, 1, 1] = sigma_y * sigma_y
        cov_cam[:, 2, 2] = sigma_z * sigma_z

        R = R_world_from_cam.to(dtype=torch.float32)
        cov_world = torch.bmm(R, torch.bmm(cov_cam, R.transpose(1, 2)))

        if self.alpha_depth_scale > 0.0:
            a = self.alpha_init * torch.exp(-z / self.alpha_depth_scale)
        else:
            a = torch.full((N,), float(self.alpha_init), device=device, dtype=torch.float32)

        a = torch.clamp(a, min=self.alpha_min, max=self.alpha_max).unsqueeze(1)
        return cov_world, a

    # -------- packing --------
    def _pack_voxels(self, vox_xyz_cpu: torch.Tensor) -> torch.Tensor:
        off = self._pack_offset
        base = self._pack_base
        x = vox_xyz_cpu[:, 0] + off
        y = vox_xyz_cpu[:, 1] + off
        z = vox_xyz_cpu[:, 2] + off
        return x * (base * base) + y * base + z

    @staticmethod
    def _unique_with_first_index_sorted(keys_cpu: torch.Tensor):
        sorted_keys, sorted_idx = torch.sort(keys_cpu)
        if sorted_keys.numel() == 0:
            return sorted_keys, sorted_idx

        keep = torch.ones(sorted_keys.shape[0], dtype=torch.bool, device="cpu")
        keep[1:] = sorted_keys[1:] != sorted_keys[:-1]

        unique_keys = sorted_keys[keep]
        first_idx = sorted_idx[keep]
        return unique_keys, first_idx

    # -------- novelty filter (fast) --------
    @torch.no_grad()
    def novelty_filter_fast(self, points: torch.Tensor, colors: torch.Tensor, voxel: float):
        vox = torch.floor(points / voxel).to(torch.int64).detach().to("cpu")
        keys = self._pack_voxels(vox)

        keys_unique, first_idx = self._unique_with_first_index_sorted(keys)
        if keys_unique.numel() == 0:
            return None, None

        if self.seen_keys.numel() == 0:
            is_new = torch.ones_like(keys_unique, dtype=torch.bool, device="cpu")
        else:
            is_new = ~torch.isin(keys_unique, self.seen_keys)

        if int(is_new.sum().item()) == 0:
            return None, None

        new_keys = keys_unique[is_new]
        new_first_idx = first_idx[is_new]

        rep_idx = new_first_idx.to(points.device)
        out_points = points[rep_idx]
        out_colors = colors[rep_idx]

        self.seen_keys = torch.cat([self.seen_keys, new_keys])

        self._frame_since_dedup += 1
        if self._frame_since_dedup >= self._dedup_every:
            self.seen_keys = torch.unique(self.seen_keys)
            self._frame_since_dedup = 0

        return out_points, out_colors

    # -------- NEW: novelty filter that also filters cov+alpha --------
    @torch.no_grad()
    def novelty_filter_fast_with_gaussians(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
        cov_world: torch.Tensor,
        alpha: torch.Tensor,
        voxel: float,
    ):
        vox = torch.floor(points / voxel).to(torch.int64).detach().to("cpu")
        keys = self._pack_voxels(vox)

        keys_unique, first_idx = self._unique_with_first_index_sorted(keys)
        if keys_unique.numel() == 0:
            return None, None, None, None

        if self.seen_keys.numel() == 0:
            is_new = torch.ones_like(keys_unique, dtype=torch.bool, device="cpu")
        else:
            is_new = ~torch.isin(keys_unique, self.seen_keys)

        if int(is_new.sum().item()) == 0:
            return None, None, None, None

        new_keys = keys_unique[is_new]
        new_first_idx = first_idx[is_new]

        rep_idx = new_first_idx.to(points.device)

        out_points = points[rep_idx]
        out_colors = colors[rep_idx]
        out_cov = cov_world[rep_idx]
        out_alpha = alpha[rep_idx]

        self.seen_keys = torch.cat([self.seen_keys, new_keys])

        self._frame_since_dedup += 1
        if self._frame_since_dedup >= self._dedup_every:
            self.seen_keys = torch.unique(self.seen_keys)
            self._frame_since_dedup = 0

        return out_points, out_colors, out_cov, out_alpha

    # -------- voxel averaging --------
    @torch.no_grad()
    def voxel_filter(self, points: torch.Tensor, colors: torch.Tensor, voxel: float):
        if points.numel() == 0:
            return points, colors

        vox = torch.floor(points / voxel).to(torch.int64)
        _, inverse = torch.unique(vox, dim=0, return_inverse=True)
        num = int(inverse.max().item()) + 1

        filtered_points = torch.zeros((num, 3), device=points.device, dtype=points.dtype)
        filtered_colors = torch.zeros((num, 3), device=colors.device, dtype=colors.dtype)
        counts = torch.zeros((num, 1), device=points.device, dtype=points.dtype)

        filtered_points.scatter_add_(0, inverse[:, None].expand(-1, 3), points)
        filtered_colors.scatter_add_(0, inverse[:, None].expand(-1, 3), colors)
        counts.scatter_add_(
            0,
            inverse[:, None],
            torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype),
        )

        filtered_points = filtered_points / counts
        filtered_colors = filtered_colors / counts
        return filtered_points, filtered_colors

    # -------- NEW: voxel filter that also merges cov+alpha --------
    @torch.no_grad()
    def voxel_filter_with_gaussians(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
        cov_world: torch.Tensor,
        alpha: torch.Tensor,
        voxel: float,
    ):
        if points.numel() == 0:
            return points, colors, cov_world, alpha

        vox = torch.floor(points / voxel).to(torch.int64)
        _, inverse = torch.unique(vox, dim=0, return_inverse=True)
        num = int(inverse.max().item()) + 1

        # points/colors mean
        filtered_points = torch.zeros((num, 3), device=points.device, dtype=points.dtype)
        filtered_colors = torch.zeros((num, 3), device=colors.device, dtype=colors.dtype)
        filtered_alpha = torch.zeros((num, 1), device=alpha.device, dtype=alpha.dtype)

        # covariance: simple average of covariance matrices in voxel
        filtered_cov = torch.zeros((num, 3, 3), device=cov_world.device, dtype=cov_world.dtype)

        counts = torch.zeros((num, 1), device=points.device, dtype=points.dtype)

        filtered_points.scatter_add_(0, inverse[:, None].expand(-1, 3), points)
        filtered_colors.scatter_add_(0, inverse[:, None].expand(-1, 3), colors)
        filtered_alpha.scatter_add_(0, inverse[:, None], alpha)

        # for [N,3,3], flatten to [N,9] for scatter_add, then reshape back
        cov_flat = cov_world.reshape(-1, 9)
        filtered_cov_flat = torch.zeros((num, 9), device=cov_world.device, dtype=cov_world.dtype)
        filtered_cov_flat.scatter_add_(0, inverse[:, None].expand(-1, 9), cov_flat)

        counts.scatter_add_(
            0,
            inverse[:, None],
            torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype),
        )

        filtered_points = filtered_points / counts
        filtered_colors = filtered_colors / counts
        filtered_alpha = filtered_alpha / counts

        filtered_cov = (filtered_cov_flat / counts).reshape(num, 3, 3)

        return filtered_points, filtered_colors, filtered_cov, filtered_alpha

    def update_full_pointcloud(self, rgb_keyframes, depth_keyframes, poses):
        if not depth_keyframes:
            return None, None, None, None

        self.add_keyframes_batch_gpu(
            rgb_keyframes,
            depth_keyframes,
            poses,
        )

        depth_keyframes.clear()

        if not self.points:
            return None, None, None, None

        self.all_points = torch.cat(self.points, dim=0)
        self.all_colors = torch.cat(self.colors, dim=0)
        self.all_covariances = torch.cat(self.gaussian_covariances, dim=0) if self.gaussian_covariances else None
        self.all_alpha = torch.cat(self.gaussian_alpha, dim=0) if self.gaussian_alpha else None

        return self.all_points, self.all_colors, self.all_covariances, self.all_alpha