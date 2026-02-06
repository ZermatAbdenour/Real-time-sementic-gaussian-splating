import numpy as np
import torch


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

        # storage (CPU)
        self.points = []
        self.colors = []
        self.all_points = None
        self.all_colors = None

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

        z = depth_t
        mask = z > 0

        if self.pixel_subsample < 1.0:
            rnd = torch.rand_like(z, dtype=torch.float32)
            mask = mask & (rnd < self.pixel_subsample)

        if mask.sum() == 0:
            return

        # Get indices where mask is True
        indices = torch.where(mask)
        v_indices = indices[0]  # row indices (y)
        u_indices = indices[1]  # column indices (x)
        
        # Use these indices directly
        u = u_indices.float()
        v = v_indices.float()
        z = depth_t[indices]

        # back-project
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        points_cam = torch.stack((x, y, z), dim=1)

        # to world
        points_world = (R @ points_cam.T).T + t
        
        # Get colors using the same indices
        points_colors = img[v_indices, u_indices]  # This is the fix!

        # novelty gating
        points_world, points_colors = self.novelty_filter_fast(
            points_world, points_colors, voxel=self.novelty_voxel
        )
        if points_world is None:
            return

        # regularize density for the new points
        points_world, points_colors = self.voxel_filter(points_world, points_colors, voxel=self.voxel_size)
        # store CPU
        self.points.append(points_world.detach().cpu())
        self.colors.append(points_colors.detach().cpu())

        if torch.cuda.is_available() and (self.frame_count % 10 == 0):
            torch.cuda.empty_cache()

    # -------- packing --------
    def _pack_voxels(self, vox_xyz_cpu: torch.Tensor) -> torch.Tensor:
        off = self._pack_offset
        base = self._pack_base
        x = vox_xyz_cpu[:, 0] + off
        y = vox_xyz_cpu[:, 1] + off
        z = vox_xyz_cpu[:, 2] + off
        return x * (base * base) + y * base + z

    # -------- helpers: unique-with-first-index (no return_index needed) --------
    @staticmethod
    def _unique_with_first_index_sorted(keys_cpu: torch.Tensor):
        """
        keys_cpu: (N,) int64 on CPU
        Returns:
          unique_keys: (M,) int64 CPU
          first_idx: (M,) int64 CPU indices into original (unsorted) keys
        Method:
          sort keys, then take first of each run.
        """
        # stable sort not required; we just need any representative index per key
        sorted_keys, sorted_idx = torch.sort(keys_cpu)  # both CPU
        if sorted_keys.numel() == 0:
            return sorted_keys, sorted_idx

        # run boundaries
        keep = torch.ones(sorted_keys.shape[0], dtype=torch.bool, device="cpu")
        keep[1:] = sorted_keys[1:] != sorted_keys[:-1]

        unique_keys = sorted_keys[keep]
        first_idx = sorted_idx[keep]  # indices into original keys
        return unique_keys, first_idx

    # -------- novelty filter (fast, compatible) --------
    @torch.no_grad()
    def novelty_filter_fast(self, points: torch.Tensor, colors: torch.Tensor, voxel: float):
        # voxel coords on GPU -> CPU
        vox = torch.floor(points / voxel).to(torch.int64).detach().to("cpu")  # (N,3)
        keys = self._pack_voxels(vox)  # (N,) CPU int64

        # Reduce within-frame duplicates: 1 representative per voxel key
        keys_unique, first_idx = self._unique_with_first_index_sorted(keys)

        if keys_unique.numel() == 0:
            return None, None

        # Vectorized membership test
        if self.seen_keys.numel() == 0:
            is_new = torch.ones_like(keys_unique, dtype=torch.bool, device="cpu")
        else:
            is_new = ~torch.isin(keys_unique, self.seen_keys)

        if int(is_new.sum().item()) == 0:
            return None, None

        new_keys = keys_unique[is_new]
        new_first_idx = first_idx[is_new]  # CPU indices into original points/colors

        # gather on GPU
        rep_idx = new_first_idx.to(points.device)
        out_points = points[rep_idx]
        out_colors = colors[rep_idx]

        # update seen
        self.seen_keys = torch.cat([self.seen_keys, new_keys])

        # periodic dedup
        self._frame_since_dedup += 1
        if self._frame_since_dedup >= self._dedup_every:
            self.seen_keys = torch.unique(self.seen_keys)
            self._frame_since_dedup = 0

        return out_points, out_colors

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

    def update_full_pointcloud(self):
        if not self.points:
            return None, None
        self.all_points = torch.cat(self.points, dim=0)
        self.all_colors = torch.cat(self.colors, dim=0)
        return self.all_points, self.all_colors