import torch
import torch.nn.functional as F
from gsplat import rendering
import numpy as np
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
from RTSGS.DataLoader.DataLoader import DataLoader
from RTSGS.GaussianSplatting.PointCloud import PointCloud
from RTSGS.Tracker.Tracker import Tracker

class GaussianSplatting:
    def __init__(self, pcd: PointCloud, dataset: DataLoader, tracker: Tracker, learning_rate: float = 1e-3):
        self.pcd = pcd
        self.dataset = dataset
        self.device = pcd.device
        self.learning_rate = learning_rate
        self.tracker = tracker
        self.width, self.height = tracker.config.get('width'), tracker.config.get('height')
        
        self.num_points_optimized = 0
        self.optimizer = None
        self.debug_done = False

    def _setup_optimizer(self):
        """ 
        Note: Resetting the optimizer wipes momentum. 
        In SLAM, points are added constantly, so this 'wiping' is 
        what makes the foreground look frozen.
        """
        if self.pcd.all_points is None: return

        # Ensure parameters are leaf nodes
        for attr in ["all_points", "all_colors", "all_scales", "all_quaternions", "all_alpha"]:
            val = getattr(self.pcd, attr)
            if not isinstance(val, torch.nn.Parameter):
                setattr(self.pcd, attr, torch.nn.Parameter(val.detach().requires_grad_(True)))

        params = [
            {'params': [self.pcd.all_points], 'lr': self.learning_rate * 0.1, "name": "points"},
            {'params': [self.pcd.all_colors], 'lr': self.learning_rate, "name": "rgb"},
            # Increase this! Scales usually need 5x-10x the base LR
            {'params': [self.pcd.all_scales], 'lr': self.learning_rate * 5.0, "name": "scales"},
            {'params': [self.pcd.all_quaternions], 'lr': self.learning_rate * 0.5, "name": "quats"},
            {'params': [self.pcd.all_alpha], 'lr': self.learning_rate, "name": "alphas"},
        ]
        self.optimizer = torch.optim.Adam(params)
        self.num_points_optimized = self.pcd.all_points.shape[0]

    def training_step(self):
        if self.pcd.all_points is None or not self.tracker.keyframes_poses:
            return 0.0

        # Re-init optimizer only if point count changes significantly
        if self.optimizer is None or self.pcd.all_points.shape[0] != self.num_points_optimized:
            self._setup_optimizer()

        self.optimizer.zero_grad()

        # 1. Scale Guard (Keeps Gaussians sharp)

        # 2. Keyframe Subsampling
        all_idx = list(range(len(self.tracker.keyframes_poses)))
        sample_idx = np.random.choice(all_idx, min(2, len(all_idx)), replace=False)

        gt_rgbs = torch.stack([torch.from_numpy(self.dataset.rgb_keyframes[i]).to(self.device).float() / 255.0 for i in sample_idx])
        gt_depths = torch.stack([torch.from_numpy(self.dataset.depth_keyframes[i]).to(self.device).float() / self.pcd.depth_scale for i in sample_idx])
        
        # --- CAMERA POSE FIX ---
        # tracker.keyframes_poses are C2W (Camera-to-World).
        # gsplat.rasterization requires W2C (World-to-Camera).
        viewmats = []
        for i in sample_idx:
            c2w = torch.from_numpy(self.tracker.keyframes_poses[i]).to(self.device).float()
            # Simple inverse works because your points are likely already in OpenCV space.
            # Do NOT apply 'flip' here, or you will get a black screen again.
            w2c = torch.inverse(c2w)
            viewmats.append(w2c)
        viewmats = torch.stack(viewmats)

        # 3. Render
        K = torch.eye(3, device=self.device)
        K[0,0], K[1,1], K[0,2], K[1,2] = self.pcd.fx, self.pcd.fy, self.pcd.cx, self.pcd.cy
        Ks = K.unsqueeze(0).expand(len(sample_idx), -1, -1)

        rendered_rgb, rendered_depth, _ = rendering.rasterization(
            means=self.pcd.all_points,
            quats=F.normalize(self.pcd.all_quaternions, p=2, dim=-1),
            scales=torch.exp(self.pcd.all_scales), 
            opacities=torch.sigmoid(self.pcd.all_alpha).squeeze(-1),
            colors=self.pcd.all_colors,
            viewmats=viewmats,
            Ks=Ks,
            width=self.width,
            height=self.height,
        )

        # --- DEBUG WINDOW (Runs once after 30k points) ---
        if not self.debug_done and self.num_points_optimized > 30000:
            self.debug_done = True
            print(f"\n[DEBUG] Verifying Camera Pose for {self.num_points_optimized} points...")
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1); plt.title("Current Render (Model)")
            plt.imshow(torch.clamp(rendered_rgb[0], 0, 1).detach().cpu().numpy())
            plt.subplot(1, 2, 2); plt.title("Ground Truth (Target)")
            plt.imshow(gt_rgbs[0].cpu().numpy())
            plt.show()

        # 4. Losses
        l1_loss = F.l1_loss(rendered_rgb, gt_rgbs)
        ssim_val = ssim(rendered_rgb.permute(0, 3, 1, 2), gt_rgbs.permute(0, 3, 1, 2), data_range=1.0)
        rgb_total = 0.8 * l1_loss + 0.2 * (1.0 - ssim_val)

        # 5. Depth Loss (Prioritizing Foreground movement)
        rendered_depth = rendered_depth.squeeze(-1)
        mask = gt_depths > 0
        depth_error = torch.abs(rendered_depth[mask] - gt_depths[mask])
        
        # Weighted loss: Closer points (smaller depth) get higher gradients
        weights = 1.0 / (gt_depths[mask] + 0.1) 
        depth_loss = (depth_error * weights).mean()

        total_loss = rgb_total + 0.8 * depth_loss

        if total_loss > 0:
            total_loss.backward()
            self.optimizer.step()
                
        return total_loss.item()