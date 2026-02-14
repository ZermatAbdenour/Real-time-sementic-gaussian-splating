import torch
import torch.nn.functional as F
from gsplat import rendering
import numpy as np
from pytorch_msssim import ssim
from RTSGS.DataLoader.DataLoader import DataLoader
from RTSGS.GaussianSplatting.PointCloud import PointCloud
from RTSGS.Tracker.Tracker import Tracker

class GaussianSplatting:
    def __init__(self, pcd: PointCloud, dataset: DataLoader, tracker: Tracker, learning_rate: float = 1e-3):
        self.pcd = pcd
        self.dataset = dataset
        self.device = pcd.device
        self.base_lr = learning_rate
        self.tracker = tracker
        self.width, self.height = tracker.config.get('width'), tracker.config.get('height')
        
        self.num_points_optimized = 0
        self.optimizer = None
        self.iteration_count = 0
        
        # Hyperparameters
        self.densify_scale_threshold = 0.01

    def _setup_optimizer(self):
        if self.pcd.all_points is None: return

        # Ensure parameters are leaf nodes
        for attr in ["all_points", "all_colors", "all_scales", "all_quaternions", "all_alpha"]:
            val = getattr(self.pcd, attr)
            if not isinstance(val, torch.nn.Parameter):
                setattr(self.pcd, attr, torch.nn.Parameter(val.detach().requires_grad_(True)))

        # Stabilized LR: Points (positions) move slower to prevent jitter
        params = [
            {'params': [self.pcd.all_points], 'lr': self.base_lr * 0.2, "name": "points"},
            {'params': [self.pcd.all_colors], 'lr': self.base_lr * 2.0, "name": "rgb"},
            {'params': [self.pcd.all_scales], 'lr': self.base_lr * 5.0, "name": "scales"},
            {'params': [self.pcd.all_quaternions], 'lr': self.base_lr * 0.5, "name": "quats"},
            {'params': [self.pcd.all_alpha], 'lr': self.base_lr, "name": "alphas"},
        ]
        self.optimizer = torch.optim.Adam(params)
        self.num_points_optimized = self.pcd.all_points.shape[0]

    def training_step(self):
        if self.pcd.all_points is None or not self.tracker.keyframes_poses:
            return 0.0

        self.iteration_count += 1
        
        # Re-init optimizer only if point count changed
        if self.optimizer is None or self.pcd.all_points.shape[0] != self.num_points_optimized:
            self._setup_optimizer()

        # Exponential LR Decay for point positions
        if self.iteration_count % 1000 == 0:
            for g in self.optimizer.param_groups:
                if g['name'] == 'points':
                    g['lr'] *= 0.8 # Gradually slow down points to freeze them

        self.optimizer.zero_grad()

        # 1. Camera setup
        sample_idx = np.random.choice(len(self.tracker.keyframes_poses), min(2, len(self.tracker.keyframes_poses)), replace=False)
        gt_rgbs = torch.stack([torch.from_numpy(self.dataset.rgb_keyframes[i]).to(self.device).float() / 255.0 for i in sample_idx])
        viewmats = torch.stack([torch.inverse(torch.from_numpy(self.tracker.keyframes_poses[i]).to(self.device).float()) for i in sample_idx])
        
        K = torch.eye(3, device=self.device)
        K[0,0], K[1,1], K[0,2], K[1,2] = self.pcd.fx, self.pcd.fy, self.pcd.cx, self.pcd.cy
        Ks = K.unsqueeze(0).expand(len(sample_idx), -1, -1)

        # 2. Render
        rendered_rgb, _, info = rendering.rasterization(
            means=self.pcd.all_points,
            quats=F.normalize(self.pcd.all_quaternions, p=2, dim=-1),
            scales=torch.exp(self.pcd.all_scales), 
            opacities=torch.sigmoid(self.pcd.all_alpha).squeeze(-1),
            colors=torch.sigmoid(self.pcd.all_colors),
            viewmats=viewmats,
            Ks=Ks,
            width=self.width,
            height=self.height,
        )

        if "means2d" in info:
            info["means2d"].retain_grad()

        l1_loss = F.l1_loss(rendered_rgb, gt_rgbs)
        ssim_val = ssim(rendered_rgb, gt_rgbs, data_range=1.0)
        rgb_loss = 0.8 * l1_loss + 0.2 * (1.0 - ssim_val)

        # B. Isotropy Loss (Anti-glitch/Anti-needle)
        # Forces the 3 scale dimensions to be similar
        scales = self.pcd.all_scales
        mean_scale = scales.mean(dim=-1, keepdim=True)
        isotropy_loss = torch.mean((scales - mean_scale)**2)

        # Total Loss
        total_loss = rgb_loss + 0.1 * isotropy_loss

        if total_loss > 0:
            total_loss.backward()
            self.optimizer.step()
                
        return total_loss.item()
                
        return total_loss.item()