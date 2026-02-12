import torch
import torch.nn.functional as F
from gsplat import rendering
import numpy as np
from pytorch_msssim import ssim
from RTSGS.DataLoader.DataLoader import DataLoader
from RTSGS.GaussianSplatting.PointCloud import PointCloud
from RTSGS.Tracker.Tracker import Tracker

class GaussianSplatting:
    def __init__(self, pcd: PointCloud, dataset: DataLoader,tracker:Tracker, learning_rate: float = 1e-3):
        self.pcd = pcd
        self.dataset = dataset
        self.device = pcd.device
        self.learning_rate = learning_rate
        self.tracker = tracker
        self.width, self.height = tracker.config.get('width'), tracker.config.get('height')
        # Keep track of how many points we are currently optimizing
        self.num_points_optimized = 0
        self.optimizer = None

    def _setup_optimizer(self):
            if self.pcd.all_points is None: return

            # Leaf parameter check
            for attr in ["all_points", "all_colors", "all_scales", "all_quaternions", "all_alpha"]:
                if not isinstance(getattr(self.pcd, attr), torch.nn.Parameter):
                    setattr(self.pcd, attr, torch.nn.Parameter(getattr(self.pcd, attr)))

            params = [
                {'params': [self.pcd.all_points], 'lr': self.learning_rate, "name": "points"},
                {'params': [self.pcd.all_colors], 'lr': self.learning_rate, "name": "rgb"},
                # KEEP THIS LOW: Prevents the orange blobs from growing too fast
                {'params': [self.pcd.all_scales], 'lr': self.learning_rate * 0.01, "name": "scales"},
                {'params': [self.pcd.all_quaternions], 'lr': self.learning_rate * 0.5, "name": "quats"},
                {'params': [self.pcd.all_alpha], 'lr': self.learning_rate, "name": "alphas"},
            ]
            self.optimizer = torch.optim.Adam(params)
            self.num_points_optimized = self.pcd.all_points.shape[0]
    def training_step(self):
            if self.pcd.all_points is None or not self.tracker.keyframes_poses:
                return 0.0

            if self.optimizer is None or self.pcd.all_points.shape[0] != self.num_points_optimized:
                self._setup_optimizer()

            self.optimizer.zero_grad()

            # 1. Scale Guard: Manually cap the physical size of points
            with torch.no_grad():
                self.pcd.all_scales.clamp_(max=-3.0)

            # 2. Keyframe Subsampling (Crucial for 6GB VRAM)
            all_idx = list(range(len(self.tracker.keyframes_poses)))
            sample_idx = np.random.choice(all_idx, min(2, len(all_idx)), replace=False)

            gt_rgbs = torch.stack([torch.from_numpy(self.dataset.rgb_keyframes[i]).to(self.device).float() / 255.0 for i in sample_idx])
            gt_depths = torch.stack([torch.from_numpy(self.dataset.depth_keyframes[i]).to(self.device).float() / self.pcd.depth_scale for i in sample_idx])
            viewmats = torch.stack([torch.from_numpy(self.tracker.keyframes_poses[i]).to(self.device).float() for i in sample_idx])

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

            # 4. SSIM & L1 Loss
            # pytorch-msssim expects [B, C, H, W]
            rend_img = rendered_rgb.permute(0, 3, 1, 2)
            gt_img = gt_rgbs.permute(0, 3, 1, 2)
            
            l1_loss = F.l1_loss(rendered_rgb, gt_rgbs)
            ssim_val = ssim(rend_img, gt_img, data_range=1.0, size_average=True)
            
            # Standard 3DGS balance: 0.8 L1 + 0.2 (1-SSIM)
            rgb_total = 0.8 * l1_loss + 0.2 * (1.0 - ssim_val)

            # 5. Depth Loss
            rendered_depth = rendered_depth.squeeze(-1)
            mask = gt_depths > 0
            depth_loss = F.l1_loss(rendered_depth[mask], gt_depths[mask])

            # Final Combined Loss
            total_loss = rgb_total + 0.6 * depth_loss

            if total_loss > 0:
                total_loss.backward()
                self.optimizer.step()
                
            return total_loss.item()