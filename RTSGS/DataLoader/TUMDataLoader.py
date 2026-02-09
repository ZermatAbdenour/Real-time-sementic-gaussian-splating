import os
import numpy as np
from RTSGS.DataLoader.DataLoader import DataLoader
import cv2

class TUMDataLoader(DataLoader):
    def __init__(self, rgb_path, depth_path=None, gt_path=None, stream=False):
        super().__init__(rgb_path, depth_path, stream)
        self._gt_path = gt_path

        self.gt_timestamps = None
        self.gt_poses = None
        self.gt_vec = None

    @staticmethod
    def _load_tum_gt_file(gt_path: str):
        if gt_path is None:
            return None, None

        ts_list = []
        vec_list = []
        with open(gt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 8:
                    continue
                t = float(parts[0])
                tx, ty, tz = map(float, parts[1:4])
                qx, qy, qz, qw = map(float, parts[4:8])
                ts_list.append(t)
                vec_list.append([tx, ty, tz, qx, qy, qz, qw])

        if not ts_list:
            return None, None
        ts = np.asarray(ts_list, dtype=np.float64)
        vec = np.asarray(vec_list, dtype=np.float32)
        return ts, vec

    @staticmethod
    def _quat_xyzw_to_R(qx, qy, qz, qw):
        x, y, z, w = qx, qy, qz, qw
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return np.array([
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
            [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
        ], dtype=np.float32)

    @classmethod
    def _vec_to_T44(cls, tx, ty, tz, qx, qy, qz, qw):
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = cls._quat_xyzw_to_R(qx, qy, qz, qw)
        T[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
        return T

    def load_data(self, limit=-1, max_dt=0.02):
        # Sort files numerically by timestamp, not lexicographically!
        rgb_files = sorted(os.listdir(self._rgb_path), key=lambda f: float(f[:-4]))
        depth_files = sorted(os.listdir(self._depth_path), key=lambda f: float(f[:-4]))
        rgb_ts = np.array([float(f[:-4]) for f in rgb_files], dtype=np.float64)
        depth_ts = np.array([float(f[:-4]) for f in depth_files], dtype=np.float64)
        gt_ts, gt_vec = self._load_tum_gt_file(self._gt_path)

        pairs = []
        used_rgb_ts, used_gt_ts, used_gt_vec, used_gt_T = [], [], [], []
        skipped_rgb_depth, skipped_gt = 0, 0
        j_depth = 0

        for i, t_rgb in enumerate(rgb_ts):
            if limit != -1 and len(pairs) >= limit:
                break
            # Find nearest depth frame
            while (
                j_depth + 1 < len(depth_ts)
                and abs(depth_ts[j_depth + 1] - t_rgb) < abs(depth_ts[j_depth] - t_rgb)
            ):
                j_depth += 1
            # Require depth within max_dt
            if abs(depth_ts[j_depth] - t_rgb) >= max_dt:
                skipped_rgb_depth += 1
                continue

            rgb_file_path = os.path.join(self._rgb_path, rgb_files[i])
            depth_file_path = os.path.join(self._depth_path, depth_files[j_depth])

            pairs.append((rgb_file_path, depth_file_path))
            used_rgb_ts.append(t_rgb)

            # Find GT nearest (may be unmatched)
            if gt_ts is not None:
                idx_gt = np.argmin(np.abs(gt_ts - t_rgb))
                err = abs(gt_ts[idx_gt] - t_rgb)
                if err < max_dt:
                    used_gt_ts.append(gt_ts[idx_gt])
                    used_gt_vec.append(gt_vec[idx_gt])
                    tx, ty, tz, qx, qy, qz, qw = gt_vec[idx_gt]
                    used_gt_T.append(self._vec_to_T44(tx, ty, tz, qx, qy, qz, qw))
                else:
                    skipped_gt += 1
                    used_gt_ts.append(np.nan)
                    used_gt_vec.append([np.nan]*7)
                    used_gt_T.append(np.full((4,4), np.nan, dtype=np.float32))
            else:
                used_gt_ts.append(np.nan)
                used_gt_vec.append([np.nan]*7)
                used_gt_T.append(np.full((4,4), np.nan, dtype=np.float32))

        # Store results
        self.RGBD_pairs = pairs
        self.time_stamps = np.asarray(used_rgb_ts, dtype=np.float64)
        if gt_ts is not None:
            self.gt_timestamps = np.asarray(used_gt_ts, dtype=np.float64)
            self.gt_vec = np.asarray(used_gt_vec, dtype=np.float32)
            self.gt_poses = np.asarray(used_gt_T, dtype=np.float32)
        else:
            self.gt_timestamps = None
            self.gt_vec = None
            self.gt_poses = None

        print(f"\nSkipped {skipped_rgb_depth} frames due to RGB/depth timestamp mismatch.")
        if gt_ts is not None:
            print(f"GT loaded from: {self._gt_path}")
            print(f"GT unmatched for {skipped_gt} frames (stored NaNs).")
        print("RGB timestamps:", rgb_ts[:10])
        print("GT timestamps:", gt_ts[:10] if gt_ts is not None else None)
        print("Differences:", rgb_ts[:10] - gt_ts[:10] if gt_ts is not None else None)

        print("\n================= Loaded Data Summary =================")
        for idx, ((rgb_file, depth_file), t_rgb, t_gt, gt_pose) in enumerate(zip(
            self.RGBD_pairs, self.time_stamps, 
            self.gt_timestamps if self.gt_timestamps is not None else [None]*len(self.RGBD_pairs),
            self.gt_poses if self.gt_poses is not None else [None]*len(self.RGBD_pairs),
        )):
            gt_str = f"GT time: {t_gt:.6f}" if isinstance(t_gt, float) and not np.isnan(t_gt) else "GT: None"
            print(f"[{idx:4d}] RGB: {os.path.basename(rgb_file)}, "
                  f"Depth: {os.path.basename(depth_file)}, "
                  f"RGB TS: {t_rgb:.6f} {gt_str}")

        print(f"\nTotal loaded pairs: {len(self.RGBD_pairs)}")