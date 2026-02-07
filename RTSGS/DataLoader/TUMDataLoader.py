import os
import numpy as np
from RTSGS.DataLoader.DataLoader import DataLoader
import cv2


class TUMDataLoader(DataLoader):
    def __init__(self, rgb_path, depth_path=None, gt_path=None, stream=False):
        super().__init__(rgb_path, depth_path, stream)
        self._gt_path = gt_path

        # Ground truth (aligned to loaded frames)
        self.gt_timestamps = None                  # (N,)
        self.gt_poses = None                       # (N, 4, 4) or None if gt not available
        self.gt_vec = None                         # (N, 7) [tx,ty,tz,qx,qy,qz,qw]

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

        if len(ts_list) == 0:
            return None, None

        ts = np.asarray(ts_list, dtype=np.float64)
        vec = np.asarray(vec_list, dtype=np.float32)
        return ts, vec

    @staticmethod
    def _quat_xyzw_to_R(qx, qy, qz, qw):
        """
        Quaternion (x,y,z,w) -> 3x3 rotation matrix.
        """
        x, y, z, w = qx, qy, qz, qw
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        R = np.array([
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
            [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
        ], dtype=np.float32)
        return R

    @classmethod
    def _vec_to_T44(cls, tx, ty, tz, qx, qy, qz, qw):
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = cls._quat_xyzw_to_R(qx, qy, qz, qw)
        T[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
        return T

    def load_data(self, limit=-1):
        max_dt = 0.02

        # List and sort files
        rgb_files = sorted(os.listdir(self._rgb_path))
        depth_files = sorted(os.listdir(self._depth_path))

        rgb_ts = np.array([float(f[:-4]) for f in rgb_files], dtype=np.float64)
        depth_ts = np.array([float(f[:-4]) for f in depth_files], dtype=np.float64)

        # Load GT once
        gt_ts, gt_vec = self._load_tum_gt_file(self._gt_path)

        pairs = []                     # (rgb_path, depth_path)
        used_rgb_ts = []
        used_gt_ts = []
        used_gt_vec = []
        used_gt_T = []

        skipped_rgb_depth = 0
        skipped_gt = 0

        j_depth = 0
        j_gt = 0

        for i, t_rgb in enumerate(rgb_ts):
            if limit != -1 and len(pairs) >= limit:
                break

            print(
                f"Loading frame {i+1}/{min(limit if limit != -1 else len(rgb_files), len(rgb_files))}",
                end="\r"
            )

            # --- match depth to rgb (nearest) ---
            while (
                j_depth + 1 < len(depth_ts)
                and abs(depth_ts[j_depth + 1] - t_rgb) < abs(depth_ts[j_depth] - t_rgb)
            ):
                j_depth += 1

            if abs(depth_ts[j_depth] - t_rgb) >= max_dt:
                skipped_rgb_depth += 1
                continue

            rgb_file_path = os.path.join(self._rgb_path, rgb_files[i])
            depth_file_path = os.path.join(self._depth_path, depth_files[j_depth])

            # Store file paths only, not images
            pairs.append((rgb_file_path, depth_file_path))
            used_rgb_ts.append(t_rgb)

            # --- match GT to rgb (nearest) ---
            if gt_ts is not None:
                while (
                    j_gt + 1 < len(gt_ts)
                    and abs(gt_ts[j_gt + 1] - t_rgb) < abs(gt_ts[j_gt] - t_rgb)
                ):
                    j_gt += 1

                if abs(gt_ts[j_gt] - t_rgb) < max_dt:
                    used_gt_ts.append(gt_ts[j_gt])
                    used_gt_vec.append(gt_vec[j_gt])

                    tx, ty, tz, qx, qy, qz, qw = gt_vec[j_gt]
                    used_gt_T.append(self._vec_to_T44(tx, ty, tz, qx, qy, qz, qw))
                else:
                    skipped_gt += 1
                    used_gt_ts.append(np.nan)
                    used_gt_vec.append([np.nan] * 7)
                    used_gt_T.append(np.full((4, 4), np.nan, dtype=np.float32))

        # Save loaded pairs
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

        print()
        print(f"Skipped {skipped_rgb_depth} frames due to RGB/depth timestamp mismatch.")
        if gt_ts is not None:
            print(f"GT loaded from: {self._gt_path}")
            print(f"GT unmatched for {skipped_gt} frames (stored NaNs).")
