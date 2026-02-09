import numpy as np
import open3d as o3d
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import os
import matplotlib.pyplot as plt

# =============================================================================
# CONVENTION FIX (ONLY ADDITION):
# Export poses as T_wc in OpenCV camera convention (x right, y down, z forward)
# while keeping everything else in Habitat world coords (Y-up) exactly the same.
# =============================================================================

def habitat_wc_quat_to_opencv_wc_quat(q_wc_hab_xyzw: np.ndarray) -> np.ndarray:
    """
    Convert camera-to-world quaternion from Habitat/OpenGL-like camera convention
    (camera forward = -Z) to OpenCV camera convention (camera forward = +Z,
    y down).
    Both are camera-frame conventions; world stays the same.

    R_wc_ocv = R_wc_hab @ diag(1, -1, -1)
    """
    R_wc_hab = R.from_quat(q_wc_hab_xyzw).as_matrix().astype(np.float32)
    R_chab__cocv = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    R_wc_ocv = R_wc_hab @ R_chab__cocv
    return R.from_matrix(R_wc_ocv).as_quat().astype(np.float32)


class VelocityControlledTrajectoryGenerator:
    """
    Generates a smooth indoor camera trajectory with controlled velocity.

    World convention (kept as in your code):
      - Habitat/Replica world is Y-up
      - Position is [x, y, z] in meters
      - y is height

    Export convention (changed to match your reprojection):
      - Pose is T_wc (camera-to-world)
      - Camera frame is OpenCV: x right, y down, z forward (+Z)
      - Quaternion written as (qx qy qz qw) == (x,y,z,w)
    """

    def __init__(self, scene_path=None, margin=0.3):
        self.scene = None
        self.scene_bounds = None
        self.floor_height = None
        self.ceiling_height = None
        self.margin = margin

        if scene_path and scene_path != "":
            self.load_scene(scene_path)
        else:
            self.set_default_bounds()

    def set_default_bounds(self):
        print("Using default Replica room_0 bounds...")
        self.scene_bounds = {
            "min": np.array([-0.88, 0.0, -1.19], dtype=np.float32),
            "max": np.array([6.89, 2.5, 3.51], dtype=np.float32),
            "center": np.array([3.12, 1.25, 1.12], dtype=np.float32),
        }
        self.floor_height = 0.0
        self.ceiling_height = 2.5
        self.eye_height_min = 1.4
        self.eye_height_max = 1.8

    def load_scene(self, scene_path):
        if not os.path.exists(scene_path):
            print("Scene file not found, using default bounds")
            self.set_default_bounds()
            return

        try:
            mesh = o3d.io.read_triangle_mesh(scene_path)
            if len(mesh.vertices) == 0:
                pcd = o3d.io.read_point_cloud(scene_path)
                points = np.asarray(pcd.points)
            else:
                points = np.asarray(mesh.vertices)
        except Exception:
            self.set_default_bounds()
            return

        if points.size == 0:
            self.set_default_bounds()
            return

        mins = np.min(points, axis=0).astype(np.float32)
        maxs = np.max(points, axis=0).astype(np.float32)

        mins_m = mins + self.margin
        maxs_m = maxs - self.margin
        if np.any(mins_m >= maxs_m):
            mins_m, maxs_m = mins, maxs

        self.scene_bounds = {
            "min": mins_m,
            "max": maxs_m,
            "center": ((mins_m + maxs_m) / 2.0).astype(np.float32),
        }

        y_values = points[:, 1]
        self.floor_height = float(np.percentile(y_values, 5))
        self.ceiling_height = float(np.percentile(y_values, 95))

        self.eye_height_min = max(self.floor_height + 1.4, self.floor_height)
        self.eye_height_max = min(self.floor_height + 1.8, self.ceiling_height - 0.1)

        if self.eye_height_min >= self.eye_height_max:
            self.eye_height_min = self.floor_height + 1.6
            self.eye_height_max = self.floor_height + 1.7

    def generate_smooth_velocity_profile(self, num_frames, max_speed=0.5, acceleration_time=0.2):
        accel_frames = int(num_frames * acceleration_time)
        cruise_frames = num_frames - 2 * accel_frames
        if cruise_frames < 0:
            accel_frames = num_frames // 3
            cruise_frames = num_frames - 2 * accel_frames

        t_accel = np.linspace(0, 1, accel_frames)
        accel_profile = 0.5 - 0.5 * np.cos(t_accel * np.pi)

        cruise_profile = np.ones(cruise_frames, dtype=np.float32)

        t_decel = np.linspace(0, 1, accel_frames)
        decel_profile = 0.5 + 0.5 * np.cos(t_decel * np.pi)

        velocity_profile = np.concatenate([accel_profile, cruise_profile, decel_profile]).astype(np.float32)
        velocity_profile *= float(max_speed)

        if len(velocity_profile) > num_frames:
            velocity_profile = velocity_profile[:num_frames]
        elif len(velocity_profile) < num_frames:
            pad_length = num_frames - len(velocity_profile)
            velocity_profile = np.pad(velocity_profile, (0, pad_length), mode="edge")

        return velocity_profile

    def generate_path_with_controlled_velocity(self, num_frames=300, camera_height=1.6,
                                             max_speed=0.4, path_type="room_center"):
        camera_height = float(np.clip(camera_height, self.eye_height_min, self.eye_height_max))

        bounds = self.scene_bounds
        room_center = (bounds["min"] + bounds["max"]) / 2.0
        room_size = bounds["max"] - bounds["min"]

        velocity_profile = self.generate_smooth_velocity_profile(
            num_frames, max_speed=max_speed, acceleration_time=0.2
        )

        if path_type == "room_center":
            keypoints = self._generate_elliptical_keypoints(room_center, room_size, camera_height)
        elif path_type == "wall_following":
            keypoints = self._generate_wall_keypoints(bounds, room_size, camera_height)
        elif path_type == "figure8":
            keypoints = self._generate_figure8_keypoints(room_center, room_size, camera_height)
        else:
            keypoints = self._generate_elliptical_keypoints(room_center, room_size, camera_height)

        positions = self._create_bspline_path(keypoints, num_frames)
        positions = self._adjust_path_for_velocity(positions, velocity_profile)
        positions = self._clamp_to_bounds(positions)
        positions = self._apply_gentle_smoothing(positions)
        return positions, velocity_profile

    def _generate_elliptical_keypoints(self, room_center, room_size, camera_height):
        max_radius_x = min(float(room_size[0] * 0.3), 1.0)
        max_radius_z = min(float(room_size[2] * 0.3), 1.0)

        t = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        x = max_radius_x * np.cos(t)
        z = max_radius_z * np.sin(t)
        y = camera_height * np.ones_like(t) + np.sin(t * 2) * 0.05

        keypoints = np.column_stack([x, y, z]).astype(np.float32) + room_center.astype(np.float32)
        keypoints = np.vstack([keypoints, keypoints[0]])
        return keypoints

    def _generate_wall_keypoints(self, bounds, room_size, camera_height):
        wall_distance = min(float(room_size[0]), float(room_size[2])) * 0.4

        corners = [
            [bounds["min"][0] + wall_distance, camera_height, bounds["min"][2] + wall_distance],
            [bounds["max"][0] - wall_distance, camera_height, bounds["min"][2] + wall_distance],
            [bounds["max"][0] - wall_distance, camera_height, bounds["max"][2] - wall_distance],
            [bounds["min"][0] + wall_distance, camera_height, bounds["max"][2] - wall_distance],
        ]

        keypoints = []
        for i in range(4):
            current = corners[i]
            nxt = corners[(i + 1) % 4]
            keypoints.append(current)

            midpoint = [
                (current[0] + nxt[0]) / 2.0,
                camera_height + 0.03,
                (current[2] + nxt[2]) / 2.0,
            ]
            keypoints.append(midpoint)

        keypoints.append(keypoints[0])
        return np.array(keypoints, dtype=np.float32)

    def _generate_figure8_keypoints(self, room_center, room_size, camera_height):
        max_radius_x = min(float(room_size[0] * 0.25), 0.8)
        max_radius_z = min(float(room_size[2] * 0.25), 0.8)

        t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        x = max_radius_x * np.sin(t)
        z = max_radius_z * np.sin(2 * t) * 0.6
        y = camera_height * np.ones_like(t) + np.sin(t) * 0.03

        keypoints = np.column_stack([x, y, z]).astype(np.float32) + room_center.astype(np.float32)
        keypoints = np.vstack([keypoints, keypoints[0]])
        return keypoints

    def _create_bspline_path(self, keypoints, num_frames):
        distances = np.cumsum(np.sqrt(np.sum(np.diff(keypoints, axis=0) ** 2, axis=1)))
        distances = np.insert(distances, 0, 0)
        t_norm = distances / distances[-1]
        t_new = np.linspace(0, 1, num_frames)

        positions = np.zeros((num_frames, 3), dtype=np.float32)
        for i in range(3):
            spline = CubicSpline(t_norm, keypoints[:, i], bc_type="periodic")
            positions[:, i] = spline(t_new).astype(np.float32)

        return positions

    def _adjust_path_for_velocity(self, positions, velocity_profile):
        num_frames = len(positions)

        diffs = np.diff(positions, axis=0)
        seg_lens = np.sqrt(np.sum(diffs**2, axis=1))
        cumdist = np.cumsum(seg_lens)
        cumdist = np.insert(cumdist, 0, 0)
        total_len = float(cumdist[-1])

        dt = 1.0 / (num_frames - 1)
        desired_cum = np.zeros(num_frames, dtype=np.float32)
        for i in range(1, num_frames):
            avg_v = (velocity_profile[i - 1] + velocity_profile[i]) / 2.0
            desired_cum[i] = desired_cum[i - 1] + avg_v * dt

        if desired_cum[-1] > 0:
            desired_cum *= (total_len / float(desired_cum[-1]))

        new_pos = np.zeros_like(positions)
        new_pos[0] = positions[0]

        for i in range(1, num_frames):
            target = float(desired_cum[i])
            idx = int(np.searchsorted(cumdist, target) - 1)
            idx = min(idx, len(seg_lens) - 1)

            if idx < 0:
                new_pos[i] = positions[0]
                continue

            dist_in_seg = target - float(cumdist[idx])
            if seg_lens[idx] > 0:
                alpha = np.clip(dist_in_seg / float(seg_lens[idx]), 0.0, 1.0)
                new_pos[i] = positions[idx] + alpha * (positions[idx + 1] - positions[idx])
            else:
                new_pos[i] = positions[idx]

        return new_pos

    def _apply_gentle_smoothing(self, positions):
        sm = positions.copy()
        for i in range(3):
            sm[:, i] = gaussian_filter1d(sm[:, i], sigma=1.5)
            window = min(15, len(sm) // 10)
            if window >= 5 and window % 2 == 1:
                sm[:, i] = savgol_filter(sm[:, i], window_length=window, polyorder=3)
        return sm.astype(np.float32)

    def _clamp_to_bounds(self, positions):
        cl = positions.copy()
        cl[:, 0] = np.clip(cl[:, 0], self.scene_bounds["min"][0], self.scene_bounds["max"][0])  # x
        cl[:, 2] = np.clip(cl[:, 2], self.scene_bounds["min"][2], self.scene_bounds["max"][2])  # z
        cl[:, 1] = np.clip(
            cl[:, 1],
            self.eye_height_min,
            min(self.eye_height_max, self.ceiling_height - 0.2),
        )
        return cl.astype(np.float32)

    def compute_smooth_orientations(self, positions, look_ahead_frames=10):
        """
        KEEP AS-IS:
          R_w_c = [right, up, -forward]
        This is Habitat/OpenGL-like camera convention.
        We'll convert ONLY at export to OpenCV.
        """
        num_frames = len(positions)
        look_at_points = np.zeros_like(positions)

        for i in range(num_frames):
            look_ahead_idx = min(i + look_ahead_frames, num_frames - 1)

            room_center = self.scene_bounds["center"]
            blend = 0.8
            look_at = blend * positions[look_ahead_idx] + (1 - blend) * room_center

            t = i / num_frames * 2 * np.pi
            variation = np.array([
                np.sin(t * 0.2) * 0.02,
                np.cos(t * 0.15) * 0.02,
                np.sin(t * 0.1) * 0.01
            ], dtype=np.float32)

            look_at_points[i] = look_at + variation

        for k in range(3):
            look_at_points[:, k] = gaussian_filter1d(look_at_points[:, k], sigma=3)

        quats = np.zeros((num_frames, 4), dtype=np.float32)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Y-up

        for i in range(num_frames):
            forward = (look_at_points[i] - positions[i]).astype(np.float32)
            n = float(np.linalg.norm(forward))
            if n < 1e-6:
                forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                forward /= n

            up = world_up - np.dot(world_up, forward) * forward
            up_n = float(np.linalg.norm(up))
            if up_n < 1e-6:
                up = world_up.copy()
            else:
                up /= up_n

            right = np.cross(forward, up)
            r_n = float(np.linalg.norm(right))
            if r_n < 1e-6:
                right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                right /= r_n

            up = np.cross(right, forward)

            R_w_c = np.column_stack([right, up, -forward]).astype(np.float32)
            quats[i] = R.from_matrix(R_w_c).as_quat().astype(np.float32)

        quats = self._smooth_quaternions(quats)
        return quats, look_at_points

    def _smooth_quaternions(self, quaternions):
        num_frames = len(quaternions)
        if num_frames < 10:
            return quaternions

        rotations = R.from_quat(quaternions)
        rotvecs = rotations.as_rotvec()

        sm = np.zeros_like(rotvecs)
        for k in range(3):
            sm[:, k] = gaussian_filter1d(rotvecs[:, k], sigma=2)

        sm_rots = R.from_rotvec(sm)

        key_int = max(1, num_frames // 20)
        key_idx = list(range(0, num_frames, key_int))
        if key_idx[-1] != num_frames - 1:
            key_idx.append(num_frames - 1)

        key_rots = sm_rots[key_idx]
        slerp = Slerp(key_idx, key_rots)

        times = np.arange(num_frames)
        final_rots = slerp(times)
        return final_rots.as_quat().astype(np.float32)

    def generate_trajectory(self, num_frames=300, fps=30, camera_height=1.6,
                           max_speed=0.4, path_type="room_center"):
        print(f"Generating {path_type} trajectory with max speed {max_speed} m/s (Habitat Y-up)...")

        positions, velocity_profile = self.generate_path_with_controlled_velocity(
            num_frames=num_frames,
            camera_height=camera_height,
            max_speed=max_speed,
            path_type=path_type,
        )

        print("Computing smooth orientations...")
        orientations, look_at_points = self.compute_smooth_orientations(
            positions, look_ahead_frames=int(num_frames * 0.05)
        )

        timestamps = np.arange(num_frames, dtype=np.float32) / float(fps)

        return {
            "positions": positions,
            "orientations": orientations,  # Habitat/OpenGL camera convention
            "look_at_points": look_at_points,
            "velocity_profile": velocity_profile,
            "timestamps": timestamps,
            "fps": fps,
            "num_frames": num_frames,
            "camera_height": camera_height,
            "path_type": path_type,
            "max_speed": max_speed
        }

    def analyze_velocity(self, trajectory):
        positions = trajectory["positions"]
        fps = float(trajectory["fps"])

        dt = 1.0 / fps
        v = np.diff(positions, axis=0) / dt
        speeds = np.linalg.norm(v, axis=1)

        print("\n" + "=" * 60)
        print("VELOCITY ANALYSIS")
        print("=" * 60)
        print(f"Target max speed: {trajectory['max_speed']:.3f} m/s")
        print(f"Actual max speed: {np.max(speeds):.4f} m/s")
        print(f"Mean speed: {np.mean(speeds):.4f} m/s")

        start_speed = float(speeds[0]) if len(speeds) else 0.0
        end_speed = float(speeds[-1]) if len(speeds) else 0.0
        print(f"Start speed: {start_speed:.4f} m/s")
        print(f"End speed: {end_speed:.4f} m/s")

        return {
            "max_speed": float(np.max(speeds)),
            "start_speed": start_speed,
            "end_speed": end_speed,
            "actual_speeds": speeds,
        }

    def export_tum_format(self, trajectory, output_path):
        """
        ONLY CHANGE: export quaternions converted to OpenCV camera convention
        so that your PointCloud backprojection uses the correct pose.
        """
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(output_path, "w") as f:
            f.write("# TUM trajectory (T_wc) exported for RTSGS projection convention\n")
            f.write("# World: Habitat Y-up. Camera: OpenCV (x right, y down, z forward)\n")
            f.write(f"# Frames: {trajectory['num_frames']}, FPS: {trajectory['fps']}\n")
            f.write("# timestamp tx ty tz qx qy qz qw\n")

            for i in range(trajectory["num_frames"]):
                pos = trajectory["positions"][i]
                q_hab = trajectory["orientations"][i]  # (x,y,z,w) Habitat/OpenGL-like camera
                ts = float(trajectory["timestamps"][i])

                q_ocv = habitat_wc_quat_to_opencv_wc_quat(q_hab)

                f.write(
                    f"{ts:.6f} "
                    f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                    f"{q_ocv[0]:.6f} {q_ocv[1]:.6f} {q_ocv[2]:.6f} {q_ocv[3]:.6f}\n"
                )

        print(f"Exported {trajectory['num_frames']} frames to {output_path}")

# ============================================================================
# CONFIG
# ============================================================================
SCENE_PATH = "./ThirdParty/Replica-Dataset/data/data/room_0/mesh.ply"
MARGIN = 0.3

NUM_FRAMES = 100
FPS = 30
CAMERA_HEIGHT = 1.6
MAX_SPEED = 0.3
PATH_TYPE = "figure8"

OUTPUT_TUM = "trajectory.txt"
PLOT_VELOCITY = True

# ============================================================================
# EXECUTION
# ============================================================================
print("=" * 70)
print("VELOCITY CONTROLLED TRAJECTORY GENERATOR (Habitat world, OpenCV camera export)")
print("=" * 70)

generator = VelocityControlledTrajectoryGenerator(SCENE_PATH, margin=MARGIN)

trajectory = generator.generate_trajectory(
    num_frames=NUM_FRAMES,
    fps=FPS,
    camera_height=CAMERA_HEIGHT,
    max_speed=MAX_SPEED,
    path_type=PATH_TYPE
)

velocity_analysis = generator.analyze_velocity(trajectory)
generator.export_tum_format(trajectory, OUTPUT_TUM)

if PLOT_VELOCITY:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    positions = trajectory["positions"]
    velocity_profile = trajectory["velocity_profile"]
    actual_speeds = velocity_analysis["actual_speeds"]
    timestamps = trajectory["timestamps"]

    axes[0, 0].plot(timestamps[:-1], actual_speeds, "b-", label="Actual speed", linewidth=2)
    axes[0, 0].plot(timestamps, velocity_profile, "r--", label="Target speed", alpha=0.7)
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Speed (m/s)")
    axes[0, 0].set_title("Velocity Profile")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(timestamps, positions[:, 0], "r-", label="X", alpha=0.8)
    axes[0, 1].plot(timestamps, positions[:, 1], "g-", label="Y", alpha=0.8)
    axes[0, 1].plot(timestamps, positions[:, 2], "b-", label="Z", alpha=0.8)
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Position (m)")
    axes[0, 1].set_title("Position Over Time")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(positions[:, 0], positions[:, 2], "b-", linewidth=1.5)
    axes[1, 0].scatter(positions[0, 0], positions[0, 2], c="green", s=100, marker="o", label="Start", zorder=5)
    axes[1, 0].scatter(positions[-1, 0], positions[-1, 2], c="red", s=100, marker="s", label="End", zorder=5)
    axes[1, 0].set_xlabel("X (m)")
    axes[1, 0].set_ylabel("Z (m)")
    axes[1, 0].set_title("Top View (X-Z)")
    axes[1, 0].legend()
    axes[1, 0].axis("equal")
    axes[1, 0].grid(True, alpha=0.3)

    if len(actual_speeds) > 1:
        dt = 1.0 / FPS
        accelerations = np.diff(actual_speeds) / dt

        axes[1, 1].plot(timestamps[1:-1], accelerations, "purple", linewidth=1.5)
        axes[1, 1].axhline(y=0, color="k", linestyle="-", alpha=0.3)
        axes[1, 1].axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="Comfort limit (1 m/s²)")
        axes[1, 1].axhline(y=-1.0, color="r", linestyle="--", alpha=0.5)
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("Acceleration (m/s²)")
        axes[1, 1].set_title("Acceleration Over Time")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(
        f"Velocity Controlled Trajectory\n"
        f"Max Speed: {MAX_SPEED:.2f} m/s, Start/End Speed: {velocity_analysis['start_speed']:.3f}/{velocity_analysis['end_speed']:.3f} m/s",
        fontsize=14,
    )

    plt.tight_layout()
    plt.show()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Output file: {OUTPUT_TUM}")
print(f"Frames: {NUM_FRAMES}")
print(f"Duration: {NUM_FRAMES / FPS:.1f} seconds")
print(f"Max target speed: {MAX_SPEED:.3f} m/s")
print(f"Actual max speed: {velocity_analysis['max_speed']:.3f} m/s")
print(f"Start speed: {velocity_analysis['start_speed']:.3f} m/s")
print(f"End speed: {velocity_analysis['end_speed']:.3f} m/s")
print("=" * 70)