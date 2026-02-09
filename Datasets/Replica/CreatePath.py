import numpy as np
import open3d as o3d
from scipy.interpolate import CubicSpline, BSpline
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class VelocityControlledTrajectoryGenerator:
    def __init__(self, scene_path=None, margin=0.3):
        """
        Generate smooth indoor camera trajectory with controlled velocity
        
        Args:
            scene_path: Path to scene file
            margin: Safety margin from walls (meters)
        """
        self.scene = None
        self.scene_center = None
        self.scene_bounds = None
        self.floor_height = None
        self.ceiling_height = None
        self.margin = margin
        
        if scene_path and scene_path != "":
            self.load_scene(scene_path)
        else:
            self.set_default_bounds()
    
    def set_default_bounds(self):
        """Set default bounds based on Replica room_0"""
        print("Using default Replica room_0 bounds...")
        self.scene_bounds = {
            'min': np.array([-0.88, -1.19, 0.0]),
            'max': np.array([6.89, 3.51, 2.5]),
            'center': np.array([3.12, 1.12, 1.25])
        }
        self.floor_height = 0.0
        self.ceiling_height = 2.5
        self.eye_height_min = 1.4
        self.eye_height_max = 1.8
    
    def load_scene(self, scene_path):
        """Load scene and compute room boundaries"""
        if not os.path.exists(scene_path):
            print(f"Scene file not found, using default bounds")
            self.set_default_bounds()
            return
        
        try:
            self.scene = o3d.io.read_triangle_mesh(scene_path)
            if len(self.scene.vertices) == 0:
                self.scene = o3d.io.read_point_cloud(scene_path)
        except:
            self.set_default_bounds()
            return
        
        if isinstance(self.scene, o3d.geometry.PointCloud):
            points = np.asarray(self.scene.points)
        else:
            points = np.asarray(self.scene.vertices)
        
        if len(points) == 0:
            self.set_default_bounds()
            return
        
        self.scene_bounds = {
            'min': np.min(points, axis=0) + self.margin,
            'max': np.max(points, axis=0) - self.margin,
            'center': np.mean(points, axis=0)
        }
        
        if np.any(self.scene_bounds['min'] >= self.scene_bounds['max']):
            self.scene_bounds = {
                'min': np.min(points, axis=0),
                'max': np.max(points, axis=0),
                'center': np.mean(points, axis=0)
            }
        
        z_values = points[:, 2]
        self.floor_height = np.percentile(z_values, 5)
        self.ceiling_height = np.percentile(z_values, 95)
        
        self.eye_height_min = max(self.floor_height + 1.4, self.floor_height)
        self.eye_height_max = min(self.floor_height + 1.8, self.ceiling_height - 0.1)
        
        if self.eye_height_min >= self.eye_height_max:
            self.eye_height_min = self.floor_height + 1.6
            self.eye_height_max = self.floor_height + 1.7
    
    def generate_smooth_velocity_profile(self, num_frames, max_speed=0.5, acceleration_time=0.2):
        """
        Generate smooth velocity profile with controlled acceleration/deceleration
        
        Args:
            num_frames: Number of frames
            max_speed: Maximum speed in m/s
            acceleration_time: Time to accelerate/decelerate (as fraction of total time)
        """
        total_time = num_frames  # In frames, will convert to seconds later
        accel_frames = int(num_frames * acceleration_time)
        cruise_frames = num_frames - 2 * accel_frames
        
        # Ensure we have positive cruise time
        if cruise_frames < 0:
            accel_frames = num_frames // 3
            cruise_frames = num_frames - 2 * accel_frames
        
        # Create acceleration phase (ramp up)
        t_accel = np.linspace(0, 1, accel_frames)
        # Smooth S-curve for acceleration
        accel_profile = 0.5 - 0.5 * np.cos(t_accel * np.pi)
        
        # Cruise phase (constant speed)
        cruise_profile = np.ones(cruise_frames)
        
        # Deceleration phase (ramp down)
        t_decel = np.linspace(0, 1, accel_frames)
        decel_profile = 0.5 + 0.5 * np.cos(t_decel * np.pi)
        
        # Combine all phases
        velocity_profile = np.concatenate([
            accel_profile,
            cruise_profile,
            decel_profile
        ])
        
        # Normalize to desired max speed
        velocity_profile = velocity_profile * max_speed
        
        # Ensure correct length
        if len(velocity_profile) > num_frames:
            velocity_profile = velocity_profile[:num_frames]
        elif len(velocity_profile) < num_frames:
            # Pad with zeros
            pad_length = num_frames - len(velocity_profile)
            velocity_profile = np.pad(velocity_profile, (0, pad_length), mode='edge')
        
        return velocity_profile
    
    def generate_path_with_controlled_velocity(self, num_frames=300, camera_height=1.6,
                                             max_speed=0.4, path_type="room_center"):
        """
        Generate path with controlled velocity (starts and ends at zero speed)
        """
        camera_height = np.clip(camera_height, self.eye_height_min, self.eye_height_max)
        bounds = self.scene_bounds
        room_center = (bounds['min'] + bounds['max']) / 2
        room_size = bounds['max'] - bounds['min']
        
        # Generate velocity profile
        velocity_profile = self.generate_smooth_velocity_profile(
            num_frames, max_speed=max_speed, acceleration_time=0.2
        )
        
        # Generate base path points
        if path_type == "room_center":
            keypoints = self._generate_elliptical_keypoints(room_center, room_size, camera_height)
        elif path_type == "wall_following":
            keypoints = self._generate_wall_keypoints(bounds, room_size, camera_height)
        elif path_type == "figure8":
            keypoints = self._generate_figure8_keypoints(room_center, room_size, camera_height)
        else:
            keypoints = self._generate_elliptical_keypoints(room_center, room_size, camera_height)
        
        # Create smooth B-spline through keypoints
        positions = self._create_bspline_path(keypoints, num_frames)
        
        # Adjust path length to match velocity profile
        positions = self._adjust_path_for_velocity(positions, velocity_profile)
        
        # Ensure within bounds
        positions = self._clamp_to_bounds(positions)
        
        # Apply final smoothing
        positions = self._apply_gentle_smoothing(positions)
        
        return positions, velocity_profile
    
    def _generate_elliptical_keypoints(self, room_center, room_size, camera_height):
        """Generate keypoints for elliptical path"""
        max_radius_x = min(room_size[0] * 0.3, 1.0)
        max_radius_y = min(room_size[1] * 0.3, 1.0)
        
        # Generate points around ellipse
        t = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        x = max_radius_x * np.cos(t)
        y = max_radius_y * np.sin(t)
        z = camera_height * np.ones_like(t) + np.sin(t * 2) * 0.05
        
        keypoints = np.column_stack([x, y, z]) + room_center
        
        # Close the loop by adding first point at the end
        keypoints = np.vstack([keypoints, keypoints[0]])
        
        return keypoints
    
    def _generate_wall_keypoints(self, bounds, room_size, camera_height):
        """Generate keypoints for wall-following path"""
        wall_distance = min(room_size[0], room_size[1]) * 0.4
        
        # Create rounded rectangle corners
        corners = [
            [bounds['min'][0] + wall_distance, bounds['min'][1] + wall_distance, camera_height],
            [bounds['max'][0] - wall_distance, bounds['min'][1] + wall_distance, camera_height],
            [bounds['max'][0] - wall_distance, bounds['max'][1] - wall_distance, camera_height],
            [bounds['min'][0] + wall_distance, bounds['max'][1] - wall_distance, camera_height],
        ]
        
        # Add intermediate points for smoother corners
        keypoints = []
        for i in range(4):
            current = corners[i]
            next_corner = corners[(i + 1) % 4]
            
            # Add current corner
            keypoints.append(current)
            
            # Add midpoint for smoother turn
            midpoint = [
                (current[0] + next_corner[0]) / 2,
                (current[1] + next_corner[1]) / 2,
                camera_height + 0.03  # Slight lift at corners
            ]
            keypoints.append(midpoint)
        
        # Close the loop
        keypoints.append(keypoints[0])
        
        return np.array(keypoints)
    
    def _generate_figure8_keypoints(self, room_center, room_size, camera_height):
        """Generate keypoints for figure-8 path"""
        max_radius_x = min(room_size[0] * 0.25, 0.8)
        max_radius_y = min(room_size[1] * 0.25, 0.8)
        
        t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        x = max_radius_x * np.sin(t)
        y = max_radius_y * np.sin(2 * t) * 0.6
        z = camera_height * np.ones_like(t) + np.sin(t) * 0.03
        
        keypoints = np.column_stack([x, y, z]) + room_center
        keypoints = np.vstack([keypoints, keypoints[0]])  # Close loop
        
        return keypoints
    
    def _create_bspline_path(self, keypoints, num_frames):
        """Create smooth B-spline path through keypoints"""
        # Parameterize by cumulative distance
        distances = np.cumsum(np.sqrt(np.sum(np.diff(keypoints, axis=0)**2, axis=1)))
        distances = np.insert(distances, 0, 0)
        t_normalized = distances / distances[-1]
        
        # Create B-spline
        t_new = np.linspace(0, 1, num_frames)
        
        positions = np.zeros((num_frames, 3))
        for i in range(3):
            # Use cubic spline for smoothness
            spline = CubicSpline(t_normalized, keypoints[:, i], bc_type='periodic')
            positions[:, i] = spline(t_new)
        
        return positions
    
    def _adjust_path_for_velocity(self, positions, velocity_profile):
        """
        Adjust path sampling to match desired velocity profile
        Uses arc-length parameterization
        """
        num_frames = len(positions)
        
        # Calculate cumulative distance along path
        diffs = np.diff(positions, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        cumulative_dist = np.cumsum(segment_lengths)
        cumulative_dist = np.insert(cumulative_dist, 0, 0)
        total_length = cumulative_dist[-1]
        
        # Calculate desired cumulative distance based on velocity profile
        dt = 1.0 / (num_frames - 1)  # Normalized time step
        desired_cumulative_dist = np.zeros(num_frames)
        
        # Integrate velocity to get distance
        for i in range(1, num_frames):
            # Trapezoidal integration
            avg_velocity = (velocity_profile[i-1] + velocity_profile[i]) / 2
            desired_cumulative_dist[i] = desired_cumulative_dist[i-1] + avg_velocity * dt
        
        # Normalize desired distance to match total path length
        if desired_cumulative_dist[-1] > 0:
            desired_cumulative_dist = desired_cumulative_dist * (total_length / desired_cumulative_dist[-1])
        
        # Re-sample positions based on desired distance
        new_positions = np.zeros_like(positions)
        new_positions[0] = positions[0]
        
        for i in range(1, num_frames):
            target_dist = desired_cumulative_dist[i]
            
            # Find segment containing target distance
            idx = np.searchsorted(cumulative_dist, target_dist) - 1
            idx = min(idx, len(segment_lengths) - 1)
            
            if idx < 0:
                new_positions[i] = positions[0]
                continue
            
            # Linear interpolation within segment
            dist_in_segment = target_dist - cumulative_dist[idx]
            if segment_lengths[idx] > 0:
                t = dist_in_segment / segment_lengths[idx]
                t = np.clip(t, 0, 1)
                new_positions[i] = positions[idx] + t * (positions[idx + 1] - positions[idx])
            else:
                new_positions[i] = positions[idx]
        
        return new_positions
    
    def _apply_gentle_smoothing(self, positions):
        """Apply minimal smoothing to maintain velocity control"""
        smoothed = positions.copy()
        
        # Very gentle smoothing to remove any high-frequency noise
        for i in range(3):
            smoothed[:, i] = gaussian_filter1d(smoothed[:, i], sigma=1.5)
            # Savitzky-Golay preserves edges better
            window = min(15, len(smoothed) // 10)
            if window >= 5 and window % 2 == 1:
                smoothed[:, i] = savgol_filter(smoothed[:, i], window_length=window, polyorder=3)
        
        return smoothed
    
    def _clamp_to_bounds(self, positions):
        """Ensure positions stay within room bounds"""
        clamped = positions.copy()
        
        clamped[:, 0] = np.clip(clamped[:, 0], 
                               self.scene_bounds['min'][0],
                               self.scene_bounds['max'][0])
        
        clamped[:, 1] = np.clip(clamped[:, 1],
                               self.scene_bounds['min'][1],
                               self.scene_bounds['max'][1])
        
        clamped[:, 2] = np.clip(clamped[:, 2],
                               self.eye_height_min,
                               min(self.eye_height_max, self.ceiling_height - 0.2))
        
        return clamped
    
    def compute_smooth_orientations(self, positions, look_ahead_frames=10):
        """
        Compute orientations with smooth transitions
        """
        num_frames = len(positions)
        
        # Generate smooth look-at points
        look_at_points = np.zeros_like(positions)
        
        for i in range(num_frames):
            # Look ahead along the path
            look_ahead_idx = min(i + look_ahead_frames, num_frames - 1)
            
            # Blend between looking ahead and looking at room center for stability
            room_center = self.scene_bounds['center']
            blend = 0.8  # 80% look ahead, 20% room center
            look_at = blend * positions[look_ahead_idx] + (1 - blend) * room_center
            
            # Very subtle variation
            t = i / num_frames * 2 * np.pi
            variation = np.array([
                np.sin(t * 0.2) * 0.02,
                np.cos(t * 0.15) * 0.02,
                np.sin(t * 0.1) * 0.01
            ])
            
            look_at_points[i] = look_at + variation
        
        # Smooth look-at points
        for i in range(3):
            look_at_points[:, i] = gaussian_filter1d(look_at_points[:, i], sigma=3)
        
        # Compute orientations
        orientations = []
        
        for i in range(num_frames):
            forward = look_at_points[i] - positions[i]
            forward_norm = np.linalg.norm(forward)
            
            if forward_norm < 0.001:
                forward = np.array([1, 0, 0])
            else:
                forward = forward / forward_norm
            
            # Stable up vector
            up = np.array([0, 0, 1])
            
            # Ensure up is perpendicular to forward
            up = up - np.dot(up, forward) * forward
            up_norm = np.linalg.norm(up)
            if up_norm < 0.001:
                up = np.array([0, 0, 1])
            else:
                up = up / up_norm
            
            right = np.cross(forward, up)
            right_norm = np.linalg.norm(right)
            if right_norm < 0.001:
                right = np.array([0, 1, 0])
                right = right - np.dot(right, forward) * forward
                right_norm = np.linalg.norm(right)
                if right_norm > 0.001:
                    right = right / right_norm
                else:
                    right = np.array([1, 0, 0])
            else:
                right = right / right_norm
            
            up = np.cross(right, forward)
            
            rotation_matrix = np.column_stack([right, up, -forward])
            rotation = R.from_matrix(rotation_matrix)
            quaternion = rotation.as_quat()
            
            orientations.append(quaternion)
        
        orientations = np.array(orientations)
        
        # Smooth orientations
        orientations = self._smooth_quaternions(orientations)
        
        return orientations, look_at_points
    
    def _smooth_quaternions(self, quaternions):
        """Smooth quaternion orientations"""
        num_frames = len(quaternions)
        
        if num_frames < 10:
            return quaternions
        
        # Convert to rotation vectors for smoothing
        rotations = R.from_quat(quaternions)
        rotvecs = rotations.as_rotvec()
        
        # Apply smoothing
        smoothed_rotvecs = np.zeros_like(rotvecs)
        for i in range(3):
            smoothed_rotvecs[:, i] = gaussian_filter1d(rotvecs[:, i], sigma=2)
        
        # Convert back
        smoothed_rotations = R.from_rotvec(smoothed_rotvecs)
        
        # Use SLERP for final smoothness
        keyframe_interval = max(1, num_frames // 20)
        keyframe_indices = list(range(0, num_frames, keyframe_interval))
        if keyframe_indices[-1] != num_frames - 1:
            keyframe_indices.append(num_frames - 1)
        
        keyframe_rots = smoothed_rotations[keyframe_indices]
        slerp = Slerp(keyframe_indices, keyframe_rots)
        
        times = np.arange(num_frames)
        final_rotations = slerp(times)
        
        return final_rotations.as_quat()
    
    def generate_trajectory(self, num_frames=300, fps=30, camera_height=1.6,
                           max_speed=0.4, path_type="room_center"):
        """Generate complete trajectory with controlled velocity"""
        print(f"Generating {path_type} trajectory with max speed {max_speed}m/s...")
        
        # Generate positions with controlled velocity
        positions, velocity_profile = self.generate_path_with_controlled_velocity(
            num_frames=num_frames,
            camera_height=camera_height,
            max_speed=max_speed,
            path_type=path_type
        )
        
        print("Computing smooth orientations...")
        orientations, look_at_points = self.compute_smooth_orientations(
            positions, look_ahead_frames=int(num_frames * 0.05)
        )
        
        timestamps = np.arange(num_frames) / fps
        
        trajectory = {
            'positions': positions,
            'orientations': orientations,
            'look_at_points': look_at_points,
            'velocity_profile': velocity_profile,
            'timestamps': timestamps,
            'fps': fps,
            'num_frames': num_frames,
            'camera_height': camera_height,
            'path_type': path_type,
            'max_speed': max_speed
        }
        
        return trajectory
    
    def analyze_velocity(self, trajectory):
        """Analyze velocity profile of trajectory"""
        positions = trajectory['positions']
        velocity_profile = trajectory['velocity_profile']
        fps = trajectory['fps']
        
        # Calculate actual velocity from positions
        dt = 1.0 / fps
        actual_velocities = np.diff(positions, axis=0) / dt
        actual_speeds = np.linalg.norm(actual_velocities, axis=1)
        
        print("\n" + "="*60)
        print("VELOCITY ANALYSIS")
        print("="*60)
        print(f"Target max speed: {trajectory['max_speed']:.3f} m/s")
        print(f"\nActual velocity statistics:")
        print(f"  Max speed: {np.max(actual_speeds):.4f} m/s")
        print(f"  Min speed: {np.min(actual_speeds):.4f} m/s")
        print(f"  Mean speed: {np.mean(actual_speeds):.4f} m/s")
        
        # Check start and end speeds
        start_speed = actual_speeds[0] if len(actual_speeds) > 0 else 0
        end_speed = actual_speeds[-1] if len(actual_speeds) > 0 else 0
        
        print(f"\nStart/End speeds:")
        print(f"  Start speed: {start_speed:.4f} m/s")
        print(f"  End speed: {end_speed:.4f} m/s")
        
        if start_speed < 0.01 and end_speed < 0.01:
            print("  Smooth start and end (zero velocity)")
        else:
            print("  Non-zero start/end velocity detected")
        
        # Check acceleration
        if len(actual_speeds) > 1:
            accelerations = np.diff(actual_speeds) / dt
            max_accel = np.max(np.abs(accelerations))
            print(f"\nAcceleration analysis:")
            print(f"  Max acceleration: {max_accel:.4f} m/s²")
            
            if max_accel < 1.0:
                print("   Gentle acceleration (comfortable)")
            elif max_accel < 2.0:
                print("   Moderate acceleration")
            else:
                print("   High acceleration detected")
        
        return {
            'max_speed': np.max(actual_speeds),
            'start_speed': start_speed,
            'end_speed': end_speed,
            'actual_speeds': actual_speeds
        }
    
    def export_tum_format(self, trajectory, output_path):
        """Export in TUM RGB-D format"""
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("# TUM trajectory with controlled velocity\n")
            f.write(f"# Max speed: {trajectory['max_speed']:.3f} m/s, Path: {trajectory['path_type']}\n")
            f.write(f"# Frames: {trajectory['num_frames']}, FPS: {trajectory['fps']}\n")
            f.write("# timestamp tx ty tz qx qy qz qw\n")
            
            for i in range(trajectory['num_frames']):
                pos = trajectory['positions'][i]
                quat = trajectory['orientations'][i]
                timestamp = trajectory['timestamps'][i]
                
                f.write(f"{timestamp:.6f} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                       f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n")
        
        print(f"Exported {trajectory['num_frames']} frames to {output_path}")


# ============================================================================
# CONFIGURATION
# ============================================================================

SCENE_PATH = "./ThirdParty/Replica-Dataset/data/data/room_0/mesh.ply"
MARGIN = 0.3

# Trajectory configuration
NUM_FRAMES = 2000  # 15 seconds at 30fps
FPS = 30
CAMERA_HEIGHT = 1.6
MAX_SPEED = 0.3  # Maximum speed in m/s (comfortable walking speed)
PATH_TYPE = "figure8"  # Options: "room_center", "wall_following", "figure8"

# Output configuration
OUTPUT_TUM = "trajectory.txt"
PLOT_VELOCITY = True

# ============================================================================
# EXECUTION
# ============================================================================

print("=" * 70)
print("VELOCITY CONTROLLED TRAJECTORY GENERATOR")
print("=" * 70)

# Initialize generator
generator = VelocityControlledTrajectoryGenerator(SCENE_PATH, margin=MARGIN)

# Generate trajectory with controlled velocity
trajectory = generator.generate_trajectory(
    num_frames=NUM_FRAMES,
    fps=FPS,
    camera_height=CAMERA_HEIGHT,
    max_speed=MAX_SPEED,
    path_type=PATH_TYPE
)

# Analyze velocity
velocity_analysis = generator.analyze_velocity(trajectory)

# Export
generator.export_tum_format(trajectory, OUTPUT_TUM)

# Plot velocity profile if requested
if PLOT_VELOCITY:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    positions = trajectory['positions']
    velocity_profile = trajectory['velocity_profile']
    actual_speeds = velocity_analysis['actual_speeds']
    
    # Velocity profile comparison
    axes[0, 0].plot(trajectory['timestamps'][:-1], actual_speeds, 'b-', label='Actual speed', linewidth=2)
    axes[0, 0].plot(trajectory['timestamps'], velocity_profile, 'r--', label='Target speed', alpha=0.7)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Speed (m/s)')
    axes[0, 0].set_title('Velocity Profile')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Position over time
    axes[0, 1].plot(trajectory['timestamps'], positions[:, 0], 'r-', label='X', alpha=0.8)
    axes[0, 1].plot(trajectory['timestamps'], positions[:, 1], 'g-', label='Y', alpha=0.8)
    axes[0, 1].plot(trajectory['timestamps'], positions[:, 2], 'b-', label='Z', alpha=0.8)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Position (m)')
    axes[0, 1].set_title('Position Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Top view
    axes[1, 0].plot(positions[:, 0], positions[:, 1], 'b-', linewidth=1.5)
    axes[1, 0].scatter(positions[0, 0], positions[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    axes[1, 0].scatter(positions[-1, 0], positions[-1, 1], c='red', s=100, marker='s', label='End', zorder=5)
    
    # Plot velocity vectors
    step = NUM_FRAMES // 20
    for i in range(0, NUM_FRAMES - 1, step):
        dx = positions[i+1, 0] - positions[i, 0]
        dy = positions[i+1, 1] - positions[i, 1]
        speed = np.sqrt(dx**2 + dy**2) * FPS
        color = plt.cm.viridis(speed / MAX_SPEED)
        axes[1, 0].arrow(positions[i, 0], positions[i, 1], dx*0.8, dy*0.8,
                        head_width=0.05, head_length=0.1, fc=color, ec=color, alpha=0.7)
    
    axes[1, 0].set_xlabel('X (m)')
    axes[1, 0].set_ylabel('Y (m)')
    axes[1, 0].set_title('Top View with Velocity Vectors')
    axes[1, 0].legend()
    axes[1, 0].axis('equal')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Acceleration analysis
    if len(actual_speeds) > 1:
        dt = 1.0 / FPS
        accelerations = np.diff(actual_speeds) / dt
        
        axes[1, 1].plot(trajectory['timestamps'][1:-1], accelerations, 'purple', linewidth=1.5)
        axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Comfort limit (1 m/s²)')
        axes[1, 1].axhline(y=-1.0, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Acceleration (m/s²)')
        axes[1, 1].set_title('Acceleration Over Time')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Velocity Controlled Trajectory\n'
                f'Max Speed: {MAX_SPEED:.2f} m/s, Start/End Speed: {velocity_analysis["start_speed"]:.3f}/{velocity_analysis["end_speed"]:.3f} m/s',
                fontsize=14)
    
    plt.tight_layout()
    plt.savefig('velocity_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Output file: {OUTPUT_TUM}")
print(f"Frames: {NUM_FRAMES}")
print(f"Duration: {NUM_FRAMES/FPS:.1f} seconds")
print(f"Max target speed: {MAX_SPEED:.3f} m/s")
print(f"Actual max speed: {velocity_analysis['max_speed']:.3f} m/s")
print(f"Start speed: {velocity_analysis['start_speed']:.3f} m/s")
print(f"End speed: {velocity_analysis['end_speed']:.3f} m/s")

if velocity_analysis['start_speed'] < 0.01 and velocity_analysis['end_speed'] < 0.01:
    print("Perfect: Starts and ends at zero velocity")
else:
    print("Warning: Non-zero start/end velocity")

print("=" * 70)