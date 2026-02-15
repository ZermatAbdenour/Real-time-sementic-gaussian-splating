import os
import numpy as np
import magnum as mn
import habitat_sim
import imageio.v2 as imageio
import open3d as o3d
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
from habitat_sim.agent import AgentConfiguration
from habitat_sim.utils.common import quat_from_coeffs

# =============================================================================
# 1. COORDINATE CONVERSION (The "Secret Sauce")
# =============================================================================

def convert_habitat_to_opencv_quat(q_hab_xyzw):
    """
    Converts Habitat/OpenGL (-Z forward, Y up) to OpenCV (+Z forward, Y down).
    This ensures your saved poses match your Gaussian Splatting / PointCloud logic.
    """
    R_hab = R.from_quat(q_hab_xyzw).as_matrix()
    # Rotation to flip Y and Z axes
    R_chab_cocv = np.diag([1.0, -1.0, -1.0])
    R_ocv = R_hab @ R_chab_cocv
    return R.from_matrix(R_ocv).as_quat()

# =============================================================================
# 2. TRAJECTORY GENERATOR (Now with Robust Loading)
# =============================================================================

class VelocityControlledTrajectoryGenerator:
    def __init__(self, scene_path, margin=0.3):
        self.margin = margin
        self.load_scene_robustly(scene_path)

    def load_scene_robustly(self, scene_path):
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"Could not find mesh at {scene_path}")

        print(f"[INFO] Attempting to load geometry from {scene_path}...")
        
        # We try Point Cloud first because it's more permissive than Triangle Mesh
        pcd = o3d.io.read_point_cloud(scene_path)
        points = np.asarray(pcd.points)

        if points.shape[0] == 0:
            print("[WARNING] Point cloud empty, trying Mesh loader (permitting non-triangles)...")
            mesh = o3d.io.read_triangle_mesh(scene_path)
            points = np.asarray(mesh.vertices)

        if points.shape[0] == 0:
            raise ValueError("Failed to extract any vertices from the PLY file. Check file path/integrity.")

        print(f"[OK] Successfully loaded {points.shape[0]} vertices.")
        
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        
        self.scene_bounds = {
            "min": mins + self.margin,
            "max": maxs - self.margin,
            "center": (mins + maxs) / 2.0
        }
        y_vals = points[:, 1]
        self.floor_height = np.percentile(y_vals, 5)
        self.ceiling_height = np.percentile(y_vals, 95)
        self.eye_height_min = self.floor_height + 1.3
        self.eye_height_max = self.ceiling_height - 0.2

    def generate_trajectory(self, num_frames, camera_height, max_speed):
        # Create a smooth velocity profile (S-curve)
        t_prof = np.linspace(0, 1, num_frames)
        velocity_profile = (0.5 - 0.5 * np.cos(t_prof * 2 * np.pi)) * max_speed
        
        center = self.scene_bounds["center"]
        rad_x, rad_z = 0.7, 0.4 # Keeping it tight for figure-8
        
        # Figure 8 Math
        t_key = np.linspace(0, 2 * np.pi, 24)
        x = rad_x * np.sin(t_key) + center[0]
        z = rad_z * np.sin(2 * t_key) * 0.6 + center[2]
        y = np.full_like(t_key, camera_height)
        keypoints = np.column_stack([x, y, z])

        # Interpolate positions
        spline = CubicSpline(np.linspace(0, 1, len(keypoints)), keypoints, bc_type='periodic')
        positions = spline(np.linspace(0, 1, num_frames))
        
        # Smooth the resulting path
        for i in range(3):
            positions[:, i] = gaussian_filter1d(positions[:, i], sigma=1.2)

        # Generate Orientations (Habitat convention for capture)
        orientations = []
        world_up = np.array([0, 1, 0])
        for i in range(num_frames):
            look_idx = min(i + 12, num_frames - 1)
            forward = positions[look_idx] - positions[i]
            forward /= (np.linalg.norm(forward) + 1e-6)
            
            right = np.cross(forward, world_up)
            right /= (np.linalg.norm(right) + 1e-6)
            up = np.cross(right, forward)
            
            # Habitat basis: X=right, Y=up, Z=-forward
            R_mat = np.column_stack([right, up, -forward])
            orientations.append(R.from_matrix(R_mat).as_quat())

        return positions, np.array(orientations)

# =============================================================================
# 3. SETTINGS & EXECUTION
# =============================================================================

SCENE_PATH = "./ThirdParty/Replica-Dataset/data/data/room_0/mesh.ply"
DATASET_CONFIG = "ThirdParty/Replica-Dataset/data/data/replica.scene_dataset_config.json"
SCENE_NAME = "room_0"

OUTPUT_DIR = "habitat_capture"
RGB_DIR, DEPTH_DIR = os.path.join(OUTPUT_DIR, "rgb"), os.path.join(OUTPUT_DIR, "depth")
TRAJECTORY_FILE = os.path.join(OUTPUT_DIR, "poses_opencv.txt")

for d in [RGB_DIR, DEPTH_DIR]: os.makedirs(d, exist_ok=True)

# Path parameters
NUM_FRAMES, FPS = 150, 30
CAMERA_HEIGHT, MAX_SPEED = 1.5, 0.3

# --- Step A: Generate Trajectory ---
generator = VelocityControlledTrajectoryGenerator(SCENE_PATH)
pos, ori_hab = generator.generate_trajectory(NUM_FRAMES, CAMERA_HEIGHT, MAX_SPEED)

# --- Step B: Setup Simulator ---
sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_dataset_config_file = DATASET_CONFIG
sim_cfg.scene_id = SCENE_NAME
sim_cfg.gpu_device_id = 0

sensor_specs = []
for uid, s_type in [("rgb", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH)]:
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid, spec.sensor_type = uid, s_type
    spec.resolution, spec.hfov = [480, 640], 90.0
    sensor_specs.append(spec)

agent_cfg = AgentConfiguration()
agent_cfg.sensor_specifications = sensor_specs
sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
agent = sim.get_agent(0)

# --- Step C: Capture Loop & Export ---
print(f"[START] Capturing {NUM_FRAMES} frames...")
with open(TRAJECTORY_FILE, "w") as f:
    f.write("# timestamp tx ty tz qx qy qz qw (OpenCV T_wc)\n")
    
    for i in range(NUM_FRAMES):
        # 1. Update Habitat Camera
        state = habitat_sim.AgentState()
        state.position = pos[i]
        state.rotation = quat_from_coeffs(ori_hab[i])
        agent.set_state(state)

        # 2. Extract Data
        obs = sim.get_sensor_observations()
        rgb = obs["rgb"][:, :, :3]  # No alpha
        depth = (obs["depth"] * 1000).astype(np.uint16) # Millimeters
        
        # 3. Save Files
        ts_str = f"{i/FPS:.6f}"
        imageio.imwrite(os.path.join(RGB_DIR, f"{ts_str}.png"), rgb)
        imageio.imwrite(os.path.join(DEPTH_DIR, f"{ts_str}.png"), depth)

        # 4. Export Pose in OpenCV convention
        q_ocv = convert_habitat_to_opencv_quat(ori_hab[i])
        f.write(f"{ts_str} {pos[i][0]} {pos[i][1]} {pos[i][2]} {q_ocv[0]} {q_ocv[1]} {q_ocv[2]} {q_ocv[3]}\n")

        if i % 25 == 0: print(f"Processing frame {i}...")

sim.close()
print(f"\n[DONE] Saved {NUM_FRAMES} frames and poses to: {OUTPUT_DIR}")