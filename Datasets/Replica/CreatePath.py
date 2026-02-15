import os
import numpy as np
import habitat_sim
import imageio.v2 as imageio
import open3d as o3d
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
from habitat_sim.agent import AgentConfiguration
from habitat_sim.utils.common import quat_from_coeffs

# ----------------------------
# Reconstruction constants
# ----------------------------
def build_R_fix():
    ax, ay = np.radians(-90.0), np.radians(180.0)
    Rx = np.array(
        [[1, 0, 0],
         [0, np.cos(ax), -np.sin(ax)],
         [0, np.sin(ax),  np.cos(ax)]],
        dtype=np.float64
    )
    Ry = np.array(
        [[ np.cos(ay), 0, np.sin(ay)],
         [0,           1, 0],
         [-np.sin(ay), 0, np.cos(ay)]],
        dtype=np.float64
    )
    return (Ry @ Rx)

R_fix = build_R_fix()
R_fix_inv = R_fix.T

# OpenCV cam -> Habitat cam axis flip
S_cv_to_hcam = np.diag([1.0, -1.0, -1.0])

def write_tum_line(f, ts, t, Rm):
    q = R.from_matrix(Rm).as_quat()  # [x,y,z,w]
    f.write(f"{ts:.6f} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]}\n")

def lookat_quat_habitat(position, target, world_up):
    forward = target - position
    forward = forward / (np.linalg.norm(forward) + 1e-9)

    right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + 1e-9)

    up = np.cross(right, forward)
    up = up / (np.linalg.norm(up) + 1e-9)

    R_mat = np.column_stack([right, up, -forward])  # Habitat basis
    return R.from_matrix(R_mat).as_quat()

def habitat_quat_to_R_world_from_hcam(q_hab_xyzw):
    return R.from_quat(q_hab_xyzw).as_matrix()

def desired_Rt_world_from_cv(pos_world, q_hab_xyzw):
    R_wh = habitat_quat_to_R_world_from_hcam(q_hab_xyzw)
    R_des = R_wh @ S_cv_to_hcam
    t_des = pos_world
    return R_des, t_des

def raw_Rt_for_reconstruction(R_des, t_des):
    R_raw = R_fix_inv @ R_des
    t_raw = R_fix_inv @ t_des
    return R_raw, t_raw

# ----------------------------
# Up-axis detection by "floor band" density
# ----------------------------
def detect_up_axis_by_floor_band(points, band_frac=0.02):
    """
    Choose the axis that most looks like "vertical":
    Floors create a dense band near the minimum of the vertical axis.
    We score each axis by how many points lie within a small band above its low percentile.
    """
    scores = []
    for axis in range(3):
        vals = points[:, axis]
        lo = np.percentile(vals, 2.0)
        hi = np.percentile(vals, 98.0)
        span = max(hi - lo, 1e-6)
        band = lo + band_frac * span
        score = float(np.mean(vals <= band))  # fraction of points in the bottom band
        scores.append(score)

    up_axis = int(np.argmax(scores))
    return up_axis, scores

class TrajectoryGeneratorRobustUp:
    def __init__(self, scene_path, margin=0.3):
        self.margin = margin
        self._load(scene_path)

    def _load(self, scene_path):
        if not os.path.exists(scene_path):
            raise FileNotFoundError(scene_path)

        pcd = o3d.io.read_point_cloud(scene_path)
        pts = np.asarray(pcd.points)
        if pts.shape[0] == 0:
            mesh = o3d.io.read_triangle_mesh(scene_path)
            pts = np.asarray(mesh.vertices)
        if pts.shape[0] == 0:
            raise ValueError("Empty mesh/pcd")

        self.pts = pts
        self.mins = pts.min(axis=0)
        self.maxs = pts.max(axis=0)
        self.center = (self.mins + self.maxs) / 2.0

        self.up_axis, scores = detect_up_axis_by_floor_band(pts)
        self.ground_axes = [a for a in [0, 1, 2] if a != self.up_axis]

        print("[INFO] floor-band scores (x,y,z):", scores)
        print("[INFO] detected up axis:", ["x", "y", "z"][self.up_axis])
        print("[INFO] ground axes:", [ ["x","y","z"][a] for a in self.ground_axes ])

        # Estimate floor on that axis
        vals = pts[:, self.up_axis]
        self.floor = float(np.percentile(vals, 2.0))

    def generate(self, num_frames, eye_height=1.5, rad_a=0.7, rad_b=0.4):
        """
        Keep up-axis constant at floor + eye_height.
        Generate figure-8 on the other two axes (ground plane).
        """
        t_key = np.linspace(0, 2 * np.pi, 24)

        a = rad_a * np.sin(t_key)
        b = (rad_b * np.sin(2 * t_key) * 0.6)

        keypoints = np.zeros((len(t_key), 3), dtype=np.float64)
        keypoints[:] = self.center

        ax0, ax1 = self.ground_axes
        keypoints[:, ax0] = self.center[ax0] + a
        keypoints[:, ax1] = self.center[ax1] + b

        # Clamp "height" so it cannot bob up/down
        keypoints[:, self.up_axis] = self.floor + eye_height

        spline = CubicSpline(np.linspace(0, 1, len(keypoints)), keypoints, bc_type="periodic")
        pos = spline(np.linspace(0, 1, num_frames))
        for k in range(3):
            pos[:, k] = gaussian_filter1d(pos[:, k], sigma=1.2)

        # Force the up-axis to be EXACT constant (kills any spline overshoot)
        pos[:, self.up_axis] = self.floor + eye_height

        world_up = np.zeros(3, dtype=np.float64)
        world_up[self.up_axis] = 1.0

        q = []
        for i in range(num_frames):
            look_idx = min(i + 12, num_frames - 1)
            q.append(lookat_quat_habitat(pos[i], pos[look_idx], world_up))

        print("[DEBUG] pos ranges:",
              "x", float(pos[:, 0].min()), float(pos[:, 0].max()),
              "y", float(pos[:, 1].min()), float(pos[:, 1].max()),
              "z", float(pos[:, 2].min()), float(pos[:, 2].max()))

        return pos, np.array(q)

# ----------------------------
# SETTINGS
# ----------------------------
SCENE_PATH = "./ThirdParty/Replica-Dataset/data/data/room_0/mesh.ply"
DATASET_CONFIG = "ThirdParty/Replica-Dataset/data/data/replica.scene_dataset_config.json"
SCENE_NAME = "room_0"

OUTPUT_DIR = "habitat_capture"
RGB_DIR = os.path.join(OUTPUT_DIR, "rgb")
DEPTH_DIR = os.path.join(OUTPUT_DIR, "depth")
POSES_TUM = os.path.join(OUTPUT_DIR, "poses.txt")

os.makedirs(RGB_DIR, exist_ok=True)
os.makedirs(DEPTH_DIR, exist_ok=True)

NUM_FRAMES, FPS = 150, 30
EYE_HEIGHT = 1.5  # meters above detected floor (on detected up-axis)

gen = TrajectoryGeneratorRobustUp(SCENE_PATH)
pos_world, q_hab = gen.generate(NUM_FRAMES, eye_height=EYE_HEIGHT)

# Simulator
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

print(f"[START] Capturing {NUM_FRAMES} frames...")

with open(POSES_TUM, "w", encoding="utf-8", newline="\n") as f:
    f.write("# timestamp tx ty tz qx qy qz qw\n")

    for i in range(NUM_FRAMES):
        state = habitat_sim.AgentState()
        state.position = pos_world[i].astype(np.float32)
        state.rotation = quat_from_coeffs(q_hab[i].astype(np.float32))
        agent.set_state(state)

        obs = sim.get_sensor_observations()
        rgb = obs["rgb"][:, :, :3]
        depth = (obs["depth"] * 1000).astype(np.uint16)

        ts = i / FPS
        ts_str = f"{ts:.6f}"
        imageio.imwrite(os.path.join(RGB_DIR, f"{ts_str}.png"), rgb)
        imageio.imwrite(os.path.join(DEPTH_DIR, f"{ts_str}.png"), depth)

        R_des, t_des = desired_Rt_world_from_cv(pos_world[i], q_hab[i])
        R_raw, t_raw = raw_Rt_for_reconstruction(R_des, t_des)
        write_tum_line(f, ts, t_raw, R_raw)

        if i % 25 == 0:
            print(f"Processing frame {i}...")

sim.close()
print(f"\n[DONE] Saved frames to: {OUTPUT_DIR}")
print(f"[DONE] Saved poses to: {POSES_TUM}")