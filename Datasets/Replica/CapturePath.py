import os
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ["OMP_NUM_THREADS"] = "1"

import habitat_sim
import magnum as mn
import numpy as np
import imageio.v2 as imageio

from habitat_sim.agent import AgentConfiguration
from habitat_sim.utils.common import quat_from_coeffs

# =====================================================
# USER PATHS
# =====================================================
DATASET_CONFIG = "ThirdParty/Replica-Dataset/data/data/replica.scene_dataset_config.json"
SCENE_NAME = "room_0"

TRAJ_PATH = "trajectory.txt"

OUTPUT_DIR = "habitat_capture"
RGB_DIR = os.path.join(OUTPUT_DIR, "rgb")
DEPTH_DIR = os.path.join(OUTPUT_DIR, "depth")
POSES_OUT = os.path.join(OUTPUT_DIR, "poses_habitat_tum.txt")  # saved as TUM (t tx ty tz qx qy qz qw)

os.makedirs(RGB_DIR, exist_ok=True)
os.makedirs(DEPTH_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# SETTINGS
# =====================================================
# If your trajectory.txt is in the usual TUM/OpenCV camera convention,
# set this True to convert to Habitat/OpenGL-like camera convention.
CONVERT_TUM_OPENCV_TO_HABITAT = True

# =====================================================
# LOAD TUM TRAJECTORY
# =====================================================
def load_tum_trajectory(path):
    traj = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            t, tx, ty, tz, qx, qy, qz, qw = map(float, line.split())
            traj.append({
                "position": np.array([tx, ty, tz], dtype=np.float32),
                "quat_xyzw": np.array([qx, qy, qz, qw], dtype=np.float32),
                "time": float(t),
            })
    return traj

trajectory = load_tum_trajectory(TRAJ_PATH)
assert len(trajectory) > 0, "Trajectory is empty"
print(f"[OK] Loaded {len(trajectory)} trajectory poses from {TRAJ_PATH}")

# =====================================================
# POSE / ROTATION HELPERS
# =====================================================
def quat_xyzw_to_R(q):
    """Quaternion (x,y,z,w) -> 3x3 rotation matrix."""
    x, y, z, w = q
    # Normalize to avoid drift / bad inputs
    n = np.sqrt(x*x + y*y + z*z + w*w)
    if n == 0:
        raise ValueError("Zero-norm quaternion")
    x, y, z, w = x/n, y/n, z/n, w/n

    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),   2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),       1 - 2*(xx + yy)],
    ], dtype=np.float32)

def R_to_quat_xyzw(R):
    """3x3 rotation matrix -> quaternion (x,y,z,w)."""
    # Robust conversion
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S

    q = np.array([x, y, z, w], dtype=np.float32)
    q /= np.linalg.norm(q)
    return q

def convert_tum_opencv_to_habitat_pose(t_w_c, q_w_c_xyzw):
    """
    Convert a pose expressed as T_w_c in OpenCV camera convention
    (camera looks +Z, x right, y down)
    to Habitat/OpenGL-like camera convention (camera looks -Z, y up).

    This is done by right-multiplying with a fixed camera-frame transform:
      R_copencv__chab = diag(1, -1, -1)
    so:
      R_w_chab = R_w_copencv * R_copencv__chab
    Translation in world stays the same (same camera center).
    """
    R_w_c = quat_xyzw_to_R(q_w_c_xyzw)

    R_copencv__chab = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    R_w_chab = R_w_c @ R_copencv__chab

    q_w_chab = R_to_quat_xyzw(R_w_chab)
    return t_w_c.astype(np.float32), q_w_chab

# =====================================================
# SIMULATOR CONFIG
# =====================================================
sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_dataset_config_file = DATASET_CONFIG
sim_cfg.scene_id = SCENE_NAME
sim_cfg.enable_physics = False

# =====================================================
# SENSORS
# =====================================================
sensor_specs = []

rgb_sensor = habitat_sim.CameraSensorSpec()
rgb_sensor.uuid = "rgb"
rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
rgb_sensor.resolution = [480, 640]  # H, W
rgb_sensor.position = mn.Vector3(0.0, 0.0, 0.0)
rgb_sensor.hfov = 90.0
sensor_specs.append(rgb_sensor)

depth_sensor = habitat_sim.CameraSensorSpec()
depth_sensor.uuid = "depth"
depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
depth_sensor.resolution = [480, 640]  # H, W
depth_sensor.position = mn.Vector3(0.0, 0.0, 0.0)
depth_sensor.hfov = 90.0
depth_sensor.min_depth = 0.1
depth_sensor.max_depth = 10.0
depth_sensor.normalize_depth = False
sensor_specs.append(depth_sensor)

# =====================================================
# AGENT
# =====================================================
agent_cfg = AgentConfiguration()
agent_cfg.sensor_specifications = sensor_specs

# =====================================================
# CREATE SIMULATOR
# =====================================================
sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
agent = sim.get_agent(0)

print(f"[OK] Loaded scene '{SCENE_NAME}'")

# Save the actual poses we used (after conversion) for debugging
poses_f = open(POSES_OUT, "w")
poses_f.write("# t tx ty tz qx qy qz qw   (poses actually used for Habitat)\n")

# =====================================================
# RUN TRAJECTORY + CAPTURE
# =====================================================
for i, frame in enumerate(trajectory):
    t = frame["time"]
    t_w_c = frame["position"]
    q_w_c = frame["quat_xyzw"]

    if CONVERT_TUM_OPENCV_TO_HABITAT:
        t_w_c, q_w_c = convert_tum_opencv_to_habitat_pose(t_w_c, q_w_c)

    # Habitat expects AgentState.rotation as a Magnum quaternion.
    # quat_from_coeffs expects (x, y, z, w).
    state = habitat_sim.AgentState()
    state.position = t_w_c
    state.rotation = quat_from_coeffs(q_w_c)

    agent.set_state(state, reset_sensors=True)

    obs = sim.get_sensor_observations()
    rgb = obs["rgb"]  # uint8 HxWx4 (RGBA) in many setups; keep as-is or strip alpha if needed
    depth = obs["depth"].astype(np.float32)  # meters

    ts = f"{t:.6f}"

    imageio.imwrite(os.path.join(RGB_DIR, f"{ts}.png"), rgb)
    imageio.imwrite(
        os.path.join(DEPTH_DIR, f"{ts}.png"),
        (depth * 1000.0).astype(np.uint16)
    )

    # Save pose used
    poses_f.write(f"{ts} {t_w_c[0]} {t_w_c[1]} {t_w_c[2]} {q_w_c[0]} {q_w_c[1]} {q_w_c[2]} {q_w_c[3]}\n")

    if i % 50 == 0:
        print(f"Captured {i}/{len(trajectory)}")

poses_f.close()

# =====================================================
# CLEANUP
# =====================================================
sim.close()
print(f"DONE — RGB + Depth captured with timestamps")
print(f"[OK] Wrote poses used for capture to: {POSES_OUT}")