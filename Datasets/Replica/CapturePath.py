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

os.makedirs(RGB_DIR, exist_ok=True)
os.makedirs(DEPTH_DIR, exist_ok=True)

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
                "quat": np.array([qx, qy, qz, qw], dtype=np.float32),
                "time": t
            })
    return traj

trajectory = load_tum_trajectory(TRAJ_PATH)
assert len(trajectory) > 0, "Trajectory is empty"
print(f"[OK] Loaded {len(trajectory)} trajectory poses")

# =====================================================
# SIMULATOR CONFIG (same as ./build/viewer)
# =====================================================
sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_dataset_config_file = DATASET_CONFIG
sim_cfg.scene_id = SCENE_NAME
sim_cfg.enable_physics = False

# =====================================================
# SENSORS
# =====================================================
sensor_specs = []

# --- RGB ---
rgb_sensor = habitat_sim.CameraSensorSpec()
rgb_sensor.uuid = "rgb"
rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
rgb_sensor.resolution = [480, 640]
rgb_sensor.position = mn.Vector3(0.0, 0.0, 0.0)
rgb_sensor.hfov = 90.0
sensor_specs.append(rgb_sensor)

# --- DEPTH ---
depth_sensor = habitat_sim.CameraSensorSpec()
depth_sensor.uuid = "depth"
depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
depth_sensor.resolution = [480, 640]
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

# =====================================================
# RUN TRAJECTORY + CAPTURE
# =====================================================
for i, frame in enumerate(trajectory):
    state = habitat_sim.AgentState()
    state.position = frame["position"]
    state.rotation = quat_from_coeffs(frame["quat"])

    agent.set_state(state, reset_sensors=True)

    obs = sim.get_sensor_observations()

    rgb = obs["rgb"]                          # uint8
    depth = obs["depth"].astype(np.float32)  # meters

    imageio.imwrite(
        os.path.join(RGB_DIR, f"{i:06d}.png"),
        rgb
    )

    imageio.imwrite(
        os.path.join(DEPTH_DIR, f"{i:06d}.png"),
        (depth * 1000.0).astype(np.uint16)    # mm (TUM style)
    )

    if i % 50 == 0:
        print(f"Captured {i}/{len(trajectory)}")

# =====================================================
# CLEANUP
# =====================================================
sim.close()
print("DONE — RGB + Depth captured correctly")
