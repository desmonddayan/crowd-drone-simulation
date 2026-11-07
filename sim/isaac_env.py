"""NVIDIA Isaac Sim 5.1.0 scene builder, crowd/drone runner, and ZeroMQ publisher.

Verified against installed Isaac Sim version:
    `5.1.0-rc.19+release.26219.9c81211b.gl` (`C:\\isaacsim`)

Uses official Isaac Sim 5.1 APIs:
    - `from isaacsim import SimulationApp`
    - `from isaacsim.core.api import World`
    - `from isaacsim.core.api.objects import VisualCuboid, VisualCapsule`
    - `from isaacsim.sensors.camera import Camera`
    - `import isaacsim.core.utils.numpy.rotations as rot_utils`
    - `from pxr import Gf, Sdf, UsdGeom, UsdLux`

Run with Isaac Sim's bundled Python:
    `C:\\isaacsim\\python.bat -m sim.isaac_env --drones 4 --people 150`

Inputs:
    CLI arguments (`--drones`, `--people`, `--headless`, `--steps`) and `DroneCommand`
    messages received over ZeroMQ (`ZMQ_CMD_PORT`).
Outputs:
    Publishes `FrameMessage` (`ZMQ_FRAME_PORT`), `DroneState` (`ZMQ_STATE_PORT`),
    and evaluation-only ground truth (`ZMQ_GT_PORT`).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

# Ensure repository root is on sys.path when invoked via `C:\isaacsim\python.bat`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from interfaces.contract import FrameMessage
from sim.crowd_dynamics import CrowdAndDroneWorld
from sim.zmq_bridge import SimZmqEndpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isaac Sim 5.1 Crowd & Drone Fleet Simulator")
    parser.add_argument("--drones", type=int, default=config.N_DRONES, help="Number of drones")
    parser.add_argument("--people", type=int, default=config.N_PEOPLE, help="Number of people")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless")
    parser.add_argument("--steps", type=int, default=0, help="Max physics steps (0 = infinite)")
    parser.add_argument("--surge-time", type=float, default=config.SURGE_START_SECONDS)
    args, _ = parser.parse_known_args()
    return args


def run_isaac_sim() -> None:
    args = parse_args()

    # 1. Launch Isaac Sim 5.1 SimulationApp BEFORE importing omni / pxr / isaacsim.core
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": bool(args.headless)})

    import isaacsim.core.utils.numpy.rotations as rot_utils
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import VisualCapsule, VisualCuboid
    from isaacsim.sensors.camera import Camera
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    # 2. Night-time concert scene: ground plane, stage, barriers, and low ambient dome light
    world.scene.add_default_ground_plane()

    dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/NightAmbientLight"))
    dome_light.CreateIntensityAttr(65.0)  # Dim night lighting so drone spotlights stand out
    dome_light.CreateColorAttr(Gf.Vec3f(0.15, 0.18, 0.28))

    # Concert Stage cuboid
    scx, scy = config.STAGE_CENTER
    slx, sly, slz = config.STAGE_SIZE
    world.scene.add(
        VisualCuboid(
            prim_path="/World/ConcertStage",
            name="concert_stage",
            position=np.array([scx, scy, slz * 0.5]),
            scale=np.array([slx, sly, slz]),
            size=1.0,
            color=np.array([140, 50, 120]),
        )
    )

    # Barriers (front-of-stage pen + Exit A funnel walls)
    for b_idx, ((x1, y1), (x2, y2)) in enumerate(config.BARRIER_SEGMENTS):
        mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        length = math.hypot(x2 - x1, y2 - y1)
        yaw_deg = math.degrees(math.atan2(y2 - y1, x2 - x1))
        quat = rot_utils.euler_angles_to_quats(np.array([0.0, 0.0, yaw_deg]), degrees=True)
        world.scene.add(
            VisualCuboid(
                prim_path=f"/World/Barriers/Barrier_{b_idx}",
                name=f"barrier_{b_idx}",
                position=np.array([mx, my, 0.6]),
                orientation=quat,
                scale=np.array([length, 0.35, 1.2]),
                size=1.0,
                color=np.array([180, 185, 195]),
            )
        )

    # 3. Spawn P people as bright warm-coloured capsules with semantic class "person"
    dynamics = CrowdAndDroneWorld(
        n_drones=args.drones,
        n_people=args.people,
        surge_start_seconds=args.surge_time,
    )
    person_prims: List[VisualCapsule] = []
    for p_idx in range(args.people):
        px, py = dynamics.person_pos[p_idx]
        prim_path = f"/World/Crowd/Person_{p_idx:03d}"
        cap = world.scene.add(
            VisualCapsule(
                prim_path=prim_path,
                name=f"person_{p_idx:03d}",
                position=np.array([px, py, 0.9]),
                radius=0.24,
                height=1.1,
                color=np.array([255, 205, 55]),
            )
        )
        # Attach custom USD attribute for semantic label "person"
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            prim.CreateAttribute("semantic:Semantics:params:semanticType", Sdf.ValueTypeNames.String).Set("class")
            prim.CreateAttribute("semantic:Semantics:params:semanticData", Sdf.ValueTypeNames.String).Set("person")
        person_prims.append(cap)

    # 4. Spawn N kinematic drones with downward Camera and downward UsdLux SphereLight/ShapingAPI spotlight
    drone_prims: List[VisualCuboid] = []
    cameras: List[Camera] = []
    spotlights: Dict[int, UsdLux.SphereLight] = {}
    down_quat = rot_utils.euler_angles_to_quats(np.array([0.0, 90.0, 0.0]), degrees=True)

    for d_idx in range(args.drones):
        dx, dy, dz = dynamics.drone_pos[d_idx]
        drone_path = f"/World/Drones/Drone_{d_idx:02d}"
        drone_obj = world.scene.add(
            VisualCuboid(
                prim_path=drone_path,
                name=f"drone_{d_idx:02d}",
                position=np.array([dx, dy, dz]),
                scale=np.array([0.65, 0.65, 0.22]),
                size=1.0,
                color=np.array([40, 220, 100]),
            )
        )
        drone_prims.append(drone_obj)

        # Child downward camera
        cam_path = f"{drone_path}/DownwardCamera"
        cam = Camera(
            prim_path=cam_path,
            position=np.array([dx, dy, dz - 0.15]),
            frequency=int(config.CAMERA_CAPTURE_HZ),
            resolution=(config.CAMERA_WIDTH, config.CAMERA_HEIGHT),
            orientation=down_quat,
        )
        cameras.append(cam)

        # Child downward spotlight using UsdLux.SphereLight + ShapingAPI cone
        light_path = Sdf.Path(f"{drone_path}/Spotlight")
        s_light = UsdLux.SphereLight.Define(stage, light_path)
        s_light.CreateIntensityAttr(35000.0)
        s_light.CreateRadiusAttr(0.25)
        s_light.CreateColorAttr(Gf.Vec3f(0.1, 0.9, 0.2))
        shaping = UsdLux.ShapingAPI.Apply(s_light.GetPrim())
        shaping.CreateShapingConeAngleAttr(config.LIGHT_CONE_HALF_ANGLE_DEG)
        shaping.CreateShapingConeSoftnessAttr(0.25)
        spotlights[d_idx] = s_light

    world.reset()
    for cam in cameras:
        cam.initialize()

    zmq_bridge = SimZmqEndpoint()
    intrinsics = config.get_camera_intrinsics()
    rr_cursor = 0
    step_count = 0

    try:
        while simulation_app.is_running() and (args.steps <= 0 or step_count < args.steps):
            # Poll latest commands from agent process
            incoming_cmds = zmq_bridge.poll_commands()
            for cmd in incoming_cmds:
                dynamics.apply_command(cmd)

            # Step shared crowd social-force & kinematic drone physics
            dynamics.step(dt=0.1)

            # Update person USD prim positions
            for p_idx, cap in enumerate(person_prims):
                px, py = dynamics.person_pos[p_idx]
                cap.set_world_pose(position=np.array([px, py, 0.9]))

            # Update drone USD prim positions, cameras, and spotlight parameters
            for d_idx, drone_obj in enumerate(drone_prims):
                dx, dy, dz = dynamics.drone_pos[d_idx]
                drone_obj.set_world_pose(position=np.array([dx, dy, dz]))
                cameras[d_idx].set_world_pose(
                    position=np.array([dx, dy, dz - 0.15]),
                    orientation=down_quat,
                )

                cmd = dynamics.drone_commands[d_idx]
                # Photosensitivity safety: intensity modulation uses smooth sine <= 2.0 Hz
                mod_intensity = dynamics.get_modulated_light_intensity(d_idx)
                s_light = spotlights[d_idx]
                s_light.GetColorAttr().Set(
                    Gf.Vec3f(
                        float(cmd.light_colour[0]),
                        float(cmd.light_colour[1]),
                        float(cmd.light_colour[2]),
                    )
                )
                s_light.GetIntensityAttr().Set(float(80000.0 * mod_intensity))

            world.step(render=True)

            # Publish DroneState telemetry and evaluation-only ground truth
            zmq_bridge.publish_states(dynamics.get_drone_states())
            zmq_bridge.publish_ground_truth(
                dynamics.sim_time,
                [(float(p[0]), float(p[1])) for p in dynamics.person_pos],
            )

            # Capture downward cameras (round-robin subset if N > MAX_CAMERAS_PER_STEP)
            n_d = args.drones
            batch_size = min(n_d, config.MAX_CAMERAS_PER_STEP)
            for k in range(batch_size):
                cam_id = (rr_cursor + k) % n_d
                rgba = cameras[cam_id].get_rgba()
                if rgba is not None and rgba.size > 0:
                    rgb = rgba[:, :, :3].astype(np.uint8)
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    if ok:
                        dx, dy, dz = dynamics.drone_pos[cam_id]
                        zmq_bridge.publish_frame(
                            FrameMessage(
                                drone_id=cam_id,
                                timestamp=dynamics.sim_time,
                                jpeg_bytes=buf.tobytes(),
                                camera_position_xyz=(float(dx), float(dy), float(dz)),
                                camera_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
                                intrinsics_3x3=intrinsics,
                                width=config.CAMERA_WIDTH,
                                height=config.CAMERA_HEIGHT,
                            )
                        )
            rr_cursor = (rr_cursor + batch_size) % n_d
            step_count += 1
    finally:
        zmq_bridge.close()
        simulation_app.close()


if __name__ == "__main__":
    run_isaac_sim()
