"""NVIDIA Isaac Sim 5.1.0 3D Concert Venue, 3D Crowd & Quadcopter Fleet Simulator.

Verified against installed Isaac Sim version:
    `5.1.0-rc.19+release.26219.9c81211b.gl` (`C:\\isaacsim`)

Builds a rich 3D outdoor concert environment in USD:
- Concert Stage with main deck, rear LED backdrop wall, 4 truss towers, overhead roof
  truss, and coloured stage spotlights.
- Metallic crowd barriers forming the front-of-stage pen and the Exit A bottleneck funnel,
  plus illuminated Exit A and Exit B portals.
- 150 3D human figures (head + torso + legs, oriented along their walking velocity vector).
- N 3D quadcopter drones (central chassis + 4 rotor arms + 4 rotor discs + downward camera
  + downward UsdLux spotlight with ShapingAPI cone).

Run with Isaac Sim's bundled Python:
    `C:\\isaacsim\\python.bat -m sim.isaac_env --drones 4 --people 150`
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from interfaces.contract import FrameMessage
from sim.crowd_dynamics import CrowdAndDroneWorld
from sim.zmq_bridge import SimZmqEndpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isaac Sim 5.1 3D Concert Crowd & Drone Simulator")
    parser.add_argument("--drones", type=int, default=config.N_DRONES, help="Number of drones")
    parser.add_argument("--people", type=int, default=config.N_PEOPLE, help="Number of people")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless")
    parser.add_argument("--steps", type=int, default=0, help="Max physics steps (0 = infinite)")
    parser.add_argument("--surge-time", type=float, default=config.SURGE_START_SECONDS)
    args, _ = parser.parse_known_args()
    return args


def run_isaac_sim() -> None:
    args = parse_args()

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": bool(args.headless)})

    import isaacsim.core.utils.numpy.rotations as rot_utils
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import VisualCapsule, VisualCuboid, VisualCylinder, VisualSphere
    from isaacsim.core.prims import SingleXFormPrim
    from isaacsim.core.utils import viewports
    from isaacsim.sensors.camera import Camera
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    # Position the main 3D perspective camera overlooking the concert arena
    viewports.set_camera_view(
        eye=np.array([-14.0, -16.0, 38.0]),
        target=np.array([42.0, 30.0, 2.0]),
    )

    # 1. Ground plane & night-time stadium lighting
    world.scene.add_default_ground_plane()

    dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/NightAmbientLight"))
    dome_light.CreateIntensityAttr(420.0)
    dome_light.CreateColorAttr(Gf.Vec3f(0.48, 0.52, 0.68))

    # 2. 3D Concert Stage (deck, backdrop screen, truss towers, roof, and stage lights)
    scx, scy = config.STAGE_CENTER
    slx, sly, slz = config.STAGE_SIZE
    world.scene.add(
        VisualCuboid(
            prim_path="/World/ConcertStage/Deck",
            name="stage_deck",
            position=np.array([scx, scy, slz * 0.5]),
            scale=np.array([slx, sly, slz]),
            size=1.0,
            color=np.array([0.18, 0.18, 0.22]),
        )
    )
    world.scene.add(
        VisualCuboid(
            prim_path="/World/ConcertStage/Backdrop",
            name="stage_backdrop",
            position=np.array([scx - slx * 0.42, scy, slz + 4.5]),
            scale=np.array([0.5, sly * 0.9, 9.0]),
            size=1.0,
            color=np.array([0.55, 0.12, 0.75]),
        )
    )
    # 4 Truss pillars + roof canopy
    for t_idx, (tx, ty) in enumerate(
        [
            (scx - slx * 0.4, scy - sly * 0.45),
            (scx - slx * 0.4, scy + sly * 0.45),
            (scx + slx * 0.4, scy - sly * 0.45),
            (scx + slx * 0.4, scy + sly * 0.45),
        ]
    ):
        world.scene.add(
            VisualCylinder(
                prim_path=f"/World/ConcertStage/Pillar_{t_idx}",
                name=f"stage_pillar_{t_idx}",
                position=np.array([tx, ty, slz + 5.0]),
                radius=0.22,
                height=10.0,
                color=np.array([0.75, 0.78, 0.82]),
            )
        )
    world.scene.add(
        VisualCuboid(
            prim_path="/World/ConcertStage/RoofCanopy",
            name="stage_roof",
            position=np.array([scx, scy, slz + 10.2]),
            scale=np.array([slx * 1.05, sly * 1.02, 0.5]),
            size=1.0,
            color=np.array([0.22, 0.22, 0.26]),
        )
    )

    # Stage light beams illuminating the concert arena
    for l_idx, ly in enumerate([scy - 7.0, scy, scy + 7.0]):
        st_light = UsdLux.SphereLight.Define(
            stage, Sdf.Path(f"/World/ConcertStage/StageLight_{l_idx}")
        )
        UsdGeom.XformCommonAPI(st_light).SetTranslate(Gf.Vec3d(scx + 3.0, ly, slz + 9.5))
        st_light.CreateIntensityAttr(45000.0)
        st_light.CreateRadiusAttr(0.4)
        st_light.CreateColorAttr(
            Gf.Vec3f(0.9, 0.3, 0.8) if l_idx != 1 else Gf.Vec3f(0.2, 0.6, 1.0)
        )

    # 3. Illuminated Exit Portals (Exit_A and Exit_B)
    for exit_name, poly in config.EXITS.items():
        pts = np.asarray(poly, dtype=np.float64)
        ex_cx, ex_cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
        ex_wx = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
        ex_wy = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
        world.scene.add(
            VisualCuboid(
                prim_path=f"/World/Exits/{exit_name}_Pad",
                name=f"exit_pad_{exit_name.lower()}",
                position=np.array([ex_cx, ex_cy, 0.06]),
                scale=np.array([max(ex_wx, 2.0), max(ex_wy, 2.0), 0.12]),
                size=1.0,
                color=np.array([0.10, 0.88, 0.35]),
            )
        )

    # 4. Metallic Crowd Barriers (front-of-stage pen + Exit A funnel walls)
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
                scale=np.array([length, 0.30, 1.2]),
                size=1.0,
                color=np.array([0.72, 0.76, 0.82]),
            )
        )

    # 5. Spawn P 3D Human Figures (Root XForm with LowerBody, Torso, and Head)
    dynamics = CrowdAndDroneWorld(
        n_drones=args.drones,
        n_people=args.people,
        surge_start_seconds=args.surge_time,
    )
    person_roots: List[SingleXFormPrim] = []
    # High-visibility warm jacket/hat palette so downward cameras detect every person reliably
    jacket_palette = [
        np.array([1.0, 0.78, 0.20]),
        np.array([0.98, 0.62, 0.18]),
        np.array([1.0, 0.86, 0.28]),
        np.array([0.95, 0.70, 0.22]),
    ]

    for p_idx in range(args.people):
        px, py = dynamics.person_pos[p_idx]
        root_path = f"/World/Crowd/Person_{p_idx:03d}"
        root_xform = SingleXFormPrim(
            prim_path=root_path,
            name=f"person_root_{p_idx:03d}",
            position=np.array([px, py, 0.0]),
        )
        # Legs
        VisualCapsule(
            prim_path=f"{root_path}/Legs",
            name=f"person_legs_{p_idx:03d}",
            translation=np.array([0.0, 0.0, 0.42]),
            radius=0.14,
            height=0.55,
            color=np.array([0.20, 0.24, 0.35]),
        )
        # Torso / shoulders
        VisualCapsule(
            prim_path=f"{root_path}/Torso",
            name=f"person_torso_{p_idx:03d}",
            translation=np.array([0.0, 0.0, 1.05]),
            radius=0.22,
            height=0.52,
            color=jacket_palette[p_idx % len(jacket_palette)],
        )
        # Head (bright skin/hat tone visible from overhead camera)
        VisualSphere(
            prim_path=f"{root_path}/Head",
            name=f"person_head_{p_idx:03d}",
            translation=np.array([0.0, 0.0, 1.58]),
            radius=0.16,
            color=np.array([1.0, 0.92, 0.50]),
        )
        prim = stage.GetPrimAtPath(root_path)
        if prim.IsValid():
            prim.CreateAttribute(
                "semantic:Semantics:params:semanticType", Sdf.ValueTypeNames.String
            ).Set("class")
            prim.CreateAttribute(
                "semantic:Semantics:params:semanticData", Sdf.ValueTypeNames.String
            ).Set("person")
        person_roots.append(root_xform)

    # 6. Spawn N 3D Quadcopter Drones (Root XForm with Body, 4 Arms, 4 Rotors, Camera, Spotlight)
    drone_roots: List[SingleXFormPrim] = []
    drone_fuselages: List[VisualCuboid] = []
    cameras: List[Camera] = []
    spotlights: Dict[int, UsdLux.SphereLight] = {}
    down_quat = rot_utils.euler_angles_to_quats(np.array([0.0, 90.0, 0.0]), degrees=True)

    for d_idx in range(args.drones):
        dx, dy, dz = dynamics.drone_pos[d_idx]
        drone_path = f"/World/Drones/Drone_{d_idx:02d}"
        d_root = SingleXFormPrim(
            prim_path=drone_path,
            name=f"drone_root_{d_idx:02d}",
            position=np.array([dx, dy, dz]),
        )
        drone_roots.append(d_root)

        fuselage = VisualCuboid(
            prim_path=f"{drone_path}/Fuselage",
            name=f"drone_fuselage_{d_idx:02d}",
            translation=np.array([0.0, 0.0, 0.0]),
            scale=np.array([0.55, 0.55, 0.22]),
            size=1.0,
            color=np.array([0.15, 0.90, 0.35]),
        )
        drone_fuselages.append(fuselage)

        # 4 quadcopter rotor discs at corners
        for r_idx, (rx, ry) in enumerate(
            [(-0.45, -0.45), (0.45, -0.45), (0.45, 0.45), (-0.45, 0.45)]
        ):
            VisualCylinder(
                prim_path=f"{drone_path}/Rotor_{r_idx}",
                name=f"drone_{d_idx:02d}_rotor_{r_idx}",
                translation=np.array([rx, ry, 0.12]),
                radius=0.28,
                height=0.04,
                color=np.array([0.25, 0.28, 0.32]),
            )

        # Downward camera
        cam_path = f"{drone_path}/DownwardCamera"
        cam = Camera(
            prim_path=cam_path,
            position=np.array([dx, dy, dz - 0.18]),
            frequency=int(config.CAMERA_CAPTURE_HZ),
            resolution=(config.CAMERA_WIDTH, config.CAMERA_HEIGHT),
            orientation=down_quat,
        )
        cameras.append(cam)

        # Downward spotlight
        light_path = Sdf.Path(f"{drone_path}/Spotlight")
        s_light = UsdLux.SphereLight.Define(stage, light_path)
        s_light.CreateIntensityAttr(50000.0)
        s_light.CreateRadiusAttr(0.25)
        s_light.CreateColorAttr(Gf.Vec3f(0.1, 0.9, 0.2))
        shaping = UsdLux.ShapingAPI.Apply(s_light.GetPrim())
        shaping.CreateShapingConeAngleAttr(config.LIGHT_CONE_HALF_ANGLE_DEG)
        shaping.CreateShapingConeSoftnessAttr(0.25)
        spotlights[d_idx] = s_light

    world.reset()
    for cam in cameras:
        cam.initialize()

    viewports.set_camera_view(
        eye=np.array([-14.0, -16.0, 38.0]),
        target=np.array([42.0, 30.0, 2.0]),
    )
    usd_out_path = str(REPO_ROOT / "DroneIsaacSim.usd")
    try:
        stage.GetRootLayer().Export(usd_out_path)
    except Exception:
        pass

    zmq_bridge = SimZmqEndpoint()
    intrinsics = config.get_camera_intrinsics()
    rr_cursor = 0
    step_count = 0

    try:
        while simulation_app.is_running() and (args.steps <= 0 or step_count < args.steps):
            incoming_cmds = zmq_bridge.poll_commands()
            for cmd in incoming_cmds:
                dynamics.apply_command(cmd)

            dynamics.step(dt=0.1)

            # Update 3D human positions & walking yaw orientation
            for p_idx, p_root in enumerate(person_roots):
                px, py = dynamics.person_pos[p_idx]
                vx, vy = dynamics.person_vel[p_idx]
                if math.hypot(vx, vy) > 0.1:
                    yaw = math.degrees(math.atan2(vy, vx))
                    p_quat = rot_utils.euler_angles_to_quats(
                        np.array([0.0, 0.0, yaw]), degrees=True
                    )
                    p_root.set_world_pose(
                        position=np.array([px, py, 0.0]), orientation=p_quat
                    )
                else:
                    p_root.set_world_pose(position=np.array([px, py, 0.0]))

            # Update 3D quadcopter positions, cameras, and spotlight beams
            for d_idx, d_root in enumerate(drone_roots):
                dx, dy, dz = dynamics.drone_pos[d_idx]
                d_root.set_world_pose(position=np.array([dx, dy, dz]))
                cameras[d_idx].set_world_pose(
                    position=np.array([dx, dy, dz - 0.18]),
                    orientation=down_quat,
                )

                cmd = dynamics.drone_commands[d_idx]
                mod_intensity = dynamics.get_modulated_light_intensity(d_idx)
                s_light = spotlights[d_idx]
                s_light.GetColorAttr().Set(
                    Gf.Vec3f(
                        float(cmd.light_colour[0]),
                        float(cmd.light_colour[1]),
                        float(cmd.light_colour[2]),
                    )
                )
                s_light.GetIntensityAttr().Set(float(90000.0 * mod_intensity))

            world.step(render=True)

            zmq_bridge.publish_states(dynamics.get_drone_states())
            zmq_bridge.publish_ground_truth(
                dynamics.sim_time,
                [(float(p[0]), float(p[1])) for p in dynamics.person_pos],
            )

            n_d = args.drones
            batch_size = min(n_d, config.MAX_CAMERAS_PER_STEP)
            for k in range(batch_size):
                cam_id = (rr_cursor + k) % n_d
                rgba = cameras[cam_id].get_rgba()
                if rgba is not None and rgba.size > 0:
                    rgb = rgba[:, :, :3].astype(np.uint8)
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
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
