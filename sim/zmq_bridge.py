"""ZeroMQ communication bridge between the Simulator process and the Agent/Perception process.

Topics & Sockets:
- Frame Publisher (Sim -> Agent): PUB on `ZMQ_FRAME_PORT` (topic `frame`)
- State Publisher (Sim -> Agent): PUB on `ZMQ_STATE_PORT` (topic `state`)
- Command Channel (Agent -> Sim): PUSH on Agent / PULL on Sim at `ZMQ_CMD_PORT`
- Ground-Truth Publisher (Sim -> Evaluation ONLY): PUB on `ZMQ_GT_PORT` (topic `gt`)
  NOTE: The agent process NEVER subscribes to `ZMQ_GT_PORT`.

Inputs:
    `FrameMessage`, `DroneState`, `DroneCommand` instances.
Outputs:
    Non-blocking ZeroMQ message transmission and reception.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import zmq

import config
from interfaces.contract import DroneCommand, DroneState, FrameMessage


class SimZmqEndpoint:
    """ZeroMQ sockets hosted by the simulator process (Isaac Sim or Mock Sim)."""

    def __init__(
        self,
        host: str = config.ZMQ_HOST,
        frame_port: int = config.ZMQ_FRAME_PORT,
        state_port: int = config.ZMQ_STATE_PORT,
        cmd_port: int = config.ZMQ_CMD_PORT,
        gt_port: int = config.ZMQ_GT_PORT,
    ) -> None:
        self.ctx = zmq.Context.instance()

        self.frame_pub = self.ctx.socket(zmq.PUB)
        self.frame_pub.setsockopt(zmq.SNDHWM, 50)
        self.frame_pub.bind(f"tcp://{host}:{frame_port}")

        self.state_pub = self.ctx.socket(zmq.PUB)
        self.state_pub.setsockopt(zmq.SNDHWM, 50)
        self.state_pub.bind(f"tcp://{host}:{state_port}")

        self.cmd_pull = self.ctx.socket(zmq.PULL)
        self.cmd_pull.setsockopt(zmq.RCVHWM, 100)
        self.cmd_pull.bind(f"tcp://{host}:{cmd_port}")

        self.gt_pub = self.ctx.socket(zmq.PUB)
        self.gt_pub.setsockopt(zmq.SNDHWM, 10)
        self.gt_pub.bind(f"tcp://{host}:{gt_port}")

    def publish_frame(self, frame_msg: FrameMessage) -> None:
        payload = json.dumps(frame_msg.to_dict()).encode("utf-8")
        self.frame_pub.send_multipart([b"frame", payload], flags=zmq.NOBLOCK)

    def publish_states(self, states: List[DroneState]) -> None:
        payload = json.dumps([s.to_dict() for s in states]).encode("utf-8")
        self.state_pub.send_multipart([b"state", payload], flags=zmq.NOBLOCK)

    def publish_ground_truth(self, sim_time: float, person_positions: List[Tuple[float, float]]) -> None:
        """Publish ground truth on separate port for offline evaluation ONLY."""
        payload = json.dumps(
            {"timestamp": float(sim_time), "positions_xy": person_positions}
        ).encode("utf-8")
        self.gt_pub.send_multipart([b"gt", payload], flags=zmq.NOBLOCK)

    def poll_commands(self) -> List[DroneCommand]:
        commands: List[DroneCommand] = []
        while True:
            try:
                raw = self.cmd_pull.recv(flags=zmq.NOBLOCK)
                data = json.loads(raw.decode("utf-8"))
                if isinstance(data, list):
                    commands.extend(DroneCommand.from_dict(d) for d in data)
                else:
                    commands.append(DroneCommand.from_dict(data))
            except zmq.Again:
                break
        return commands

    def close(self) -> None:
        for sock in (self.frame_pub, self.state_pub, self.cmd_pull, self.gt_pub):
            sock.close(0)


class AgentZmqEndpoint:
    """ZeroMQ sockets used by the autonomous agent & perception process.

    Note: Never connects to `ZMQ_GT_PORT` (ground-truth isolation hard rule).
    """

    def __init__(
        self,
        host: str = config.ZMQ_HOST,
        frame_port: int = config.ZMQ_FRAME_PORT,
        state_port: int = config.ZMQ_STATE_PORT,
        cmd_port: int = config.ZMQ_CMD_PORT,
    ) -> None:
        self.ctx = zmq.Context.instance()

        self.frame_sub = self.ctx.socket(zmq.SUB)
        self.frame_sub.setsockopt(zmq.RCVHWM, 50)
        self.frame_sub.connect(f"tcp://{host}:{frame_port}")
        self.frame_sub.setsockopt(zmq.SUBSCRIBE, b"frame")

        self.state_sub = self.ctx.socket(zmq.SUB)
        self.state_sub.setsockopt(zmq.RCVHWM, 50)
        self.state_sub.connect(f"tcp://{host}:{state_port}")
        self.state_sub.setsockopt(zmq.SUBSCRIBE, b"state")

        self.cmd_push = self.ctx.socket(zmq.PUSH)
        self.cmd_push.setsockopt(zmq.SNDHWM, 100)
        self.cmd_push.connect(f"tcp://{host}:{cmd_port}")

    def poll_frames(self, max_frames: int = 25) -> List[FrameMessage]:
        frames: List[FrameMessage] = []
        for _ in range(max_frames):
            try:
                _, payload = self.frame_sub.recv_multipart(flags=zmq.NOBLOCK)
                frames.append(FrameMessage.from_dict(json.loads(payload.decode("utf-8"))))
            except zmq.Again:
                break
        return frames

    def poll_states(self) -> Dict[int, DroneState]:
        latest: Dict[int, DroneState] = {}
        while True:
            try:
                _, payload = self.state_sub.recv_multipart(flags=zmq.NOBLOCK)
                items = json.loads(payload.decode("utf-8"))
                for d in items:
                    st = DroneState.from_dict(d)
                    latest[st.drone_id] = st
            except zmq.Again:
                break
        return latest

    def send_commands(self, commands: List[DroneCommand]) -> None:
        payload = json.dumps([c.to_dict() for c in commands]).encode("utf-8")
        try:
            self.cmd_push.send(payload, flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

    def close(self) -> None:
        for sock in (self.frame_sub, self.state_sub, self.cmd_push):
            sock.close(0)
