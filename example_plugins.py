"""Runnable mock plugins and a template for real policy adapters."""

from __future__ import annotations

import time

import numpy as np

from policy_switcher import PolicyAdapter, PolicyCommand, RobotBackend, RobotState


class MockBackend(RobotBackend):
    """Small first-order backend used to test switching without a robot."""

    def __init__(self, motor_count: int = 6, control_dt: float = 0.02):
        super().__init__(motor_count, control_dt)
        self.q = np.zeros(motor_count, dtype=np.float32)
        self.dq = np.zeros(motor_count, dtype=np.float32)

    def read_state(self) -> RobotState:
        return RobotState(time.monotonic(), self.q.copy(), self.dq.copy())

    def write_command(self, command: PolicyCommand) -> None:
        previous = self.q.copy()
        self.q += 0.08 * (command.q - self.q)
        self.dq = (self.q - previous) / self.control_dt

    def damping(self) -> None:
        print("Mock backend entered damping mode.")


class HoldPositionAdapter(PolicyAdapter):
    """Functional example. Replace inference() for a real policy."""

    def __init__(
        self,
        name: str,
        target: list[float],
        kp: float = 100.0,
        kd: float = 5.0,
    ):
        target_array = np.asarray(target, dtype=np.float32)
        super().__init__(name, len(target_array))
        self.target = target_array
        self.kp = np.full(self.motor_count, kp, dtype=np.float32)
        self.kd = np.full(self.motor_count, kd, dtype=np.float32)
        self.last_executed = self.target.copy()

    def step(self, state: RobotState, command_scale: float) -> PolicyCommand:
        # A real adapter builds its own observation here, runs its model, maps
        # actions into physical motor order, and returns full PD targets.
        target = state.q * (1.0 - command_scale) + self.target * command_scale
        return PolicyCommand(target.astype(np.float32), self.kp.copy(), self.kd.copy())

    def accept_executed_command(self, command: PolicyCommand) -> None:
        # Use this to synchronize a policy's last-action observation while it is
        # inactive or partially blended.
        self.last_executed = command.q.copy()
