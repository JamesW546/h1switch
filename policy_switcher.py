"""Generic hot policy switcher for position-PD robot controllers."""

from __future__ import annotations

import argparse
import importlib
import select
import sys
import termios
import time
import tty
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


Array = np.ndarray


@dataclass
class RobotState:
    """Robot feedback shared by every policy adapter."""

    timestamp: float
    q: Array
    dq: Array
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self, motor_count: int) -> None:
        if self.q.shape != (motor_count,) or self.dq.shape != (motor_count,):
            raise ValueError(
                f"Expected q/dq shape ({motor_count},), got {self.q.shape}/{self.dq.shape}"
            )


@dataclass
class PolicyCommand:
    """Canonical motor command produced by every policy adapter."""

    q: Array
    kp: Array
    kd: Array
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, motor_count: int) -> None:
        expected = (motor_count,)
        if self.q.shape != expected or self.kp.shape != expected or self.kd.shape != expected:
            raise ValueError(
                f"Expected command arrays with shape {expected}, got "
                f"q={self.q.shape}, kp={self.kp.shape}, kd={self.kd.shape}"
            )
        if not all(np.all(np.isfinite(x)) for x in (self.q, self.kp, self.kd)):
            raise ValueError("Policy command contains NaN or Inf")


class PolicyAdapter(ABC):
    """Implement this interface for each policy observation/action contract."""

    def __init__(self, name: str, motor_count: int):
        self.name = name
        self.motor_count = int(motor_count)

    def reset(self, state: RobotState) -> None:
        """Reset recurrent state, action history, phase, or motion index."""

    @abstractmethod
    def step(self, state: RobotState, command_scale: float) -> PolicyCommand:
        """Infer once from live state and return full motor-space PD targets."""

    def ready(self, state: RobotState, command: PolicyCommand) -> bool:
        """Return False to delay a requested handoff until this policy is safe."""

        return True

    def accept_executed_command(self, command: PolicyCommand) -> None:
        """Receive the mixed command for last-action/history synchronization."""


class RobotBackend(ABC):
    """Owns robot communication. Exactly one backend writes motor commands."""

    def __init__(self, motor_count: int, control_dt: float):
        self.motor_count = int(motor_count)
        self.control_dt = float(control_dt)

    def connect(self) -> None:
        pass

    @abstractmethod
    def read_state(self) -> RobotState:
        pass

    @abstractmethod
    def write_command(self, command: PolicyCommand) -> None:
        pass

    def damping(self) -> None:
        pass

    def close(self) -> None:
        pass


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def blend_commands(first: PolicyCommand, second: PolicyCommand, second_weight: float) -> PolicyCommand:
    """Blend targets and gains in physical motor order."""

    weight = float(np.clip(second_weight, 0.0, 1.0))
    first_weight = 1.0 - weight
    return PolicyCommand(
        q=first.q * first_weight + second.q * weight,
        kp=first.kp * first_weight + second.kp * weight,
        kd=first.kd * first_weight + second.kd * weight,
        metadata={"blend_weight": weight},
    )


class CommandLimiter:
    def __init__(
        self,
        motor_count: int,
        q_lower: list[float] | None = None,
        q_upper: list[float] | None = None,
        max_target_delta: float | list[float] | None = None,
    ):
        self.motor_count = motor_count
        self.q_lower = self._array(q_lower, -np.inf)
        self.q_upper = self._array(q_upper, np.inf)
        self.max_delta = self._array(max_target_delta, np.inf)
        self.previous_q: Array | None = None

    def _array(self, value: Any, default: float) -> Array:
        if value is None:
            return np.full(self.motor_count, default, dtype=np.float32)
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 0:
            array = np.full(self.motor_count, float(array), dtype=np.float32)
        if array.shape != (self.motor_count,):
            raise ValueError(f"Safety array must have {self.motor_count} elements")
        return array

    def reset(self, current_q: Array) -> None:
        self.previous_q = np.asarray(current_q, dtype=np.float32).copy()

    def apply(self, command: PolicyCommand) -> PolicyCommand:
        q = np.clip(command.q, self.q_lower, self.q_upper)
        if self.previous_q is not None:
            q = np.clip(q, self.previous_q - self.max_delta, self.previous_q + self.max_delta)
        self.previous_q = q.copy()
        return PolicyCommand(q=q, kp=command.kp.copy(), kd=command.kd.copy(), metadata=dict(command.metadata))


class PolicySwitcher:
    """Runs two policies every tick and switches without stopping motor output."""

    def __init__(
        self,
        first: PolicyAdapter,
        second: PolicyAdapter,
        initial_policy: str,
        transition_seconds: float,
        control_dt: float,
        stable_seconds: float = 0.0,
        max_target_gap: float | None = None,
        limiter: CommandLimiter | None = None,
    ):
        if first.name == second.name:
            raise ValueError("Policy names must be different")
        if first.motor_count != second.motor_count:
            raise ValueError("Both policies must control the same motor count")
        self.policies = {first.name: first, second.name: second}
        if initial_policy not in self.policies:
            raise ValueError(f"Unknown initial policy: {initial_policy}")
        self.names = (first.name, second.name)
        self.motor_count = first.motor_count
        self.active = initial_policy
        self.transition_seconds = max(float(transition_seconds), 0.0)
        self.control_dt = float(control_dt)
        self.stable_seconds = max(float(stable_seconds), 0.0)
        self.max_target_gap = max_target_gap
        self.limiter = limiter
        self.pending: str | None = None
        self.ready_time = 0.0
        self.transition_source: str | None = None
        self.transition_target: str | None = None
        self.transition_elapsed = 0.0

    def reset(self, state: RobotState) -> None:
        state.validate(self.motor_count)
        for policy in self.policies.values():
            policy.reset(state)
        if self.limiter is not None:
            self.limiter.reset(state.q)

    def request_switch(self, target: str | None = None) -> bool:
        if self.transition_target is not None or self.pending is not None:
            return False
        if target is None:
            target = self.names[1] if self.active == self.names[0] else self.names[0]
        if target not in self.policies or target == self.active:
            return False
        self.pending = target
        self.ready_time = 0.0
        return True

    def _start_transition(self) -> None:
        self.transition_source = self.active
        self.transition_target = self.pending
        self.pending = None
        self.transition_elapsed = 0.0

    def _weights(self) -> dict[str, float]:
        if self.transition_target is None:
            return {name: float(name == self.active) for name in self.names}
        if self.transition_seconds == 0.0:
            weight = 1.0
        else:
            weight = smoothstep(self.transition_elapsed / self.transition_seconds)
        return {
            self.transition_source: 1.0 - weight,
            self.transition_target: weight,
        }

    def step(self, state: RobotState) -> tuple[PolicyCommand, dict[str, Any]]:
        state.validate(self.motor_count)
        weights = self._weights()
        commands = {
            name: self.policies[name].step(state, weights.get(name, 0.0))
            for name in self.names
        }
        for command in commands.values():
            command.validate(self.motor_count)

        if self.pending is not None:
            target_ready = self.policies[self.pending].ready(state, commands[self.pending])
            source_ready = self.policies[self.active].ready(state, commands[self.active])
            gap = float(np.max(np.abs(commands[self.active].q - commands[self.pending].q)))
            gap_ready = self.max_target_gap is None or gap <= self.max_target_gap
            if target_ready and source_ready and gap_ready:
                self.ready_time += self.control_dt
                if self.ready_time + 1e-9 >= self.stable_seconds:
                    self._start_transition()
                    weights = self._weights()
            else:
                self.ready_time = 0.0

        first_name, second_name = self.names
        mixed = blend_commands(commands[first_name], commands[second_name], weights[second_name])
        if self.limiter is not None:
            mixed = self.limiter.apply(mixed)
        mixed.validate(self.motor_count)

        for policy in self.policies.values():
            policy.accept_executed_command(mixed)

        event = None
        if self.transition_target is not None:
            transition_complete = (
                self.transition_seconds == 0.0
                or self.transition_elapsed + 1e-9 >= self.transition_seconds
            )
            if transition_complete:
                self.active = self.transition_target
                event = f"settled:{self.active}"
                self.transition_source = None
                self.transition_target = None
                self.transition_elapsed = 0.0
            else:
                self.transition_elapsed += self.control_dt

        status = {
            "active": self.active,
            "pending": self.pending,
            "transition_target": self.transition_target,
            "weights": weights,
            "event": event,
        }
        return mixed, status


class TerminalKeyWatcher:
    def __init__(self, key: str):
        if len(key) != 1:
            raise ValueError("switch_key must be one character")
        self.key = key.lower()
        self.old_settings: list[Any] | None = None
        self.enabled = False

    def __enter__(self) -> "TerminalKeyWatcher":
        if sys.stdin.isatty():
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            self.enabled = True
            print(f"Press '{self.key}' to switch policies. Ctrl+C to stop.")
        else:
            print("stdin is not a TTY; keyboard switching is disabled.")
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.enabled and self.old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def pressed(self) -> bool:
        hit = False
        while self.enabled and select.select([sys.stdin], [], [], 0.0)[0]:
            if sys.stdin.read(1).lower() == self.key:
                hit = True
        return hit


def load_class(path: str) -> type[Any]:
    module_name, separator, class_name = path.partition(":")
    if not separator:
        raise ValueError(f"Class path must use module:ClassName syntax: {path}")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def build_plugin(spec: dict[str, Any]) -> Any:
    plugin_class = load_class(spec["class"])
    return plugin_class(**spec.get("kwargs", {}))


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    return config


def run(config: dict[str, Any], max_seconds: float | None = None) -> None:
    backend: RobotBackend = build_plugin(config["backend"])
    policy_specs = config["policies"]
    if len(policy_specs) != 2:
        raise ValueError("Exactly two policies are required")
    policies = []
    for name, spec in policy_specs.items():
        kwargs = dict(spec.get("kwargs", {}))
        kwargs.setdefault("name", name)
        policy_class = load_class(spec["class"])
        policies.append(policy_class(**kwargs))

    transition = config.get("transition", {})
    safety = config.get("safety", {})
    limiter = CommandLimiter(
        backend.motor_count,
        q_lower=safety.get("q_lower"),
        q_upper=safety.get("q_upper"),
        max_target_delta=safety.get("max_target_delta"),
    )
    switcher = PolicySwitcher(
        policies[0],
        policies[1],
        initial_policy=config.get("initial_policy", policies[0].name),
        transition_seconds=transition.get("seconds", 2.0),
        control_dt=backend.control_dt,
        stable_seconds=transition.get("stable_seconds", 0.0),
        max_target_gap=transition.get("max_target_gap"),
        limiter=limiter,
    )

    switch_key = config.get("switch_key", "s")
    backend.connect()
    state = backend.read_state()
    switcher.reset(state)
    started = time.monotonic()
    next_tick = started
    last_event = None
    try:
        with TerminalKeyWatcher(switch_key) as watcher:
            while max_seconds is None or time.monotonic() - started < max_seconds:
                if watcher.pressed() and switcher.request_switch():
                    print("Switch requested.")
                state = backend.read_state()
                command, status = switcher.step(state)
                backend.write_command(command)
                if status["event"] and status["event"] != last_event:
                    print(status["event"])
                    last_event = status["event"]
                next_tick += backend.control_dt
                time.sleep(max(0.0, next_tick - time.monotonic()))
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        backend.damping()
        backend.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="YAML configuration file")
    parser.add_argument("--max-seconds", type=float, default=None, help="optional test duration")
    args = parser.parse_args()
    run(load_config(args.config), max_seconds=args.max_seconds)


if __name__ == "__main__":
    main()
