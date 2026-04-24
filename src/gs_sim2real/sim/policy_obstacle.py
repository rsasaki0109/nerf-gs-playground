"""ObstaclePolicy protocol for agent-driven dynamic obstacles.

The first-generation :class:`DynamicObstacle` interpolates between
hard-coded waypoints, and the second generation grew two reactive
primitives (``chase_target_agent`` / ``flee_from_agent``) that are pure
functions of the queried agent position. This module introduces the
third generation: every obstacle can carry an opt-in
:class:`ObstaclePolicy` callable that picks the next world-frame
position from a small observation context, including the latest known
positions of peer obstacles. Multi-obstacle scenarios can therefore
express coordinated behaviour (e.g. maintaining separation, following
the agent in formation) without bolting more flags onto the obstacle
dataclass.

Design notes:

* ``ObstaclePolicy`` is a pure callable. Stateful policies are still
  expressible — the policy object can hold its own state — but the
  interface itself is stateless from the caller's perspective so
  scenario replays stay deterministic.
* Policies are runtime-only on :class:`DynamicObstacle`. They are
  intentionally not part of the JSON serialisation surface in this
  release: the existing ``chase_target_agent`` / ``flee_from_agent`` /
  waypoint fields stay the source of truth on disk so v1 scenario CI
  artifacts keep loading unchanged. Code-level callers attach a policy
  after construction (or via :func:`default_obstacle_policy`).
* ``peer_positions`` in :class:`ObstaclePolicyContext` carries the
  *previous*-step positions of sibling obstacles. Resolving against the
  previous step avoids dependency cycles when several obstacles consult
  each other.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ObstaclePolicyContext:
    """Inputs an :class:`ObstaclePolicy` may consult to choose its next position.

    ``current_position`` is what the obstacle would have produced under
    its default (waypoint / chase / flee) behaviour at ``step_index`` —
    policies can use it as a starting point and nudge from there, or
    ignore it entirely. ``agent_position`` is the world-frame agent
    position consulted by the collision query (``None`` when the env
    has not provided one — e.g. headless renderer warm-ups).
    ``peer_positions`` maps peer obstacle ids to their resolved
    positions at the previous step.
    """

    obstacle_id: str
    step_index: int
    current_position: tuple[float, float, float]
    agent_position: tuple[float, float, float] | None = None
    peer_positions: Mapping[str, tuple[float, float, float]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ObstaclePolicyDecision:
    """Output of an :class:`ObstaclePolicy` call.

    ``next_position`` is the obstacle's chosen world-frame position
    after the current step. Headless collision queries treat this as
    the centre of the obstacle sphere.
    """

    next_position: tuple[float, float, float]


@runtime_checkable
class ObstaclePolicy(Protocol):
    """Callable that maps an observation context to a next-step position."""

    def __call__(self, context: ObstaclePolicyContext) -> ObstaclePolicyDecision: ...


class WaypointInterpolationObstaclePolicy:
    """Replays the default waypoint linear-interpolation behaviour.

    Constructed from the obstacle's own ``waypoints`` tuple. Useful as a
    base layer when wrapping with a multi-agent modifier such as
    :class:`MaintainSeparationObstaclePolicy`.
    """

    __slots__ = ("_waypoints",)

    def __init__(self, waypoints: Sequence[tuple[int, tuple[float, float, float]]]) -> None:
        cleaned: list[tuple[int, tuple[float, float, float]]] = []
        for waypoint in waypoints:
            step_index, position = waypoint
            if int(step_index) < 0:
                raise ValueError("waypoint step_index must be non-negative")
            if len(position) != 3:
                raise ValueError("waypoint position must be a 3-tuple of floats")
            cleaned.append((int(step_index), tuple(float(c) for c in position)))
        if not cleaned:
            raise ValueError("waypoint policy requires at least one waypoint")
        cleaned.sort(key=lambda item: item[0])
        if len({step for step, _ in cleaned}) != len(cleaned):
            raise ValueError("waypoint step indices must be unique")
        self._waypoints: tuple[tuple[int, tuple[float, float, float]], ...] = tuple(cleaned)

    def __call__(self, context: ObstaclePolicyContext) -> ObstaclePolicyDecision:
        step_index = int(context.step_index)
        waypoints = self._waypoints
        if step_index <= waypoints[0][0]:
            return ObstaclePolicyDecision(next_position=waypoints[0][1])
        if step_index >= waypoints[-1][0]:
            return ObstaclePolicyDecision(next_position=waypoints[-1][1])
        for previous, current in zip(waypoints, waypoints[1:]):
            if previous[0] <= step_index <= current[0]:
                span = current[0] - previous[0]
                if span <= 0:
                    return ObstaclePolicyDecision(next_position=current[1])
                alpha = (step_index - previous[0]) / span
                interpolated = (
                    previous[1][0] + alpha * (current[1][0] - previous[1][0]),
                    previous[1][1] + alpha * (current[1][1] - previous[1][1]),
                    previous[1][2] + alpha * (current[1][2] - previous[1][2]),
                )
                return ObstaclePolicyDecision(next_position=interpolated)
        return ObstaclePolicyDecision(next_position=waypoints[-1][1])


class ChaseAgentObstaclePolicy:
    """Walks from ``start_position`` toward the agent at a capped speed.

    Matches :meth:`DynamicObstacle._chase_position`. When
    ``agent_position`` is ``None`` the policy returns
    ``start_position``, mirroring the existing fallback used by
    headless renderer warm-ups.
    """

    __slots__ = ("_start", "_speed_m_per_step")

    def __init__(
        self,
        start_position: tuple[float, float, float],
        speed_m_per_step: float,
    ) -> None:
        if len(start_position) != 3:
            raise ValueError("start_position must be a 3-tuple of floats")
        speed = float(speed_m_per_step)
        if not math.isfinite(speed) or speed < 0.0:
            raise ValueError("speed_m_per_step must be non-negative and finite")
        self._start: tuple[float, float, float] = tuple(float(c) for c in start_position)
        self._speed_m_per_step = speed

    def __call__(self, context: ObstaclePolicyContext) -> ObstaclePolicyDecision:
        if context.agent_position is None:
            return ObstaclePolicyDecision(next_position=self._start)
        target = tuple(float(c) for c in context.agent_position)
        delta_x = target[0] - self._start[0]
        delta_y = target[1] - self._start[1]
        delta_z = target[2] - self._start[2]
        gap = math.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z)
        if gap <= 0.0:
            return ObstaclePolicyDecision(next_position=self._start)
        max_travel = max(0.0, float(context.step_index)) * self._speed_m_per_step
        distance = min(gap, max_travel)
        fraction = distance / gap
        return ObstaclePolicyDecision(
            next_position=(
                self._start[0] + fraction * delta_x,
                self._start[1] + fraction * delta_y,
                self._start[2] + fraction * delta_z,
            )
        )


class FleeAgentObstaclePolicy:
    """Walks away from the agent at a fixed speed, no upper-bound clamp.

    Matches :meth:`DynamicObstacle._flee_position`. The flee direction
    is ``start_position - agent_position``; the obstacle keeps
    retreating without bound.
    """

    __slots__ = ("_start", "_speed_m_per_step")

    def __init__(
        self,
        start_position: tuple[float, float, float],
        speed_m_per_step: float,
    ) -> None:
        if len(start_position) != 3:
            raise ValueError("start_position must be a 3-tuple of floats")
        speed = float(speed_m_per_step)
        if not math.isfinite(speed) or speed < 0.0:
            raise ValueError("speed_m_per_step must be non-negative and finite")
        self._start: tuple[float, float, float] = tuple(float(c) for c in start_position)
        self._speed_m_per_step = speed

    def __call__(self, context: ObstaclePolicyContext) -> ObstaclePolicyDecision:
        if context.agent_position is None:
            return ObstaclePolicyDecision(next_position=self._start)
        origin = tuple(float(c) for c in context.agent_position)
        delta_x = self._start[0] - origin[0]
        delta_y = self._start[1] - origin[1]
        delta_z = self._start[2] - origin[2]
        gap = math.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z)
        if gap <= 0.0:
            return ObstaclePolicyDecision(next_position=self._start)
        distance = max(0.0, float(context.step_index)) * self._speed_m_per_step
        fraction = distance / gap
        return ObstaclePolicyDecision(
            next_position=(
                self._start[0] + fraction * delta_x,
                self._start[1] + fraction * delta_y,
                self._start[2] + fraction * delta_z,
            )
        )


class MaintainSeparationObstaclePolicy:
    """Wraps an inner policy and pushes the result away from peers.

    For each peer in ``context.peer_positions`` whose distance from the
    inner policy's chosen position is less than
    ``min_separation_meters``, the obstacle is nudged outward along the
    ``peer → me`` direction by the missing distance. Multiple peer
    constraints are applied in id-sorted order so the result is
    deterministic. Peers without a recorded position are ignored.
    """

    __slots__ = ("_inner", "_min_separation")

    def __init__(self, inner_policy: ObstaclePolicy, min_separation_meters: float) -> None:
        if not callable(inner_policy):
            raise TypeError("inner_policy must be callable")
        separation = float(min_separation_meters)
        if not math.isfinite(separation) or separation < 0.0:
            raise ValueError("min_separation_meters must be non-negative and finite")
        self._inner = inner_policy
        self._min_separation = separation

    def __call__(self, context: ObstaclePolicyContext) -> ObstaclePolicyDecision:
        decision = self._inner(context)
        if self._min_separation <= 0.0 or not context.peer_positions:
            return decision
        position = decision.next_position
        for peer_id in sorted(context.peer_positions):
            peer = context.peer_positions[peer_id]
            delta_x = position[0] - peer[0]
            delta_y = position[1] - peer[1]
            delta_z = position[2] - peer[2]
            distance = math.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z)
            if distance >= self._min_separation:
                continue
            if distance <= 0.0:
                # Degenerate: peer overlaps me. Push along +X by the full
                # separation so we leave the singularity deterministically.
                position = (
                    position[0] + self._min_separation,
                    position[1],
                    position[2],
                )
                continue
            push = (self._min_separation - distance) / distance
            position = (
                position[0] + delta_x * push,
                position[1] + delta_y * push,
                position[2] + delta_z * push,
            )
        return ObstaclePolicyDecision(next_position=position)
