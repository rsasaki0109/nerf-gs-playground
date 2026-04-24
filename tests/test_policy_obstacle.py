"""Tests for the ObstaclePolicy protocol and reference implementations."""

from __future__ import annotations

import math

import pytest

from gs_sim2real.sim import (
    ChaseAgentObstaclePolicy,
    DynamicObstacle,
    DynamicObstacleTimeline,
    DynamicObstacleWaypoint,
    FleeAgentObstaclePolicy,
    MaintainSeparationObstaclePolicy,
    ObstaclePolicy,
    ObstaclePolicyContext,
    ObstaclePolicyDecision,
    WaypointInterpolationObstaclePolicy,
)


def _waypoint_obstacle(obstacle_id: str = "wp", radius: float = 0.5) -> DynamicObstacle:
    return DynamicObstacle(
        obstacle_id=obstacle_id,
        waypoints=(
            DynamicObstacleWaypoint(step_index=0, position=(0.0, 0.0, 0.0)),
            DynamicObstacleWaypoint(step_index=10, position=(10.0, 0.0, 0.0)),
        ),
        radius_meters=radius,
    )


def test_obstacle_policy_protocol_is_runtime_checkable() -> None:
    waypoints = (
        DynamicObstacleWaypoint(step_index=0, position=(0.0, 0.0, 0.0)),
        DynamicObstacleWaypoint(step_index=4, position=(2.0, 0.0, 0.0)),
    )
    policy = WaypointInterpolationObstaclePolicy(((wp.step_index, wp.position) for wp in waypoints))
    assert isinstance(policy, ObstaclePolicy)
    assert isinstance(ChaseAgentObstaclePolicy((0.0, 0.0, 0.0), 1.0), ObstaclePolicy)
    assert isinstance(FleeAgentObstaclePolicy((0.0, 0.0, 0.0), 1.0), ObstaclePolicy)
    assert isinstance(
        MaintainSeparationObstaclePolicy(ChaseAgentObstaclePolicy((0.0, 0.0, 0.0), 1.0), 1.0),
        ObstaclePolicy,
    )


def test_waypoint_policy_matches_existing_inline_interpolation() -> None:
    obstacle = _waypoint_obstacle()
    policy = WaypointInterpolationObstaclePolicy(tuple((wp.step_index, wp.position) for wp in obstacle.waypoints))
    for step in (0, 1, 5, 10, 12):
        baseline = obstacle.position_at_step(step)
        decision = policy(
            ObstaclePolicyContext(
                obstacle_id=obstacle.obstacle_id,
                step_index=step,
                current_position=baseline,
            )
        )
        assert decision.next_position == pytest.approx(baseline)


def test_chase_policy_matches_existing_chase_inline_logic() -> None:
    chase_obstacle = DynamicObstacle(
        obstacle_id="chase",
        waypoints=(DynamicObstacleWaypoint(step_index=0, position=(0.0, 0.0, 0.0)),),
        radius_meters=0.4,
        chase_target_agent=True,
        chase_speed_m_per_step=0.3,
    )
    policy = ChaseAgentObstaclePolicy(start_position=(0.0, 0.0, 0.0), speed_m_per_step=0.3)
    agent = (5.0, 0.0, 0.0)
    for step in (0, 3, 100):
        baseline = chase_obstacle.position_at_step(step, agent_position=agent)
        decision = policy(
            ObstaclePolicyContext(
                obstacle_id="chase",
                step_index=step,
                current_position=baseline,
                agent_position=agent,
            )
        )
        assert decision.next_position == pytest.approx(baseline)


def test_flee_policy_matches_existing_flee_inline_logic() -> None:
    flee_obstacle = DynamicObstacle(
        obstacle_id="flee",
        waypoints=(DynamicObstacleWaypoint(step_index=0, position=(2.0, 0.0, 0.0)),),
        radius_meters=0.4,
        flee_from_agent=True,
        chase_speed_m_per_step=0.5,
    )
    policy = FleeAgentObstaclePolicy(start_position=(2.0, 0.0, 0.0), speed_m_per_step=0.5)
    agent = (0.0, 0.0, 0.0)
    for step in (0, 2, 9):
        baseline = flee_obstacle.position_at_step(step, agent_position=agent)
        decision = policy(
            ObstaclePolicyContext(
                obstacle_id="flee",
                step_index=step,
                current_position=baseline,
                agent_position=agent,
            )
        )
        assert decision.next_position == pytest.approx(baseline)


def test_dynamic_obstacle_policy_short_circuits_default_behavior() -> None:
    obstacle = _waypoint_obstacle()

    class _ConstantPolicy:
        def __call__(self, context: ObstaclePolicyContext) -> ObstaclePolicyDecision:
            return ObstaclePolicyDecision(next_position=(7.0, 8.0, 9.0))

    policy_obstacle = DynamicObstacle(
        obstacle_id=obstacle.obstacle_id,
        waypoints=obstacle.waypoints,
        radius_meters=obstacle.radius_meters,
        policy=_ConstantPolicy(),
    )

    # Without policy: linear interpolation between (0,0,0) and (10,0,0).
    assert obstacle.position_at_step(5) == pytest.approx((5.0, 0.0, 0.0))
    # With policy: constant override regardless of step / waypoints.
    assert policy_obstacle.position_at_step(5) == pytest.approx((7.0, 8.0, 9.0))
    assert policy_obstacle.position_at_step(0) == pytest.approx((7.0, 8.0, 9.0))


def test_dynamic_obstacle_policy_receives_agent_and_peer_positions() -> None:
    captured: list[ObstaclePolicyContext] = []

    class _RecorderPolicy:
        def __call__(self, context: ObstaclePolicyContext) -> ObstaclePolicyDecision:
            captured.append(context)
            return ObstaclePolicyDecision(next_position=context.current_position)

    obstacle = DynamicObstacle(
        obstacle_id="rec",
        waypoints=(DynamicObstacleWaypoint(step_index=0, position=(1.0, 2.0, 3.0)),),
        radius_meters=0.5,
        policy=_RecorderPolicy(),
    )

    obstacle.position_at_step(
        4,
        agent_position=(10.0, 11.0, 12.0),
        peer_positions={"peer-a": (4.0, 5.0, 6.0)},
    )

    assert len(captured) == 1
    context = captured[0]
    assert context.obstacle_id == "rec"
    assert context.step_index == 4
    assert context.current_position == pytest.approx((1.0, 2.0, 3.0))
    assert context.agent_position == pytest.approx((10.0, 11.0, 12.0))
    assert context.peer_positions == {"peer-a": (4.0, 5.0, 6.0)}


def test_maintain_separation_pushes_obstacle_away_from_close_peer() -> None:
    inner = WaypointInterpolationObstaclePolicy(((0, (0.0, 0.0, 0.0)),))
    policy = MaintainSeparationObstaclePolicy(inner, min_separation_meters=2.0)
    decision = policy(
        ObstaclePolicyContext(
            obstacle_id="me",
            step_index=0,
            current_position=(0.0, 0.0, 0.0),
            peer_positions={"peer": (1.0, 0.0, 0.0)},
        )
    )
    distance = math.dist(decision.next_position, (1.0, 0.0, 0.0))
    assert distance == pytest.approx(2.0)
    # Pushed along -X (peer is at +1 X relative to me, so I move to -1 X).
    assert decision.next_position[0] == pytest.approx(-1.0)


def test_maintain_separation_is_a_noop_when_peer_is_far_enough() -> None:
    inner = WaypointInterpolationObstaclePolicy(((0, (0.0, 0.0, 0.0)),))
    policy = MaintainSeparationObstaclePolicy(inner, min_separation_meters=1.0)
    decision = policy(
        ObstaclePolicyContext(
            obstacle_id="me",
            step_index=0,
            current_position=(0.0, 0.0, 0.0),
            peer_positions={"peer": (5.0, 0.0, 0.0)},
        )
    )
    assert decision.next_position == pytest.approx((0.0, 0.0, 0.0))


def test_maintain_separation_handles_overlap_singularity_deterministically() -> None:
    inner = WaypointInterpolationObstaclePolicy(((0, (0.0, 0.0, 0.0)),))
    policy = MaintainSeparationObstaclePolicy(inner, min_separation_meters=2.0)
    decision = policy(
        ObstaclePolicyContext(
            obstacle_id="me",
            step_index=0,
            current_position=(0.0, 0.0, 0.0),
            peer_positions={"peer": (0.0, 0.0, 0.0)},
        )
    )
    # Degenerate-overlap escape pushes along +X by full separation.
    assert decision.next_position == pytest.approx((2.0, 0.0, 0.0))


def test_timeline_step_positions_threads_previous_peer_state() -> None:
    inner = WaypointInterpolationObstaclePolicy(((0, (0.0, 0.0, 0.0)),))
    obstacle_a = DynamicObstacle(
        obstacle_id="a",
        waypoints=(DynamicObstacleWaypoint(step_index=0, position=(0.0, 0.0, 0.0)),),
        radius_meters=0.5,
        policy=MaintainSeparationObstaclePolicy(inner, min_separation_meters=2.0),
    )
    obstacle_b = DynamicObstacle(
        obstacle_id="b",
        waypoints=(DynamicObstacleWaypoint(step_index=0, position=(1.0, 0.0, 0.0)),),
        radius_meters=0.5,
    )
    timeline = DynamicObstacleTimeline(timeline_id="pair", obstacles=(obstacle_a, obstacle_b))

    initial = timeline.step_positions(0)
    assert initial["a"] == pytest.approx((0.0, 0.0, 0.0))  # no peers known yet
    assert initial["b"] == pytest.approx((1.0, 0.0, 0.0))

    next_step = timeline.step_positions(1, previous_positions=initial)
    # A consults B's previous-step position (1, 0, 0) and pushes itself away.
    assert math.dist(next_step["a"], initial["b"]) == pytest.approx(2.0)
    assert next_step["b"] == pytest.approx((1.0, 0.0, 0.0))


def test_timeline_blocking_obstacle_forwards_peer_positions() -> None:
    captured: list[ObstaclePolicyContext] = []

    class _RecorderPolicy:
        def __call__(self, context: ObstaclePolicyContext) -> ObstaclePolicyDecision:
            captured.append(context)
            return ObstaclePolicyDecision(next_position=(0.0, 0.0, 0.0))

    obstacle = DynamicObstacle(
        obstacle_id="me",
        waypoints=(DynamicObstacleWaypoint(step_index=0, position=(0.0, 0.0, 0.0)),),
        radius_meters=1.0,
        policy=_RecorderPolicy(),
    )
    other = DynamicObstacle(
        obstacle_id="peer",
        waypoints=(DynamicObstacleWaypoint(step_index=0, position=(5.0, 0.0, 0.0)),),
        radius_meters=1.0,
    )
    timeline = DynamicObstacleTimeline(timeline_id="t", obstacles=(obstacle, other))

    timeline.blocking_obstacle(
        (0.0, 0.0, 0.0),
        0,
        peer_positions={"me": (0.0, 0.0, 0.0), "peer": (5.0, 0.0, 0.0)},
    )

    # First call (for "me") should NOT include itself in peers.
    assert captured[0].obstacle_id == "me"
    assert "me" not in captured[0].peer_positions
    assert captured[0].peer_positions == {"peer": (5.0, 0.0, 0.0)}


def test_dynamic_obstacle_to_dict_omits_runtime_policy() -> None:
    class _ConstantPolicy:
        def __call__(self, context: ObstaclePolicyContext) -> ObstaclePolicyDecision:
            return ObstaclePolicyDecision(next_position=(0.0, 0.0, 0.0))

    obstacle = DynamicObstacle(
        obstacle_id="serial",
        waypoints=(DynamicObstacleWaypoint(step_index=0, position=(0.0, 0.0, 0.0)),),
        radius_meters=0.5,
        policy=_ConstantPolicy(),
    )
    payload = obstacle.to_dict()
    assert "policy" not in payload


def test_invalid_separation_and_speed_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        ChaseAgentObstaclePolicy((0.0, 0.0, 0.0), -1.0)
    with pytest.raises(ValueError):
        FleeAgentObstaclePolicy((0.0, 0.0, 0.0), float("nan"))
    with pytest.raises(ValueError):
        MaintainSeparationObstaclePolicy(ChaseAgentObstaclePolicy((0.0, 0.0, 0.0), 0.0), -0.1)
    with pytest.raises(TypeError):
        MaintainSeparationObstaclePolicy("not-callable", 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        WaypointInterpolationObstaclePolicy(())
