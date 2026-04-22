"""Tests for Physical AI occupancy helpers."""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pytest

from gs_sim2real.sim import (
    AgentAction,
    HeadlessPhysicalAIEnvironment,
    Observation,
    Pose3D,
    VoxelOccupancyGrid,
    build_occupancy_grid_from_lidar_observation,
    build_simulation_catalog,
    decode_lidar_points_observation,
    point_to_voxel_cell,
)


def test_decode_lidar_observation_and_build_voxel_occupancy_grid() -> None:
    points = np.array(
        [
            [0.05, 0.05, 0.05],
            [0.55, 0.05, 0.05],
        ],
        dtype="<f4",
    )
    observation = lidar_observation(points)

    decoded = decode_lidar_points_observation(observation)
    grid = build_occupancy_grid_from_lidar_observation(
        observation,
        voxel_size_meters=0.5,
        inflation_radius_meters=0.0,
    )

    np.testing.assert_allclose(decoded, points)
    assert grid.source == "lidar-ray-proxy:unit-scene"
    assert grid.point_count == 2
    assert grid.cell_count == 2
    assert grid.contains_point((0.05, 0.05, 0.05))
    assert grid.contains_point((0.55, 0.05, 0.05))
    assert not grid.contains_point((0.95, 0.95, 0.95))
    assert point_to_voxel_cell((0.55, 0.05, 0.05), 0.5) == (1, 0, 0)


def test_voxel_occupancy_grid_can_inflate_cells() -> None:
    grid = VoxelOccupancyGrid.from_points(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        voxel_size_meters=0.5,
        inflation_radius_meters=0.5,
        source="unit",
    )

    assert grid.contains_point((0.0, 0.0, 0.0))
    assert grid.contains_point((0.55, 0.0, 0.0))
    assert grid.cell_count > 1


def test_headless_environment_uses_occupancy_grid_for_collision() -> None:
    env = HeadlessPhysicalAIEnvironment(build_unit_catalog())
    env.reset("unit-scene")
    grid = VoxelOccupancyGrid.from_points(
        np.array([[0.75, 0.0, 0.0]], dtype=np.float32),
        voxel_size_meters=0.25,
        source="unit-grid",
    )
    env.set_occupancy_grid(grid)

    free_pose = Pose3D(position=(0.0, 0.0, 0.0), orientation_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="generic_world")
    occupied_pose = Pose3D(
        position=(0.75, 0.0, 0.0),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        frame_id="generic_world",
    )

    free = env.query_collision(free_pose)
    occupied = env.query_collision(occupied_pose)

    assert free.collides is False
    assert free.reason == "free-voxel:unit-grid"
    assert occupied.collides is True
    assert occupied.reason == "occupied-voxel:unit-grid"

    blocked = env.step(AgentAction("teleport", {"x": 0.75, "y": 0.0, "z": 0.0}))
    assert blocked["applied"] is False
    assert blocked["collision"]["reason"] == "occupied-voxel:unit-grid"
    assert env.state.pose.position == free_pose.position

    env.set_occupancy_grid(None)
    cleared = env.query_collision(occupied_pose)
    assert cleared.collides is False
    assert cleared.reason == "inside-bounds"


def test_decode_lidar_observation_rejects_malformed_payloads() -> None:
    pose = Pose3D(position=(0.0, 0.0, 0.0), orientation_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="generic_world")
    with pytest.raises(ValueError, match="sensor_id"):
        decode_lidar_points_observation(Observation(sensor_id="rgb-forward", pose=pose, outputs={}))

    bad_bytes_payload = Observation(
        sensor_id="lidar-ray-proxy",
        pose=pose,
        outputs={"points": {"pointsBase64": base64.b64encode(b"\x00").decode("ascii")}},
    )
    with pytest.raises(ValueError, match="float32 width"):
        decode_lidar_points_observation(bad_bytes_payload)

    bad_shape_payload = Observation(
        sensor_id="lidar-ray-proxy",
        pose=pose,
        outputs={"points": {"pointsBase64": base64.b64encode(np.array([1.0], dtype="<f4").tobytes()).decode("ascii")}},
    )
    with pytest.raises(ValueError, match="divisible by 3"):
        decode_lidar_points_observation(bad_shape_payload)


def build_unit_catalog():
    return build_simulation_catalog(
        {
            "scenes": [
                {
                    "url": "assets/unit-scene/unit-scene.splat",
                    "label": "Unit Scene",
                    "summary": "Generic unit scene",
                }
            ]
        },
        docs_root=Path("."),
        site_url="https://example.test/gs/",
    )


def lidar_observation(points: np.ndarray) -> Observation:
    return Observation(
        sensor_id="lidar-ray-proxy",
        pose=Pose3D(position=(0.0, 0.0, 0.0), orientation_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="generic_world"),
        outputs={
            "sceneId": "unit-scene",
            "points": {
                "encoding": "float32-le-xyz",
                "pointsBase64": base64.b64encode(np.asarray(points, dtype="<f4").tobytes()).decode("ascii"),
            },
        },
    )
