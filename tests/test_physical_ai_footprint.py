"""Tests for Physical AI robot footprint collision checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gs_sim2real.sim import (
    AgentAction,
    HeadlessPhysicalAIEnvironment,
    Pose3D,
    RobotFootprint,
    VoxelOccupancyGrid,
    build_simulation_catalog,
)


def test_robot_footprint_samples_horizontal_radius_and_height() -> None:
    footprint = RobotFootprint(radius_meters=0.25, height_meters=0.5)
    pose = Pose3D(position=(0.0, 0.0, 0.0), orientation_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="generic_world")

    cells = set(footprint.cells_for_pose(pose, voxel_size_meters=0.25))

    assert (0, 0, 0) in cells
    assert (1, 0, 0) in cells
    assert (0, 0, 1) in cells
    assert (0, 2, 0) in cells
    assert (0, -1, 0) not in cells


def test_voxel_occupancy_grid_queries_multiple_footprint_cells() -> None:
    grid = VoxelOccupancyGrid.from_points(
        np.array([[0.5, 0.0, 0.0]], dtype=np.float32),
        voxel_size_meters=0.25,
        source="unit-grid",
    )

    free_center = grid.query_pose(
        Pose3D(position=(0.0, 0.0, 0.0), orientation_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="generic_world")
    )
    footprint_hit = grid.query_cells(
        ((0, 0, 0), (1, 0, 0), (2, 0, 0)),
        reference_cell=(0, 0, 0),
        occupied_reason="occupied-footprint-voxel",
        free_reason="free-footprint",
    )

    assert free_center.occupied is False
    assert footprint_hit.occupied is True
    assert footprint_hit.cell == (2, 0, 0)
    assert footprint_hit.reason == "occupied-footprint-voxel"
    assert footprint_hit.checked_cell_count == 3


def test_headless_environment_uses_robot_footprint_for_collision() -> None:
    env = HeadlessPhysicalAIEnvironment(build_unit_catalog())
    env.reset("unit-scene")
    env.set_occupancy_grid(
        VoxelOccupancyGrid.from_points(
            np.array([[0.5, 0.0, 0.0]], dtype=np.float32),
            voxel_size_meters=0.25,
            source="unit-grid",
        )
    )

    center_pose = Pose3D(position=(0.0, 0.0, 0.0), orientation_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="generic_world")
    point_query = env.query_collision(center_pose)
    assert point_query.collides is False
    assert point_query.reason == "free-voxel:unit-grid"

    env.set_robot_footprint(RobotFootprint(radius_meters=0.5))
    footprint_query = env.query_collision(center_pose)
    assert footprint_query.collides is True
    assert footprint_query.reason == "occupied-footprint-voxel:unit-grid"

    blocked = env.step(AgentAction("teleport", {"x": 0.0, "y": 0.0, "z": 0.0}))
    assert blocked["applied"] is False
    assert blocked["collision"]["reason"] == "occupied-footprint-voxel:unit-grid"

    env.set_robot_footprint(None)
    cleared = env.query_collision(center_pose)
    assert cleared.collides is False
    assert cleared.reason == "free-voxel:unit-grid"


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
