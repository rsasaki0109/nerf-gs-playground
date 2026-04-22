"""Occupancy helpers for Physical AI simulation."""

from __future__ import annotations

import base64
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .interfaces import Observation, Pose3D


VoxelCell = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class OccupancyQuery:
    """Result from a voxel occupancy lookup."""

    occupied: bool
    cell: VoxelCell
    reason: str
    clearance_meters: float | None = None
    checked_cell_count: int = 1


@dataclass(frozen=True, slots=True)
class VoxelOccupancyGrid:
    """Sparse voxel occupancy grid used for lightweight collision checks."""

    voxel_size_meters: float
    occupied_cells: frozenset[VoxelCell]
    source: str = "unknown"
    point_count: int = 0
    inflation_radius_meters: float = 0.0

    @classmethod
    def from_points(
        cls,
        points: Sequence[Sequence[float]] | np.ndarray,
        *,
        voxel_size_meters: float,
        inflation_radius_meters: float = 0.0,
        source: str = "points",
    ) -> VoxelOccupancyGrid:
        """Build an occupancy grid from world-frame XYZ points."""

        voxel_size = _positive_float(voxel_size_meters, "voxel_size_meters")
        inflation_radius = max(float(inflation_radius_meters), 0.0)
        points_array = _normalize_points(points)
        cells: set[VoxelCell] = set()
        inflation_offsets = tuple(_inflation_offsets(voxel_size, inflation_radius))
        for point in points_array:
            base_cell = point_to_voxel_cell(point, voxel_size)
            for offset in inflation_offsets:
                cells.add((base_cell[0] + offset[0], base_cell[1] + offset[1], base_cell[2] + offset[2]))
        return cls(
            voxel_size_meters=voxel_size,
            occupied_cells=frozenset(cells),
            source=str(source or "points"),
            point_count=int(points_array.shape[0]),
            inflation_radius_meters=inflation_radius,
        )

    @property
    def cell_count(self) -> int:
        return len(self.occupied_cells)

    def contains_point(self, point: Sequence[float]) -> bool:
        """Return whether ``point`` falls into an occupied voxel."""

        return point_to_voxel_cell(point, self.voxel_size_meters) in self.occupied_cells

    def query_pose(self, pose: Pose3D) -> OccupancyQuery:
        """Query occupancy for a pose position."""

        cell = point_to_voxel_cell(pose.position, self.voxel_size_meters)
        return self.query_cells(
            (cell,),
            reference_cell=cell,
            occupied_reason="occupied-voxel",
            free_reason="free-voxel",
        )

    def query_cells(
        self,
        cells: Iterable[VoxelCell],
        *,
        reference_cell: VoxelCell,
        occupied_reason: str = "occupied-voxel",
        free_reason: str = "free-voxel",
    ) -> OccupancyQuery:
        """Query occupancy for a set of cells."""

        checked_cells = tuple(dict.fromkeys(cells))
        if not checked_cells:
            raise ValueError("cells must contain at least one voxel cell")
        for cell in checked_cells:
            if cell in self.occupied_cells:
                return OccupancyQuery(
                    occupied=True,
                    cell=cell,
                    reason=occupied_reason,
                    clearance_meters=0.0,
                    checked_cell_count=len(checked_cells),
                )
        return OccupancyQuery(
            occupied=False,
            cell=reference_cell,
            reason=free_reason,
            clearance_meters=self._nearest_cell_set_distance(checked_cells),
            checked_cell_count=len(checked_cells),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "voxelSizeMeters": self.voxel_size_meters,
            "inflationRadiusMeters": self.inflation_radius_meters,
            "pointCount": self.point_count,
            "cellCount": self.cell_count,
        }

    def _nearest_cell_set_distance(self, cells: Sequence[VoxelCell]) -> float | None:
        if not self.occupied_cells:
            return None
        nearest = min(
            math.dist(cell, occupied_cell) * self.voxel_size_meters
            for cell in cells
            for occupied_cell in self.occupied_cells
        )
        return float(nearest)


def build_occupancy_grid_from_lidar_observation(
    observation: Observation,
    *,
    voxel_size_meters: float,
    inflation_radius_meters: float = 0.0,
) -> VoxelOccupancyGrid:
    """Decode a `lidar-ray-proxy` observation into a sparse occupancy grid."""

    points = decode_lidar_points_observation(observation)
    outputs = observation.outputs
    scene_id = outputs.get("sceneId", "unknown") if isinstance(outputs, Mapping) else "unknown"
    return VoxelOccupancyGrid.from_points(
        points,
        voxel_size_meters=voxel_size_meters,
        inflation_radius_meters=inflation_radius_meters,
        source=f"lidar-ray-proxy:{scene_id}",
    )


def decode_lidar_points_observation(observation: Observation) -> np.ndarray:
    """Decode float32 XYZ points from a `lidar-ray-proxy` observation."""

    if observation.sensor_id != "lidar-ray-proxy":
        raise ValueError("observation sensor_id must be 'lidar-ray-proxy'")
    points_payload = observation.outputs.get("points")
    if not isinstance(points_payload, Mapping):
        raise ValueError("lidar observation must include a points payload")
    encoded = points_payload.get("pointsBase64")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("lidar points payload must include pointsBase64")
    points_bytes = base64.b64decode(encoded, validate=True)
    if len(points_bytes) % np.dtype("<f4").itemsize != 0:
        raise ValueError("lidar points payload byte size must be a multiple of float32 width")
    points = np.frombuffer(points_bytes, dtype="<f4")
    if points.size % 3 != 0:
        raise ValueError("lidar points payload size must be divisible by 3 float32 values")
    return points.reshape(-1, 3).copy()


def point_to_voxel_cell(point: Sequence[float] | np.ndarray, voxel_size_meters: float) -> VoxelCell:
    """Quantize a world point into a voxel cell index."""

    voxel_size = _positive_float(voxel_size_meters, "voxel_size_meters")
    vector = np.asarray(point, dtype=np.float32)
    if vector.shape != (3,):
        raise ValueError("point must contain exactly three values")
    cell = np.floor(vector / voxel_size).astype(np.int64)
    return (int(cell[0]), int(cell[1]), int(cell[2]))


def _normalize_points(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    points_array = np.asarray(points, dtype=np.float32)
    if points_array.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError("points must be an array of shape (N, 3)")
    return points_array


def _inflation_offsets(voxel_size_meters: float, inflation_radius_meters: float) -> Iterable[VoxelCell]:
    radius_cells = int(math.ceil(inflation_radius_meters / voxel_size_meters))
    for x_offset in range(-radius_cells, radius_cells + 1):
        for y_offset in range(-radius_cells, radius_cells + 1):
            for z_offset in range(-radius_cells, radius_cells + 1):
                distance = math.dist((0, 0, 0), (x_offset, y_offset, z_offset)) * voxel_size_meters
                if distance <= inflation_radius_meters + 1e-9:
                    yield (x_offset, y_offset, z_offset)


def _positive_float(value: float, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return normalized
