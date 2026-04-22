"""Robot footprint helpers for Physical AI simulation."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .interfaces import Pose3D
from .occupancy import VoxelCell, point_to_voxel_cell


@dataclass(frozen=True, slots=True)
class RobotFootprint:
    """Conservative circular footprint sampled into occupancy grid cells."""

    radius_meters: float
    height_meters: float = 0.0
    vertical_axis: str = "y"

    def __post_init__(self) -> None:
        _positive_float(self.radius_meters, "radius_meters", allow_zero=True)
        _positive_float(self.height_meters, "height_meters", allow_zero=True)
        if self.vertical_axis not in ("x", "y", "z"):
            raise ValueError("vertical_axis must be x, y, or z")

    def cells_for_pose(self, pose: Pose3D, voxel_size_meters: float) -> tuple[VoxelCell, ...]:
        """Return occupancy cells touched by this footprint at ``pose``."""

        voxel_size = _positive_float(voxel_size_meters, "voxel_size_meters")
        center_cell = point_to_voxel_cell(pose.position, voxel_size)
        axis_index = _axis_index(self.vertical_axis)
        offsets = _footprint_offsets(
            radius_meters=self.radius_meters,
            height_meters=self.height_meters,
            voxel_size_meters=voxel_size,
            vertical_axis_index=axis_index,
        )
        return tuple(
            (
                center_cell[0] + offset[0],
                center_cell[1] + offset[1],
                center_cell[2] + offset[2],
            )
            for offset in offsets
        )

    def to_dict(self) -> dict[str, float | str]:
        return {
            "radiusMeters": self.radius_meters,
            "heightMeters": self.height_meters,
            "verticalAxis": self.vertical_axis,
        }


def _footprint_offsets(
    *,
    radius_meters: float,
    height_meters: float,
    voxel_size_meters: float,
    vertical_axis_index: int,
) -> tuple[VoxelCell, ...]:
    radius_cells = int(math.ceil(radius_meters / voxel_size_meters))
    height_cells = int(math.ceil(height_meters / voxel_size_meters))
    horizontal_axes = tuple(index for index in range(3) if index != vertical_axis_index)
    horizontal_padding = voxel_size_meters * math.sqrt(2.0) * 0.5
    offsets: list[VoxelCell] = []
    for vertical_offset in range(0, height_cells + 1):
        for first_offset in range(-radius_cells, radius_cells + 1):
            for second_offset in range(-radius_cells, radius_cells + 1):
                horizontal_distance = math.hypot(first_offset * voxel_size_meters, second_offset * voxel_size_meters)
                if horizontal_distance > radius_meters + horizontal_padding + 1e-9:
                    continue
                offset = [0, 0, 0]
                offset[vertical_axis_index] = vertical_offset
                offset[horizontal_axes[0]] = first_offset
                offset[horizontal_axes[1]] = second_offset
                offsets.append((offset[0], offset[1], offset[2]))
    return tuple(offsets)


def _axis_index(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[axis]


def _positive_float(value: float, field_name: str, *, allow_zero: bool = False) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0 or (not allow_zero and normalized == 0.0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{field_name} must be {qualifier}")
    return normalized
