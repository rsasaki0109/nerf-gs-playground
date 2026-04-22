"""Renderer adapters for Physical AI simulation observations."""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image as PILImage

from .contract import SceneEnvironment
from .interfaces import Observation, ObservationRequest, Pose3D


@dataclass(frozen=True, slots=True)
class SplatRenderConfig:
    """Camera and raster settings for local .splat observation renders."""

    width: int = 320
    height: int = 240
    fov_degrees: float = 60.0
    near_clip: float = 0.05
    far_clip: float = 80.0
    point_radius: int = 1
    jpeg_quality: int = 85
    max_gaussians: int | None = 120_000


@dataclass(frozen=True, slots=True)
class SplatPointCloud:
    """Decoded subset of an antimatter15-style .splat asset."""

    path: Path
    positions: np.ndarray
    colors: np.ndarray
    opacities: np.ndarray
    gaussian_count: int
    loaded_count: int


class ObservationRenderer(Protocol):
    """Renderer adapter used by concrete Physical AI environments."""

    def can_render(self, scene: SceneEnvironment, request: ObservationRequest) -> bool:
        """Return whether this adapter can render the request."""

    def render_observation(self, scene: SceneEnvironment, request: ObservationRequest) -> Observation:
        """Render one observation for ``scene`` and ``request``."""


class SplatAssetObservationRenderer:
    """Render RGB observations from bundled `.splat` assets using a local rasterizer."""

    def __init__(self, docs_root: str | Path, *, config: SplatRenderConfig | None = None):
        self.docs_root = Path(docs_root)
        self.config = config or SplatRenderConfig()
        self._cache: dict[Path, SplatPointCloud] = {}

    def can_render(self, scene: SceneEnvironment, request: ObservationRequest) -> bool:
        return (
            request.sensor_id == "rgb-forward"
            and tuple(request.outputs) == ("rgb",)
            and scene.asset_url.endswith(".splat")
        )

    def render_observation(self, scene: SceneEnvironment, request: ObservationRequest) -> Observation:
        if not self.can_render(scene, request):
            raise ValueError(f"splat asset renderer cannot render {request.sensor_id} outputs {request.outputs}")

        cloud = self._load_cloud(scene)
        rgb, depth = render_splat_point_cloud(cloud, request.pose, self.config)
        jpeg_bytes = encode_rgb_to_jpeg(rgb, quality=self.config.jpeg_quality)
        valid_depth = depth[depth < self.config.far_clip]
        valid_pixel_count = int(valid_depth.size)
        return Observation(
            sensor_id=request.sensor_id,
            pose=request.pose,
            outputs={
                "mode": "splat-raster",
                "sceneId": scene.scene_id,
                "assetUrl": scene.asset_url,
                "renderer": "splat-asset-simple",
                "gaussianCount": cloud.gaussian_count,
                "loadedGaussianCount": cloud.loaded_count,
                "rgb": {
                    "encoding": "jpeg",
                    "width": int(self.config.width),
                    "height": int(self.config.height),
                    "jpegBase64": base64.b64encode(jpeg_bytes).decode("ascii"),
                    "byteLength": len(jpeg_bytes),
                },
                "cameraInfo": build_camera_info(
                    width=self.config.width,
                    height=self.config.height,
                    fov_degrees=self.config.fov_degrees,
                    frame_id=request.pose.frame_id,
                ),
                "depthStats": {
                    "validPixelCount": valid_pixel_count,
                    "validPixelRatio": valid_pixel_count / float(depth.size),
                    "minMeters": float(valid_depth.min()) if valid_pixel_count else None,
                    "maxMeters": float(valid_depth.max()) if valid_pixel_count else None,
                    "farClipMeters": float(self.config.far_clip),
                },
            },
        )

    def _load_cloud(self, scene: SceneEnvironment) -> SplatPointCloud:
        path = resolve_scene_asset_path(self.docs_root, scene.asset_url)
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        cloud = load_splat_point_cloud(path, max_gaussians=self.config.max_gaussians)
        self._cache[path] = cloud
        return cloud


def resolve_scene_asset_path(docs_root: str | Path, asset_url: str) -> Path:
    """Resolve a scene asset URL under a local docs root."""

    root = Path(docs_root)
    path = (root / asset_url).resolve()
    root_resolved = root.resolve()
    if root_resolved not in path.parents and path != root_resolved:
        raise ValueError(f"scene asset escapes docs root: {asset_url}")
    return path


def load_splat_point_cloud(path: str | Path, *, max_gaussians: int | None = None) -> SplatPointCloud:
    """Load position/color/opacity fields from an antimatter15 `.splat` asset."""

    asset_path = Path(path)
    if asset_path.suffix != ".splat":
        raise ValueError(f"expected a .splat asset, got: {asset_path}")
    if not asset_path.is_file():
        raise FileNotFoundError(f"splat asset not found: {asset_path}")

    dtype = np.dtype(
        [
            ("position", "<f4", 3),
            ("scale", "<f4", 3),
            ("rgba", "u1", 4),
            ("rotation", "u1", 4),
        ]
    )
    asset_size = asset_path.stat().st_size
    if asset_size <= 0 or asset_size % dtype.itemsize != 0:
        raise ValueError(f"splat asset must be non-empty and {dtype.itemsize}-byte aligned: {asset_path}")

    raw = np.fromfile(asset_path, dtype=dtype)
    gaussian_count = int(raw.shape[0])
    if max_gaussians is not None and max_gaussians > 0 and gaussian_count > max_gaussians:
        indices = np.linspace(0, gaussian_count - 1, num=int(max_gaussians), dtype=np.int64)
        raw = raw[indices]

    rgba = raw["rgba"].astype(np.float32) / 255.0
    return SplatPointCloud(
        path=asset_path,
        positions=raw["position"].astype(np.float32),
        colors=rgba[:, :3],
        opacities=rgba[:, 3],
        gaussian_count=gaussian_count,
        loaded_count=int(raw.shape[0]),
    )


def render_splat_point_cloud(
    cloud: SplatPointCloud,
    pose: Pose3D,
    config: SplatRenderConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a simple RGB/depth frame by projecting .splat centers."""

    width, height = _validate_image_shape(config.width, config.height)
    near_clip = float(config.near_clip)
    far_clip = float(config.far_clip)
    if near_clip <= 0.0 or far_clip <= near_clip:
        raise ValueError("clip planes must satisfy 0 < near_clip < far_clip")

    fx, fy, cx, cy = compute_camera_intrinsics(width, height, config.fov_degrees)
    rotation_world_to_camera, translation = world_to_camera_transform(pose)
    camera_space = cloud.positions @ rotation_world_to_camera.T + translation
    depths = camera_space[:, 2]
    valid = (depths > near_clip) & (depths < far_clip)

    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.full((height, width), far_clip, dtype=np.float32)
    if not np.any(valid):
        return rgb, depth

    camera_space = camera_space[valid]
    depths = depths[valid]
    colors = cloud.colors[valid]
    opacities = cloud.opacities[valid]

    px = np.rint((camera_space[:, 0] / depths) * fx + cx).astype(np.int32)
    py = np.rint(cy - (camera_space[:, 1] / depths) * fy).astype(np.int32)
    px, py, depths, colors, opacities = expand_point_footprints(
        px,
        py,
        depths,
        colors,
        opacities,
        point_radius=config.point_radius,
    )

    in_bounds = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    if not np.any(in_bounds):
        return rgb, depth

    px = px[in_bounds]
    py = py[in_bounds]
    depths = depths[in_bounds]
    colors = colors[in_bounds]
    opacities = opacities[in_bounds]

    pixel_indices = py * width + px
    sort_order = np.lexsort((depths, pixel_indices))
    pixel_indices = pixel_indices[sort_order]
    depths = depths[sort_order]
    colors = colors[sort_order]
    opacities = opacities[sort_order]

    _, first_indices = np.unique(pixel_indices, return_index=True)
    pixel_indices = pixel_indices[first_indices]
    depths = depths[first_indices]
    colors = colors[first_indices]
    opacities = opacities[first_indices]

    rgb.reshape(-1, 3)[pixel_indices] = np.clip(colors * opacities[:, None] * 255.0, 0.0, 255.0).astype(np.uint8)
    depth.reshape(-1)[pixel_indices] = depths.astype(np.float32)
    return rgb, depth


def compute_camera_intrinsics(width: int, height: int, fov_degrees: float) -> tuple[float, float, float, float]:
    """Compute pinhole intrinsics from vertical field of view."""

    width, height = _validate_image_shape(width, height)
    if not math.isfinite(fov_degrees) or fov_degrees <= 0.0 or fov_degrees >= 179.0:
        raise ValueError("fov_degrees must be within (0, 179)")
    fov_radians = math.radians(float(fov_degrees))
    fy = height / (2.0 * math.tan(fov_radians * 0.5))
    fx = fy * (width / height)
    return (float(fx), float(fy), width * 0.5, height * 0.5)


def build_camera_info(width: int, height: int, fov_degrees: float, frame_id: str) -> dict[str, object]:
    """Build a compact camera-info payload for observation metadata."""

    fx, fy, cx, cy = compute_camera_intrinsics(width, height, fov_degrees)
    return {
        "frameId": frame_id,
        "width": int(width),
        "height": int(height),
        "distortionModel": "plumb_bob",
        "d": [0.0, 0.0, 0.0, 0.0, 0.0],
        "k": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
    }


def world_to_camera_transform(pose: Pose3D) -> tuple[np.ndarray, np.ndarray]:
    """Return world-to-camera rotation and translation for a camera-to-world pose."""

    rotation_camera_to_world = quaternion_to_rotation_matrix(pose.orientation_xyzw)
    position = np.asarray(pose.position, dtype=np.float32)
    rotation_world_to_camera = rotation_camera_to_world.T
    translation = -rotation_world_to_camera @ position
    return rotation_world_to_camera, translation


def quaternion_to_rotation_matrix(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    """Convert an `[x, y, z, w]` quaternion to a 3x3 rotation matrix."""

    q = np.asarray(quaternion, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-8:
        raise ValueError("quaternion norm must be positive")
    x, y, z, w = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def expand_point_footprints(
    px: np.ndarray,
    py: np.ndarray,
    depths: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    *,
    point_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Broadcast projected points over a circular pixel footprint."""

    offsets = build_footprint_offsets(point_radius)
    if offsets.shape[0] <= 1:
        return px, py, depths, colors, opacities

    px = px[:, None] + offsets[None, :, 0]
    py = py[:, None] + offsets[None, :, 1]
    depths = np.broadcast_to(depths[:, None], px.shape).reshape(-1)
    colors = np.broadcast_to(colors[:, None, :], (colors.shape[0], offsets.shape[0], 3)).reshape(-1, 3)
    opacities = np.broadcast_to(opacities[:, None], px.shape).reshape(-1)
    return px.reshape(-1), py.reshape(-1), depths, colors, opacities


def build_footprint_offsets(point_radius: int) -> np.ndarray:
    """Build integer pixel offsets for a circular point footprint."""

    radius = max(int(point_radius), 0)
    if radius == 0:
        return np.array([[0, 0]], dtype=np.int32)

    radius_sq = radius * radius
    offsets = [
        (x_offset, y_offset)
        for y_offset in range(-radius, radius + 1)
        for x_offset in range(-radius, radius + 1)
        if x_offset * x_offset + y_offset * y_offset <= radius_sq
    ]
    return np.asarray(offsets, dtype=np.int32)


def encode_rgb_to_jpeg(rgb_image: np.ndarray, *, quality: int = 85) -> bytes:
    """Encode an RGB uint8 array into JPEG bytes."""

    image = PILImage.fromarray(np.asarray(rgb_image, dtype=np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality), optimize=False)
    return buffer.getvalue()


def _validate_image_shape(width: int, height: int) -> tuple[int, int]:
    resolved_width = int(width)
    resolved_height = int(height)
    if resolved_width <= 0 or resolved_height <= 0:
        raise ValueError("image width and height must be positive")
    return resolved_width, resolved_height
