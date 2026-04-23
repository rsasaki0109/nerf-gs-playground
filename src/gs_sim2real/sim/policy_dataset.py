"""Replay-friendly dataset export for route policy episodes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
from typing import Any

from .gym_adapter import RoutePolicyAction, RoutePolicyGymAdapter
from .interfaces import Pose3D
from .route_planning import RouteCandidate


ROUTE_POLICY_DATASET_VERSION = "gs-mapper-route-policy-dataset/v1"

RoutePolicyCallable = Callable[[Mapping[str, float], Mapping[str, Any]], RoutePolicyAction]
RoutePolicyGoal = Pose3D | Mapping[str, Any] | Sequence[float] | None


@dataclass(frozen=True, slots=True)
class RoutePolicyTransitionRecord:
    """One replay transition emitted by a route policy adapter."""

    episode_id: str
    scene_id: str
    episode_index: int
    step_index: int
    observation: Mapping[str, float]
    action: Mapping[str, Any]
    reward: float
    next_observation: Mapping[str, float]
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-transition",
            "episodeId": self.episode_id,
            "sceneId": self.scene_id,
            "episodeIndex": self.episode_index,
            "stepIndex": self.step_index,
            "observation": _feature_payload(self.observation),
            "action": _json_mapping(self.action),
            "reward": float(self.reward),
            "nextObservation": _feature_payload(self.next_observation),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "info": _json_mapping(self.info),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyEpisodeRecord:
    """Replay episode made from Gymnasium-style route policy transitions."""

    episode_id: str
    scene_id: str
    episode_index: int
    seed: int | None
    initial_observation: Mapping[str, float]
    reset_info: Mapping[str, Any]
    transitions: tuple[RoutePolicyTransitionRecord, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def step_count(self) -> int:
        return len(self.transitions)

    @property
    def total_reward(self) -> float:
        return sum(transition.reward for transition in self.transitions)

    @property
    def terminated(self) -> bool:
        return bool(self.transitions and self.transitions[-1].terminated)

    @property
    def truncated(self) -> bool:
        return bool(self.transitions and self.transitions[-1].truncated)

    def summary(self) -> dict[str, Any]:
        final_info = dict(self.transitions[-1].info) if self.transitions else {}
        return {
            "stepCount": self.step_count,
            "totalReward": self.total_reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "goalReached": bool(final_info.get("goal_reached", False)),
            "blocked": bool(final_info.get("blocked", False)),
            "terminationReason": final_info.get("termination_reason"),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-episode",
            "episodeId": self.episode_id,
            "sceneId": self.scene_id,
            "episodeIndex": self.episode_index,
            "seed": self.seed,
            "summary": self.summary(),
            "initialObservation": _feature_payload(self.initial_observation),
            "resetInfo": _json_mapping(self.reset_info),
            "metadata": _json_mapping(self.metadata),
            "transitions": [transition.to_dict() for transition in self.transitions],
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyDatasetExport:
    """Collection of route policy episodes ready for JSON or JSONL export."""

    dataset_id: str
    episodes: tuple[RoutePolicyEpisodeRecord, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = ROUTE_POLICY_DATASET_VERSION

    @property
    def transition_count(self) -> int:
        return sum(episode.step_count for episode in self.episodes)

    def transition_rows(self) -> tuple[dict[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        for episode in self.episodes:
            for transition in episode.transitions:
                row = transition.to_dict()
                row["datasetId"] = self.dataset_id
                rows.append(row)
        return tuple(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-dataset",
            "version": self.version,
            "datasetId": self.dataset_id,
            "episodeCount": len(self.episodes),
            "transitionCount": self.transition_count,
            "metadata": _json_mapping(self.metadata),
            "episodes": [episode.to_dict() for episode in self.episodes],
        }


def collect_route_policy_episode(
    adapter: RoutePolicyGymAdapter,
    policy: RoutePolicyCallable,
    *,
    seed: int | None = None,
    goal: RoutePolicyGoal = None,
    episode_id: str | None = None,
    reset_options: Mapping[str, Any] | None = None,
    max_steps: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RoutePolicyEpisodeRecord:
    """Collect one adapter episode into replay transitions."""

    step_limit = _positive_int(max_steps if max_steps is not None else adapter.config.max_steps, "max_steps")
    observation, reset_info = adapter.reset(seed=seed, options=reset_options, goal=goal)
    scene_id = str(reset_info["sceneId"])
    episode_index = int(reset_info["episodeIndex"])
    resolved_episode_id = episode_id or f"{scene_id}-episode-{episode_index}"

    transitions: list[RoutePolicyTransitionRecord] = []
    current_observation: Mapping[str, float] = observation
    current_info: Mapping[str, Any] = reset_info
    for _ in range(step_limit):
        action = policy(current_observation, current_info)
        next_observation, reward, terminated, truncated, step_info = adapter.step(action)
        step_index = int(current_info.get("stepIndex", len(transitions)))
        transition = RoutePolicyTransitionRecord(
            episode_id=resolved_episode_id,
            scene_id=scene_id,
            episode_index=episode_index,
            step_index=step_index,
            observation=current_observation,
            action=serialize_route_policy_action(action),
            reward=reward,
            next_observation=next_observation,
            terminated=terminated,
            truncated=truncated,
            info=step_info,
        )
        transitions.append(transition)
        current_observation = next_observation
        current_info = step_info
        if terminated or truncated:
            break
    if transitions and not transitions[-1].terminated and not transitions[-1].truncated:
        forced_info = {
            **dict(transitions[-1].info),
            "truncated": True,
            "done": True,
            "termination_reason": "collector-max-steps",
        }
        transitions[-1] = replace(transitions[-1], truncated=True, info=forced_info)

    return RoutePolicyEpisodeRecord(
        episode_id=resolved_episode_id,
        scene_id=scene_id,
        episode_index=episode_index,
        seed=seed,
        initial_observation=observation,
        reset_info=reset_info,
        transitions=tuple(transitions),
        metadata=_json_mapping(metadata or {}),
    )


def collect_route_policy_dataset(
    adapters: Sequence[RoutePolicyGymAdapter],
    policy: RoutePolicyCallable,
    *,
    episode_count: int,
    dataset_id: str = "route-policy-rollouts",
    seed_start: int = 0,
    goals: Sequence[RoutePolicyGoal] | None = None,
    max_steps: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RoutePolicyDatasetExport:
    """Collect episodes across one or more adapters in round-robin order."""

    count = _non_negative_int(episode_count, "episode_count")
    if count and not adapters:
        raise ValueError("adapters must contain at least one adapter when episode_count is positive")

    episodes: list[RoutePolicyEpisodeRecord] = []
    for index in range(count):
        adapter_index = index % len(adapters)
        episode = collect_route_policy_episode(
            adapters[adapter_index],
            policy,
            seed=seed_start + index,
            goal=_goal_for_index(goals, index),
            episode_id=f"{dataset_id}-episode-{index:06d}",
            max_steps=max_steps,
            metadata={
                "collectorIndex": index,
                "adapterIndex": adapter_index,
            },
        )
        episodes.append(episode)
    return RoutePolicyDatasetExport(
        dataset_id=dataset_id,
        episodes=tuple(episodes),
        metadata=_json_mapping(metadata or {}),
    )


def write_route_policy_dataset_json(path: str | Path, dataset: RoutePolicyDatasetExport) -> Path:
    """Write a full episode dataset as JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dataset.to_dict(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_route_policy_transitions_jsonl(path: str | Path, dataset: RoutePolicyDatasetExport) -> Path:
    """Write replay transitions as newline-delimited JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = dataset.transition_rows()
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return output_path


def serialize_route_policy_action(action: RoutePolicyAction) -> dict[str, Any]:
    """Convert an adapter action into a stable JSON-friendly payload."""

    if isinstance(action, RouteCandidate):
        return {"kind": "route-candidate", "payload": action.to_dict()}
    if isinstance(action, Pose3D):
        return {"kind": "pose", "payload": action.to_dict()}
    if isinstance(action, Mapping):
        return {"kind": "mapping", "payload": _json_mapping(action)}
    if _is_position_sequence(action):
        return {"kind": "position", "payload": [float(component) for component in action]}
    if _is_sequence(action):
        return {"kind": "sequence", "payload": _json_value(action)}
    raise TypeError("unsupported route policy action")


def _goal_for_index(goals: Sequence[RoutePolicyGoal] | None, index: int) -> RoutePolicyGoal:
    if not goals:
        return None
    return goals[index % len(goals)]


def _feature_payload(features: Mapping[str, float]) -> dict[str, float]:
    payload: dict[str, float] = {}
    for key, value in sorted(features.items()):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"feature {key!r} must be finite")
        payload[str(key)] = normalized
    return payload


def _json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}


def _json_value(value: Any) -> Any:
    if isinstance(value, Pose3D):
        return value.to_dict()
    if isinstance(value, RouteCandidate):
        return value.to_dict()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return _json_mapping(value)
    if _is_sequence(value):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _is_position_sequence(value: Any) -> bool:
    if not _is_sequence(value) or len(value) != 3:
        return False
    return all(isinstance(component, (int, float)) for component in value)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _positive_int(value: int, field_name: str) -> int:
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{field_name} must be positive")
    return normalized


def _non_negative_int(value: int, field_name: str) -> int:
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return normalized
