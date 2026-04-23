"""Replay loading and offline training batches for route policy datasets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import random
from typing import Any

from .policy_dataset import (
    ROUTE_POLICY_DATASET_VERSION,
    RoutePolicyDatasetExport,
    RoutePolicyEpisodeRecord,
    RoutePolicyTransitionRecord,
)


ROUTE_POLICY_REPLAY_VERSION = "gs-mapper-route-policy-replay/v1"


@dataclass(frozen=True, slots=True)
class RoutePolicyTransitionTable:
    """Flat transition table loaded from JSONL or derived from an episode dataset."""

    dataset_id: str
    transitions: tuple[RoutePolicyTransitionRecord, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = ROUTE_POLICY_REPLAY_VERSION

    @property
    def transition_count(self) -> int:
        return len(self.transitions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-transition-table",
            "version": self.version,
            "datasetId": self.dataset_id,
            "transitionCount": self.transition_count,
            "metadata": _json_mapping(self.metadata),
            "transitions": [transition.to_dict() for transition in self.transitions],
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyReplayFeatureSchema:
    """Ordered numeric feature schema used by replay training batches."""

    observation_keys: tuple[str, ...]
    action_keys: tuple[str, ...]
    next_observation_keys: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = ROUTE_POLICY_REPLAY_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "observation_keys", _unique_key_tuple(self.observation_keys, "observation_keys"))
        object.__setattr__(self, "action_keys", _unique_key_tuple(self.action_keys, "action_keys"))
        object.__setattr__(
            self,
            "next_observation_keys",
            _unique_key_tuple(self.next_observation_keys, "next_observation_keys"),
        )

    @property
    def observation_feature_count(self) -> int:
        return len(self.observation_keys)

    @property
    def action_feature_count(self) -> int:
        return len(self.action_keys)

    @property
    def next_observation_feature_count(self) -> int:
        return len(self.next_observation_keys)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-replay-feature-schema",
            "version": self.version,
            "observationKeys": list(self.observation_keys),
            "actionKeys": list(self.action_keys),
            "nextObservationKeys": list(self.next_observation_keys),
            "metadata": _json_mapping(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyReplaySample:
    """One vectorized transition row for offline route policy training."""

    dataset_id: str
    episode_id: str
    scene_id: str
    episode_index: int
    step_index: int
    observation_vector: tuple[float, ...]
    action_vector: tuple[float, ...]
    reward: float
    next_observation_vector: tuple[float, ...]
    terminated: bool
    truncated: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-replay-sample",
            "datasetId": self.dataset_id,
            "episodeId": self.episode_id,
            "sceneId": self.scene_id,
            "episodeIndex": self.episode_index,
            "stepIndex": self.step_index,
            "observationVector": list(self.observation_vector),
            "actionVector": list(self.action_vector),
            "reward": float(self.reward),
            "nextObservationVector": list(self.next_observation_vector),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "done": self.done,
            "metadata": _json_mapping(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyReplayBatch:
    """Schema-bound batch for imitation learning or offline RL input pipelines."""

    schema: RoutePolicyReplayFeatureSchema
    samples: tuple[RoutePolicyReplaySample, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = ROUTE_POLICY_REPLAY_VERSION

    @property
    def size(self) -> int:
        return len(self.samples)

    @property
    def observation_matrix(self) -> tuple[tuple[float, ...], ...]:
        return tuple(sample.observation_vector for sample in self.samples)

    @property
    def action_matrix(self) -> tuple[tuple[float, ...], ...]:
        return tuple(sample.action_vector for sample in self.samples)

    @property
    def reward_vector(self) -> tuple[float, ...]:
        return tuple(sample.reward for sample in self.samples)

    @property
    def next_observation_matrix(self) -> tuple[tuple[float, ...], ...]:
        return tuple(sample.next_observation_vector for sample in self.samples)

    @property
    def terminated_vector(self) -> tuple[bool, ...]:
        return tuple(sample.terminated for sample in self.samples)

    @property
    def truncated_vector(self) -> tuple[bool, ...]:
        return tuple(sample.truncated for sample in self.samples)

    @property
    def done_vector(self) -> tuple[bool, ...]:
        return tuple(sample.done for sample in self.samples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-replay-batch",
            "version": self.version,
            "size": self.size,
            "schema": self.schema.to_dict(),
            "observationMatrix": [list(row) for row in self.observation_matrix],
            "actionMatrix": [list(row) for row in self.action_matrix],
            "rewardVector": list(self.reward_vector),
            "nextObservationMatrix": [list(row) for row in self.next_observation_matrix],
            "terminatedVector": list(self.terminated_vector),
            "truncatedVector": list(self.truncated_vector),
            "doneVector": list(self.done_vector),
            "samples": [sample.to_dict() for sample in self.samples],
            "metadata": _json_mapping(self.metadata),
        }


RoutePolicyReplaySource = RoutePolicyDatasetExport | RoutePolicyTransitionTable | Sequence[RoutePolicyTransitionRecord]


def load_route_policy_dataset_json(path: str | Path) -> RoutePolicyDatasetExport:
    """Load a full episode dataset previously written by ``write_route_policy_dataset_json``."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return route_policy_dataset_from_dict(payload)


def load_route_policy_transitions_jsonl(
    path: str | Path,
    *,
    dataset_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RoutePolicyTransitionTable:
    """Load newline-delimited replay transitions into a flat transition table."""

    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL transition at line {line_number}") from exc
        rows.append(_mapping(payload, f"line {line_number}"))
    return route_policy_transition_table_from_rows(rows, dataset_id=dataset_id, metadata=metadata)


def route_policy_dataset_from_dict(payload: Mapping[str, Any]) -> RoutePolicyDatasetExport:
    """Rebuild route policy dataset dataclasses from their stable JSON envelope."""

    _record_type(payload, "route-policy-dataset")
    version = str(payload.get("version", ROUTE_POLICY_DATASET_VERSION))
    if version != ROUTE_POLICY_DATASET_VERSION:
        raise ValueError(f"unsupported route policy dataset version: {version}")
    episodes = tuple(
        _episode_from_payload(_mapping(item, "episode")) for item in _sequence(payload.get("episodes", ()))
    )
    expected_episode_count = payload.get("episodeCount")
    if expected_episode_count is not None and int(expected_episode_count) != len(episodes):
        raise ValueError("episodeCount does not match loaded episodes")
    dataset = RoutePolicyDatasetExport(
        dataset_id=str(payload["datasetId"]),
        episodes=episodes,
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
        version=version,
    )
    expected_transition_count = payload.get("transitionCount")
    if expected_transition_count is not None and int(expected_transition_count) != dataset.transition_count:
        raise ValueError("transitionCount does not match loaded transitions")
    return dataset


def route_policy_transition_table_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RoutePolicyTransitionTable:
    """Build a transition table from decoded JSONL rows."""

    source_dataset_ids = tuple(sorted({str(row["datasetId"]) for row in rows if row.get("datasetId") is not None}))
    resolved_dataset_id = _resolve_transition_table_dataset_id(dataset_id, source_dataset_ids)
    transitions = tuple(_transition_from_payload(row) for row in rows)
    return RoutePolicyTransitionTable(
        dataset_id=resolved_dataset_id,
        transitions=transitions,
        metadata={
            "sourceDatasetIds": list(source_dataset_ids),
            **_json_mapping(metadata or {}),
        },
    )


def route_policy_transition_table_from_dataset(dataset: RoutePolicyDatasetExport) -> RoutePolicyTransitionTable:
    """Flatten a full episode dataset into a transition table without losing the dataset id."""

    transitions = tuple(transition for episode in dataset.episodes for transition in episode.transitions)
    return RoutePolicyTransitionTable(
        dataset_id=dataset.dataset_id,
        transitions=transitions,
        metadata={
            "source": "route-policy-dataset",
            "episodeCount": len(dataset.episodes),
            **_json_mapping(dataset.metadata),
        },
    )


def build_route_policy_replay_schema(
    source: RoutePolicyReplaySource,
    *,
    observation_keys: Sequence[str] | None = None,
    action_keys: Sequence[str] | None = None,
    next_observation_keys: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RoutePolicyReplayFeatureSchema:
    """Infer or pin the ordered numeric feature schema for replay batches."""

    _, transitions = _source_context(source)
    return RoutePolicyReplayFeatureSchema(
        observation_keys=(
            _unique_key_tuple(observation_keys, "observation_keys")
            if observation_keys is not None
            else _sorted_feature_keys(transition.observation for transition in transitions)
        ),
        action_keys=(
            _unique_key_tuple(action_keys, "action_keys")
            if action_keys is not None
            else _sorted_feature_keys(_flatten_numeric_mapping(transition.action) for transition in transitions)
        ),
        next_observation_keys=(
            _unique_key_tuple(next_observation_keys, "next_observation_keys")
            if next_observation_keys is not None
            else _sorted_feature_keys(transition.next_observation for transition in transitions)
        ),
        metadata=_json_mapping(metadata or {}),
    )


def build_route_policy_replay_batch(
    source: RoutePolicyReplaySource,
    *,
    schema: RoutePolicyReplayFeatureSchema | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RoutePolicyReplayBatch:
    """Vectorize replay transitions into one schema-bound offline training batch."""

    dataset_id, transitions = _source_context(source)
    resolved_schema = schema or build_route_policy_replay_schema(source)
    samples = tuple(_replay_sample(dataset_id, transition, resolved_schema) for transition in transitions)
    return RoutePolicyReplayBatch(
        schema=resolved_schema,
        samples=samples,
        metadata={
            "datasetId": dataset_id,
            **_json_mapping(metadata or {}),
        },
    )


def iter_route_policy_replay_batches(
    source: RoutePolicyReplaySource,
    *,
    batch_size: int,
    schema: RoutePolicyReplayFeatureSchema | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    drop_remainder: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> Iterable[RoutePolicyReplayBatch]:
    """Yield replay batches with a shared schema and deterministic optional shuffling."""

    resolved_batch_size = _positive_int(batch_size, "batch_size")
    dataset_id, transitions = _source_context(source)
    resolved_schema = schema or build_route_policy_replay_schema(source)
    ordered = list(transitions)
    if shuffle:
        random.Random(seed).shuffle(ordered)
    for offset in range(0, len(ordered), resolved_batch_size):
        chunk = tuple(ordered[offset : offset + resolved_batch_size])
        if drop_remainder and len(chunk) < resolved_batch_size:
            continue
        yield build_route_policy_replay_batch(
            RoutePolicyTransitionTable(dataset_id=dataset_id, transitions=chunk),
            schema=resolved_schema,
            metadata={
                "batchOffset": offset,
                "batchSize": resolved_batch_size,
                **_json_mapping(metadata or {}),
            },
        )


def _replay_sample(
    dataset_id: str,
    transition: RoutePolicyTransitionRecord,
    schema: RoutePolicyReplayFeatureSchema,
) -> RoutePolicyReplaySample:
    return RoutePolicyReplaySample(
        dataset_id=dataset_id,
        episode_id=transition.episode_id,
        scene_id=transition.scene_id,
        episode_index=transition.episode_index,
        step_index=transition.step_index,
        observation_vector=_feature_vector(transition.observation, schema.observation_keys),
        action_vector=_feature_vector(_flatten_numeric_mapping(transition.action), schema.action_keys),
        reward=_finite_float(transition.reward, "reward"),
        next_observation_vector=_feature_vector(transition.next_observation, schema.next_observation_keys),
        terminated=transition.terminated,
        truncated=transition.truncated,
        metadata={
            "actionKind": str(transition.action.get("kind", "unknown")),
            "terminationReason": transition.info.get("termination_reason"),
        },
    )


def _episode_from_payload(payload: Mapping[str, Any]) -> RoutePolicyEpisodeRecord:
    _record_type(payload, "route-policy-episode")
    transitions = tuple(
        _transition_from_payload(_mapping(item, "transition")) for item in _sequence(payload.get("transitions", ()))
    )
    summary = _mapping(payload.get("summary", {}), "summary")
    if summary.get("stepCount") is not None and int(summary["stepCount"]) != len(transitions):
        raise ValueError("episode summary stepCount does not match transitions")
    return RoutePolicyEpisodeRecord(
        episode_id=str(payload["episodeId"]),
        scene_id=str(payload["sceneId"]),
        episode_index=int(payload["episodeIndex"]),
        seed=None if payload.get("seed") is None else int(payload["seed"]),
        initial_observation=_feature_mapping(_mapping(payload.get("initialObservation", {}), "initialObservation")),
        reset_info=_json_mapping(_mapping(payload.get("resetInfo", {}), "resetInfo")),
        transitions=transitions,
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
    )


def _transition_from_payload(payload: Mapping[str, Any]) -> RoutePolicyTransitionRecord:
    _record_type(payload, "route-policy-transition")
    return RoutePolicyTransitionRecord(
        episode_id=str(payload["episodeId"]),
        scene_id=str(payload["sceneId"]),
        episode_index=int(payload["episodeIndex"]),
        step_index=int(payload["stepIndex"]),
        observation=_feature_mapping(_mapping(payload.get("observation", {}), "observation")),
        action=_json_mapping(_mapping(payload.get("action", {}), "action")),
        reward=_finite_float(payload["reward"], "reward"),
        next_observation=_feature_mapping(_mapping(payload.get("nextObservation", {}), "nextObservation")),
        terminated=_bool_value(payload["terminated"], "terminated"),
        truncated=_bool_value(payload["truncated"], "truncated"),
        info=_json_mapping(_mapping(payload.get("info", {}), "info")),
    )


def _source_context(source: RoutePolicyReplaySource) -> tuple[str, tuple[RoutePolicyTransitionRecord, ...]]:
    if isinstance(source, RoutePolicyDatasetExport):
        table = route_policy_transition_table_from_dataset(source)
        return table.dataset_id, table.transitions
    if isinstance(source, RoutePolicyTransitionTable):
        return source.dataset_id, source.transitions
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        return "route-policy-replay", tuple(source)
    raise TypeError("unsupported route policy replay source")


def _resolve_transition_table_dataset_id(dataset_id: str | None, source_dataset_ids: Sequence[str]) -> str:
    if dataset_id is not None:
        return str(dataset_id)
    if len(source_dataset_ids) == 1:
        return source_dataset_ids[0]
    if len(source_dataset_ids) > 1:
        raise ValueError("dataset_id is required when JSONL rows contain multiple dataset ids")
    return "route-policy-transitions"


def _sorted_feature_keys(rows: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(sorted({str(key) for row in rows for key in row}))


def _feature_vector(features: Mapping[str, Any], keys: Sequence[str]) -> tuple[float, ...]:
    return tuple(_finite_float(features.get(key, 0.0), key) for key in keys)


def _feature_mapping(value: Mapping[str, Any]) -> dict[str, float]:
    return {str(key): _finite_float(item, str(key)) for key, item in sorted(value.items())}


def _flatten_numeric_mapping(value: Mapping[str, Any]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    _flatten_numeric_value("", value, flattened)
    return dict(sorted(flattened.items()))


def _flatten_numeric_value(prefix: str, value: Any, output: dict[str, float]) -> None:
    if isinstance(value, Mapping):
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_numeric_value(child_prefix, item, output)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            _flatten_numeric_value(child_prefix, item, output)
        return
    if isinstance(value, bool):
        output[prefix] = 1.0 if value else 0.0
        return
    if isinstance(value, (int, float)):
        output[prefix] = _finite_float(value, prefix)


def _record_type(payload: Mapping[str, Any], expected: str) -> None:
    record_type = payload.get("recordType")
    if record_type != expected:
        raise ValueError(f"expected {expected!r}, got {record_type!r}")


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise TypeError(f"{field_name} must be a mapping")


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    raise TypeError("value must be a sequence")


def _unique_key_tuple(keys: Sequence[str], field_name: str) -> tuple[str, ...]:
    normalized = tuple(str(key) for key in keys)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicate keys")
    return normalized


def _json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _json_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return _finite_float(value, "json-value")
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _finite_float(value: Any, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


def _bool_value(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError(f"{field_name} must be a boolean")


def _positive_int(value: int, field_name: str) -> int:
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{field_name} must be positive")
    return normalized
