"""Strict JSON configuration with paths independent of the working directory."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "so101_act.json"
LEROBOT_REVISION = "eacddcb9cff5e033c7811daa15d30f5debcc9a7b"

# This schema also rejects misspelled keys instead of silently ignoring them.
SCHEMA = {
    "schema_version": int,
    "project_root": str,
    "task": (str, type(None)),
    "hardware": {
        "follower_port": str, "follower_id": str,
        "leader_port": str, "leader_id": str,
        "use_degrees": (bool, type(None)),
        "wrist_camera": int, "front_camera": int,
        "width": int, "height": int, "fps": int,
    },
    "dataset": {"repo_id": str, "root": str},
    "record": {"episodes": int, "episode_seconds": int, "reset_seconds": int},
    "train": {
        "output_dir": str, "job_name": str, "steps": int,
        "batch_size": int, "save_freq": int, "num_workers": int,
        "device": str, "use_amp": bool,
    },
    "rollout": {"policy_path": str, "duration_seconds": int, "device": str},
    "logging": {"root": str},
}


class ConfigError(ValueError):
    """A configuration cannot be used safely or unambiguously."""


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"Expected a JSON object in {path}")
    return value


def _validate(value: dict, schema: dict, prefix: str = "") -> None:
    missing, extra = schema.keys() - value.keys(), value.keys() - schema.keys()
    if missing or extra:
        raise ConfigError(f"{prefix or 'config'}: missing keys {sorted(missing)}, unknown keys {sorted(extra)}")
    for key, expected in schema.items():
        name, item = prefix + key, value[key]
        if isinstance(expected, dict):
            if not isinstance(item, dict):
                raise ConfigError(f"{name} must be an object")
            _validate(item, expected, name + ".")
        else:
            allowed = expected if isinstance(expected, tuple) else (expected,)
            if type(item) not in allowed:  # bool is not accepted as an integer
                raise ConfigError(f"{name}: invalid type {type(item).__name__}")
            if isinstance(item, str) and not item.strip():
                raise ConfigError(f"{name} must not be empty")


def apply_overrides(data: dict, overrides: list[str]) -> dict:
    result = copy.deepcopy(data)
    for override in overrides:
        key, sep, raw = override.partition("=")
        if not sep:
            raise ConfigError("Use --set section.key=value")
        target = result
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ConfigError(f"Unknown config key: {key}")
            target = target[part]
        if parts[-1] not in target or isinstance(target[parts[-1]], dict):
            raise ConfigError(f"Unknown or non-leaf config key: {key}")
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw
        target[parts[-1]] = value
    return result


@dataclass(frozen=True)
class ExperimentConfig:
    source: Path
    root: Path
    data: dict[str, Any]

    def path(self, section: str, key: str) -> Path:
        path = Path(os.path.expandvars(self.data[section][key])).expanduser()
        return (path if path.is_absolute() else self.root / path).resolve()

    def snapshot(self) -> dict:
        data = copy.deepcopy(self.data)
        data["project_root"] = str(self.root)
        for section, key in (("dataset", "root"), ("train", "output_dir"),
                             ("rollout", "policy_path"), ("logging", "root")):
            data[section][key] = str(self.path(section, key))
        return data

    def validate_stage(self, stage: str) -> None:
        if stage in {"record", "rollout"}:
            if self.data["hardware"]["use_degrees"] is None:
                raise ConfigError("Set hardware.use_degrees to true or false after confirming joint units; see docs/setup.md")
            if self.data["task"] is None:
                raise ConfigError("Set task to describe your actual pick-and-place task")


def load_config(path: Path = DEFAULT_CONFIG, overrides: list[str] | None = None) -> ExperimentConfig:
    source = path.expanduser().resolve()
    data = apply_overrides(read_json(source), overrides or [])
    _validate(data, SCHEMA)
    if data["schema_version"] != 1:
        raise ConfigError("Only schema_version=1 is supported")
    for section, keys in {
        "hardware": ["width", "height", "fps"],
        "record": ["episodes", "episode_seconds", "reset_seconds"],
        "train": ["steps", "batch_size", "save_freq"],
        "rollout": ["duration_seconds"],
    }.items():
        for key in keys:
            if data[section][key] <= 0:
                raise ConfigError(f"{section}.{key} must be positive")
    for key in ("wrist_camera", "front_camera"):
        if data["hardware"][key] < 0:
            raise ConfigError(f"hardware.{key} must be nonnegative")
    if data["hardware"]["wrist_camera"] == data["hardware"]["front_camera"]:
        raise ConfigError("Wrist and front cameras must have different indices")
    if data["hardware"]["follower_port"].casefold() == data["hardware"]["leader_port"].casefold():
        raise ConfigError("Leader and follower must have different serial ports")
    if data["train"]["num_workers"] < 0:
        raise ConfigError("train.num_workers must be nonnegative")
    for section in ("train", "rollout"):
        if data[section]["device"] not in {"cuda", "cpu"}:
            raise ConfigError(f"{section}.device must be cuda or cpu")
    if data["train"]["device"] == "cpu" and data["train"]["use_amp"]:
        raise ConfigError("Set train.use_amp=false when using CPU")
    repo = data["dataset"]["repo_id"].split("/")
    if len(repo) != 2 or not all(repo):
        raise ConfigError("dataset.repo_id must be owner/name")
    root = Path(os.path.expandvars(data["project_root"])).expanduser()
    root = (root if root.is_absolute() else source.parent / root).resolve()
    return ExperimentConfig(source, root, data)
