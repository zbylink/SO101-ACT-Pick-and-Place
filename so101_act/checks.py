"""Read-only preflight checks. No robot/camera connections or model inference."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import ConfigError, ExperimentConfig, LEROBOT_REVISION, read_json


@dataclass(frozen=True)
class Check:
    level: str
    message: str


def environment_info() -> dict:
    info = {"python": sys.version, "executable": sys.executable, "packages": {}}
    for package in ("lerobot", "torch", "torchvision", "huggingface-hub", "safetensors", "av"):
        try:
            info["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            info["packages"][package] = None
    spec = importlib.util.find_spec("lerobot")
    if spec and spec.origin:
        source = Path(spec.origin).resolve()
        info["lerobot_source"] = str(source)
        for root in source.parents:
            if (root / ".git").exists():
                try:
                    result = subprocess.run(
                        ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), "rev-parse", "HEAD"],
                        capture_output=True, text=True, timeout=10, check=False,
                    )
                    if result.returncode == 0:
                        info["lerobot_revision"] = result.stdout.strip()
                except (OSError, subprocess.TimeoutExpired):
                    pass
                break
    return info


def check_environment(require_cuda: bool = False) -> list[Check]:
    info = environment_info()
    results = []
    if sys.version_info < (3, 12):
        results.append(Check("error", "The pinned LeRobot environment requires Python 3.12+"))
    if not info["packages"]["lerobot"]:
        results.append(Check("error", "LeRobot is not installed in this Python. Follow docs/setup.md"))
    elif info.get("lerobot_revision") != LEROBOT_REVISION:
        results.append(Check("error", f"Cannot verify the supported LeRobot revision {LEROBOT_REVISION}; install the pinned editable checkout"))
    else:
        results.append(Check("ok", f"LeRobot source revision verified: {LEROBOT_REVISION}"))
    if require_cuda:
        try:
            result = subprocess.run(
                [sys.executable, "-c", "import torch; print('SO101_CUDA=' + str(torch.cuda.is_available()))"],
                capture_output=True, text=True, timeout=45, check=False,
            )
            available = result.returncode == 0 and "SO101_CUDA=True" in result.stdout.splitlines()
            results.append(Check("ok" if available else "error", "CUDA available" if available else "CUDA unavailable in the active Python environment"))
        except (OSError, subprocess.TimeoutExpired) as exc:
            results.append(Check("error", f"CUDA probe failed: {exc}"))
    return results


def _features(features: dict, hardware: dict, policy: bool = False) -> None:
    for name in ("wrist", "front"):
        key = f"observation.images.{name}"
        shape = [3, hardware["height"], hardware["width"]] if policy else [hardware["height"], hardware["width"], 3]
        if features.get(key, {}).get("shape") != shape:
            raise ConfigError(f"{key} must have shape {shape}")
    if features.get("observation.state", {}).get("shape") != [6]:
        raise ConfigError("observation.state must have shape [6]")


def validate_dataset(root: Path, hardware: dict) -> dict:
    info = read_json(root / "meta" / "info.json")
    if info.get("codebase_version") != "v3.0":
        raise ConfigError("Expected a LeRobot v3.0 dataset; convert other formats using the pinned source")
    if type(info.get("total_episodes")) is not int or info["total_episodes"] <= 0:
        raise ConfigError("Dataset has no episodes")
    if info.get("fps") != hardware["fps"]:
        raise ConfigError(f"Dataset FPS {info.get('fps')} does not match hardware.fps={hardware['fps']}")
    _features(info.get("features", {}), hardware)
    if info.get("features", {}).get("action", {}).get("shape") != [6]:
        raise ConfigError("Dataset action must have shape [6]")
    for folder, pattern in (("data", "*.parquet"), ("meta/episodes", "*.parquet")):
        if not any((root / folder).rglob(pattern)):
            raise ConfigError(f"Missing {folder} Parquet files")
    for name in ("wrist", "front"):
        if not any((root / "videos" / f"observation.images.{name}").rglob("*.mp4")):
            raise ConfigError(f"Missing {name} camera videos")
    return info


def validate_safetensors(path: Path) -> None:
    """Check header and data bounds without loading 200 MB of tensors into RAM."""
    try:
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("missing header")
            size = struct.unpack("<Q", prefix)[0]
            if not 2 <= size <= 100_000_000:
                raise ValueError("invalid header length")
            header = json.loads(stream.read(size))
        tensors = [v for k, v in header.items() if k != "__metadata__"]
        if not tensors:
            raise ValueError("no tensors")
        payload = path.stat().st_size - 8 - size
        for tensor in tensors:
            start, end = tensor["data_offsets"]
            if not (0 <= start <= end <= payload):
                raise ValueError("truncated or invalid tensor payload")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ConfigError(f"Invalid safetensors file {path}: {exc}") from exc


def validate_checkpoint(root: Path, hardware: dict | None = None) -> dict:
    config = read_json(root / "config.json")
    if config.get("type") != "act":
        raise ConfigError("Checkpoint must contain an ACT policy")
    if hardware:
        _features(config.get("input_features", {}), hardware, policy=True)
    if config.get("output_features", {}).get("action", {}).get("shape") != [6]:
        raise ConfigError("Checkpoint action must have shape [6]")
    validate_safetensors(root / "model.safetensors")
    for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
        processor = read_json(root / filename)
        steps = processor.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ConfigError(f"{filename} has no processor steps")
        for step in steps:
            if not isinstance(step, dict):
                raise ConfigError(f"Invalid processor step in {filename}")
            if "state_file" in step:
                if not isinstance(step["state_file"], str) or not step["state_file"]:
                    raise ConfigError(f"Invalid processor state filename in {filename}")
                target = (root / step["state_file"]).resolve()
                if not target.is_relative_to(root.resolve()):
                    raise ConfigError(f"Processor state file escapes checkpoint directory: {target}")
                validate_safetensors(target)
    return config


def preflight(stage: str, config: ExperimentConfig, runtime: bool = True) -> list[Check]:
    results = []
    if runtime:
        device = config.data.get(stage, {}).get("device")
        results.extend(check_environment(require_cuda=device == "cuda"))
    try:
        config.validate_stage(stage)
        dataset = config.path("dataset", "root")
        output = config.path("train", "output_dir")
        logroot = config.path("logging", "root")
        policy = config.path("rollout", "policy_path")
        if dataset.is_relative_to(output) or output.is_relative_to(dataset):
            raise ConfigError("Dataset and training output directories must not overlap")
        if any(logroot.is_relative_to(p) or p.is_relative_to(logroot) for p in (dataset, output, policy)):
            raise ConfigError("Logging directory must be separate from datasets, training outputs and checkpoints")
        if stage == "record":
            if dataset.exists():
                raise ConfigError("Dataset root exists. Select a fresh path; recording will not overwrite or resume it")
            results.append(Check("warning", "Ports, camera images and calibration require a manual check; no hardware was opened"))
        elif stage == "train":
            info = validate_dataset(dataset, config.data["hardware"])
            results.append(Check("ok", f"Dataset structure: {info['total_episodes']} episodes at {info['fps']} FPS (not a full frame/decode audit)"))
            if info["total_episodes"] != 40:
                results.append(Check("warning", "Dataset episode count differs from the documented 40-episode experiment"))
            if output.exists():
                raise ConfigError("Training output already exists. Choose a new directory; automatic resume is not supported")
        elif stage == "rollout":
            validate_checkpoint(policy, config.data["hardware"])
            results.append(Check("ok", "ACT checkpoint, processor state files and feature dimensions checked"))
            results.append(Check("warning", "Checkpoint metadata cannot establish historical joint units or calibration; verify them before motion"))
    except (ConfigError, OSError, TypeError) as exc:
        results.append(Check("error", str(exc)))
    return results
