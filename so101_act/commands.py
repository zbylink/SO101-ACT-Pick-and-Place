"""Pure command builders: no imports from LeRobot and no hardware access."""

import json
import sys

from .config import ExperimentConfig


def _flag(name: str, value) -> str:
    if isinstance(value, bool):
        value = str(value).lower()
    return f"--{name}={value}"


def build_command(stage: str, config: ExperimentConfig) -> list[str]:
    if stage not in {"record", "train", "rollout"}:
        raise ValueError(f"Unknown stage: {stage}")
    config.validate_stage(stage)
    d, hw = config.data, config.data["hardware"]
    # The active Python also runs LeRobot, avoiding a CLI from another Conda env.
    command = [sys.executable, "-m", f"lerobot.scripts.lerobot_{stage}"]
    values = {}
    if stage in {"record", "rollout"}:
        cameras = {
            name: {"type": "opencv", "index_or_path": hw[f"{name}_camera"],
                   "width": hw["width"], "height": hw["height"], "fps": hw["fps"]}
            for name in ("wrist", "front")
        }
        values.update({
            "robot.type": "so101_follower", "robot.port": hw["follower_port"],
            "robot.id": hw["follower_id"], "robot.use_degrees": hw["use_degrees"],
            "robot.cameras": json.dumps(cameras), "display_data": False, "play_sounds": False,
        })
    if stage in {"record", "train"}:
        values.update({"dataset.repo_id": d["dataset"]["repo_id"], "dataset.root": config.path("dataset", "root")})
    if stage == "record":
        values.update({
            "teleop.type": "so101_leader", "teleop.port": hw["leader_port"],
            "teleop.id": hw["leader_id"], "teleop.use_degrees": hw["use_degrees"],
            "dataset.single_task": d["task"], "dataset.fps": hw["fps"],
            "dataset.num_episodes": d["record"]["episodes"],
            "dataset.episode_time_s": d["record"]["episode_seconds"],
            "dataset.reset_time_s": d["record"]["reset_seconds"],
            "dataset.vcodec": "h264", "dataset.push_to_hub": False,
        })
    elif stage == "train":
        train = d["train"]
        values.update({
            "dataset.streaming": False, "policy.type": "act", "output_dir": config.path("train", "output_dir"),
            "job_name": train["job_name"], "policy.device": train["device"],
            "wandb.enable": False, "policy.push_to_hub": False, "policy.use_amp": train["use_amp"],
            **{key: train[key] for key in ("steps", "batch_size", "save_freq", "num_workers")},
        })
    else:
        values.update({
            "strategy.type": "base", "inference.type": "sync",
            "policy.path": config.path("rollout", "policy_path"), "device": d["rollout"]["device"],
            "fps": hw["fps"], "duration": d["rollout"]["duration_seconds"], "task": d["task"],
            "return_to_initial_position": False,
        })
    return command + [_flag(key, value) for key, value in values.items()]
