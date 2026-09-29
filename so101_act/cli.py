"""Command-line interface for reproducible SO-101 experiments."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .checks import check_environment, preflight
from .commands import build_command
from .config import ConfigError, DEFAULT_CONFIG, PROJECT_ROOT, load_config, read_json
from .models import DEFAULT_MANIFEST, DEFAULT_MODEL_DIR, download_model, load_manifest, verify_model
from .runner import run_experiment


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="SO-101 + ACT experiment manager (LeRobot backend)")
    sub = root.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init-config", help="Create an editable local configuration")
    init.add_argument("--output", type=Path, default=PROJECT_ROOT / "configs" / "local.json")
    for name in ("check", "record", "train", "rollout"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        command.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                             help="Override a leaf key, e.g. --set train.batch_size=4 (repeatable)")
        if name == "check":
            command.add_argument("--stage", choices=("environment", "record", "train", "rollout", "all"), default="environment")
            command.add_argument("--files-only", action="store_true", help="Skip environment/CUDA probes; never opens hardware")
        else:
            command.add_argument("--dry-run", action="store_true", help="Print argument list; do not run LeRobot or write logs")
    for name in ("model-info", "download-model", "verify-model"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        if name != "model-info":
            command.add_argument("--output", type=Path, default=DEFAULT_MODEL_DIR, help="Local pretrained_model directory")
    return root


def _report(checks) -> bool:
    for check in checks:
        print(f"[{check.level.upper()}] {check.message}")
    return not any(check.level == "error" for check in checks)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "init-config":
            target = args.output.expanduser().resolve()
            data = read_json(DEFAULT_CONFIG)
            try:
                data["project_root"] = os.path.relpath(PROJECT_ROOT, target.parent)
            except ValueError:  # A config and checkout can be on different Windows drives.
                data["project_root"] = str(PROJECT_ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8") as stream:
                json.dump(data, stream, indent=2)
                stream.write("\n")
            print(f"Created {target}. Set task, hardware.use_degrees and your hardware/path values.")
            return 0
        if args.command in {"model-info", "download-model", "verify-model"}:
            manifest = load_manifest(args.manifest)
            if args.command == "model-info":
                print(json.dumps(manifest, indent=2))
            else:
                folder = args.output.expanduser().resolve()
                if args.command == "download-model":
                    download_model(folder, manifest)
                else:
                    verify_model(folder, manifest)
                print(f"Verified complete model bundle: {folder}")
            return 0
        config = load_config(args.config, args.set)
        if args.command == "check":
            if args.stage == "environment":
                if args.files_only:
                    raise ConfigError("--files-only requires --stage record, train, rollout or all")
                return 0 if _report(check_environment(require_cuda=True)) else 1
            stages = ("record", "train", "rollout") if args.stage == "all" else (args.stage,)
            checks = []
            if not args.files_only:
                checks += check_environment(require_cuda=any(config.data.get(s, {}).get("device") == "cuda" for s in stages))
            for stage in stages:
                checks += preflight(stage, config, runtime=False)
            return 0 if _report(checks) else 1
        command = build_command(args.command, config)
        if args.dry_run:
            print(json.dumps({"stage": args.command, "cwd": str(config.root), "argv": command}, indent=2))
            return 0
        if not _report(preflight(args.command, config)):
            return 1
        return run_experiment(args.command, config, command)
    except (ConfigError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
