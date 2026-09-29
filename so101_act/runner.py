"""Run LeRobot without a shell; persist configuration, provenance and output."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .checks import environment_info
from .config import ExperimentConfig


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_experiment(stage: str, config: ExperimentConfig, command: list[str]) -> int:
    logroot = config.path("logging", "root")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + stage + "_" + uuid.uuid4().hex[:8]
    folder = logroot / run_id
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "config.json", config.snapshot())
    metadata = {
        "stage": stage, "started_at": _now(), "status": "starting",
        "command": command, "cwd": str(config.root), "environment": environment_info(),
    }
    write_json(folder / "run.json", metadata)
    print(f"Experiment log: {folder}", flush=True)
    process = None
    code = 1
    env = os.environ.copy()
    env.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    try:
        with (folder / "console.log").open("wb") as log:
            process = subprocess.Popen(
                command, cwd=config.root, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                # stdin stays attached for LeRobot's interactive prompts.
                shell=False,
            )
            metadata["status"] = "running"
            metadata["pid"] = process.pid
            write_json(folder / "run.json", metadata)
            while chunk := process.stdout.read1(4096):
                log.write(chunk)
                log.flush()
                if hasattr(sys.stdout, "buffer"):
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                else:
                    sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                    sys.stdout.flush()
            code = process.wait()
            metadata["status"] = "completed" if code == 0 else "failed"
    except KeyboardInterrupt:
        metadata["status"] = "interrupted"
        code = 130
        if process is not None and process.poll() is None:
            # Ctrl+C reaches the child in the same console/process group first.
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    except OSError as exc:
        metadata.update(status="failed", error=str(exc))
        print(f"Could not run experiment: {exc}", file=sys.stderr)
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
    finally:
        if process is not None and process.stdout is not None:
            process.stdout.close()
        metadata.update(finished_at=_now(), exit_code=code)
        write_json(folder / "run.json", metadata)
    return code
