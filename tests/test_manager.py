"""Offline tests: no robot, camera, CUDA, LeRobot or network required."""

import contextlib
import io
import json
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from so101_act.checks import preflight, validate_checkpoint, validate_dataset
from so101_act.cli import main
from so101_act.commands import build_command
from so101_act.config import ConfigError, DEFAULT_CONFIG, load_config, read_json
from so101_act.models import download_model, load_manifest, sha256, verify_model
from so101_act.runner import run_experiment


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def tensor_file(path):
    header = json.dumps({"test": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0))


class ManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.config_file = self.root / "settings" / "experiment.json"
        self.data = read_json(DEFAULT_CONFIG)
        self.data.update(project_root="..", task="Pick the test object")
        self.data["hardware"]["use_degrees"] = False
        write_json(self.config_file, self.data)

    def config(self, *overrides):
        return load_config(self.config_file, list(overrides))

    def checkpoint(self):
        folder = self.root / "checkpoint"
        folder.mkdir()
        features = {"observation.state": {"shape": [6]}}
        features.update({f"observation.images.{name}": {"shape": [3, 480, 640]} for name in ("wrist", "front")})
        write_json(folder / "config.json", {"type": "act", "input_features": features, "output_features": {"action": {"shape": [6]}}})
        tensor_file(folder / "model.safetensors")
        tensor_file(folder / "state.safetensors")
        for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
            write_json(folder / name, {"steps": [{"state_file": "state.safetensors"}]})
        return folder

    def dataset(self):
        folder = self.root / "data" / "so101_test"
        features = {"observation.state": {"shape": [6]}, "action": {"shape": [6]}}
        for name in ("wrist", "front"):
            features[f"observation.images.{name}"] = {"shape": [480, 640, 3]}
            path = folder / "videos" / f"observation.images.{name}" / "chunk-000" / "file-000.mp4"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"fixture: structure only")
        write_json(folder / "meta" / "info.json", {"codebase_version": "v3.0", "fps": 30, "total_episodes": 40, "features": features})
        for name in ("data", "meta/episodes"):
            path = folder / name / "chunk-000" / "file-000.parquet"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"fixture: structure only")
        return folder

    def test_paths_follow_config_not_cwd(self):
        config = self.config()
        self.assertEqual(config.root, self.root)
        self.assertEqual(config.path("dataset", "root"), self.root / "data" / "so101_test")

    def test_overrides_are_typed_and_unknown_keys_fail(self):
        self.assertEqual(self.config("train.batch_size=4").data["train"]["batch_size"], 4)
        for override in ("train.batchsize=4", "train.batch_size=true", "train.steps=-1", "train.use_amp=wrong"):
            with self.subTest(override=override), self.assertRaises(ConfigError):
                self.config(override)

    def test_missing_units_blocks_motion_but_not_training_preview(self):
        config = self.config("hardware.use_degrees=null")
        with self.assertRaisesRegex(ConfigError, "joint units"):
            build_command("rollout", config)
        self.assertIn("--steps=120000", build_command("train", config))

    def test_duplicate_cameras_and_ports_fail(self):
        for override in ("hardware.front_camera=0", "hardware.leader_port=com7"):
            with self.assertRaises(ConfigError):
                self.config(override)

    def test_cpu_amp_requires_explicit_change(self):
        with self.assertRaises(ConfigError):
            self.config("train.device=cpu")
        self.config("train.device=cpu", "train.use_amp=false")

    def test_argument_boundaries_and_camera_json(self):
        text = 'Pick "red cube"; $(do not execute)'
        command = build_command("record", self.config("task=" + text, "dataset.root=data with spaces"))
        self.assertIn("--dataset.single_task=" + text, command)
        self.assertIn("--dataset.root=" + str(self.root / "data with spaces"), command)
        cameras = json.loads(next(x.split("=", 1)[1] for x in command if x.startswith("--robot.cameras=")))
        self.assertEqual(cameras["wrist"]["index_or_path"], 0)
        self.assertEqual(cameras["front"]["index_or_path"], 1)
        self.assertEqual(command[:3], [sys.executable, "-m", "lerobot.scripts.lerobot_record"])

    def test_rollout_uses_separate_cli_and_finite_duration(self):
        command = build_command("rollout", self.config("rollout.duration_seconds=12"))
        self.assertIn("lerobot.scripts.lerobot_rollout", command)
        self.assertIn("--duration=12", command)
        self.assertIn("--return_to_initial_position=false", command)
        self.assertFalse(any(arg.startswith("--teleop") for arg in command))

    def test_existing_record_directory_rejected(self):
        self.config().path("dataset", "root").mkdir(parents=True)
        result = preflight("record", self.config(), runtime=False)
        self.assertTrue(any(c.level == "error" and "exists" in c.message for c in result))

    def test_logs_cannot_create_training_output_before_launch(self):
        config = self.config("logging.root=outputs/train/act_so101_test_120k/logs")
        result = preflight("train", config, runtime=False)
        self.assertTrue(any(c.level == "error" and "Logging" in c.message for c in result))

    def test_dataset_catches_missing_video_and_fps_mismatch(self):
        folder = self.dataset()
        validate_dataset(folder, self.data["hardware"])
        hw = dict(self.data["hardware"], fps=25)
        with self.assertRaisesRegex(ConfigError, "FPS"):
            validate_dataset(folder, hw)
        next((folder / "videos" / "observation.images.front").rglob("*.mp4")).unlink()
        with self.assertRaisesRegex(ConfigError, "front camera"):
            validate_dataset(folder, self.data["hardware"])

    def test_checkpoint_catches_missing_normalizer_and_truncated_weights(self):
        folder = self.checkpoint()
        validate_checkpoint(folder, self.data["hardware"])
        (folder / "state.safetensors").unlink()
        with self.assertRaises(ConfigError):
            validate_checkpoint(folder)
        tensor_file(folder / "state.safetensors")
        weights = folder / "model.safetensors"
        weights.write_bytes(weights.read_bytes()[:-1])
        with self.assertRaisesRegex(ConfigError, "truncated"):
            validate_checkpoint(folder)

    def test_checkpoint_rejects_processor_path_escape(self):
        folder = self.checkpoint()
        write_json(folder / "policy_preprocessor.json", {"steps": [{"state_file": "../external.safetensors"}]})
        with self.assertRaisesRegex(ConfigError, "escapes"):
            validate_checkpoint(folder)

    def test_model_checksums_catch_same_size_corruption(self):
        folder = self.checkpoint()
        manifest = {"files": {p.name: {"size": p.stat().st_size, "sha256": sha256(p)} for p in folder.iterdir()}}
        verify_model(folder, manifest)
        weights = folder / "model.safetensors"
        content = bytearray(weights.read_bytes())
        content[-1] ^= 1
        weights.write_bytes(content)
        with self.assertRaisesRegex(ConfigError, "SHA-256 mismatch"):
            verify_model(folder, manifest)

    def test_manifest_rejects_path_traversal(self):
        path = self.root / "manifest.json"
        write_json(path, {"schema_version": 1, "files": {"../secret": {"size": 1, "sha256": "0" * 64}}})
        with self.assertRaises(ConfigError):
            load_manifest(path)

    def test_download_pins_revision_verifies_and_reuses_existing_bundle(self):
        source = self.checkpoint()
        destination = self.root / "downloaded"
        manifest = {"repo_id": "test/model", "revision": "a" * 40,
                    "files": {p.name: {"size": p.stat().st_size, "sha256": sha256(p)} for p in source.iterdir()}}
        def fake_download(**kwargs):
            self.assertEqual(kwargs["revision"], "a" * 40)
            self.assertIs(kwargs["token"], False)
            for item in source.iterdir():
                shutil.copyfile(item, kwargs["local_dir"] / item.name)
        download = Mock(side_effect=fake_download)
        with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}):
            download_model(destination, manifest)
            download_model(destination, manifest)
        self.assertEqual(download.call_count, 1)
        verify_model(destination, manifest)

    def test_download_never_publishes_corrupt_staging_directory(self):
        destination = self.root / "downloaded"
        manifest = {"repo_id": "test/model", "revision": "a" * 40,
                    "files": {"model.safetensors": {"size": 1, "sha256": "0" * 64}}}
        with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=lambda **kw: None)}):
            with self.assertRaisesRegex(ConfigError, "partial files retained"):
                download_model(destination, manifest)
        self.assertFalse(destination.exists())
        self.assertEqual(len(list(self.root.glob("downloaded.partial-*"))), 1)

    def test_unpublished_model_cannot_download_mutable_main(self):
        with self.assertRaisesRegex(ConfigError, "pinned Hub commit"):
            download_model(self.root / "unused", {"revision": None})

    def test_dry_run_has_no_execution_or_log_side_effects(self):
        with patch("so101_act.cli.run_experiment") as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["train", "--config", str(self.config_file), "--dry-run"]), 0)
        run.assert_not_called()
        self.assertFalse((self.root / "runs").exists())

    def test_runner_logs_output_exit_code_and_resolved_config(self):
        for code in (0, 7):
            with self.subTest(code=code), patch("so101_act.runner.environment_info", return_value={}), contextlib.redirect_stdout(io.StringIO()):
                command = [sys.executable, "-c", f"import sys; print(sys.argv[1]); sys.exit({code})", "literal ; $value with spaces"]
                self.assertEqual(run_experiment("train", self.config(), command), code)
        runs = list((self.root / "runs").iterdir())
        self.assertEqual(len(runs), 2)
        states = [read_json(run / "run.json") for run in runs]
        self.assertEqual({s["status"] for s in states}, {"completed", "failed"})
        self.assertEqual({s["exit_code"] for s in states}, {0, 7})
        for run in runs:
            self.assertIn("literal ; $value with spaces", (run / "console.log").read_text())
            self.assertEqual(read_json(run / "config.json")["project_root"], str(self.root))

    def test_runner_reports_missing_executable(self):
        with patch("so101_act.runner.environment_info", return_value={}), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = run_experiment("train", self.config(), [str(self.root / "missing-executable")])
        self.assertEqual(code, 1)
        run = next((self.root / "runs").iterdir())
        self.assertEqual(read_json(run / "run.json")["status"], "failed")

    def test_interrupted_run_records_status_and_closes_stream(self):
        process = Mock()
        process.pid = 123
        process.stdout.read1.side_effect = KeyboardInterrupt
        process.poll.return_value = None
        process.wait.return_value = 0
        with patch("so101_act.runner.environment_info", return_value={}), patch("so101_act.runner.subprocess.Popen", return_value=process), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_experiment("train", self.config(), ["unused"]), 130)
        record = read_json(next((self.root / "runs").iterdir()) / "run.json")
        self.assertEqual(record["status"], "interrupted")
        process.stdout.close.assert_called_once()

    def test_init_config_never_overwrites(self):
        target = self.root / "new.json"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["init-config", "--output", str(target)]), 0)
            original = target.read_bytes()
            self.assertEqual(main(["init-config", "--output", str(target)]), 1)
        self.assertEqual(target.read_bytes(), original)

    def test_init_config_supports_different_windows_drives(self):
        target = self.root / "other-drive.json"
        with patch("so101_act.cli.os.path.relpath", side_effect=ValueError("different drives")), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["init-config", "--output", str(target)]), 0)
        self.assertTrue(Path(read_json(target)["project_root"]).is_absolute())


if __name__ == "__main__":
    unittest.main()
