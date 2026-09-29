# Python experiment manager

The project provides a Python layer for configuration, preflight checks, execution
logs and verified model downloads. LeRobot still supplies ACT training, camera and
motor drivers, calibration and policy inference. The original PowerShell scripts
remain available and unchanged; they do not read the Python JSON configuration.

## Install and configure

Use the pinned LeRobot environment in [setup.md](setup.md). The manager itself and
its offline tests use Python's standard library; downloading a model additionally
uses `huggingface-hub`, already required by that LeRobot environment. Run from the
repository root with the activated Conda Python:

```powershell
python main.py --help
python main.py init-config
python main.py check
```

Edit `configs/local.json` (ignored by Git). Set `task` to describe your object and
destination. Set `hardware.use_degrees` to a **confirmed** `true` or `false`; `null`
intentionally blocks record/rollout. Select your own COM ports, calibration IDs,
camera indices, dataset location and checkpoint path. See the hardware section of
[setup.md](setup.md) for how to establish these settings.

Configuration paths resolve relative to `project_root`, which resolves relative to
the configuration file's directory, **not the shell's current directory**.
`init-config` sets this relationship for you. Absolute paths and environment
variables are also supported. Misspelled/unknown keys, incorrect types, invalid
numeric ranges and duplicate camera indices/ports are rejected. JSON booleans must
be lowercase `true`/`false`, not quoted strings.

To override individual values without modifying a file:

```powershell
python main.py train --config configs/local.json --set train.batch_size=4 --set train.steps=1000 --dry-run
python main.py train --config configs/local.json --set 'dataset.root=D:\robot data\so101' --dry-run
```

`--set` accepts `section.key=value` and can be repeated. Values are parsed as JSON
where possible, otherwise as text. These examples change the experiment; the
default training configuration retains the documented 120k steps and batch size 8.

## Record, train, deploy

After completing hardware configuration/calibration:

```powershell
python main.py check --stage record --config configs/local.json
python main.py record --config configs/local.json --dry-run
python main.py record --config configs/local.json

python main.py check --stage train --config configs/local.json
python main.py train --config configs/local.json --dry-run
python main.py train --config configs/local.json

python main.py check --stage rollout --config configs/local.json
python main.py rollout --config configs/local.json --dry-run
python main.py rollout --config configs/local.json
```

The state transitions matter: recording requires a **new** dataset directory;
training requires a **complete existing** dataset and a **new** output directory;
rollout requires a complete checkpoint. `check --stage all` reports every stage's
current readiness, so all stages will not necessarily be ready simultaneously.
This release starts fresh recordings/training runs and does not implement resume.

`--dry-run` validates configuration and prints the exact argument list. It does not
require LeRobot/CUDA/data to exist, run preflight filesystem checks, create logs,
open hardware or execute a child process. It is a preview, not proof of readiness.
`check --stage train --files-only` checks local files without probing LeRobot/CUDA.
Actual record/train/rollout always run their preflight checks.

Preflight verifies the installed editable source's Git HEAD against the pinned
revision. It probes CUDA in a subprocess for CUDA training/rollout. Dataset checks
cover v3.0 metadata, six-dimensional state/action, camera keys, image dimensions,
FPS and presence of episode/data/video files. They do not decode every frame or
prove demonstrations are high quality. Checkpoint checks cover ACT metadata,
feature shapes, safetensors headers/data bounds and referenced processor state
files. Checks never open cameras or serial ports; camera images, calibration and
joint-unit compatibility remain manual checks. HEAD verification does not prove
the source working tree is unmodified; keep your pinned checkout clean.

## Logs and failure handling

Each actual execution creates a unique directory under `runs/`:

```text
runs/20260929T...Z_train_<id>/
  config.json    # resolved configuration, including command-line overrides
  run.json       # argument list, interpreter/packages/source revision, timestamps, status, exit code
  console.log    # combined LeRobot stdout/stderr, also streamed to the terminal
```

The child runs as `sys.executable -m lerobot.scripts.lerobot_<stage>`, using the same
Python environment as the manager. Arguments are passed as a list with
`shell=False`, preserving spaces/quotes without shell interpolation. LeRobot keeps
interactive stdin. No token/environment-variable dump is saved in the run record;
logs and resolved paths may still contain local information, so review them before
sharing. Run directories are ignored by Git.

The manager propagates the child's exit code. Ctrl+C records an interrupted run
and allows a short shutdown grace period before terminating a child that does not
exit. It is not a hardware emergency stop. Rollout uses a finite configured
duration and disables automatic return-to-initial-position motion, matching the
PowerShell wrapper. No success detection or autonomous object reset is implemented.

## Published checkpoint

See [model provenance and download instructions](model.md). Downloading and
verifying a model do not move hardware:

```powershell
python main.py model-info
python main.py download-model
python main.py verify-model
```

Use `--output` to select another local model directory. Downloads use the immutable
Hub revision and file hashes in `models/pretrained.json`, not a mutable `main`
branch. Files are staged in a sibling `.partial-*` directory and verified before
the final directory appears. An existing complete destination is verified/reused;
an existing invalid destination is rejected rather than overwritten. Failed
downloads retain their partial directory for inspection. Public downloads do not
use your login token.

## Development and tests

```powershell
python -m unittest discover -s tests -v
```

Tests use synthetic files and short child processes. They exercise config errors,
path independence, command argument boundaries, missing/truncated processor files,
checksum corruption, logging, exit codes and dry-run side effects. GitHub Actions
runs these tests on Windows and Linux with Python 3.12, without LeRobot, GPU or
robot hardware. Linux CI validates the manager, not a Linux hardware deployment.

The Python implementation is deliberately a small orchestration layer. It does not
copy LeRobot's algorithm implementation or make claims of a new ACT architecture.
