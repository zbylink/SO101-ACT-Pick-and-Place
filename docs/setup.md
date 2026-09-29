# Reproduce the SO-101 + ACT workflow

Run these instructions in **Anaconda PowerShell on Windows**. The scripts wrap LeRobot; this repository does not duplicate its training or robot-control implementation. For the Python configuration manager, preflight checks and logging, see [python.md](python.md). The PowerShell path below remains supported. They do not upload datasets or checkpoints.

## 1. Version and environment

The author's local editable installation was inspected while preparing these scripts:

| Component | Observed value |
| --- | --- |
| LeRobot source | [JoyandAI/lerobot](https://github.com/JoyandAI/lerobot), a fork of Hugging Face LeRobot |
| Source revision | `eacddcb9cff5e033c7811daa15d30f5debcc9a7b` |
| Package version | `0.5.2` |
| Python | `3.12.13` |
| PyTorch / torchvision | `2.10.0+cu126` / `0.25.0+cu126` |
| Dataset format | LeRobot `v3.0` |

This is the **currently inspected environment**, not proof of the exact source revision or every dependency installed at the time of the video. Relevant CLI/config files had no tracked changes against that revision. The original training command is preserved in the README and matches the saved `train_config.json`. No complete historical dependency lock was available. See [environment notes](environment.md).

Use the pinned fork for these scripts. Do not substitute an arbitrary PyPI release: this fork has separate `lerobot-record` and `lerobot-rollout` commands, timestamped recording IDs, and hardware defaults that can differ from upstream.

```powershell
git clone https://github.com/zbylink/SO101-ACT-Pick-and-Place.git
git clone https://github.com/JoyandAI/lerobot.git lerobot-source
git -C lerobot-source checkout eacddcb9cff5e033c7811daa15d30f5debcc9a7b
conda create -n so101-act python=3.12.13 -y
conda activate so101-act
$env:PYTHONIOENCODING = 'utf-8'
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e './lerobot-source[core_scripts,training,feetech]'
python -m pip check
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
lerobot-record --help
lerobot-train --help
lerobot-rollout --help
cd SO101-ACT-Pick-and-Place
```

CUDA must report `True`; install a compatible NVIDIA driver before training. The UTF-8 setting avoids a GBK encoding error observed when redirecting the training CLI help on Chinese Windows. The pinned source selects PyAV for Windows video decoding; its dataset extra installs PyAV. Do not replace this with Linux-only TorchCodec installation instructions. If PowerShell blocks local scripts, review them and use `Set-ExecutionPolicy -Scope Process Bypass` for this session only.

## 2. Hardware setup and calibration

Assemble and configure the motors before collection, following the [SO-101 hardware guide](https://huggingface.co/docs/lerobot/so101). Connect and power the arms, keep the workspace clear, and have the power disconnect accessible. These scripts can move real hardware; the documentation checks did not execute motion.

| Setting | Confirmed experiment value |
| --- | --- |
| Follower | `so101_follower`, ID `myfollower01`, port `COM7` |
| Leader | `so101_leader`, ID `myleader01`, port `COM8` |
| Wrist / front camera | OpenCV index `0` / `1` |
| Camera image size / dataset FPS | `640 x 480` / `30` (verified in dataset metadata) |

Find your own ports with `lerobot-find-port` or Windows Device Manager. Use `lerobot-find-cameras opencv` to identify cameras; verify the actual wrist/front images, not only their numeric indices. Keep camera names **wrist** and **front**, image orientation, placement and lighting consistent between recording and rollout. Backend/FourCC were not recorded as historical experiment settings: the scripts use the pinned implementation's defaults. If capture fails, inspect `lerobot-record --help` and the [OpenCV config](https://github.com/JoyandAI/lerobot/blob/eacddcb9cff5e033c7811daa15d30f5debcc9a7b/src/lerobot/cameras/opencv/configuration_opencv.py) before changing the camera dictionary; do not guess a camera format.

**Joint units require an explicit choice.** `-UseDegrees true` uses degrees for arm joints; `false` uses normalized arm positions. The gripper has its own normalization. The historical value was not established from the saved training config, so neither script silently selects one. For an existing dataset/checkpoint, inspect the original record/rollout logs or command for `robot.use_degrees` and `teleop.use_degrees`, and inspect the corresponding source defaults if the flags were omitted. Calibration files alone do not establish this choice. For a new dataset choose a mode and keep it identical across leader, follower, recording and rollout. Do not deploy an old checkpoint until its units are known.

```powershell
$units = Read-Host 'Confirmed joint units: enter true for degrees or false for normalized positions'
lerobot-calibrate --robot.type=so101_follower --robot.port=COM7 --robot.id=myfollower01 --robot.use_degrees=$units
lerobot-calibrate --teleop.type=so101_leader --teleop.port=COM8 --teleop.id=myleader01 --teleop.use_degrees=$units
```

Use your own ports/IDs if different. Complete the interactive calibration procedure for both arms; keep the IDs and calibration files for subsequent sessions. Check leader/follower teleoperation with the pinned version's `lerobot-teleoperate --help` before recording. Calibration is machine/hardware specific and is not distributed here.

## 3. Record demonstrations

From this repository root, set a task description that matches your actual object and destination. The exact historical task text and episode/reset time limits were not recovered; 60-second recording/reset windows below are adjustable operational defaults, not claims about the original run.

```powershell
$task = Read-Host 'Describe the object to pick up and where to place it'
$data = Join-Path $PWD 'data/so101_test'
.\scripts\record.ps1 -Task $task -UseDegrees $units -DatasetRoot $data -DryRun
# After reviewing ports, camera indices, units and calibration:
.\scripts\record.ps1 -Task $task -UseDegrees $units -DatasetRoot $data
```

The default collects 40 episodes at 30 FPS with H.264 video and local storage. Use `-RepoId 'YOUR_HF_USERNAME/so101_test'`, `-FollowerPort`, `-LeaderPort`, `-FollowerId`, `-LeaderId`, `-WristCamera`, `-FrontCamera`, `-Episodes`, `-EpisodeSeconds` and `-ResetSeconds` as needed. The root directory must not already exist. This wrapper deliberately starts a new dataset rather than overwriting/resuming one.

The pinned fork appends a timestamp to the dataset repo ID internally. With an explicit `-DatasetRoot`, files stay at that path; preserve the stamped ID printed in the recording log if sharing/resuming the dataset later. `train.ps1` accepts `-RepoId` to match it. Follow the recording terminal's keyboard prompts to finish/retry episodes and reset the object during the reset interval.

Before training, inspect `meta/info.json`: confirm 40 usable episodes, 30 FPS, both camera keys and expected image dimensions. The original local dataset contains 18,828 frames. A new recording need not have the same frame count.

## 4. Train ACT

```powershell
.\scripts\train.ps1 -DatasetRoot $data -DryRun
.\scripts\train.ps1 -DatasetRoot $data
```

Defaults: ACT, 120,000 steps, batch size 8, checkpoint every 10,000 steps, CUDA, AMP enabled, W&B disabled, Hub upload disabled. Output: `outputs/train/act_so101_test_120k`. `-Steps`, `-BatchSize`, `-SaveFreq`, `-RepoId`, `-DatasetRoot` and `-OutputDir` are configurable. Existing output directories are rejected; choose a new one for a fresh run. This is not a resume wrapper.

To use the author's existing data instead of collecting new demonstrations:

```powershell
.\scripts\train.ps1 `
  -DatasetRoot 'C:\Users\zby\.cache\huggingface\lerobot\BoyuZhao\so101_test_20260902_014234' `
  -OutputDir 'D:\experiments\act_so101_test_120k'
```

The original output was `C:\Users\zby\lerobot\outputs\train\act_so101_test_120k`. The observed throughput was approximately 2.83 steps/s, not a performance guarantee. Reduced batch sizes/steps are new experiments, not the documented training run.

**Artifact availability:** `BoyuZhao/so101_test` identifies the experiment; public download availability was not verified. This Git repository contains demo media and experiment code, not the full training dataset. The complete final trained checkpoint is now hosted separately on Hugging Face; see [model.md](model.md). To reproduce, record your own data or obtain a complete authorized copy of the dataset. Preserve `meta/`, `data/`, and `videos/` together. The scripts intentionally require a local dataset rather than assuming this ID can be downloaded.

## 5. Run the trained policy

```powershell
.\scripts\rollout.ps1 -Task $task -UseDegrees $units -DryRun
.\scripts\rollout.ps1 -Task $task -UseDegrees $units -DurationSeconds 30
```

The default loads `outputs/train/act_so101_test_120k/checkpoints/last/pretrained_model`. Override with `-PolicyPath` for another **complete local pretrained_model directory**, including weights, configuration and processor files. Do not pass only `model.safetensors`. Use the same units, follower ID/calibration, camera names and scene as training. The leader is not required for autonomous execution.

This uses `lerobot-rollout --strategy.type=base --inference.type=sync`, CUDA, 30 Hz, and a finite 30-second duration by default. It does not record an evaluation dataset or upload anything. It explicitly disables the fork's automatic return-to-initial-position motion at shutdown. The duration and shutdown choice are new wrapper defaults, not recovered evaluation settings. Ctrl+C requests shutdown; keep physical power cutoff accessible. Reset objects only after motion has stopped, then start the next cycle. Use an external continuous video to document repeated attempts if desired.

The published video shows **8 consecutive successful autonomous pick-and-place cycles followed by failure on the 9th attempt**, with manual object resets between cycles. It is one continuous demonstration, not a statistical success-rate estimate. These scripts do not implement success detection or automatically reproduce that outcome.

## CLI evidence and validation limits

Arguments were checked against the pinned fork's [record entry point](https://github.com/JoyandAI/lerobot/blob/eacddcb9cff5e033c7811daa15d30f5debcc9a7b/src/lerobot/scripts/lerobot_record.py), [dataset config](https://github.com/JoyandAI/lerobot/blob/eacddcb9cff5e033c7811daa15d30f5debcc9a7b/src/lerobot/configs/dataset.py), [rollout config](https://github.com/JoyandAI/lerobot/blob/eacddcb9cff5e033c7811daa15d30f5debcc9a7b/src/lerobot/rollout/configs.py), and local saved training configuration. PowerShell syntax and dry-run argument construction are checked separately from real hardware. A clean installation, new training run and physical rollout must still be validated on the reproducer's machine.
