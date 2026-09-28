# SO-101 Autonomous Pick-and-Place with ACT

A real-world robotic manipulation demo using the **LeRobot SO-101** and **ACT (Action Chunking with Transformers)** for imitation learning.

> **Evaluation result:** The trained policy completed **8 consecutive autonomous pick-and-place cycles successfully**. The **9th attempt failed** and is retained as a real-world failure case.

## Demo

[![Three successful autonomous pick-and-place cycles at 1.5x speed](assets/demo-3-cycles-1.5x.gif)](assets/evaluation-full.mp4)

**Quick preview:** 3 consecutive successful cycles at **1.5× speed** (26 seconds; GIF, approximately 9.1 MB). The excerpt covers 00:06–00:45 of the original recording and retains the manual object resets between autonomous cycles.

**[Watch or download the full continuous evaluation video](assets/evaluation-full.mp4)** — original MP4, approximately 2 min 3.53 s, 19.1 MB, normal speed, with audio.

The full recording shows **8 consecutive successful autonomous pick-and-place cycles followed by a failure on the 9th attempt**. All 9 attempts are retained without cuts or speed changes. This is one continuous demonstration, not a statistical success-rate estimate.

See [media details and preview-generation parameters](assets/README.md).

## Overview

This project implements an end-to-end imitation-learning workflow for autonomous robotic manipulation:

```text
Human teleoperation
        ↓
Demonstration collection
        ↓
LeRobot dataset
        ↓
ACT policy training
        ↓
Real-robot inference
        ↓
Autonomous pick-and-place
```

The goal was not only to train a policy offline, but to deploy it on physical hardware and evaluate repeated autonomous execution.

## System

| Component | Configuration |
| --- | --- |
| Robot | LeRobot SO-101 follower arm |
| Demonstration interface | SO-101 leader arm |
| Policy | ACT (Action Chunking with Transformers) |
| Framework | Hugging Face LeRobot |
| Learning paradigm | Imitation learning |
| Visual observations | Wrist camera + front camera |
| Robot state/action | 6-DoF joint state and action |
| Dataset rate | 30 FPS |
| Training framework | PyTorch |
| Compute | NVIDIA GeForce RTX 4070 Laptop GPU |

## Workflow

### 1. Teleoperation and Data Collection

Pick-and-place demonstrations were collected by teleoperating the SO-101 follower with an SO-101 leader arm. Robot state/action data and synchronized visual observations were recorded through LeRobot.

The training dataset contains **40 teleoperated demonstration episodes**.

### 2. ACT Policy Training

The demonstrations were used to train an ACT policy. ACT predicts chunks of future robot actions rather than a single action at each inference step, making it suitable for continuous manipulation trajectories.

### 3. Real-World Deployment

The trained checkpoint was deployed back onto the physical SO-101. During rollout, the policy receives camera observations and robot state and generates actions for the follower arm.

### 4. Continuous Evaluation

The policy completed **8 consecutive autonomous pick-and-place cycles successfully**. The **9th consecutive attempt failed**.

The failure is intentionally retained in the full evaluation video rather than removed. This provides a more transparent view of both repeated successful execution and a real-world failure case.

## Dataset & Training Configuration

### Dataset

- **40 teleoperated demonstration episodes**
- Dual-camera visual observations: wrist camera + front camera
- 6-DoF robot state/action
- Recording frequency: 30 FPS

### ACT Training

The ACT policy was trained with the following LeRobot configuration:

```powershell
lerobot-train `
  --dataset.repo_id=BoyuZhao/so101_test `
  --dataset.root="C:\Users\zby\.cache\huggingface\lerobot\BoyuZhao\so101_test_20260902_014234" `
  --dataset.streaming=false `
  --policy.type=act `
  --output_dir="C:\Users\zby\lerobot\outputs\train\act_so101_test_120k" `
  --job_name=act_so101_test_120k `
  --policy.device=cuda `
  --wandb.enable=false `
  --policy.push_to_hub=false `
  --steps=120000 `
  --batch_size=8 `
  --save_freq=10000 `
  --policy.use_amp=true
```

### Training Performance

- **Training steps:** 120,000
- **Batch size:** 8
- **Device:** CUDA
- **Automatic Mixed Precision (AMP):** enabled
- **Observed training throughput:** approximately **2.83 steps/s**

The throughput above is the observed training speed on the hardware used for this experiment and may vary across systems.

## What This Project Demonstrates

- End-to-end robot imitation-learning workflow
- Leader-follower teleoperation and demonstration collection
- Multi-camera visual observations
- ACT policy training with LeRobot and PyTorch
- CUDA and mixed-precision model training
- Deployment of a learned policy on a physical robot
- Real-world debugging of robot communication, camera pipelines, and inference
- Continuous evaluation including both successful executions and a retained failure case

## Repository Structure

```text
SO101-ACT-Pick-and-Place/
├── README.md
├── LICENSE
├── .gitignore
├── assets/          # Demo media
├── scripts/         # Reproducible record/train/rollout commands
└── docs/            # Additional setup and experiment notes
```

Demo media and processing details are available in `assets/`. Additional reproducibility files will be added as the project is documented.

## Acknowledgements

This project is built with [Hugging Face LeRobot](https://github.com/huggingface/lerobot) and uses the SO-101 robot platform.

ACT is based on the Action Chunking with Transformers approach for learning fine-grained visuomotor policies from demonstrations.

## License

This repository is released under the [MIT License](LICENSE). Third-party projects and dependencies remain subject to their respective licenses.
