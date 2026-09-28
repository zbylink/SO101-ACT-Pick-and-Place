# SO-101 Autonomous Pick-and-Place with ACT

A real-world robotic manipulation demo using the **LeRobot SO-101** and **ACT (Action Chunking with Transformers)** for imitation learning.

> **Result:** The trained policy completed **9 consecutive autonomous pick-and-place cycles** in a continuous real-world rollout.

## Demo

**Full 9-cycle autonomous rollout video — coming next**

The complete continuous rollout video will be added here.

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

### 2. ACT Policy Training

The demonstrations were used to train an ACT policy. ACT predicts chunks of future robot actions rather than a single action at each inference step, making it suitable for continuous manipulation trajectories.

### 3. Real-World Deployment

The trained checkpoint was deployed back onto the physical SO-101. During rollout, the policy receives camera observations and robot state and generates actions for the follower arm.

### 4. Continuous Evaluation

The final demonstration shows **9 successful pick-and-place executions consecutively in one continuous autonomous run**.

This continuous rollout demonstrates repeatability rather than presenting only an isolated successful trial.

## What This Project Demonstrates

- End-to-end robot imitation-learning workflow
- Leader-follower teleoperation and demonstration collection
- Multi-camera visual observations
- ACT policy training with LeRobot and PyTorch
- CUDA-based model training
- Deployment of a learned policy on a physical robot
- Real-world debugging of robot communication, camera pipelines, and inference
- Repeated autonomous manipulation rather than a single successful execution

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

Media and reproducibility files will be added as the project is documented.

## Acknowledgements

This project is built with [Hugging Face LeRobot](https://github.com/huggingface/lerobot) and uses the SO-101 robot platform.

ACT is based on the Action Chunking with Transformers approach for learning fine-grained visuomotor policies from demonstrations.

## License

This repository is released under the [MIT License](LICENSE). Third-party projects and dependencies remain subject to their respective licenses.
