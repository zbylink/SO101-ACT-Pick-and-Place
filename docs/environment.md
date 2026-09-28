# Environment provenance

Use the pinned source installation commands in [setup.md](setup.md). The source `pyproject.toml` is the dependency specification; install its `core_scripts`, `training` and `feetech` extras. ACT is available in this source without an `act` extra.

The environment observed during documentation was Windows, Anaconda Python 3.12.13, LeRobot 0.5.2 from JoyandAI/lerobot commit `eacddcb9cff5e033c7811daa15d30f5debcc9a7b`, torch 2.10.0+cu126 and torchvision 0.25.0+cu126. This is a source pin and a partial environment record, not a historical full lockfile. Dependencies within upstream version ranges may resolve differently over time. No clean environment installation was performed as part of this change.

After a successful installation, capture your resolved environment alongside your experiment:

```powershell
python -m pip check
python -m pip freeze > environment-resolved.txt
conda env export --no-builds > environment-resolved.yml
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Review exports before sharing: editable installation paths and the Conda prefix can contain personal paths. Keep LeRobot's commit, recording arguments (especially joint units and cameras), calibration provenance, dataset metadata, training config and the full checkpoint with the experiment. Do not upgrade LeRobot between recording, training and rollout without checking compatibility.
