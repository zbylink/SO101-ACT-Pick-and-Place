# Public ACT checkpoint

Model repository: [BoyuZhao/so101-act-pick-and-place](https://huggingface.co/BoyuZhao/so101-act-pick-and-place).

This is the final saved `pretrained_model` bundle from the author's
`act_so101_test_120k` training output: 40 teleoperated episodes, 120,000 steps,
batch size 8, CUDA and AMP. It includes the ACT weights, policy config,
preprocessor/postprocessor definitions and their normalization state files.
The release is about 207 MB and is hosted on Hugging Face, not in ordinary Git
history. The full training dataset and optimizer/resume state are not included.

The manifest at [`models/pretrained.json`](../models/pretrained.json) records the
immutable Hub commit, sizes and SHA-256 hashes. The weights and inference configs
are unchanged from the local checkpoint. The published `train_config.json` replaces
only the original personal dataset/output paths with portable examples. The model
card and MIT license are included in the bundle.

```powershell
python main.py download-model
python main.py verify-model
```

Default destination: `outputs/pretrained_model/so101-act-pick-and-place`.
No Hugging Face login is needed for this public download. To use it after following
the calibration and joint-unit guidance:

```powershell
python main.py check --stage rollout --config configs/local.json --set rollout.policy_path=outputs/pretrained_model/so101-act-pick-and-place
python main.py rollout --config configs/local.json --set rollout.policy_path=outputs/pretrained_model/so101-act-pick-and-place --dry-run
# Remove --dry-run only after hardware/scene/units have been verified.
```

**Historical joint units are still unconfirmed.** A checkpoint's six-dimensional
action shape does not tell you whether the original recording used degrees or
normalized joint positions. Confirm the original `use_degrees` setting before
deploying these weights. Keep wrist/front camera names, viewpoints, image size,
calibration and task setup consistent with the experiment. See [setup.md](setup.md).

The evaluation video documents 8 consecutive successful cycles followed by a
failure on the 9th attempt, with manual resets. It is not a statistical success-rate
estimate. The video has no checkpoint hash proving the exact weights loaded during
filming, so the release is described as the final saved training bundle, not a
hash-verified reconstruction of that video run. No new physical evaluation was
performed as part of publication.
