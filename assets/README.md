# Demo media

- `evaluation-full.mp4`: byte-for-byte copy of `VID_20260902_214315.mp4`; no trimming, transcoding, speed changes, or audio removal. All 9 consecutive attempts are retained: 8 successful autonomous pick-and-place cycles followed by a failure on attempt 9. Objects are manually reset between autonomous cycles.
- `demo-3-cycles-1.5x.gif`: continuous source interval 00:06–00:45, showing the first 3 successful cycles including object resets, played at 1.5× speed; loops indefinitely and has no audio.

| Property | Full evaluation | README preview |
| --- | --- | --- |
| Duration | approximately 123.53 s | 26.00 s |
| Dimensions | 720 × 406 | 480 × 271 |
| Frame rate | approximately 30 fps | 10 fps |
| Encoding | H.264 High, yuv420p | GIF, 128-color palette |
| Audio | AAC LC, 48 kHz, mono | none |
| Size (bytes) | 19,105,691 | 9,076,837 |

The GIF uses Lanczos resizing, a palette generated from frame differences, and no dithering to keep its size reasonable. The original MP4 provides higher visual fidelity and the complete evaluation evidence. This single rollout is not a statistical success-rate estimate.

## Reproduce the preview

Generated with FFmpeg 7.1:

```sh
ffmpeg -ss 6 -t 39 -i evaluation-full.mp4 -filter_complex "[0:v]setpts=(PTS-STARTPTS)/1.5,fps=10,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle" -an -loop 0 demo-3-cycles-1.5x.gif
```

Original/full-video SHA-256: `06f7798a0c9b3cf722b7782a70c06b8fe4dd41292258040e323e7dfc58bb2fe9`
